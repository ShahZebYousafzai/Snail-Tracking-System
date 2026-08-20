"""Find likely-mislabeled bead crops via 5-fold cross-validation, flagging
cases where a held-out model confidently disagrees with the recorded digit
label.

IMPORTANT: this does NOT just reuse runs/bead_classifier/best.pt. That model
was trained on ~75% of this data and hit 100% train accuracy (see
training_log.csv) -- checking it against crops it was directly trained on is
useless for finding mislabels, since it will have memorized (and therefore
"agree" with) even a wrong label it saw during training. Instead this trains
K=5 fresh models, each holding out a different 1/5 of the real crops
(grouped the same leakage-safe way as train_bead_classifier.py -- a
synthetic crop always follows its real source's fold), so every real crop
gets a genuine out-of-sample prediction to check its label against.

This takes several minutes (K short training runs) -- it's an offline audit
tool, not something to run on every change.

Only checks digits 0-8 (the model's output space); any crop labeled "9" is
skipped, not flagged as an error, since the model can't have an opinion on a
class it was never trained to predict.

Output is both a human-readable review CSV and a manifest in the same
schema as manifest.csv, so the flagged crops can be re-labeled directly:

    python src/find_label_disagreements.py
    python src/label_bead_crops.py --manifest data/bead_crops/from_video_dedup/label_disagreements_manifest.csv --ignore-out
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bead_classifier import INPUT_SIZE, NUM_DIGIT_CLASSES, TwoHeadBeadNet  # noqa: E402
from train_bead_classifier import IN_SCOPE_DIGITS, BeadDataset, build_label_index, run_epoch  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kfold_assignment(real_paths: set[str], index: dict, k: int, seed: int) -> dict[str, int]:
    """Stratified-by-digit fold assignment for real crops."""
    by_digit: dict[str, list[str]] = defaultdict(list)
    for p in real_paths:
        d = index[p]["digit"]
        if d in IN_SCOPE_DIGITS:
            by_digit[d].append(p)

    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for _digit, paths in by_digit.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        for i, p in enumerate(paths):
            fold_of[p] = i % k
    return fold_of


def split_for_fold(index: dict, real_paths: set[str], fold_of: dict[str, int], held_out: int) -> tuple[list[str], list[str]]:
    train_paths, eval_paths = [], []
    for path, row in index.items():
        if row["digit"] not in IN_SCOPE_DIGITS:
            continue
        if path in real_paths:
            (eval_paths if fold_of[path] == held_out else train_paths).append(path)
        else:
            src = row["source_crop_path"]
            if src not in fold_of:
                continue
            if fold_of[src] == held_out:
                continue  # don't train on a synthetic sibling of a held-out crop -- that's leakage
            train_paths.append(path)
    return train_paths, eval_paths


def train_one_fold(train_paths: list[str], index: dict, device: torch.device,
                    epochs: int, batch_size: int, lr: float, use_class_weights: bool, seed: int) -> TwoHeadBeadNet:
    torch.manual_seed(seed)
    train_ds = BeadDataset(train_paths, index, train=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    class_weights = None
    if use_class_weights:
        counts = Counter(index[p]["digit"] for p in train_paths)
        arr = np.array([counts.get(d, 1) for d in sorted(IN_SCOPE_DIGITS)], dtype=np.float32)
        weights = arr.sum() / (len(arr) * arr)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = TwoHeadBeadNet(num_digit_classes=NUM_DIGIT_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    digit_criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _epoch in range(epochs):
        run_epoch(model, loader, device, optimizer, digit_criterion, rotation_weight=1.0)
    return model


@torch.no_grad()
def predict_all(model: TwoHeadBeadNet, device: torch.device, paths: list[str], index: dict) -> dict[str, tuple[int, float]]:
    model.eval()
    ds = BeadDataset(paths, index, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    results: dict[str, tuple[int, float]] = {}
    ptr = 0
    for images, _digits, _rot, _rot_valid in loader:
        images = images.to(device)
        digit_logits, _rot_pred = model(images)
        probs = F.softmax(digit_logits, dim=1)
        preds = probs.argmax(dim=1)
        for i in range(images.size(0)):
            results[paths[ptr]] = (int(preds[i].item()), float(probs[i, preds[i]].item()))
            ptr += 1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="default: <dir>/manifest.csv (for the relabel-ready output)")
    parser.add_argument("--k", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=25, help="training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false", default=True)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                         help="only flag disagreements at or above this out-of-fold confidence (default: 0, show all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    manifest_path = args.manifest or args.dir / "manifest.csv"
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    index, real_paths = build_label_index(args.dir)
    in_scope_real = {p for p in real_paths if index[p]["digit"] in IN_SCOPE_DIGITS}
    n_skipped = len(real_paths) - len(in_scope_real)
    print(f"{len(in_scope_real)} real crops in scope (0-8), {n_skipped} skipped (labeled '9', outside model scope)")

    fold_of = kfold_assignment(real_paths, index, args.k, args.seed)

    out_of_fold: dict[str, tuple[int, float]] = {}
    for fold in range(args.k):
        train_paths, eval_paths = split_for_fold(index, real_paths, fold_of, fold)
        print(f"\n=== fold {fold + 1}/{args.k}: {len(train_paths)} train crops, {len(eval_paths)} held out ===", flush=True)
        model = train_one_fold(train_paths, index, device, args.epochs, args.batch_size, args.lr,
                                args.class_weights, args.seed + fold)
        preds = predict_all(model, device, eval_paths, index)
        out_of_fold.update(preds)
        correct = sum(1 for p, (pred, _c) in preds.items() if str(pred) == index[p]["digit"])
        print(f"  held-out accuracy: {correct / len(preds):.1%}")

    disagreements = []
    for path, (pred, conf) in out_of_fold.items():
        recorded = index[path]["digit"]
        if str(pred) != recorded and conf >= args.min_confidence:
            disagreements.append({
                "crop_path": path, "recorded_digit": recorded,
                "model_prediction": str(pred), "confidence": round(conf, 4),
            })
    disagreements.sort(key=lambda r: -r["confidence"])

    review_path = args.dir / "label_disagreements.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "recorded_digit", "model_prediction", "confidence"])
        writer.writeheader()
        writer.writerows(disagreements)

    manifest_rows = {r["crop_path"]: r for r in load_rows(manifest_path)}
    relabel_manifest_path = args.dir / "label_disagreements_manifest.csv"
    fieldnames = ["crop_path", "video", "frame", "tag_idx", "bead_idx", "sam3_score"]
    with open(relabel_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in disagreements:
            row = manifest_rows.get(d["crop_path"])
            if row is not None:
                writer.writerow({k: row[k] for k in fieldnames})

    overall_correct = sum(1 for p, (pred, _c) in out_of_fold.items() if str(pred) == index[p]["digit"])
    print(f"\noverall out-of-fold accuracy: {overall_correct / len(out_of_fold):.1%} ({len(out_of_fold)} real crops)")
    print(f"found {len(disagreements)} disagreements ({len(disagreements) / len(out_of_fold):.1%})")
    print(f"\ntop disagreements (crop_path | recorded -> model prediction | confidence):")
    for d in disagreements[:25]:
        print(f"  {d['crop_path']}: {d['recorded_digit']} -> {d['model_prediction']}  (conf {d['confidence']:.3f})")

    print(f"\nfull review list: {review_path}")
    print(f"relabel-ready manifest: {relabel_manifest_path}")
    print(f"\nto review/correct them:")
    print(f"  python src/label_bead_crops.py --manifest {relabel_manifest_path} --ignore-out")


if __name__ == "__main__":
    main()
