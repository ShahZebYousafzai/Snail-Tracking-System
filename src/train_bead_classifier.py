"""Train the two-head bead classifier (digit 0-8 + rotation angle) on the
labeled crops under data/bead_crops/from_video_dedup, with a 75/25 train/val
split.

The split is done PER SOURCE crop (real manual/PaddleOCR-labeled crops),
stratified by digit so minority classes keep proportional representation in
both splits -- then every synthetic crop from src/augment_minority_classes.py
is assigned to whichever split its source_crop_path landed in, so a real
crop and its synthetic siblings never end up on opposite sides of the split
(that would leak near-duplicate information into validation and inflate the
val accuracy estimate).

Rotation labels are only trustworthy where a human actually set them: manual
labels have a real rotation_deg; PaddleOCR-accepted labels don't (OCR gives
no rotation signal) and are defaulted to 0 by augment_minority_classes.py --
those crops (and any synthetic crops derived from them) are excluded from
the rotation loss/metric via a rotation_valid flag, but still contribute to
the digit loss.
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bead_classifier import INPUT_SIZE, NUM_DIGIT_CLASSES, TwoHeadBeadNet  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "bead_classifier"
IN_SCOPE_DIGITS = [str(d) for d in range(9)]  # 0-8


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_label_index(data_dir: Path) -> tuple[dict[str, dict], set[str]]:
    """Return ({crop_path: {digit, rotation_deg, rotation_valid, source_crop_path}}, real_paths)."""
    manual = [r for r in load_rows(data_dir / "manual_labels.csv") if r.get("digit")]
    paddle = [r for r in load_rows(data_dir / "paddleocr_labels.csv") if r.get("digit")]
    augmented = load_rows(data_dir / "augmented" / "augmented_labels.csv")

    index: dict[str, dict] = {}
    for r in paddle:
        index[r["crop_path"]] = {
            "digit": r["digit"], "rotation_deg": 0.0, "rotation_valid": False, "source_crop_path": None,
        }
    for r in manual:  # manual overrides paddleocr on overlap, same convention as check_label_balance.py
        index[r["crop_path"]] = {
            "digit": r["digit"], "rotation_deg": float(r["rotation_deg"]), "rotation_valid": True,
            "source_crop_path": None,
        }

    real_paths = set(index)  # snapshot before adding synthetic rows
    for r in augmented:
        src = r["source_crop_path"]
        src_valid = index.get(src, {}).get("rotation_valid", False)
        index[r["crop_path"]] = {
            "digit": r["digit"], "rotation_deg": float(r["rotation_deg"]), "rotation_valid": src_valid,
            "source_crop_path": src,
        }

    return index, real_paths


def stratified_split(real_paths: set[str], index: dict, val_frac: float, seed: int) -> tuple[set[str], set[str]]:
    by_digit: dict[str, list[str]] = defaultdict(list)
    for p in real_paths:
        d = index[p]["digit"]
        if d in IN_SCOPE_DIGITS:
            by_digit[d].append(p)

    rng = random.Random(seed)
    train_real, val_real = set(), set()
    for _digit, paths in by_digit.items():
        paths = sorted(paths)  # deterministic order before shuffling
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * val_frac))
        val_real.update(paths[:n_val])
        train_real.update(paths[n_val:])
    return train_real, val_real


def assign_splits(index: dict, real_paths: set[str], train_real: set[str], val_real: set[str]) -> tuple[list[str], list[str]]:
    train_paths, val_paths = [], []
    for path, row in index.items():
        if row["digit"] not in IN_SCOPE_DIGITS:
            continue
        if path in real_paths:
            (train_paths if path in train_real else val_paths).append(path)
        else:  # synthetic: follow its source crop's split
            src = row["source_crop_path"]
            if src in train_real:
                train_paths.append(path)
            elif src in val_real:
                val_paths.append(path)
    return train_paths, val_paths


class BeadDataset(Dataset):
    def __init__(self, paths: list[str], index: dict, train: bool):
        self.paths = paths
        self.index = index
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        row = self.index[path]
        img = cv2.imread(str(PROJECT_ROOT / path))
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.train:
            img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)  # brightness jitter, label-agnostic

        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        digit = int(row["digit"])
        angle_rad = np.deg2rad(row["rotation_deg"])
        target_sin_cos = torch.tensor([np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32)
        rotation_valid = torch.tensor(1.0 if row["rotation_valid"] else 0.0)
        return tensor, digit, target_sin_cos, rotation_valid


def angular_error_deg(pred_sin_cos: torch.Tensor, true_sin_cos: torch.Tensor) -> torch.Tensor:
    pred_angle = torch.atan2(pred_sin_cos[:, 0], pred_sin_cos[:, 1])
    true_angle = torch.atan2(true_sin_cos[:, 0], true_sin_cos[:, 1])
    diff = torch.rad2deg(pred_angle - true_angle)
    diff = (diff + 180) % 360 - 180
    return diff.abs()


def run_epoch(model, loader, device, optimizer, digit_criterion, rotation_weight: float) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_digit_loss = total_rot_loss = 0.0
    correct = total = 0
    rot_errors = []

    for images, digits, target_rot, rot_valid in loader:
        images, digits = images.to(device), digits.to(device)
        target_rot, rot_valid = target_rot.to(device), rot_valid.to(device)

        with torch.set_grad_enabled(train_mode):
            digit_logits, pred_rot = model(images)
            digit_loss = digit_criterion(digit_logits, digits)

            sq_err = ((pred_rot - target_rot) ** 2).sum(dim=1)
            rot_loss = (sq_err * rot_valid).sum() / rot_valid.sum() if rot_valid.sum() > 0 else torch.tensor(0.0, device=device)

            loss = digit_loss + rotation_weight * rot_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_digit_loss += digit_loss.item() * images.size(0)
        total_rot_loss += rot_loss.item() * images.size(0)
        correct += (digit_logits.argmax(1) == digits).sum().item()
        total += images.size(0)
        if rot_valid.sum() > 0:
            mask = rot_valid.bool()
            rot_errors.append(angular_error_deg(pred_rot[mask], target_rot[mask]).detach().cpu())

    mean_rot_err = torch.cat(rot_errors).mean().item() if rot_errors else float("nan")
    return {
        "digit_loss": total_digit_loss / total,
        "rot_loss": total_rot_loss / total,
        "digit_acc": correct / total,
        "rot_mae_deg": mean_rot_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--val-frac", type=float, default=0.25, help="75/25 split by default")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false", default=True,
                         help="disable inverse-frequency class weighting for the digit loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    index, real_paths = build_label_index(args.dir)
    train_real, val_real = stratified_split(real_paths, index, args.val_frac, args.seed)
    train_paths, val_paths = assign_splits(index, real_paths, train_real, val_real)

    print(f"device: {device}")
    print(f"train: {len(train_paths)} crops ({len(train_real)} real sources)")
    print(f"val  : {len(val_paths)} crops ({len(val_real)} real sources)")

    train_digit_counts = Counter(index[p]["digit"] for p in train_paths)
    val_digit_counts = Counter(index[p]["digit"] for p in val_paths)
    print("\nper-class train/val counts:")
    for d in IN_SCOPE_DIGITS:
        print(f"  {d}: train={train_digit_counts.get(d, 0):4d}  val={val_digit_counts.get(d, 0):4d}")

    train_ds = BeadDataset(train_paths, index, train=True)
    val_ds = BeadDataset(val_paths, index, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = None
    if args.class_weights:
        counts = np.array([train_digit_counts.get(d, 1) for d in IN_SCOPE_DIGITS], dtype=np.float32)
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"\nclass weights (inverse frequency): {[round(w, 2) for w in weights.tolist()]}")

    model = TwoHeadBeadNet(num_digit_classes=NUM_DIGIT_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    digit_criterion = nn.CrossEntropyLoss(weight=class_weights)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "training_log.csv"
    best_val_acc = -1.0
    best_path = args.run_dir / "best.pt"

    with open(log_path, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["epoch", "train_digit_loss", "train_rot_loss", "train_digit_acc", "train_rot_mae_deg",
                              "val_digit_loss", "val_rot_loss", "val_digit_acc", "val_rot_mae_deg"])

        print(f"\n{'epoch':>5} {'tr_loss':>8} {'tr_acc':>7} {'tr_mae':>7} | {'val_loss':>8} {'val_acc':>7} {'val_mae':>7}")
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, device, optimizer, digit_criterion, args.rotation_weight)
            va = run_epoch(model, val_loader, device, None, digit_criterion, args.rotation_weight)

            tr_loss = tr["digit_loss"] + args.rotation_weight * tr["rot_loss"]
            va_loss = va["digit_loss"] + args.rotation_weight * va["rot_loss"]
            print(f"{epoch:5d} {tr_loss:8.3f} {tr['digit_acc']:7.1%} {tr['rot_mae_deg']:7.1f} | "
                  f"{va_loss:8.3f} {va['digit_acc']:7.1%} {va['rot_mae_deg']:7.1f}")

            log_writer.writerow([epoch, tr["digit_loss"], tr["rot_loss"], tr["digit_acc"], tr["rot_mae_deg"],
                                  va["digit_loss"], va["rot_loss"], va["digit_acc"], va["rot_mae_deg"]])
            f.flush()

            if va["digit_acc"] > best_val_acc:
                best_val_acc = va["digit_acc"]
                torch.save({
                    "model_state": model.state_dict(),
                    "num_digit_classes": NUM_DIGIT_CLASSES,
                    "input_size": INPUT_SIZE,
                    "val_digit_acc": va["digit_acc"],
                    "val_rot_mae_deg": va["rot_mae_deg"],
                    "epoch": epoch,
                }, best_path)

    torch.save({"model_state": model.state_dict(), "num_digit_classes": NUM_DIGIT_CLASSES, "input_size": INPUT_SIZE},
               args.run_dir / "last.pt")

    print(f"\nDONE. best val digit acc: {best_val_acc:.1%} -> {best_path}")
    print(f"training log: {log_path}")


if __name__ == "__main__":
    main()
