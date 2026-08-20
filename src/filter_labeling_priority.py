"""Build a priority labeling queue: crops NOT yet labeled whose PaddleOCR
prediction (from the full, unfiltered run -- paddleocr_labels_all.csv) was
one of the target under-represented digits, e.g. 5 or 7.

This does NOT trust those predictions as labels (that's exactly why they
weren't accepted into paddleocr_labels.csv in the first place -- see
src/paddleocr_autolabel_beads.py, digit "0" alone had an ~83% error rate
even at high confidence, and other digits weren't spot-checked at all below
threshold). It only uses them as a cheap hint to reorder the labeling queue
so a human hits more actual 5s/7s per minute instead of labeling randomly --
every crop still needs a real human decision.

Output is a manifest with the same schema as manifest.csv, so it can be
pointed at directly with label_bead_crops.py --manifest.
"""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"


def load_labeled_paths(*csv_paths: Path) -> set[str]:
    labeled = set()
    for p in csv_paths:
        if p and p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    labeled.add(row["crop_path"])
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="default: <dir>/manifest.csv")
    parser.add_argument("--all-predictions", type=Path, default=None, help="default: <dir>/paddleocr_labels_all.csv")
    parser.add_argument("--manual-labels", type=Path, default=None, help="default: <dir>/manual_labels.csv")
    parser.add_argument("--paddleocr-labels", type=Path, default=None, help="default: <dir>/paddleocr_labels.csv")
    parser.add_argument("--digits", nargs="+", default=["5", "7"], help="target digits to prioritize (default: 5 7)")
    parser.add_argument("--out", type=Path, default=None, help="default: <dir>/priority_manifest.csv")
    args = parser.parse_args()

    manifest_path = args.manifest or args.dir / "manifest.csv"
    all_pred_path = args.all_predictions or args.dir / "paddleocr_labels_all.csv"
    manual_path = args.manual_labels or args.dir / "manual_labels.csv"
    paddle_path = args.paddleocr_labels or args.dir / "paddleocr_labels.csv"
    out_path = args.out or args.dir / "priority_manifest.csv"
    targets = set(args.digits)

    already_labeled = load_labeled_paths(manual_path, paddle_path)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest_rows = {r["crop_path"]: r for r in csv.DictReader(f)}

    with open(all_pred_path, newline="", encoding="utf-8") as f:
        predictions = list(csv.DictReader(f))

    candidates = []
    for pred in predictions:
        path = pred["crop_path"]
        if path in already_labeled:
            continue
        if pred["pred_text"] not in targets:
            continue
        row = manifest_rows.get(path)
        if row is None:
            continue
        candidates.append((float(pred["score"]), pred["pred_text"], row))

    candidates.sort(key=lambda c: -c[0])

    fieldnames = ["crop_path", "video", "frame", "tag_idx", "bead_idx", "sam3_score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _score, _digit, row in candidates:
            writer.writerow({k: row[k] for k in fieldnames})

    from collections import Counter
    by_digit = Counter(digit for _s, digit, _r in candidates)
    print(f"{len(predictions)} total predictions, {len(already_labeled)} crops already labeled")
    print(f"{len(candidates)} unlabeled crops flagged as a likely {'/'.join(sorted(targets))}")
    for d in sorted(by_digit):
        print(f"  predicted '{d}': {by_digit[d]}")
    print(f"priority manifest: {out_path}")
    print(f"\nlabel them with:")
    print(f"  python src/label_bead_crops.py --manifest {out_path}")


if __name__ == "__main__":
    main()
