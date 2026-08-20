"""Report per-digit class balance across the labeled bead crops (manual +
trusted PaddleOCR labels), so the imbalance can be inspected before this
data is used to train the two-head classifier.

Also folds in src/augment_minority_classes.py's synthetic crops (if any
exist yet) as a separate "effective" distribution -- real counts are what
was actually observed and drive labeling priorities; effective counts (real
+ synthetic) are what the model will actually train on. Keeping both
visible avoids the "training says 291 but this report says 165" confusion
of only tracking one.

Currently scoped to digits 0-8: digit 9 hasn't been observed in the labeled
set yet (see report output), so --classes lets that scope be widened once
some show up, without hardcoding 10 classes prematurely.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"


def load_digit_labels(csv_path: Path) -> list[tuple[str, str]]:
    """Return [(crop_path, digit), ...], skipping unreadable/skip rows (empty digit)."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["crop_path"], r["digit"]) for r in rows if r.get("digit")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--manual-labels", type=Path, default=None, help="default: <dir>/manual_labels.csv")
    parser.add_argument("--paddleocr-labels", type=Path, default=None, help="default: <dir>/paddleocr_labels.csv")
    parser.add_argument("--augmented-labels", type=Path, default=None,
                         help="default: <dir>/augmented/augmented_labels.csv (skipped if missing)")
    parser.add_argument("--classes", default="012345678", help="digits to include in the in-scope report (default: 0-8)")
    args = parser.parse_args()

    manual_path = args.manual_labels or args.dir / "manual_labels.csv"
    paddle_path = args.paddleocr_labels or args.dir / "paddleocr_labels.csv"
    augmented_path = args.augmented_labels or args.dir / "augmented" / "augmented_labels.csv"
    in_scope = set(args.classes)

    manual = load_digit_labels(manual_path)
    paddle = load_digit_labels(paddle_path)

    # manual labels win on any overlap (shouldn't normally happen -- the
    # labeling tool skips crops already in paddleocr_labels.csv -- but be
    # defensive rather than silently double-counting a crop).
    combined: dict[str, str] = {}
    for path, digit in paddle:
        combined[path] = digit
    overwritten = 0
    for path, digit in manual:
        if path in combined and combined[path] != digit:
            overwritten += 1
        combined[path] = digit

    counts = Counter(combined.values())
    total_labeled = len(combined)
    total_unreadable = sum(1 for _p, d in manual if not d)  # already excluded above, recompute for reporting
    with open(manual_path, newline="", encoding="utf-8") as f:
        manual_rows = list(csv.DictReader(f))
    n_unreadable = sum(1 for r in manual_rows if not r.get("digit"))

    print(f"manual labels   : {manual_path}  ({len(manual_rows)} rows, {n_unreadable} unreadable/skip)")
    print(f"paddleocr labels: {paddle_path}  ({len(paddle)} rows)")
    if overwritten:
        print(f"NOTE: {overwritten} crops had conflicting labels between the two sources; manual label kept")
    print(f"\ntotal usable labels: {total_labeled}")

    all_digits = sorted(counts, key=lambda d: (len(d) > 1, d))
    print("\nfull distribution (all digits seen):")
    max_count = max(counts.values()) if counts else 0
    bar_width = 40
    for d in all_digits:
        c = counts[d]
        bar = "#" * max(1, round(bar_width * c / max_count)) if max_count else ""
        flag = "  <-- OUT OF SCOPE for now" if d not in in_scope else ""
        print(f"  {d:>2}: {c:5d}  {bar}{flag}")

    missing = sorted(in_scope - set(counts))
    if missing:
        print(f"\nNo labeled examples yet for: {', '.join(missing)}")

    scoped_counts = {d: counts.get(d, 0) for d in sorted(in_scope)}
    scoped_total = sum(scoped_counts.values())
    print(f"\n--- in-scope REAL-label report (digits {''.join(sorted(in_scope))}) ---")
    print(f"total in-scope real labels: {scoped_total}")
    if scoped_total == 0:
        return

    nonzero = {d: c for d, c in scoped_counts.items() if c > 0}
    max_c, min_c = max(nonzero.values()), min(nonzero.values())
    mean_c = scoped_total / len(scoped_counts)
    print(f"per-class: min={min_c}  max={max_c}  mean={mean_c:.1f}  imbalance ratio (max/min)={max_c / min_c:.1f}x")

    print("\nper-class detail:")
    for d in sorted(scoped_counts):
        c = scoped_counts[d]
        bar = "#" * max(1, round(bar_width * c / max_c)) if c else ""
        pct_of_mean = (c / mean_c * 100) if mean_c else 0
        under = "  <-- <50% of mean, likely needs more labeling" if c and pct_of_mean < 50 else ""
        zero = "  <-- ZERO examples" if c == 0 else ""
        print(f"  {d}: {c:5d} ({pct_of_mean:5.0f}% of mean)  {bar}{under}{zero}")

    synthetic = load_digit_labels(augmented_path)
    if not synthetic:
        print(f"\n(no augmented crops found at {augmented_path} -- effective distribution == real distribution)")
        return

    synthetic_counts = Counter(digit for _path, digit in synthetic)
    effective_counts = {d: scoped_counts[d] + synthetic_counts.get(d, 0) for d in sorted(in_scope)}
    effective_total = sum(effective_counts.values())
    eff_max, eff_min = max(effective_counts.values()), min(v for v in effective_counts.values() if v > 0)
    eff_mean = effective_total / len(effective_counts)

    print(f"\n--- in-scope EFFECTIVE report (real + synthetic, digits {''.join(sorted(in_scope))}) ---")
    print(f"augmented labels: {augmented_path}  ({len(synthetic)} synthetic crops)")
    print(f"total effective labels: {effective_total}")
    print(f"per-class: min={eff_min}  max={eff_max}  mean={eff_mean:.1f}  imbalance ratio (max/min)={eff_max / eff_min:.1f}x")

    print("\nper-class detail (real + synthetic = effective):")
    for d in sorted(effective_counts):
        real_c = scoped_counts[d]
        synth_c = synthetic_counts.get(d, 0)
        eff_c = effective_counts[d]
        bar = "#" * max(1, round(bar_width * eff_c / eff_max)) if eff_c else ""
        note = "  <-- topped up with synthetic crops" if synth_c else ""
        print(f"  {d}: {real_c:5d} + {synth_c:5d} = {eff_c:5d}  {bar}{note}")


if __name__ == "__main__":
    main()
