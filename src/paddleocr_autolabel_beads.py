"""Auto-label a subset of bead crops with PaddleOCR, accepting only
high-confidence single-digit predictions.

An earlier experiment (src/test_paddleocr_on_tags.py) ran PaddleOCR on whole
TAG crops (5 rotated, jumbled beads at once) and it failed almost entirely.
Isolated single-bead crops are a much easier target -- there's exactly one
character, no layout to parse -- so this uses PaddleOCR's standalone
TextRecognition (recognition only, no text-detection stage, since we already
know where the digit is: the whole crop).

Verified on a 200-crop sample: predictions scoring >=0.75 were correct in
every manually-checked case (6/6); by ~0.55-0.6 confidence, wrong/ambiguous
predictions start showing up. Default threshold is deliberately strict
(0.75) to prioritize label correctness over coverage -- a wrong auto-label
poisons training data more than a missed one costs in manual labeling time.
Expect a low yield (roughly 1-3% of crops), by design: this is meant to give
the manual labeling pass a head start, not replace it.

Digit "0" is excluded from acceptance regardless of confidence. Verified on
the full run: non-zero digit predictions (3/4/6/9) were correct in all 12
manually-checked cases across the whole confidence range, but 5/6 checked
"0" predictions (at 0.75-0.88 confidence) were wrong -- the model appears to
default to "0" as a generic fallback on edge-on/unreadable crops (a thin
vertical sliver, not an actual disc face), and confidence score does not
reliably separate real "0"s from this failure mode.
"""

import argparse
import csv
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup" / "manifest.csv"

UPSCALE_TARGET_HEIGHT = 160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None, help="default: <manifest_dir>/paddleocr_labels.csv")
    parser.add_argument("--threshold", type=float, default=0.75, help="min confidence to accept a single-digit prediction")
    parser.add_argument("--exclude-digits", nargs="*", default=["0"],
                         help="digits never auto-accepted regardless of confidence (default: '0', see module docstring)")
    parser.add_argument("--from-existing", type=Path, default=None,
                         help="re-filter an existing *_all.csv instead of re-running the model")
    args = parser.parse_args()

    out_path = args.out or args.manifest.parent / "paddleocr_labels.csv"
    all_out = out_path.with_name(out_path.stem + "_all.csv")

    if args.from_existing:
        with open(args.from_existing, newline="", encoding="utf-8") as f:
            all_results = list(csv.DictReader(f))
        print(f"re-filtering {len(all_results)} existing predictions from {args.from_existing}", flush=True)
    else:
        from paddleocr import TextRecognition
        model = TextRecognition(model_name="PP-OCRv5_mobile_rec")

        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"loaded {len(rows)} crops from {args.manifest}", flush=True)

        all_results = []
        for i, row in enumerate(rows):
            crop_path = PROJECT_ROOT / row["crop_path"]
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = max(1, UPSCALE_TARGET_HEIGHT // max(h, 1))
            big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

            for res in model.predict(big):
                text = res.get("rec_text", "")
                score = float(res.get("rec_score", 0.0))
                all_results.append({"crop_path": row["crop_path"], "pred_text": text, "score": round(score, 4)})

            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(rows)}", flush=True)

        with open(all_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["crop_path", "pred_text", "score"])
            writer.writeheader()
            writer.writerows(all_results)

    accepted = []
    for r in all_results:
        text = r["pred_text"]
        score = float(r["score"])
        is_digit = len(text) == 1 and text.isdigit()
        if is_digit and text not in args.exclude_digits and score >= args.threshold:
            accepted.append({"crop_path": r["crop_path"], "digit": text, "score": round(score, 4)})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "digit", "score"])
        writer.writeheader()
        writer.writerows(accepted)

    print(f"\nDONE. {len(accepted)}/{len(all_results)} crops auto-labeled ({len(accepted) / len(all_results):.1%}) "
          f"at threshold {args.threshold} (excluding {args.exclude_digits})")
    print("per-digit counts:")
    from collections import Counter
    counts = Counter(r["digit"] for r in accepted)
    for d in sorted(counts):
        print(f"  {d}: {counts[d]}")
    print(f"\naccepted labels: {out_path}")
    print(f"all predictions (for review): {all_out}")


if __name__ == "__main__":
    main()
