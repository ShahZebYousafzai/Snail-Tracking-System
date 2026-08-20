"""Generate augmented variants of minority-class (default: 5, 7) bead crops
to close the class-imbalance gap found by src/check_label_balance.py, via
augmentation rather than SMOTE -- SMOTE's pixel-space interpolation doesn't
respect the image manifold and produces blurry ghosting artifacts; this
instead applies realistic photometric/geometric jitter to real crops.

Augmentations applied per synthetic copy (randomly composed):
  - small rotation (+-15 deg)
  - brightness/contrast jitter
  - slight gaussian blur
  - slight scale jitter (zoom in/out a little)
No flips: unlike natural photos, a mirrored digit is a DIFFERENT (usually
invalid) character, so flipping would inject wrong labels.

Rotation is itself a label here (the model's second head), not just a
photometric nuisance -- so the applied rotation delta is ADDED to each
source crop's recorded rotation_deg, not discarded, and every synthetic row
records its source_crop_path so a later train/val split can group by source
and avoid leaking near-duplicate augmented siblings across the split.

Default target count per class is the mean count across --classes in the
current combined labels (manual + trusted PaddleOCR) -- bringing the
minority classes up to par with the *average* class, not the single largest
one, since fully matching the max (537) would mean ~6x synthetic copies per
unique 86-image source for digit 5 alone.
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"


def load_digit_labels(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("digit")]


def augment_once(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]

    rotation_delta = rng.uniform(-15, 15)
    scale = rng.uniform(0.92, 1.08)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_delta, scale)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.9, 1.1)
    out = np.clip(out.astype(np.float32) * contrast + (brightness - 1) * 60, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out, rotation_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--manual-labels", type=Path, default=None, help="default: <dir>/manual_labels.csv")
    parser.add_argument("--paddleocr-labels", type=Path, default=None, help="default: <dir>/paddleocr_labels.csv")
    parser.add_argument("--classes", default="012345678", help="reference class scope for computing the default target count")
    parser.add_argument("--digits", nargs="+", default=["5", "7"], help="digits to augment (default: 5 7)")
    parser.add_argument("--target-count", type=int, default=None,
                         help="synthetic copies bring each target digit up to this count (default: mean count over --classes)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <dir>/augmented")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manual_path = args.manual_labels or args.dir / "manual_labels.csv"
    paddle_path = args.paddleocr_labels or args.dir / "paddleocr_labels.csv"
    out_dir = args.out_dir or args.dir / "augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    combined: dict[str, dict] = {}
    for r in load_digit_labels(paddle_path):
        combined[r["crop_path"]] = {"digit": r["digit"], "rotation_deg": 0}
    for r in load_digit_labels(manual_path):
        combined[r["crop_path"]] = {"digit": r["digit"], "rotation_deg": float(r.get("rotation_deg") or 0)}

    from collections import Counter, defaultdict
    counts = Counter(v["digit"] for v in combined.values())
    scope = set(args.classes)
    scoped_counts = {d: counts.get(d, 0) for d in scope}
    target_count = args.target_count or round(sum(scoped_counts.values()) / len(scoped_counts))
    print(f"reference mean count over classes {sorted(scope)}: {target_count}")

    by_digit: dict[str, list[str]] = defaultdict(list)
    for path, v in combined.items():
        by_digit[v["digit"]].append(path)

    # Load any synthetic crops from a previous run so this one MERGES rather
    # than clobbers -- and so re-running (or running a different --digits
    # subset later) is idempotent per digit instead of stacking more copies
    # on top of an already-met target every time.
    #
    # Per-source next-free-index is tracked explicitly (not just a total
    # count) so this stays correct even after someone deletes specific stale
    # rows out of augmented_labels.csv (e.g. after correcting a mislabeled
    # source crop, see src/find_label_disagreements.py) -- a count-based
    # "continue from N" scheme recomputes indices that collide with
    # still-valid files once there are gaps, silently overwriting one and
    # leaving a duplicate manifest row pointing at the same image (verified:
    # this actually happened once before this fix).
    manifest_path = out_dir / "augmented_labels.csv"
    existing_rows = load_digit_labels(manifest_path)
    existing_synthetic_counts = Counter(r["digit"] for r in existing_rows)

    next_aug_index: dict[str, int] = defaultdict(int)
    for r in existing_rows:
        stem_out = Path(r["crop_path"]).stem
        if "_aug" in stem_out:
            try:
                idx = int(stem_out.rsplit("_aug", 1)[1])
                next_aug_index[r["source_crop_path"]] = max(next_aug_index[r["source_crop_path"]], idx + 1)
            except ValueError:
                pass

    manifest_rows = list(existing_rows)
    for digit in args.digits:
        sources = by_digit.get(digit, [])
        current = len(sources)
        already_synthetic = existing_synthetic_counts.get(digit, 0)
        needed = max(0, target_count - current - already_synthetic)
        print(f"digit {digit}: {current} real examples, {already_synthetic} existing synthetic, "
              f"generating {needed} more synthetic copies")
        if not sources or needed == 0:
            continue

        generated = 0
        src_ptr = 0
        while generated < needed:
            source_path = sources[src_ptr % len(sources)]
            src_ptr += 1
            aug_index = next_aug_index[source_path]
            next_aug_index[source_path] += 1

            img = cv2.imread(str(PROJECT_ROOT / source_path))
            if img is None:
                continue

            aug_img, rotation_delta = augment_once(img, rng)
            base_rotation = combined[source_path]["rotation_deg"]
            new_rotation = (base_rotation + rotation_delta) % 360

            stem = Path(source_path).stem
            out_name = f"{stem}_aug{aug_index}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), aug_img)

            manifest_rows.append({
                "crop_path": str(out_path.relative_to(PROJECT_ROOT)),
                "digit": digit,
                "rotation_deg": round(new_rotation, 1),
                "source_crop_path": source_path,
            })
            generated += 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "digit", "rotation_deg", "source_crop_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDONE. {len(manifest_rows)} synthetic crops -> {out_dir}")
    print(f"manifest: {manifest_path}")
    print("\nNOTE: when splitting train/val, group by source_crop_path so a real crop and its")
    print("synthetic siblings always land in the same split (otherwise near-duplicates leak across it).")


if __name__ == "__main__":
    main()
