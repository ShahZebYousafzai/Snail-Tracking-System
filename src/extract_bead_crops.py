"""Extract individual bead crops from the existing tag crops, using the
trained bead detector (weights/detector_beads.pt), for digit-classifier
training data.

Unlike the tag-level extraction (src/extract_tag_crops.py, grouped by
tracked snail), bead crops here are pooled across ALL tracks/snails --
digit identity (0-9) is universal, a "6" from any snail is valid training
data for the "6" class, so there's no need to keep per-snail structure.
"""

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bead_reader import detect_beads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-crops-dir", type=Path, default=PROJECT_ROOT / "data" / "tag_crops")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "bead_crops" / "unlabeled")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--per-track-limit", type=int, default=150, help="max crops sampled per track folder")
    parser.add_argument("--pad-frac", type=float, default=0.15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted([d for d in args.tag_crops_dir.iterdir() if d.is_dir()])
    total_beads = 0
    total_crops_processed = 0

    for track_dir in track_dirs:
        files = sorted(track_dir.glob("*.jpg"))
        if not files:
            continue
        import numpy as np
        idxs = np.linspace(0, len(files) - 1, min(args.per_track_limit, len(files)), dtype=int)

        for i in idxs:
            crop_path = files[i]
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            beads = detect_beads(crop, conf=args.conf)
            h, w = crop.shape[:2]
            for j, b in enumerate(beads):
                pad = b.radius * args.pad_frac
                x0 = max(0, int(b.cx - b.radius - pad))
                y0 = max(0, int(b.cy - b.radius - pad))
                x1 = min(w, int(b.cx + b.radius + pad))
                y1 = min(h, int(b.cy + b.radius + pad))
                bead_crop = crop[y0:y1, x0:x1]
                if bead_crop.size == 0:
                    continue
                out_name = f"{track_dir.name}_{crop_path.stem}_b{j}.jpg"
                cv2.imwrite(str(args.out_dir / out_name), bead_crop)
                total_beads += 1
            total_crops_processed += 1

        print(f"{track_dir.name}: processed {len(idxs)} tag crops, {total_beads} bead crops so far", flush=True)

    print(f"\nDONE. {total_beads} bead crops extracted from {total_crops_processed} tag crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
