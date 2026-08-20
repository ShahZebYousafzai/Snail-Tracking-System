"""Extract tag crops from a video for digit-classifier ground-truth labeling.

Tracks snails with OC-SORT (so crops from the same physical snail land in
the same folder) and associates each snail with its tag box per frame by
containment. Crops are grouped by track segment under
`data/tag_crops/track_<id>/`, so labeling is one decision per folder
(rename the folder to the true 5-digit code, or `unreadable`) instead of
one decision per frame -- a track segment can't silently change identity
mid-segment (OC-SORT would have to lose and re-birth the track for that),
so grouping this way is safe.

Not part of the frozen pipeline -- this is a one-off dataset-prep script.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAIL_CLASS = 0
TAG_CLASS = 1
PAD_FRAC = 0.25

# From the earlier OC-SORT sweep: max_age=150ish materially cuts track
# fragmentation vs the default (30) without going so high it risks
# reacquiring the wrong snail after a long gap.
TRACKER_KWARGS = dict(max_age=150, iou_threshold=0.2)


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return frame[y1p:y2p, x1p:x2p]


def match_tag_to_snail(snail_box, tag_boxes):
    """Return index of the tag box whose center falls inside the snail
    box, or the highest-IoU tag box if none is fully contained."""
    sx1, sy1, sx2, sy2 = snail_box
    best_idx, best_score = -1, 0.0
    for i, (tx1, ty1, tx2, ty2) in enumerate(tag_boxes):
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
            # contained: score by tag area (prefer the match, any contained tag wins)
            return i
        ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
        ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (sx2 - sx1) * (sy2 - sy1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_score:
            best_score, best_idx = iou, i
    return best_idx if best_score > 0 else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0021.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "tag_crops")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    model = YOLO(str(args.weights))
    tracker = OcSort(min_conf=args.conf, **TRACKER_KWARGS)

    cap = cv2.VideoCapture(str(args.source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    manifest_rows = []
    track_crop_counts: dict[int, int] = {}
    n_processed = 0
    frame_idx = 0

    for result in model.predict(source=str(args.source), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False):
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        frame = result.orig_img
        boxes = result.boxes
        snail_dets = np.empty((0, 6))
        tag_boxes = np.empty((0, 4))
        tag_confs = np.empty((0,))
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            smask = clss == SNAIL_CLASS
            tmask = clss == TAG_CLASS
            if smask.any():
                snail_dets = np.concatenate([xyxy[smask], confs[smask, None], clss[smask, None]], axis=1)
            if tmask.any():
                tag_boxes = xyxy[tmask]
                tag_confs = confs[tmask]

        tracks = tracker.update(snail_dets, frame)

        for x1, y1, x2, y2, tid, conf, cls in tracks[:, :7]:
            tid = int(tid)
            match_idx = match_tag_to_snail((x1, y1, x2, y2), tag_boxes)
            if match_idx < 0:
                continue
            tx1, ty1, tx2, ty2 = tag_boxes[match_idx]
            crop = crop_with_padding(frame, tx1, ty1, tx2, ty2, PAD_FRAC)
            if crop.size == 0:
                continue

            track_dir = args.out_dir / f"track_{tid:03d}"
            track_dir.mkdir(exist_ok=True)
            crop_name = f"frame{frame_idx:05d}.jpg"
            crop_path = track_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            track_crop_counts[tid] = track_crop_counts.get(tid, 0) + 1
            manifest_rows.append({
                "track_id": tid,
                "frame": frame_idx,
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                "tag_det_conf": float(tag_confs[match_idx]),
                "snail_det_conf": float(conf),
            })

        n_processed += 1
        if n_processed % 500 == 0:
            print(f"processed {frame_idx}/{total} frames | {len(track_crop_counts)} track folders so far", flush=True)
        frame_idx += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "frame", "crop_path", "tag_det_conf", "snail_det_conf"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDONE. {len(manifest_rows)} crops across {len(track_crop_counts)} track folders under {args.out_dir}")
    print(f"manifest: {manifest_path}")
    print("\ncrops per track folder:")
    for tid, count in sorted(track_crop_counts.items()):
        print(f"  track_{tid:03d}: {count} crops")


if __name__ == "__main__":
    main()
