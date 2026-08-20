"""Quick parameter sweep for OC-SORT track fragmentation.

Runs the detector once over the video, caches per-frame snail detections,
then replays them through OC-SORT under different max_age/iou_threshold
settings to see which combination keeps track IDs most stable (fewest
unique IDs for the known 5 physical snails). Detection is the expensive
part, so it's cached and reused across the whole sweep.
"""

from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data" / "raw" / "C0025.MP4"
WEIGHTS = PROJECT_ROOT / "weights" / "detector.pt"
SNAIL_CLASS = 0
CONF = 0.25
IMGSZ = 960


def cache_detections() -> list[np.ndarray]:
    model = YOLO(str(WEIGHTS))
    cached = []
    n = 0
    for result in model.predict(source=str(SOURCE), conf=CONF, imgsz=IMGSZ, stream=True, verbose=False):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            mask = clss == SNAIL_CLASS
            dets = np.concatenate(
                [xyxy[mask], confs[mask, None], clss[mask, None]], axis=1
            ) if mask.any() else np.empty((0, 6))
        else:
            dets = np.empty((0, 6))
        cached.append(dets)
        n += 1
        if n % 1000 == 0:
            print(f"cached detections for {n} frames", flush=True)
    print(f"detection caching done: {len(cached)} frames")
    return cached


def run_config(cached: list[np.ndarray], **tracker_kwargs) -> dict:
    tracker = OcSort(min_conf=CONF, **tracker_kwargs)
    seen_ids: set[int] = set()
    id_births = 0
    prev_active: set[int] = set()
    for dets in cached:
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        tracks = tracker.update(dets, dummy_frame)
        active = set(int(t) for t in tracks[:, 4]) if len(tracks) else set()
        new_ids = active - prev_active
        id_births += len(new_ids)
        seen_ids |= active
        prev_active = active
    return {
        "unique_ids": len(seen_ids),
        "id_births": id_births,
        **tracker_kwargs,
    }


def main() -> None:
    cached = cache_detections()

    sweep = [
        dict(max_age=30, iou_threshold=0.3),   # boxmot default
        dict(max_age=60, iou_threshold=0.3),
        dict(max_age=90, iou_threshold=0.3),
        dict(max_age=125, iou_threshold=0.3),  # ~5s at 25fps
        dict(max_age=125, iou_threshold=0.2),
        dict(max_age=125, iou_threshold=0.15),
        dict(max_age=200, iou_threshold=0.2),
    ]

    print(f"\n{'max_age':>8} {'iou_thr':>8} {'unique_ids':>11} {'id_births':>10}")
    results = []
    for cfg in sweep:
        r = run_config(cached, **cfg)
        results.append(r)
        print(f"{r['max_age']:>8} {r['iou_threshold']:>8} {r['unique_ids']:>11} {r['id_births']:>10}")

    best = min(results, key=lambda r: r["unique_ids"])
    print(f"\nBest: max_age={best['max_age']} iou_threshold={best['iou_threshold']} -> {best['unique_ids']} unique ids (target: 5)")


if __name__ == "__main__":
    main()
