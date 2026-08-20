"""Quick test: run the trained detector + OC-SORT tracker (via boxmot) on a
video and write an annotated output with per-track IDs and colored trails.

This is a throwaway visualization script, not part of the frozen pipeline --
tracker.py (Day 4) will implement the real ByteTrack + identity-reconciliation
state machine per the project plan.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SNAIL_CLASS = 0
COLORS = [
    (255, 99, 71), (60, 179, 113), (255, 215, 0), (30, 144, 255),
    (238, 130, 238), (255, 140, 0), (0, 206, 209), (154, 205, 50),
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0025.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector.pt")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "C0025_ocsort.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out-width", type=int, default=1920)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    tracker = OcSort(min_conf=args.conf)

    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_w = args.out_width
    out_h = int(src_h * (out_w / src_w))

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("failed to open output video writer")

    trails: dict[int, list[tuple[int, int]]] = {}
    n = 0
    seen_ids: set[int] = set()

    for result in model.predict(source=str(args.source), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False):
        frame = result.orig_img
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

        tracks = tracker.update(dets, frame)

        for x1, y1, x2, y2, tid, conf, cls in tracks[:, :7]:
            tid = int(tid)
            seen_ids.add(tid)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trails.setdefault(tid, []).append((cx, cy))
            trails[tid] = trails[tid][-60:]

            color = color_for_id(tid)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            label = f"ID {tid} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

            pts = trails[tid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        resized = cv2.resize(frame, (out_w, out_h))
        writer.write(resized)
        n += 1
        if n % 500 == 0:
            print(f"processed {n} / {total} | unique track ids so far: {len(seen_ids)}", flush=True)

    writer.release()
    print(f"DONE, total written: {n}, unique track ids: {len(seen_ids)}")


if __name__ == "__main__":
    main()
