"""Test whether a windowed motion-heading is stable enough to use as the
rotation reference for bead position-1..5 assignment, and visualize it live.

For each tracked snail, draws two candidate reference vectors so they can be
compared by eye on real footage:
  - RED arrow: windowed motion heading (centroid now vs ~1.5s ago), held
    (dimmed) when displacement over the window is below --min-disp, instead
    of being recomputed from noise.
  - CYAN arrow: snail-body-centroid -> tag-cluster-centroid vector, available
    every frame including fully stationary ones.

Not part of the frozen pipeline -- a throwaway test script to eyeball which
signal is more usable before committing to one for bead-slot assignment.
"""

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAIL_CLASS = 0
TAG_CLASS = 1


def match_tag_to_snail(snail_box, tag_boxes):
    sx1, sy1, sx2, sy2 = snail_box
    best_idx, best_score = -1, 0.0
    for i, (tx1, ty1, tx2, ty2) in enumerate(tag_boxes):
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
            return i
        ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
        ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (sx2 - sx1) * (sy2 - sy1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_score:
            best_score, best_idx = iou, i
    return best_idx if best_score > 0 else -1


def draw_arrow(frame, origin, vec, color, length=120, thickness=4):
    norm = math.hypot(*vec)
    if norm < 1e-6:
        return
    ux, uy = vec[0] / norm, vec[1] / norm
    end = (int(origin[0] + ux * length), int(origin[1] + uy * length))
    cv2.arrowedLine(frame, (int(origin[0]), int(origin[1])), end, color, thickness, tipLength=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0025.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector_yolo.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--window-seconds", type=float, default=1.5, help="lookback window for motion heading")
    parser.add_argument("--min-disp", type=float, default=15.0, help="min pixel displacement (in native frame coords) to trust the motion heading")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save", type=Path, default=PROJECT_ROOT / "outputs" / "heading_test.mp4")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out-width", type=int, default=1920)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    tracker = OcSort(min_conf=args.conf, max_age=150, iou_threshold=0.2)

    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_frames = max(1, int(args.window_seconds * fps))
    out_w, out_h = args.out_width, int(src_h * (args.out_width / src_w))

    writer = None
    if not args.no_save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    centroid_hist: dict[int, deque] = {}   # track_id -> deque[(frame_idx, cx, cy)]
    last_heading: dict[int, tuple[float, float]] = {}

    frame_idx = 0
    scale = out_w / src_w
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        snail_dets = np.empty((0, 6))
        tag_boxes = np.empty((0, 4))
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

        tracks = tracker.update(snail_dets, frame)

        for x1, y1, x2, y2, tid, conf, _cls in tracks[:, :7]:
            tid = int(tid)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            hist = centroid_hist.setdefault(tid, deque())
            hist.append((frame_idx, cx, cy))
            while hist and frame_idx - hist[0][0] > window_frames:
                hist.popleft()

            heading = last_heading.get(tid, (0.0, 0.0))
            confident = False
            if hist and (frame_idx - hist[0][0]) >= window_frames * 0.8:
                fx0, x0, y0 = hist[0]
                disp = (cx - x0, cy - y0)
                if math.hypot(*disp) >= args.min_disp:
                    heading = disp
                    last_heading[tid] = heading
                    confident = True

            match_idx = match_tag_to_snail((x1, y1, x2, y2), tag_boxes)
            body_to_tag = None
            if match_idx >= 0:
                tx1, ty1, tx2, ty2 = tag_boxes[match_idx]
                tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
                body_to_tag = (tcx - cx, tcy - cy)

            sx1, sy1, sx2, sy2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            ocx, ocy = cx * scale, cy * scale

            cv2.rectangle(frame if scale == 1 else frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 2)
            cv2.putText(frame, f"id={tid}", (int(x1), max(0, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 3)

            heading_color = (0, 0, 255) if confident else (0, 0, 120)  # bright red vs dim red (held)
            draw_arrow(frame, (cx, cy), heading, heading_color, length=150)
            if body_to_tag is not None:
                draw_arrow(frame, (cx, cy), body_to_tag, (255, 255, 0), length=150)

            status = "MOVING" if confident else "HELD"
            cv2.putText(frame, status, (int(x1), int(y1) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, heading_color, 2)

        resized = cv2.resize(frame, (out_w, out_h))
        if writer is not None:
            writer.write(resized)
        if args.display:
            cv2.imshow("Heading test (red=motion, cyan=body->tag)", resized)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"processed {frame_idx}/{total}", flush=True)
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"DONE. {frame_idx} frames" + (f" -> {args.save}" if writer is not None else ""))


if __name__ == "__main__":
    main()
