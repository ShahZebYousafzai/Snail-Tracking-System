"""Live/saved visualization of bead detection + rotation-invariant position
assignment (src/bead_reader.py), using the body->tag vector as the
rotation reference (motion heading was tested and rejected -- see
src/test_heading_direction.py, only 0-21% confident per track).

For each tracked snail: draws the tag crop's detected beads, labels each
with its assigned position (1..5), and draws the reference axis used to
derive that assignment.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

from bead_reader import assign_positions, detect_beads

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAIL_CLASS = 0
TAG_CLASS = 1
PAD_FRAC = 0.25


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


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return frame[y1p:y2p, x1p:x2p], (x1p, y1p)


POS_COLORS = {
    1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 255), 4: (0, 255, 0), 5: (255, 0, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0025.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector_yolo.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save", type=Path, default=PROJECT_ROOT / "outputs" / "bead_positions.mp4")
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

    out_w, out_h = args.out_width, int(src_h * (args.out_width / src_w))
    writer = None
    if not args.no_save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_idx = 0
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
            body_cx, body_cy = (x1 + x2) / 2, (y1 + y2) / 2

            match_idx = match_tag_to_snail((x1, y1, x2, y2), tag_boxes)
            if match_idx < 0:
                continue
            tx1, ty1, tx2, ty2 = tag_boxes[match_idx]
            tag_cx, tag_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

            crop, (ox, oy) = crop_with_padding(frame, tx1, ty1, tx2, ty2, PAD_FRAC)
            if crop.size == 0:
                continue
            beads = detect_beads(crop, crop_offset=(ox, oy))
            if len(beads) < 3:
                continue  # too few beads found this frame, skip labeling

            ref_axis = (tag_cx - body_cx, tag_cy - body_cy)
            beads = assign_positions(beads, ref_axis, centroid=(tag_cx, tag_cy))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 2)
            cv2.arrowedLine(frame, (int(body_cx), int(body_cy)), (int(tag_cx), int(tag_cy)),
                             (255, 255, 0), 3, tipLength=0.2)

            for b in beads:
                color = POS_COLORS.get(b.position, (255, 255, 255))
                cv2.circle(frame, (int(b.cx), int(b.cy)), int(b.radius), color, 3)
                cv2.putText(frame, str(b.position), (int(b.cx) - 10, int(b.cy) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.putText(frame, f"id={tid}", (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 3)

        resized = cv2.resize(frame, (out_w, out_h))
        if writer is not None:
            writer.write(resized)
        if args.display:
            cv2.imshow("Bead positions (1=red,2=orange,3=yellow,4=green,5=blue)", resized)
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
