"""Snail / tag / bead detection + digit-classification pipeline, with visualization.

Detects snails and tags with the trained YOLO detector
(weights/detector_yolo.pt, class 0 = snail, class 1 = tag), crops each tag
region, runs the bead detector (weights/detector_beads.pt, via
src/bead_reader.py) to localize individual beads, then classifies each bead
with the trained two-head CNN (src/bead_classifier.py, runs/bead_classifier/best.pt)
to predict its digit (0-8) and rotation angle. Results are drawn on the
frame and shown live and/or saved to disk.

The main window shows an overview (snail + tag boxes only) -- bead-level
detail (digit/rotation per bead) is unreadably tiny crammed into a full 4K
frame, so each detected tag also gets its own zoomed-in live window
(--no-tag-windows to turn that off).

Usage::

    python main.py data/raw/C0025.MP4
    python main.py data/raw/C0025.MP4 --save outputs/C0025_pipeline.mp4
    python main.py data/raw/C0025.MP4 --no-display --save outputs/out.mp4
    python main.py data/tag_crops/track_001/frame00000.jpg
    python main.py data/raw/C0025.MP4 --no-classify      # skip bead classification, boxes only
    python main.py data/raw/C0025.MP4 --no-tag-windows   # single overview window only
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bead_classifier import INPUT_SIZE, NUM_DIGIT_CLASSES, TwoHeadBeadNet  # noqa: E402
from bead_reader import detect_beads  # noqa: E402

SNAIL_CLASS = 0
TAG_CLASS = 1

SNAIL_COLOR = colors(SNAIL_CLASS, True)
TAG_COLOR = colors(TAG_CLASS, True)
BEAD_COLOR = colors(2, True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

TAG_PAD_FRAC = 0.25   # padding around the detected tag box before bead detection
BEAD_PAD_FRAC = 0.15  # padding around each detected bead box before classification


class TagIdentityTracker:
    """Lightweight nearest-centroid tracker -- NOT OC-SORT (deliberately, per
    earlier discussion) -- just enough to give each tag a stable id across
    frames so its display window doesn't get a new name (and pop/close)
    every single frame. A tag keeps its id across brief misses (occlusion,
    a frame where the detector didn't fire) for up to `max_missed` updates
    before its id is freed; a detection with no nearby existing tag within
    `max_distance` pixels starts a new id.
    """

    def __init__(self, max_distance: float = 400.0, max_missed: int = 15):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 0
        self.tags: dict[int, dict] = {}  # id -> {"centroid": (cx, cy), "missed": int}

    def update(self, boxes: list[tuple[float, float, float, float, float]]) -> list[int]:
        centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2, _c in boxes]

        candidate_pairs = []
        for tid, info in self.tags.items():
            tcx, tcy = info["centroid"]
            for di, (dcx, dcy) in enumerate(centroids):
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                if dist <= self.max_distance:
                    candidate_pairs.append((dist, tid, di))
        candidate_pairs.sort(key=lambda p: p[0])

        assignments: dict[int, int] = {}  # detection index -> tag id
        matched_tags, matched_dets = set(), set()
        for _dist, tid, di in candidate_pairs:
            if tid in matched_tags or di in matched_dets:
                continue
            assignments[di] = tid
            matched_tags.add(tid)
            matched_dets.add(di)
            self.tags[tid]["centroid"] = centroids[di]
            self.tags[tid]["missed"] = 0

        for tid in list(self.tags):
            if tid not in matched_tags:
                self.tags[tid]["missed"] += 1
                if self.tags[tid]["missed"] > self.max_missed:
                    del self.tags[tid]

        for di in range(len(centroids)):
            if di not in assignments:
                tid = self.next_id
                self.next_id += 1
                self.tags[tid] = {"centroid": centroids[di], "missed": 0}
                assignments[di] = tid

        return [assignments[di] for di in range(len(centroids))]


class Pipeline:
    def __init__(self, detector: YOLO, bead_weights: Path, classifier_weights: Path | None, device: str | None):
        self.detector = detector
        self.bead_weights = bead_weights
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        if classifier_weights is not None:
            self.classifier = TwoHeadBeadNet(num_digit_classes=NUM_DIGIT_CLASSES)
            ckpt = torch.load(classifier_weights, map_location=self.device)
            self.classifier.load_state_dict(ckpt["model_state"])
            self.classifier.eval()
            self.classifier.to(self.device)
        self.tag_tracker = TagIdentityTracker()


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return frame[y1p:y2p, x1p:x2p], (x1p, y1p)


def detect_snails_and_tags(detector, frame, conf, imgsz):
    result = detector.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    snail_boxes, tag_boxes = [], []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            entry = (float(x1), float(y1), float(x2), float(y2), float(c))
            if int(cls) == SNAIL_CLASS:
                snail_boxes.append(entry)
            elif int(cls) == TAG_CLASS:
                tag_boxes.append(entry)
    return snail_boxes, tag_boxes


@torch.no_grad()
def classify_bead(pipeline: Pipeline, bead_crop) -> tuple[int, float]:
    """Return (predicted_digit, predicted_rotation_deg)."""
    img = cv2.resize(bead_crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(pipeline.device)

    digit_logits, rot = pipeline.classifier(tensor)
    digit = int(digit_logits.argmax(dim=1).item())
    sin_v, cos_v = rot[0, 0].item(), rot[0, 1].item()
    angle_deg = float(np.degrees(np.arctan2(sin_v, cos_v)) % 360)
    return digit, angle_deg


def detect_and_classify_beads(pipeline: Pipeline, frame, tag_box, args):
    x1, y1, x2, y2, _c = tag_box
    tag_crop, (ox, oy) = crop_with_padding(frame, x1, y1, x2, y2, TAG_PAD_FRAC)
    if tag_crop.size == 0:
        return []

    beads = detect_beads(tag_crop, crop_offset=(ox, oy), conf=args.bead_conf, weights_path=pipeline.bead_weights)

    results = []
    for b in beads:
        bead_crop, _ = crop_with_padding(frame, b.cx - b.radius, b.cy - b.radius, b.cx + b.radius, b.cy + b.radius, BEAD_PAD_FRAC)
        digit, angle_deg = (None, None)
        if pipeline.classifier is not None and bead_crop.size > 0:
            digit, angle_deg = classify_bead(pipeline, bead_crop)
        results.append((b, digit, angle_deg))
    return results


def annotate_frame(frame, snail_boxes, tag_boxes):
    """Overview frame: snail + tag boxes only. Bead-level detail is tiny and
    unreadable crammed into a full 4K frame -- see build_tag_views() for the
    per-tag zoomed windows that carry that detail instead."""
    annotator = Annotator(frame.copy(), line_width=None, example="snail")
    for x1, y1, x2, y2, c in snail_boxes:
        annotator.box_label((x1, y1, x2, y2), label=f"snail {c:.2f}", color=SNAIL_COLOR)
    for x1, y1, x2, y2, c in tag_boxes:
        annotator.box_label((x1, y1, x2, y2), label=f"tag {c:.2f}", color=TAG_COLOR)
    return annotator.result()


def build_tag_views(frame, tag_boxes, beads_per_tag, tag_ids, view_size: int = 420):
    """One zoomed-in, annotated image per detected tag, for display in its
    own window -- bead boxes/labels are legible here in a way they can't be
    on the full frame.

    Window name is keyed on the tag's STABLE tracked id, not its per-frame
    detection index or confidence -- either of those changes frame to frame,
    which OpenCV treats as a brand new window each time (the exact "windows
    keep popping up and closing" bug this was built to fix)."""
    views = []
    for (x1, y1, x2, y2, _conf), beads, tag_id in zip(tag_boxes, beads_per_tag, tag_ids):
        tag_crop, (ox, oy) = crop_with_padding(frame, x1, y1, x2, y2, TAG_PAD_FRAC)
        if tag_crop.size == 0:
            continue
        h, w = tag_crop.shape[:2]
        scale = view_size / max(h, w)
        resized = cv2.resize(tag_crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)

        annotator = Annotator(resized, line_width=2, example="0")
        for b, digit, angle_deg in beads:
            bx1 = (b.cx - b.radius - ox) * scale
            by1 = (b.cy - b.radius - oy) * scale
            bx2 = (b.cx + b.radius - ox) * scale
            by2 = (b.cy + b.radius - oy) * scale
            label = f"{digit} {angle_deg:.0f}deg" if digit is not None else "?"
            annotator.box_label((bx1, by1, bx2, by2), label=label, color=BEAD_COLOR)

        views.append((f"tag {tag_id}", annotator.result()))
    return views


def process_frame(pipeline: Pipeline, frame, args):
    snail_boxes, tag_boxes = detect_snails_and_tags(pipeline.detector, frame, args.conf, args.imgsz)
    tag_ids = pipeline.tag_tracker.update(tag_boxes)

    beads_per_tag = []
    if not args.no_beads:
        for tag_box in tag_boxes:
            beads_per_tag.append(detect_and_classify_beads(pipeline, frame, tag_box, args))
    else:
        beads_per_tag = [[] for _ in tag_boxes]

    annotated = annotate_frame(frame, snail_boxes, tag_boxes)
    want_tag_views = not args.no_beads and args.tag_windows and not args.no_display
    tag_views = build_tag_views(frame, tag_boxes, beads_per_tag, tag_ids) if want_tag_views else []
    return annotated, tag_views


def run_on_video(pipeline: Pipeline, args) -> int:
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"FAIL: could not open {args.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (src_w, src_h))
        print(f"saving to: {args.save}")

    window_name = "Snail / tag / bead pipeline (q/Esc to quit)"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    processed = 0
    t_start = time.perf_counter()
    processing_time = 0.0
    avg_fps = 0.0
    open_tag_windows: set[str] = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.stride == 0:
                t0 = time.perf_counter()
                annotated, tag_views = process_frame(pipeline, frame, args)
                processing_time += time.perf_counter() - t0
                processed += 1
                avg_fps = processed / processing_time if processing_time > 0 else 0.0

                # Scale text to frame width so it stays legible on both 1080p and native 4K sources.
                text_scale = annotated.shape[1] / 1920
                cv2.putText(annotated, f"avg fps: {avg_fps:.1f}", (int(10 * text_scale), int(40 * text_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), max(2, int(2 * text_scale)), cv2.LINE_AA)

                if writer is not None:
                    writer.write(annotated)

                if not args.no_display:
                    cv2.imshow(window_name, annotated)

                    for name, view in tag_views:
                        cv2.imshow(name, view)

                    # A window stays open (showing its last frame) as long as the tracker
                    # still considers that tag alive, even on frames where it wasn't
                    # re-detected -- only destroy it once the tracker actually drops the id
                    # (missed too many frames in a row), not on every single miss.
                    alive_names = {f"tag {tid}" for tid in pipeline.tag_tracker.tags}
                    for stale in open_tag_windows - alive_names:
                        cv2.destroyWindow(stale)
                    open_tag_windows = alive_names

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or Esc
                        print("stopped by user")
                        break

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"processed {frame_idx}/{total} frames | avg fps: {avg_fps:.1f}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"DONE. {frame_idx} frames read, {processed} inferred in {elapsed:.1f}s "
          f"(avg {avg_fps:.1f} processing fps)" + (f" -> {args.save}" if args.save else ""))
    return 0


def run_on_image(pipeline: Pipeline, args) -> int:
    frame = cv2.imread(str(args.source))
    if frame is None:
        print(f"FAIL: could not read {args.source}")
        return 1

    annotated, tag_views = process_frame(pipeline, frame, args)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), annotated)
        print(f"saved: {args.save}")

    if not args.no_display:
        cv2.imshow("Snail / tag / bead pipeline (any key to close)", annotated)
        for name, view in tag_views:
            cv2.imshow(name, view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?", default=PROJECT_ROOT / "data" / "raw" / "C0025.MP4",
                         help="video or image to run the pipeline on")
    parser.add_argument("--detector-weights", type=Path, default=PROJECT_ROOT / "weights" / "detector_yolo.pt",
                         help="snail+tag detector (class 0 = snail, class 1 = tag)")
    parser.add_argument("--bead-weights", type=Path, default=PROJECT_ROOT / "weights" / "detector_beads.pt")
    parser.add_argument("--classifier-weights", type=Path, default=PROJECT_ROOT / "runs" / "bead_classifier" / "best.pt",
                         help="two-head digit+rotation classifier checkpoint")
    parser.add_argument("--conf", type=float, default=0.4, help="snail/tag detector confidence threshold")
    parser.add_argument("--bead-conf", type=float, default=0.4, help="bead detector confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=1, help="run inference on every Nth frame (video only)")
    parser.add_argument("--no-beads", action="store_true", help="skip bead detection/classification entirely (snail+tag boxes only)")
    parser.add_argument("--no-classify", dest="classify", action="store_false", default=True,
                         help="detect bead boxes but skip digit/rotation classification")
    parser.add_argument("--no-tag-windows", dest="tag_windows", action="store_false", default=True,
                         help="don't open a separate zoomed-in window per detected tag (bead labels are otherwise "
                              "unreadably tiny on the full 4K frame)")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: auto-detect)")
    parser.add_argument("--save", type=Path, default=None, help="path to save the annotated video/image")
    parser.add_argument("--no-display", action="store_true", help="skip the live preview window (requires --save)")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"FAIL: source not found at {args.source}")
        return 1
    if not args.detector_weights.exists():
        print(f"FAIL: detector weights not found at {args.detector_weights}")
        return 1
    if not args.no_beads and not args.bead_weights.exists():
        print(f"FAIL: bead weights not found at {args.bead_weights}")
        return 1
    if not args.no_beads and args.classify and not args.classifier_weights.exists():
        print(f"FAIL: classifier weights not found at {args.classifier_weights} (pass --no-classify to skip, or --no-beads)")
        return 1
    if args.no_display and args.save is None:
        print("FAIL: --no-display requires --save (nothing to do otherwise)")
        return 1

    detector = YOLO(str(args.detector_weights))
    classifier_weights = args.classifier_weights if (not args.no_beads and args.classify) else None
    pipeline = Pipeline(detector, args.bead_weights, classifier_weights, args.device)

    if args.source.suffix.lower() in IMAGE_EXTS:
        return run_on_image(pipeline, args)
    return run_on_video(pipeline, args)


if __name__ == "__main__":
    raise SystemExit(main())
