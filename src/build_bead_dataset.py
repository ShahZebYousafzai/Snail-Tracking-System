"""Build an unlabeled bead-crop dataset by running the trained snail/tag
YOLO detector directly on the raw videos (native resolution), cropping each
detected tag region, then running SAM3 on the tag crop to localize
individual beads and save each one as its own crop.

This deliberately samples from the raw videos (data/raw/*.MP4, native
3840x2160) rather than the dataset_v2 Roboflow export: those images are
resized to 512x512, which shrinks each tag box to ~20px and each bead to a
handful of pixels -- nowhere near enough detail to read a digit off of.
Native video frames give ~140x130px tag boxes instead, which is the same
scale the existing bead detector (weights/detector_beads.pt) was itself
trained on (see src/extract_tag_crops.py).

Output crops are UNLABELED -- a human still needs to assign the digit (0-9)
and rotation angle per crop before they're usable to train the planned
two-head bead classifier (digit class + rotation angle). The manifest
records enough to trace each crop back to its source video/frame/tag box
for review.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"
SAM3_CHECKPOINT = PROJECT_ROOT / "weights" / "sam3_image" / "sam3.pt"
DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "detector_yolo.pt"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(SAM3_REPO))

TAG_CLASS = 1  # 0 = snail, 1 = tag (weights/detector_yolo.pt)
PROMPT = "small white circular disc with a printed digit, part of a numbered tag glued to a snail shell"

# From extract_tag_crops.py / extract_bead_crops.py conventions.
TAG_PAD_FRAC = 0.25
BEAD_PAD_FRAC = 0.15


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_sam3_processor(confidence_threshold: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(SAM3_CHECKPOINT), device=device)
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def crop_with_padding(image: np.ndarray, x1, y1, x2, y2, pad_frac):
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return image[y1p:y2p, x1p:x2p]


def detect_tag_boxes(detector, frame, conf, imgsz):
    result = detector.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    tag_boxes = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), cls in zip(xyxy, clss):
            if int(cls) == TAG_CLASS:
                tag_boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return tag_boxes


def find_beads(processor, tag_crop_rgb: Image.Image, nms_iou: float):
    base_state = processor.set_image(tag_crop_rgb)
    state = processor.set_text_prompt(PROMPT, dict(base_state))
    boxes = state.get("boxes")
    scores = state.get("scores")
    processor.reset_all_prompts(state)
    if boxes is None or len(boxes) == 0:
        return []
    boxes_f = boxes.float().cpu()
    scores_f = scores.float().cpu()
    keep = nms(boxes_f, scores_f, iou_threshold=nms_iou)
    results = []
    for idx in keep.tolist():
        x0, y0, x1, y1 = (float(v) for v in boxes_f[idx].tolist())
        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            continue
        results.append((x0, y0, x1, y1, float(scores_f[idx])))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=None, help="video filenames under data/raw (default: all .MP4 there)")
    parser.add_argument("--detector-weights", type=Path, default=DETECTOR_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.4, help="tag detector confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=25, help="sample every Nth frame (default 25 ~= 1fps at native 25fps)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "bead_crops" / "from_video_unlabeled")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="SAM3 bead confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="max sampled frames per video (0 = no limit); for smoke-testing")
    args = parser.parse_args()

    videos = args.videos or [p.name for p in sorted(RAW_DIR.glob("*.MP4"))]
    if not videos:
        print(f"FAIL: no videos found under {RAW_DIR}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    manifest_rows = []

    detector = YOLO(str(args.detector_weights))
    processor = build_sam3_processor(args.score_thresh)

    inference_ctx = torch.inference_mode()
    inference_ctx.__enter__()
    autocast_ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    autocast_ctx.__enter__()

    total_beads = 0
    total_tags = 0
    try:
        for video_name in videos:
            video_path = RAW_DIR / video_name
            if not video_path.exists():
                print(f"skip: {video_path} not found")
                continue

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{video_name}] {total_frames} frames, sampling every {args.stride}", flush=True)

            frame_idx = 0
            sampled = 0
            stop_video = False
            while not stop_video:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % args.stride == 0:
                    tag_boxes = detect_tag_boxes(detector, frame, args.conf, args.imgsz)
                    for tag_idx, (tx1, ty1, tx2, ty2) in enumerate(tag_boxes):
                        tag_crop_bgr = crop_with_padding(frame, tx1, ty1, tx2, ty2, TAG_PAD_FRAC)
                        if tag_crop_bgr.size == 0:
                            continue
                        total_tags += 1
                        tag_crop_rgb = Image.fromarray(cv2.cvtColor(tag_crop_bgr, cv2.COLOR_BGR2RGB))

                        beads = find_beads(processor, tag_crop_rgb, args.nms_iou)
                        for bead_idx, (bx1, by1, bx2, by2, score) in enumerate(beads):
                            bead_crop = crop_with_padding(tag_crop_bgr, bx1, by1, bx2, by2, BEAD_PAD_FRAC)
                            if bead_crop.size == 0:
                                continue
                            out_name = f"{video_path.stem}_frame{frame_idx:06d}_tag{tag_idx}_bead{bead_idx}.jpg"
                            out_path = args.out_dir / out_name
                            cv2.imwrite(str(out_path), bead_crop)
                            manifest_rows.append({
                                "crop_path": relative_to_project(out_path),
                                "video": video_name,
                                "frame": frame_idx,
                                "tag_idx": tag_idx,
                                "bead_idx": bead_idx,
                                "sam3_score": round(score, 4),
                            })
                            total_beads += 1

                    sampled += 1
                    if sampled % 20 == 0:
                        if torch.cuda.is_available():
                            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                            print(f"  [{video_name}] {sampled} frames sampled ({frame_idx}/{total_frames}) | "
                                  f"{total_beads} bead crops so far (VRAM: {vram_mb:.0f} MB)", flush=True)
                            torch.cuda.empty_cache()
                        else:
                            print(f"  [{video_name}] {sampled} frames sampled ({frame_idx}/{total_frames}) | "
                                  f"{total_beads} bead crops so far", flush=True)
                    if args.limit and sampled >= args.limit:
                        stop_video = True

                frame_idx += 1
            cap.release()
    finally:
        autocast_ctx.__exit__(None, None, None)
        inference_ctx.__exit__(None, None, None)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "video", "frame", "tag_idx", "bead_idx", "sam3_score"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDONE. {total_beads} bead crops from {total_tags} tag crops -> {args.out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
