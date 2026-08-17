"""Generate snail/tag box proposals for a folder of frames using a local SAM3 checkpoint.

Uses SAM3's text-prompted concept segmentation (Sam3Processor.set_text_prompt) to find
snail instances and individual number-disc instances per frame, without needing
per-frame manual clicks. Each snail's shell tag is actually a small cluster of several
single-digit discs (e.g. "1","4","6","3","0" arranged together) -- for training the
downstream tag-code classifier we want ONE crop covering the whole cluster, not one box
per digit. So individual digit-disc detections are grouped by which snail's box contains
them, and collapsed into a single union box per snail before export.

Output is a COCO-format JSON (one file covering all frames) suitable for uploading to
Roboflow as pre-annotations ahead of human correction.

This is a first-draft labeler only -- output is expected to be corrected by a human in
Roboflow before it's used for training. It intentionally does not try to be perfect; it
exists to remove the "draw every box from scratch" cost.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"
# build_sam3_image_model expects the plain "sam3" image checkpoint, not the
# "sam3.1_multiplex" one (that's for the video/multiplex builder instead).
CHECKPOINT = PROJECT_ROOT / "weights" / "sam3_image" / "sam3.pt"

sys.path.insert(0, str(SAM3_REPO))

SNAIL_CAT_ID = 1
TAG_CAT_ID = 2
CATEGORIES = [
    {"id": SNAIL_CAT_ID, "name": "snail"},
    {"id": TAG_CAT_ID, "name": "tag"},
]

# Text prompts fed to SAM3's promptable concept segmentation -- one pass per
# concept per frame, since snail/tag count and position varies frame to frame.
PROMPTS = {
    SNAIL_CAT_ID: "snail",
    TAG_CAT_ID: "small white circular disc with a printed digit, part of a numbered tag glued to a snail's shell",
}


def box_center(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2


def box_contains_point(box, point, pad=10):
    x0, y0, x1, y1 = box
    px, py = point
    return (x0 - pad) <= px <= (x1 + pad) and (y0 - pad) <= py <= (y1 + pad)


def union_box(boxes):
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def group_digits_into_tags(snail_boxes, digit_boxes_scores):
    """Assign each digit-disc box to the snail box that contains its center
    (within a small padding, since discs can sit slightly over the shell
    edge), then collapse each snail's assigned digits into one union box.
    Digits that don't fall inside any snail box are dropped as likely false
    positives (e.g. grass texture) rather than emitted as spurious tags.
    """
    assigned = [[] for _ in snail_boxes]
    for box, score in digit_boxes_scores:
        center = box_center(box)
        best_idx, best_area = None, None
        for i, sbox in enumerate(snail_boxes):
            if box_contains_point(sbox, center):
                area = (sbox[2] - sbox[0]) * (sbox[3] - sbox[1])
                if best_area is None or area < best_area:
                    best_idx, best_area = i, area
        if best_idx is not None:
            assigned[best_idx].append((box, score))

    tag_boxes_scores = []
    for digits in assigned:
        if not digits:
            continue
        boxes = [d[0] for d in digits]
        scores = [d[1] for d in digits]
        tag_boxes_scores.append((union_box(boxes), sum(scores) / len(scores)))
    return tag_boxes_scores


def build_processor(confidence_threshold: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # bpe_path intentionally omitted: build_sam3_image_model resolves it via
    # pkg_resources to sam3/assets/bpe_simple_vocab_16e6.txt.gz (inside the
    # package dir), which is where it actually ships -- not repo-root assets/.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(CHECKPOINT), device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    return processor


def run(frames_dir: Path, out_path: Path, score_thresh: float = 0.5, nms_iou: float = 0.4) -> None:
    processor = build_processor(score_thresh)

    images_meta = []
    annotations = []
    ann_id = 1

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found under {frames_dir}", file=sys.stderr)
        sys.exit(1)

    # SAM3 internally hardcodes some fused ops (e.g. sam3/perflib/fused.py's
    # addmm_act) to compute in bfloat16 regardless of the model's own dtype.
    # Running the whole forward pass under autocast(bfloat16) keeps every op
    # consistent instead of hand-casting the model -- this matches the
    # official example notebooks' own setup.
    inference_ctx = torch.inference_mode()
    inference_ctx.__enter__()
    autocast_ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    autocast_ctx.__enter__()

    for img_id, frame_path in enumerate(frame_paths, start=1):
        image = Image.open(frame_path).convert("RGB")
        w, h = image.size
        images_meta.append({
            "id": img_id,
            "file_name": frame_path.name,
            "width": w,
            "height": h,
        })

        base_state = processor.set_image(image)

        per_category_boxes = {}
        for cat_id, prompt in PROMPTS.items():
            state = processor.set_text_prompt(prompt, dict(base_state))
            boxes = state.get("boxes")
            scores = state.get("scores")
            if boxes is None or len(boxes) == 0:
                per_category_boxes[cat_id] = []
                processor.reset_all_prompts(state)
                continue
            # SAM3 doesn't dedupe its own candidate proposals -- without NMS,
            # a single physical snail/digit produces several overlapping boxes
            # of slightly different sizes, which is noise rather than a time
            # save for whoever corrects these afterward.
            boxes_f = boxes.float().cpu()
            scores_f = scores.float().cpu()
            keep = nms(boxes_f, scores_f, iou_threshold=nms_iou)
            kept = []
            for idx in keep.tolist():
                x0, y0, x1, y1 = [float(v) for v in boxes_f[idx].tolist()]
                if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                    continue
                kept.append(((x0, y0, x1, y1), float(scores_f[idx])))
            per_category_boxes[cat_id] = kept
            processor.reset_all_prompts(state)

        snail_boxes = [b for b, _ in per_category_boxes.get(SNAIL_CAT_ID, [])]
        digit_boxes_scores = per_category_boxes.get(TAG_CAT_ID, [])

        for box, score in per_category_boxes.get(SNAIL_CAT_ID, []):
            x0, y0, x1, y1 = box
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": SNAIL_CAT_ID,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
                "score": score,
            })
            ann_id += 1

        # Collapse each snail's individual digit-disc detections into one
        # union box -- that single crop is what the downstream tag-code
        # classifier will be trained on, not per-digit crops.
        for box, score in group_digits_into_tags(snail_boxes, digit_boxes_scores):
            x0, y0, x1, y1 = box
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": TAG_CAT_ID,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
                "score": score,
            })
            ann_id += 1

        if img_id % 10 == 0:
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"  processed {img_id}/{len(frame_paths)} frames (VRAM: {vram_mb:.0f} MB)")
                torch.cuda.empty_cache()
            else:
                print(f"  processed {img_id}/{len(frame_paths)} frames")

    autocast_ctx.__exit__(None, None, None)
    inference_ctx.__exit__(None, None, None)

    coco = {
        "info": {"description": "SAM3 pre-labels for Roboflow model-assisted review"},
        "licenses": [],
        "images": images_meta,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {len(annotations)} pre-annotations across {len(images_meta)} images to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", type=Path, help="Folder of frame .jpg files")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "sam3_prelabels.json")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.4, help="IoU threshold for de-duplicating overlapping candidate boxes")
    args = parser.parse_args()

    run(args.frames_dir, args.out, args.score_thresh, args.nms_iou)


if __name__ == "__main__":
    main()
