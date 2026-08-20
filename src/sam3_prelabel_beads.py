"""Generate bead (single-digit disc) box proposals for a folder of tag crops
using a local SAM3 checkpoint, for upload to Roboflow as pre-annotations.

Unlike sam3_prelabel.py (which groups individual digit-disc detections into
one union box per snail), this keeps each disc as its own "bead" box -- the
whole point here is training a detector that localizes each bead
individually, so a human can review/correct real bead boxes in Roboflow
before training.

Same first-draft-only caveat as sam3_prelabel.py: output is expected to be
corrected by a human before use.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"
CHECKPOINT = PROJECT_ROOT / "weights" / "sam3_image" / "sam3.pt"

sys.path.insert(0, str(SAM3_REPO))

BEAD_CAT_ID = 1
CATEGORIES = [{"id": BEAD_CAT_ID, "name": "bead"}]

PROMPT = "small white circular disc with a printed digit, part of a numbered tag glued to a snail shell"


def build_processor(confidence_threshold: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(CHECKPOINT), device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    return processor


def run(crops_dir: Path, out_path: Path, score_thresh: float = 0.5, nms_iou: float = 0.4) -> None:
    processor = build_processor(score_thresh)

    images_meta = []
    annotations = []
    ann_id = 1

    crop_paths = sorted(crops_dir.glob("*.jpg"))
    if not crop_paths:
        print(f"No crops found under {crops_dir}", file=sys.stderr)
        sys.exit(1)

    inference_ctx = torch.inference_mode()
    inference_ctx.__enter__()
    autocast_ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    autocast_ctx.__enter__()

    n_zero = 0
    for img_id, crop_path in enumerate(crop_paths, start=1):
        image = Image.open(crop_path).convert("RGB")
        w, h = image.size
        images_meta.append({"id": img_id, "file_name": crop_path.name, "width": w, "height": h})

        base_state = processor.set_image(image)
        state = processor.set_text_prompt(PROMPT, dict(base_state))
        boxes = state.get("boxes")
        scores = state.get("scores")

        if boxes is None or len(boxes) == 0:
            n_zero += 1
            processor.reset_all_prompts(state)
            continue

        boxes_f = boxes.float().cpu()
        scores_f = scores.float().cpu()
        keep = nms(boxes_f, scores_f, iou_threshold=nms_iou)
        for idx in keep.tolist():
            x0, y0, x1, y1 = [float(v) for v in boxes_f[idx].tolist()]
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": BEAD_CAT_ID,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
                "score": float(scores_f[idx]),
            })
            ann_id += 1
        processor.reset_all_prompts(state)

        if img_id % 20 == 0:
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"  processed {img_id}/{len(crop_paths)} crops (VRAM: {vram_mb:.0f} MB)", flush=True)
                torch.cuda.empty_cache()
            else:
                print(f"  processed {img_id}/{len(crop_paths)} crops", flush=True)

    autocast_ctx.__exit__(None, None, None)
    inference_ctx.__exit__(None, None, None)

    coco = {
        "info": {"description": "SAM3 bead pre-labels for Roboflow model-assisted review"},
        "licenses": [],
        "images": images_meta,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {len(annotations)} bead pre-annotations across {len(images_meta)} crops to {out_path}")
    print(f"{n_zero} crops got zero detections (likely genuinely unreadable/edge-on views)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("crops_dir", type=Path, help="Folder of tag-crop .jpg files")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "sam3_bead_prelabels.json")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.4)
    args = parser.parse_args()

    run(args.crops_dir, args.out, args.score_thresh, args.nms_iou)


if __name__ == "__main__":
    main()
