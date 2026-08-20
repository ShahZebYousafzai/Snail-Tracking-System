"""Quick feasibility test: run off-the-shelf PaddleOCR (zero fine-tuning) on
real tag crops pulled from the trained detector, to compare against the
closed-set classifier approach in the plan.

Samples frames spread across the video, crops every `tag` detection with
padding, runs PaddleOCR on each crop, and builds a contact sheet (crop +
predicted text + confidence) for manual visual review against the known
codes -- there's no ground-truth code label in this repo yet (see prior
discussion), so scoring is manual, not automatic.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAG_CLASS = 1
PAD_FRAC = 0.25  # padding around detected tag box, as a fraction of box size


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return frame[y1p:y2p, x1p:x2p]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0021.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "paddleocr_test")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--n-frames", type=int, default=40, help="frames sampled evenly across the video")
    parser.add_argument("--max-crops", type=int, default=40)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(args.source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total - 1, args.n_frames, dtype=int)

    model = YOLO(str(args.weights))
    # Mobile (not server) det/rec models: lighter, closer to what a real-time
    # deployment would actually use, and sidesteps a oneDNN/PIR crash hit
    # with the server models' text-detection graph on this CPU build.
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
    )

    crops = []
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, cf in zip(xyxy, clss, confs):
            if int(c) != TAG_CLASS:
                continue
            crop = crop_with_padding(frame, x1, y1, x2, y2, PAD_FRAC)
            if crop.size == 0:
                continue
            crops.append((int(fi), crop, float(cf)))
        if len(crops) >= args.max_crops:
            break
    cap.release()
    print(f"collected {len(crops)} tag crops from {len(frame_idxs)} sampled frames")

    # Upscale small crops -- OCR models expect reasonably sized text.
    results = []
    for fi, crop, det_conf in crops:
        h, w = crop.shape[:2]
        scale = max(1, 160 // max(h, 1))
        big = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        crop_path = crops_dir / f"frame{fi:05d}_conf{det_conf:.2f}.jpg"
        cv2.imwrite(str(crop_path), big)

        ocr_out = ocr.predict(big)
        texts, scores = [], []
        if ocr_out:
            page = ocr_out[0]
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
        pred_text = " ".join(texts) if texts else "(none)"
        pred_score = max(scores) if scores else 0.0
        results.append((crop_path, big, pred_text, pred_score))
        print(f"frame {fi:5d} det_conf={det_conf:.2f}  OCR: '{pred_text}'  score={pred_score:.2f}")

    # Build a contact sheet: crop thumbnail + predicted text, in a grid.
    cell_w, cell_h = 220, 200
    cols = 6
    rows = (len(results) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for i, (crop_path, big, pred_text, pred_score) in enumerate(results):
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * cell_w
        h, w = big.shape[:2]
        scale = min((cell_w - 10) / w, (cell_h - 40) / h)
        thumb = cv2.resize(big, (int(w * scale), int(h * scale)))
        th, tw = thumb.shape[:2]
        sheet[y0 + 5:y0 + 5 + th, x0 + 5:x0 + 5 + tw] = thumb
        label = f"{pred_text} ({pred_score:.2f})"
        cv2.putText(sheet, label[:22], (x0 + 5, y0 + cell_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    sheet_path = args.out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\ncontact sheet saved: {sheet_path}")
    print(f"individual crops saved under: {crops_dir}")


if __name__ == "__main__":
    main()
