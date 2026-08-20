"""Bootstrap-label bead crops with PaddleOCR (isolated single digits read
far better than the whole 5-bead cluster did, see prior test), keeping only
confident single-digit reads. Normalizes the common 'O'/'0' confusion.
Sorts accepted crops into data/bead_crops/labeled/<digit>/ (ImageFolder
layout, ready for classifier training); low-confidence/ambiguous crops are
left in data/bead_crops/unlabeled/ for manual review.
"""

import shutil
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "bead_crops" / "unlabeled"
OUT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "labeled"
MIN_SCORE = 0.5


def normalize(text: str) -> str | None:
    t = text.strip().upper()
    if t == "O":
        t = "0"
    if len(t) == 1 and t.isdigit():
        return t
    return None


def main() -> None:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
    )

    files = sorted(SRC_DIR.glob("*.jpg"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in "0123456789":
        (OUT_DIR / d).mkdir(exist_ok=True)

    counts = {d: 0 for d in "0123456789"}
    n_accepted = 0
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        scale = max(1, 200 // max(h, 1))
        big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        out = ocr.predict(big)
        texts, scores = [], []
        if out:
            texts = out[0].get("rec_texts", [])
            scores = out[0].get("rec_scores", [])

        if len(texts) == 1 and scores and scores[0] >= MIN_SCORE:
            digit = normalize(texts[0])
            if digit is not None:
                shutil.copy2(f, OUT_DIR / digit / f.name)
                counts[digit] += 1
                n_accepted += 1

        if i % 100 == 0:
            print(f"processed {i}/{len(files)}, accepted so far: {n_accepted}", flush=True)

    print(f"\nDONE. {n_accepted}/{len(files)} crops auto-labeled with confidence >= {MIN_SCORE}")
    for d in "0123456789":
        print(f"  digit {d}: {counts[d]} crops")


if __name__ == "__main__":
    main()
