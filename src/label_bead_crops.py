"""Keyboard-driven labeling tool for bead crops: assign a digit (0-9) and a
rotation angle to each crop, for training the planned two-head classifier
(digit class + rotation angle, 36 bins of 10 degrees -- see project plan in
.claude/Snail Identity Tracking.md).

Controls:
  a / Left arrow     rotate the preview -10 degrees
  d / Right arrow     rotate the preview +10 degrees
  w / Up arrow        rotate the preview +90 degrees (quick flip)
  s / Down arrow      rotate the preview -90 degrees (quick flip)
  r                    reset rotation to 0
  0-9                  confirm: digit = key pressed, rotation = current
                        display angle -- i.e. the angle you rotated the RAW
                        crop by to make the digit look upright. That's the
                        label the two-head model will learn to predict from
                        the unrotated crop.
  u                    mark unreadable (no digit label, but logged so it
                        won't be shown again)
  n / Space            skip -- leave this crop alone, nothing written, it
                        will show up again next run (unlike 'u')
  b                    go back to the previous crop (undo the last decision)
  q / Esc              quit and save

Progress is written incrementally (one line per decision, flushed
immediately) to --out, so you can quit anytime with no lost work. On
startup, anything already present in --out OR in the PaddleOCR auto-labels
(data/bead_crops/from_video_dedup/paddleocr_labels.csv) is skipped
automatically -- resuming later just picks up where you left off.

Arrow-key codes are notoriously OpenCV-build/platform dependent and this
tool hasn't been interactively tested (no live display available in the
environment it was written in) -- the a/w/s/d letters are the reliable
fallback if arrows don't register.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup" / "manifest.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup" / "manual_labels.csv"
DEFAULT_PADDLEOCR_LABELS = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup" / "paddleocr_labels.csv"

DISPLAY_SIZE = 360
FIELDNAMES = ["crop_path", "digit", "rotation_deg"]

# Common Windows/OpenCV waitKeyEx extended codes for arrow keys.
ARROW_LEFT = {2424832, 65361}
ARROW_RIGHT = {2555904, 65363}
ARROW_UP = {2490368, 65362}
ARROW_DOWN = {2621440, 65364}


def load_labeled_paths(*csv_paths: Path) -> set[str]:
    labeled = set()
    for p in csv_paths:
        if p and p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    labeled.add(row["crop_path"])
    return labeled


def prepare_base(crop: np.ndarray, display_size: int = DISPLAY_SIZE) -> np.ndarray:
    h, w = crop.shape[:2]
    side = int(max(h, w) * 1.6)
    canvas = np.full((side, side, 3), 210, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return cv2.resize(canvas, (display_size, display_size), interpolation=cv2.INTER_CUBIC)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(210, 210, 210))


def render(base: np.ndarray, angle: float, crop_path: str, idx: int, total: int, unreadable_count: int) -> np.ndarray:
    canvas = np.full((DISPLAY_SIZE + 150, DISPLAY_SIZE, 3), 30, dtype=np.uint8)
    canvas[:DISPLAY_SIZE, :] = rotate(base, angle)
    lines = [
        f"{idx}/{total}   (unreadable so far: {unreadable_count})",
        f"rotation: {int(angle % 360)} deg",
        "0-9 = confirm digit    u = unreadable    n = skip    b = back    q = quit",
        "rotate: a/d or Left/Right (+-10)   w/s or Up/Down (+-90)   r = reset",
    ]
    y = DISPLAY_SIZE + 22
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(canvas, Path(crop_path).name, (8, DISPLAY_SIZE + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 220, 150), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paddleocr-labels", type=Path, default=DEFAULT_PADDLEOCR_LABELS,
                         help="crops already auto-labeled here are skipped (pass a missing path to disable)")
    parser.add_argument("--ignore-out", action="store_true",
                         help="show crops even if already present in --out, for re-reviewing/correcting "
                              "mislabeled crops (e.g. via src/find_label_disagreements.py) -- a correction is "
                              "appended as a new row, and every downstream loader in this project treats the "
                              "LAST non-empty-digit row for a crop_path as authoritative, so relabeling with the "
                              "correct digit overrides the old one. NOTE: pressing 'u' (unreadable) during a "
                              "correction pass does NOT retract a prior wrong digit label -- downstream loaders "
                              "filter empty-digit rows out before the last-wins merge, so the old digit would "
                              "still win. Fine for the common case (wrong digit -> right digit); if you need to "
                              "retract a label to unreadable, edit manual_labels.csv directly.")
    args = parser.parse_args()

    skip_sources = [args.paddleocr_labels] if args.ignore_out else [args.paddleocr_labels, args.out]
    already_labeled = load_labeled_paths(*skip_sources)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    todo = [r["crop_path"] for r in rows if r["crop_path"] not in already_labeled]
    print(f"{len(rows)} total crops, {len(already_labeled)} already labeled, {len(todo)} remaining")

    if not todo:
        print("nothing to label")
        return

    out_exists = args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()
        out_f.flush()

    window = "bead labeler (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    idx = 0
    angle = 0.0
    unreadable_count = 0
    history: list[str] = []

    while idx < len(todo):
        crop_path = todo[idx]
        img = cv2.imread(str(PROJECT_ROOT / crop_path))
        if img is None:
            idx += 1
            continue
        base = prepare_base(img)

        advance = False
        while not advance:
            frame = render(base, angle, crop_path, idx + 1, len(todo), unreadable_count)
            cv2.imshow(window, frame)
            key = cv2.waitKeyEx(0)

            if key in (27, ord("q")):
                out_f.close()
                cv2.destroyAllWindows()
                print(f"stopped. progress saved -> {args.out}")
                return
            elif key in ARROW_LEFT or key in (ord("a"),):
                angle -= 10
            elif key in ARROW_RIGHT or key in (ord("d"),):
                angle += 10
            elif key in ARROW_UP or key in (ord("w"),):
                angle += 90
            elif key in ARROW_DOWN or key in (ord("s"),):
                angle -= 90
            elif key == ord("r"):
                angle = 0
            elif key == ord("u"):
                writer.writerow({"crop_path": crop_path, "digit": "", "rotation_deg": ""})
                out_f.flush()
                unreadable_count += 1
                history.append(crop_path)
                idx += 1
                angle = 0
                advance = True
            elif key in (ord("n"), ord(" ")):
                # Skip: nothing written, so this crop shows up again next run.
                history.append(crop_path)
                idx += 1
                angle = 0
                advance = True
            elif key == ord("b"):
                if history:
                    idx -= 1
                    history.pop()
                    angle = 0
                advance = True
            elif 0 <= key < 256 and chr(key).isdigit():
                digit = chr(key)
                writer.writerow({"crop_path": crop_path, "digit": digit, "rotation_deg": int(angle % 360)})
                out_f.flush()
                history.append(crop_path)
                idx += 1
                angle = 0
                advance = True

    out_f.close()
    cv2.destroyAllWindows()
    print(f"DONE. all {len(todo)} remaining crops labeled -> {args.out}")


if __name__ == "__main__":
    main()
