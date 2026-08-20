"""Run the trained detector on CPU and overlay live FPS on each frame.

Useful for checking whether the detector is fast enough without a GPU
(e.g. for a deployment target that won't have one). Forces device='cpu'
regardless of what's available, so results reflect true CPU-only speed.

Usage::

    python src/infer_cpu_fps.py --source data/raw/C0025.MP4 --save outputs/C0025_cpu.mp4
    python src/infer_cpu_fps.py --source data/raw/C0025.MP4 --display   # live window, run locally (not headless)
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "C0025.MP4")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "detector_yolo.pt")
    parser.add_argument("--save", type=Path, default=PROJECT_ROOT / "outputs" / "cpu_fps_test.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out-width", type=int, default=1280)
    parser.add_argument("--display", action="store_true", help="show a live window instead of/as well as saving")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = process the whole video")
    parser.add_argument("--smoothing", type=int, default=15, help="frames averaged for the displayed FPS")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    model.to("cpu")

    cap = cv2.VideoCapture(str(args.source))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = None
    out_w, out_h = args.out_width, int(src_h * (args.out_width / src_w))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("failed to open output video writer")

    frame_times: deque[float] = deque(maxlen=args.smoothing)
    n = 0
    t_start = time.perf_counter()

    for result in model.predict(source=str(args.source), conf=args.conf, imgsz=args.imgsz,
                                 device="cpu", stream=True, verbose=False, iou=0.25):
        t0 = time.perf_counter()
        annotated = result.plot()
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        inst_fps = 1.0 / (sum(frame_times) / len(frame_times))
        cv2.putText(annotated, f"CPU FPS: {inst_fps:.1f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4, cv2.LINE_AA)

        resized = cv2.resize(annotated, (out_w, out_h))
        if writer is not None:
            writer.write(resized)
        if args.display:
            cv2.imshow("CPU inference", resized)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        n += 1
        if n % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"processed {n}/{total} | running avg fps: {n/elapsed:.2f}", flush=True)
        if args.max_frames and n >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\nDONE. {n} frames in {elapsed:.1f}s -> overall avg {n/elapsed:.2f} fps (CPU, imgsz={args.imgsz})")
    if args.save:
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
