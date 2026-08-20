"""Run a trained RF-DETR checkpoint on a video and visualize detections in real time.

Opens the video, runs inference frame by frame (optionally skipping frames via
--stride for speed on high-fps/4K sources), and shows the annotated result live
in a window. Press 'q' or Esc to stop early. Pass --save to also write the
annotated video to disk; pass --no-display for a headless (save-only) run.

Usage::

    python scripts/infer_video.py videos/C0025.MP4
    python scripts/infer_video.py videos/C0025.MP4 --save runs/infer/C0025_annotated.mp4
    python scripts/infer_video.py videos/C0025.MP4 --checkpoint runs/train/checkpoint_best_ema.pth \\
        --data-yaml data/construction-ppe/data.yaml
    python scripts/infer_video.py videos/C0025.MP4 --stride 2 --output-height 1080
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def log(message: str = "") -> None:
    print(message, flush=True)


def read_class_names(data_yaml: Path) -> list[str]:
    import yaml

    with data_yaml.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(name) for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="path to a video file")
    parser.add_argument("--checkpoint", type=Path,
                         default=REPO_ROOT / "runs" / "snail_tag_v5" / "checkpoint_best_ema.pth")
    parser.add_argument("--data-yaml", type=Path, default=REPO_ROOT / "data" / "roboflow_exports" / "dataset_v2" / "data.yaml")
    parser.add_argument("--conf-threshold", type=float, default=0.75)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: auto-detect)")
    parser.add_argument("--stride", type=int, default=1, help="run inference on every Nth frame (default: 1, every frame)")
    parser.add_argument("--output-height", type=int, default=720,
                         help="resize frames to this height (preserving aspect ratio) before detecting/annotating/"
                              "displaying/saving -- keeps box/text size proportionate (0 = native resolution)")
    parser.add_argument("--save", type=Path, default=None, help="write the annotated video here (e.g. runs/infer/out.mp4)")
    parser.add_argument("--no-display", action="store_true", help="skip the live preview window (headless, requires --save)")
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True,
                         help="skip JIT-compiling the model for inference (compile is on by default for speed)")
    args = parser.parse_args(argv)

    if not args.video.exists():
        log(f"FAIL: video not found at {args.video}")
        return 1
    if not args.checkpoint.exists():
        log(f"FAIL: checkpoint not found at {args.checkpoint}")
        return 1
    if not args.data_yaml.exists():
        log(f"FAIL: data.yaml not found at {args.data_yaml}")
        return 1
    if args.no_display and args.save is None:
        log("FAIL: --no-display requires --save (nothing to do otherwise)")
        return 1

    import cv2
    import supervision as sv
    import torch

    import rfdetr

    class_names = read_class_names(args.data_yaml)

    model_kwargs = {"device": args.device} if args.device else {}
    log(f"checkpoint : {args.checkpoint}")
    model = rfdetr.from_checkpoint(args.checkpoint, trust_checkpoint=True, **model_kwargs)

    device = torch.device(args.device) if args.device else model.model.device
    if args.compile and device.type == "cuda":
        log("optimizing model for inference (jit trace, fp16)...")
        model.inference(compile=True, dtype=torch.float16)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        log(f"FAIL: could not open {args.video}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.output_height and args.output_height < src_height:
        scale = args.output_height / src_height
        out_width, out_height = int(round(src_width * scale)), args.output_height
    else:
        out_width, out_height = src_width, src_height

    log(f"video      : {args.video}  ({src_width}x{src_height} @ {src_fps:.1f}fps, {total_frames} frames)")
    log(f"output     : {out_width}x{out_height}")
    log(f"stride     : {args.stride}  (effective inference fps ~{src_fps / args.stride:.1f})")

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, src_fps / args.stride, (out_width, out_height))
        log(f"saving to  : {args.save}")

    # Boxes/text are drawn on the already-resized frame, so default sizing (tuned for
    # ~720p-1080p) stays legible -- annotating at native 4K then shrinking for display
    # made the text unreadably small.
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_padding=6)

    window_name = "RF-DETR inference (q/Esc to quit)"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    processed = 0
    last_detections = sv.Detections.empty()
    last_labels: list[str] = []
    t_start = time.monotonic()
    fps_ema = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (out_width, out_height) != (src_width, src_height):
                frame = cv2.resize(frame, (out_width, out_height))

            if frame_idx % args.stride == 0:
                t0 = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(rgb, threshold=args.conf_threshold, include_source_image=False)
                dt = time.monotonic() - t0
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps

                last_detections = detections
                last_labels = [f"{class_names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
                processed += 1

            annotated = box_annotator.annotate(scene=frame.copy(), detections=last_detections)
            annotated = label_annotator.annotate(scene=annotated, detections=last_detections, labels=last_labels)
            if fps_ema is not None:
                cv2.putText(annotated, f"{fps_ema:.1f} inference fps", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None and frame_idx % args.stride == 0:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    log("stopped by user")
                    break

            frame_idx += 1
            if frame_idx % (int(src_fps) * 10) == 0:
                elapsed = time.monotonic() - t_start
                log(f"  frame {frame_idx}/{total_frames}  ({processed} inferred, {elapsed:.1f}s elapsed)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    log("")
    log(f"done: {frame_idx} frames read, {processed} inferred in {elapsed:.1f}s "
        f"({processed / elapsed:.1f} inference fps avg)" if elapsed > 0 else "done")
    if args.save:
        log(f"saved video: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
