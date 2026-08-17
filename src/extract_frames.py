"""Extract frames from a raw video at a fixed sampling rate for annotation review."""

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


def extract(video: str, fps: float, width: int, quality: int) -> Path:
    src = RAW_DIR / video
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = FRAMES_DIR / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source is 3840x2160; downscale to `width` (height auto, even) to cut
    # per-frame size drastically while keeping the tag disc legible enough
    # for both bbox annotation and tag-code classifier crops.
    vf = f"fps={fps},scale={width}:-2"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-q:v", str(quality),
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Filename under data/raw, e.g. C0021.MP4")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate in frames per second")
    parser.add_argument("--width", type=int, default=1920,
                         help="Output frame width in px; height scales to preserve aspect (source is 3840x2160)")
    parser.add_argument("--quality", type=int, default=4,
                         help="ffmpeg -q:v JPEG quality, lower = better/larger (2=high, 4=good/smaller, 8=low)")
    args = parser.parse_args()

    out_dir = extract(args.video, args.fps, args.width, args.quality)
    n_frames = len(list(out_dir.glob("*.jpg")))
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.jpg")) / (1024 * 1024)
    print(f"Extracted {n_frames} frames to {out_dir} at {args.fps} fps, {args.width}px wide "
          f"({total_mb:.1f} MB total, {total_mb / max(n_frames, 1):.2f} MB/frame)")


if __name__ == "__main__":
    main()
