"""Probe raw video files for fps, resolution, duration, frame count, and codec."""

import sys
from pathlib import Path

import cv2

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    duration_s = frame_count / fps if fps else 0.0

    cap.release()
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_s": round(duration_s, 1),
        "duration_hms": f"{int(duration_s // 3600)}:{int((duration_s % 3600) // 60):02d}:{duration_s % 60:05.2f}",
        "fourcc": fourcc,
    }


def main() -> None:
    videos = sorted(RAW_DIR.glob("*.MP4")) + sorted(RAW_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos found under {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    for video in videos:
        info = probe(video)
        print(f"\n{info['file']}")
        for key, val in info.items():
            if key == "file":
                continue
            print(f"  {key:14s} {val}")


if __name__ == "__main__":
    main()
