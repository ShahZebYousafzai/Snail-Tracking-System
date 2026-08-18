"""Upload extracted frames plus SAM3 pre-labels to the Roboflow project as
model-assisted-labeling predictions, ready for human review/correction.
"""

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_upload_folder(frames_dir: Path, coco_json: Path, staging_dir: Path) -> Path:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Re-encode rather than copy: Roboflow's prediction/annotation cache is
    # keyed off image content hash *workspace-wide*, not per-project -- a
    # byte-identical re-upload silently reattaches whatever annotations were
    # ever associated with that hash before, ignoring the new annotation
    # file. Re-saving with a different JPEG quality changes the bytes (and
    # hash) while keeping the image visually identical, so this upload is
    # treated as genuinely new.
    for frame in frames_dir.glob("*.jpg"):
        img = Image.open(frame)
        img.save(staging_dir / frame.name, "JPEG", quality=93)

    shutil.copy2(coco_json, staging_dir / "_annotations.coco.json")
    return staging_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "data" / "frames" / "C0021")
    parser.add_argument("--coco-json", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "sam3_prelabels.json")
    parser.add_argument("--project", default="snail-tracking-azwre")
    parser.add_argument("--staging-dir", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "_upload_staging")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    args = parser.parse_args()

    from roboflow import Roboflow

    api_key = os.environ["ROBOFLOW_API_KEY"]
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace()

    upload_dir = build_upload_folder(args.frames_dir, args.coco_json, args.staging_dir)
    print(f"Staged {len(list(upload_dir.glob('*.jpg')))} images + annotations at {upload_dir}")

    ws.upload_dataset(
        dataset_path=str(upload_dir),
        project_name=args.project,
        project_type="object-detection",
        is_prediction=True,
        split=args.split,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
