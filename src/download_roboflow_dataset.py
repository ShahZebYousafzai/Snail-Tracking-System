"""Download a versioned, human-corrected dataset from Roboflow in YOLO format
for detector training.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="acuvira")
    parser.add_argument("--project", default="snail-tracking-azwre")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--format", default="yolo26")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "dataset")
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset = version.download(args.format, location=str(args.out))
    print(f"Downloaded dataset version {args.version} ({args.format}) to {dataset.location}")


if __name__ == "__main__":
    main()
