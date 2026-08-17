"""Overwrite annotations on already-uploaded Roboflow images.

Used when images are already present in the project (e.g. after a failed
delete+reupload cycle where Roboflow's content-hash dedup silently kept the
old annotations) and we need to force-replace each image's annotations with
a fresh COCO pre-label export, without re-uploading the image bytes.

Builds a minimal per-image Pascal VOC XML (a format save_annotation reliably
parses from a raw string) from the COCO boxes and pushes it with
annotation_overwrite=True.
"""

import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_voc_xml(filename: str, width: int, height: int, boxes) -> str:
    objects = []
    for cls_name, (x0, y0, x1, y1) in boxes:
        objects.append(f"""  <object>
    <name>{escape(cls_name)}</name>
    <bndbox>
      <xmin>{int(round(x0))}</xmin>
      <ymin>{int(round(y0))}</ymin>
      <xmax>{int(round(x1))}</xmax>
      <ymax>{int(round(y1))}</ymax>
    </bndbox>
  </object>""")
    return f"""<annotation>
  <filename>{escape(filename)}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
{chr(10).join(objects)}
</annotation>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", type=Path, default=PROJECT_ROOT / "data" / "roboflow_exports" / "sam3_prelabels.json")
    parser.add_argument("--project", default="snail-identity-tracking-mvilv")
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ws = rf.workspace()
    project = ws.project(args.project)

    print("Fetching current image list from Roboflow...")
    pages = list(project.search_all(fields=["id", "name"]))
    name_to_id = {item["name"]: item["id"] for page in pages for item in page}
    print(f"Found {len(name_to_id)} images in project")

    coco = json.loads(args.coco_json.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = {img["id"]: img for img in coco["images"]}
    boxes_by_image = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        box = (cats[ann["category_id"]], (x, y, x + w, y + h))
        boxes_by_image.setdefault(ann["image_id"], []).append(box)

    updated, missing = 0, 0
    for img_id, img_meta in by_image.items():
        fname = img_meta["file_name"]
        rf_id = name_to_id.get(fname)
        if rf_id is None:
            print(f"  SKIP {fname}: not found in Roboflow project")
            missing += 1
            continue

        xml = build_voc_xml(fname, img_meta["width"], img_meta["height"], boxes_by_image.get(img_id, []))
        project.save_annotation(
            annotation_path={"name": f"{Path(fname).stem}.xml", "rawText": xml},
            image_id=rf_id,
            is_prediction=True,
            annotation_overwrite=True,
        )
        updated += 1
        if updated % 20 == 0:
            print(f"  updated {updated}/{len(by_image)}")

    print(f"Done. Updated {updated}, missing {missing}.")


if __name__ == "__main__":
    main()
