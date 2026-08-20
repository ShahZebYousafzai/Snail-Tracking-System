"""Bead localization + rotation-invariant position (1..5) assignment.

Bead localization uses a trained detector (weights/detector_beads.pt,
single class "bead") -- classical CV (Hough circles, HSV threshold +
distance-transform peak-finding) was tried first and rejected: even with
per-crop adaptive threshold search exploiting the known count of 5 beads,
real crops gave 5/4/4/5/6 detections on a 5-crop sample, and every parameter
set that worked on one crop broke another. SAM3 prompted with "small white
circular disc with a printed digit..." found the 5 beads correctly on 4/5
of that same sample (the failure was a genuinely edge-on/unreadable view),
so its output was used to bootstrap a Roboflow-reviewed dataset and train
this detector -- precision 0.986 / recall 0.952 / mAP50 0.991 on the
held-out split.

Strategy for position assignment (per discussion): the 5 digit-beads are
rigidly glued together, so their positions relative to each other never
change -- only the whole cluster's rotation on screen does. The
snail-body-centroid -> tag-cluster-centroid vector is used as the rotation
reference (chosen over motion heading, which src/test_heading_direction.py
showed is only "confident" 0-21% of frames -- and 0% for two tracks that
barely moved -- because it's available every frame, including fully
stationary ones).

For each detected bead, its angle relative to that reference axis is a
rotation-invariant coordinate: sorting beads by this relative angle gives a
consistent position-1..5 assignment regardless of how the cluster is
oriented on screen in any given frame.

This is the core piece `tag_reader.py` will build on once bead crops are
labeled with digits; for now it only assigns *positions*, not digit values.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BEAD_WEIGHTS = PROJECT_ROOT / "weights" / "detector_beads.pt"

_model_cache: dict[str, YOLO] = {}


def _get_model(weights_path: Path) -> YOLO:
    key = str(weights_path)
    if key not in _model_cache:
        _model_cache[key] = YOLO(key)
    return _model_cache[key]


@dataclass
class Bead:
    cx: float          # center, in the *source frame's* coordinates (not crop-local)
    cy: float
    radius: float
    position: int = -1  # 1..5 once assigned, -1 if unassigned


def detect_beads(
    tag_crop: np.ndarray,
    crop_offset: tuple[int, int] = (0, 0),
    conf: float = 0.4,
    weights_path: Path = DEFAULT_BEAD_WEIGHTS,
) -> list[Bead]:
    """Locate digit-beads inside a tag crop using the trained bead detector."""
    model = _get_model(weights_path)
    result = model.predict(tag_crop, conf=conf, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    ox, oy = crop_offset
    xyxy = boxes.xyxy.cpu().numpy()
    beads = []
    for x1, y1, x2, y2 in xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        radius = max(x2 - x1, y2 - y1) / 2
        beads.append(Bead(cx=float(cx + ox), cy=float(cy + oy), radius=float(radius)))
    return beads


def assign_positions(beads: list[Bead], ref_axis: tuple[float, float], centroid: tuple[float, float] | None = None) -> list[Bead]:
    """Assign position 1..N by sorting beads on their angle relative to
    `ref_axis` (a vector, e.g. body-centroid -> tag-cluster-centroid),
    measured around `centroid` (defaults to the beads' own mean position).

    Position 1 = the bead whose angle is closest to the reference axis
    itself; positions 2..N proceed clockwise from there. This is an
    internally-consistent convention (stable frame-to-frame for the same
    physical cluster) -- it is not guaranteed to match a manually-chosen
    "position 1" without a one-time fixed offset calibrated per snail.
    """
    if not beads:
        return beads
    if centroid is None:
        cx = sum(b.cx for b in beads) / len(beads)
        cy = sum(b.cy for b in beads) / len(beads)
    else:
        cx, cy = centroid

    ref_angle = math.atan2(ref_axis[1], ref_axis[0])

    def relative_angle(b: Bead) -> float:
        a = math.atan2(b.cy - cy, b.cx - cx) - ref_angle
        return a % (2 * math.pi)

    ordered = sorted(beads, key=relative_angle)
    for i, b in enumerate(ordered, start=1):
        b.position = i
    return ordered
