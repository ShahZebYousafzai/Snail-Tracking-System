"""Motion tracker (OC-SORT backend) + identity-reconciliation state machine.

Two pieces, per the project plan:

- `SnailTracker` wraps the trained detector + OC-SORT to produce
  frame-to-frame `track_id`s for snail-class detections. Params default to
  the tuned values found in the OC-SORT sweep (max_age=150,
  iou_threshold=0.2), which cut ID fragmentation substantially vs OC-SORT's
  defaults on this footage.

- `IdentityReconciler` turns ephemeral `track_id`s into a persistent
  `snail_code` per track:
    - a code is only (re)assigned after `n_consecutive` consecutive reads
      that agree (avoids flapping on a single noisy read)
    - once assigned, `snail_code` is *held* across frames where the tag
      isn't currently read -- this is the occlusion-hold behavior. A read
      gap resets the pending-confirmation streak, not the held code itself.
    - a brand new `track_id` always starts with `snail_code=None` and must
      re-earn identity via the same N-consecutive rule -- this is what
      "track rebirth" handling reduces to, since OC-SORT never reuses IDs.
    - `possible_swap` / `identity_reassigned` events are logged for anything
      that looks like an identity collision, for later review.

`tag_reads` (the `{track_id: (code, confidence)}` mapping the reconciler
consumes) doesn't have a producer yet -- that's `tag_reader.py`, still
pending the tag-crop labeling step. Until then, callers pass `tag_reads=None`
and every track's `snail_code` simply stays unassigned.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort
from ultralytics import YOLO

SNAIL_CLASS = 0

# From the OC-SORT parameter sweep (src/sweep_ocsort_params.py) against
# C0025.MP4: cut unique track IDs from 74 (defaults) to 28 without pushing
# max_age so high that a lost track risks reacquiring a different snail.
DEFAULT_TRACKER_KWARGS = dict(max_age=150, iou_threshold=0.2)


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    det_conf: float
    snail_code: Optional[str] = None
    pending_code: Optional[str] = None
    pending_count: int = 0
    frames_since_code_seen: int = 0
    frames_alive: int = 0
    just_reborn: bool = False


class IdentityReconciler:
    def __init__(self, n_consecutive: int = 3, swap_window: int = 30):
        self.n_consecutive = n_consecutive
        self.swap_window = swap_window
        self._states: dict[int, TrackState] = {}
        self.events: list[dict] = []

    def update_track(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        det_conf: float,
        tag_read: Optional[tuple[str, float]] = None,
        frame_idx: int = -1,
    ) -> TrackState:
        st = self._states.get(track_id)
        if st is None:
            st = TrackState(track_id=track_id, bbox=bbox, det_conf=det_conf, just_reborn=True)
            self._states[track_id] = st
        else:
            st.bbox = bbox
            st.det_conf = det_conf
            st.just_reborn = False
        st.frames_alive += 1

        if tag_read is not None:
            code, _code_conf = tag_read
            st.frames_since_code_seen = 0
            if code == st.snail_code:
                st.pending_code, st.pending_count = None, 0
            elif code == st.pending_code:
                st.pending_count += 1
                if st.pending_count >= self.n_consecutive:
                    self._assign(st, code, frame_idx)
            else:
                st.pending_code, st.pending_count = code, 1
        else:
            st.frames_since_code_seen += 1
            st.pending_code, st.pending_count = None, 0

        return st

    def _assign(self, st: TrackState, code: str, frame_idx: int) -> None:
        holders = [
            t for t in self._states.values()
            if t.snail_code == code and t.track_id != st.track_id and t.frames_since_code_seen < self.swap_window
        ]
        if holders:
            self.events.append({
                "type": "possible_swap", "frame": frame_idx, "code": code,
                "tracks": [st.track_id] + [h.track_id for h in holders],
            })
        if st.snail_code != code:
            self.events.append({
                "type": "identity_reassigned", "frame": frame_idx,
                "track_id": st.track_id, "old_code": st.snail_code, "new_code": code,
            })
        st.snail_code = code
        st.pending_code, st.pending_count = None, 0

    def active_states(self) -> list[TrackState]:
        return list(self._states.values())


class SnailTracker:
    def __init__(
        self,
        weights_path: Path,
        conf: float = 0.4,
        imgsz: int = 960,
        n_consecutive: int = 3,
        tracker_kwargs: Optional[dict] = None,
    ):
        self.model = YOLO(str(weights_path))
        self.ocsort = OcSort(min_conf=conf, **(tracker_kwargs or DEFAULT_TRACKER_KWARGS))
        self.reconciler = IdentityReconciler(n_consecutive=n_consecutive)
        self.conf = conf
        self.imgsz = imgsz

    def update(
        self,
        frame: np.ndarray,
        tag_reads: Optional[dict[int, tuple[str, float]]] = None,
        frame_idx: int = -1,
    ) -> list[TrackState]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        boxes = result.boxes
        dets = np.empty((0, 6))
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            mask = clss == SNAIL_CLASS
            if mask.any():
                dets = np.concatenate([xyxy[mask], confs[mask, None], clss[mask, None]], axis=1)

        tracks = self.ocsort.update(dets, frame)
        states = []
        for x1, y1, x2, y2, tid, conf, _cls in tracks[:, :7]:
            tid = int(tid)
            tag_read = tag_reads.get(tid) if tag_reads else None
            st = self.reconciler.update_track(tid, (float(x1), float(y1), float(x2), float(y2)), float(conf), tag_read, frame_idx)
            states.append(st)
        return states


def _demo() -> None:
    """Run the tracker on a video and save an annotated copy, with no tag
    reads (tag_reader.py doesn't exist yet) -- so every box is labeled with
    its track_id only, `code=?` throughout. This is just to visually confirm
    the tracker itself is wired correctly; identity assignment is exercised
    separately in the reconciler's own tests until the classifier exists.
    """
    import argparse
    import time
    from collections import deque
    from pathlib import Path

    import cv2

    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=project_root / "data" / "raw" / "C0025.MP4")
    parser.add_argument("--weights", type=Path, default=project_root / "weights" / "detector_yolo.pt")
    parser.add_argument("--save", type=Path, default=project_root / "outputs" / "tracker_demo.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--out-width", type=int, default=1920)
    parser.add_argument("--display", action="store_true", help="show a live window (run locally, not headless)")
    parser.add_argument("--no-save", action="store_true", help="skip writing the annotated video")
    args = parser.parse_args()

    tracker = SnailTracker(args.weights)

    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = args.out_width, int(src_h * (args.out_width / src_w))
    writer = None
    if not args.no_save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    colors = [(255, 99, 71), (60, 179, 113), (255, 215, 0), (30, 144, 255),
              (238, 130, 238), (255, 140, 0), (0, 206, 209), (154, 205, 50)]

    frame_times: deque[float] = deque(maxlen=15)
    frame_idx = 0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        states = tracker.update(frame, tag_reads=None, frame_idx=frame_idx)
        for st in states:
            x1, y1, x2, y2 = map(int, st.bbox)
            color = colors[st.track_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"id={st.track_id} code={st.snail_code or '?'}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        frame_times.append(time.perf_counter() - t0)
        inst_fps = 1.0 / (sum(frame_times) / len(frame_times))
        cv2.putText(frame, f"FPS: {inst_fps:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4, cv2.LINE_AA)

        resized = cv2.resize(frame, (out_w, out_h))
        if writer is not None:
            writer.write(resized)
        if args.display:
            cv2.imshow("Tracker demo", resized)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"processed {frame_idx}/{total}", flush=True)
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"DONE. {frame_idx} frames" + (f" -> {args.save}" if writer is not None else ""))


if __name__ == "__main__":
    _demo()
