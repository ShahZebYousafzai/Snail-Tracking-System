# Snail Identity Tracking — Feasibility PoC (Milestone 1)

## Context

This is a paid client milestone: a feasibility proof-of-concept to track 5 individually numbered snails from fixed top-down video, reading the 5-digit number on each snail's shell tag as the source of identity (not a fixed set of pre-trained classes), and holding that identity through occlusion via motion-based tracking until the number becomes readable again. It's the gate before the client invests in live camera processing or platform integration — success here is judged by manual review of an annotated video + CSV, not an abstract accuracy metric. Fixed price $450, 7 business days from receipt of the primary video.

The working project directory `D:\Projects\Snail Tracking System` currently contains only three raw camcorder clips and nothing else — no code, no git repo, no environment. This plan takes it from that bare state to a delivered PoC.

Confirmed decisions (from user):
- **Primary video** (train/annotate): `Videos\C0021.MP4` (~3.9 GiB)
- **Validation video** (held out, honest eval only): `Videos\C0023.MP4` (~3.9 GiB)
- `Videos\C0025.MP4` (~2.2 GiB) — ignored for this milestone
- Shell tags are **fixed 5-digit codes per snail**; only on-screen viewing angle changes as the snail moves, the disc doesn't rotate on the shell
- Annotation via **CVAT.ai free cloud account** (not self-hosted Docker), accelerated with **SAM3 pre-labeling**: SAM3 (run locally, smallest available checkpoint to fit the 6 GB GPU) generates candidate masks/boxes for `snail` and `tag` instances from a handful of click/box prompts per frame plus propagation, which get imported into CVAT as pre-annotations — a human then only corrects SAM3's mistakes instead of drawing every box from scratch. Final export is still CVAT format, so the deliverable is unchanged; only the annotation time drops.
- Environment via **conda/miniconda**, reusing the user's existing **`torch_env`** conda environment (confirmed via `conda list -n torch_env`: Python 3.12.13, `torch` 2.11.0+cu128, `torchvision`/`torchaudio` matching, `opencv-python` 5.0.0.93, `ultralytics` 8.4.115 — already covers the detector/tracker stack) rather than creating a fresh env. System Python 3.14.6 on PATH is too new for these wheels and is not used directly; only `ffmpeg` and SAM3's dependencies still need to be added to `torch_env`.
- GPU: NVIDIA GTX 1660, 6 GB VRAM, driver supports CUDA 12.8 — modest, so model choices must stay small (YOLOv8n/s, not m/l/x)
- `ffmpeg`/`ffprobe` not currently installed — first-day setup item via `conda install -c conda-forge ffmpeg`

## Repo / Project Structure

Initialize git at `D:\Projects\Snail Tracking System\`. Move `Videos\*.mp4` → `data\raw\`.

D:\Projects\Snail Tracking System\
├── .git\
├── .gitignore                  # excludes data\raw, data\frames, outputs\*.mp4, conda/env cruft
├── environment.yml             # conda env spec (pinned), for reproducibility
├── README.md                   # deliverable: setup + run instructions + design notes
├── data\
│   ├── raw\                    # C0021.MP4, C0023.MP4, C0025.MP4 (not git-tracked, too large)
│   ├── frames\                 # extracted frames for annotation (not git-tracked)
│   ├── cvat_exports\           # raw CVAT XML/COCO downloads (not git-tracked)
│   └── processed\              # YOLO-format labels + train/val split (small text — tracked)
├── src\
│   ├── probe.py                 # OpenCV-based video probing (fps/res/duration/codec)
│   ├── extract_frames.py        # frame sampling for annotation
│   ├── sam3_prelabel.py         # SAM3 mask/box proposals -> CVAT pre-annotation import file
│   ├── cvat_to_yolo.py          # CVAT export -> YOLO labels + train/val split
│   ├── train_detector.py        # YOLOv8 fine-tune (snail + tag classes)
│   ├── tag_reader.py            # tag crop -> closed-set 5-digit code classifier
│   ├── tracker.py               # ByteTrack wrapper + identity-reconciliation state machine
│   ├── confidence.py            # CONFIRMED / HELD / UNCERTAIN flagging rule
│   ├── pipeline.py              # end-to-end: video in -> annotated video + CSV/JSON out
│   └── validate.py              # runs frozen pipeline on C0023.MP4, held-out only
├── weights\                     # detector.pt, tag_classifier.pt (small, git-tracked)
├── outputs\                     # annotated videos, tracks CSV/JSON, validation_summary.md
└── notebooks\                   # optional scratch, not a deliverable

## Technical Approach

**Detection**: one YOLOv8n model, two classes — `snail` (whole-body box) and `tag` (the small disc). A dedicated `tag` class gives `tag_reader.py` a tight crop directly, instead of a fixed sub-region heuristic that breaks when the snail's body rotates relative to its (non-rotating) tag. Start with `yolov8n`; step up to `yolov8s` only if tag-class recall is weak.

**Annotation budget (the load-bearing decision for a fixed-price 7-day job)**: not exhaustive frame-by-frame. Extract `C0021.MP4` at ~1 fps, hand-pick ~150–200 frames covering all 5 snails, varying tag rotations, overlap/close-contact events, motion blur, and poor lighting/reflection — prioritizing scenario diversity over raw count since occlusion frames are what the identity-recovery success criterion depends on.

Before uploading to CVAT, run `sam3_prelabel.py`: load the smallest local SAM3 checkpoint, give it a handful of point/box prompts per snail and per tag on each selected frame (or a short prompted-then-propagated run across nearby frames within the same clip segment), and convert the resulting masks to `snail`/`tag` bounding boxes. Import this as a pre-annotation file into the CVAT.ai task instead of starting from blank frames — a human then reviews and corrects only what SAM3 got wrong (missed tag, merged two overlapping snails into one mask, mislabeled edge) rather than drawing every box manually. Record the true 5-digit code as a manual attribute on readable `tag` boxes during this same review pass — this doubles as digit-classifier training data and can't be automated by SAM3 (it segments, it doesn't read digits). Export CVAT XML/COCO once corrected, convert via `cvat_to_yolo.py` into an 85/15 train/val split by frame (not by box, to avoid leakage between nearby sampled frames).

**Digit/ID reading — closed-set classifier, not general OCR**: since there are only 5 known fixed codes, treat this as a 6-way classification problem (5 codes + `unreadable`) on the cropped tag image, not open-ended OCR. A small CNN (or truncated MobileNetV3-small/ResNet18) trained on tag crops from the CVAT-labeled attribute data. This is dramatically more data-efficient than OCR at a ~150-200 frame budget, and rotation augmentation (±180°, since the tag's on-screen angle varies freely) plus blur/brightness augmentation directly target the documented failure modes. State explicitly in the README that this is a deliberate PoC-scale shortcut: `tag_reader.py` is the single drop-in swap point if the project scales to more snails or unknown codes later — nothing else in the pipeline depends on how the code is read, only that a `(code, confidence)` pair comes out per crop.

**Tracking**: ByteTrack on `snail`-class detections (via `ultralytics`'s built-in `track()`), chosen because its low-confidence second-stage matching keeps tracks alive through partial occlusion and motion blur without needing a from-scratch re-ID embedding network (which this dataset is too small to train). Identity reconciliation in `tracker.py`:
- Each track holds an ephemeral `track_id` and a persistent `snail_code` (or `None`).
- A `snail_code` is only (re)assigned after **N consecutive frames** (e.g. N=3) of a consistent high-confidence tag read — avoids flapping on a single noisy read.
- While a track stays alive but its tag isn't currently readable, `snail_code` is **held** from the last confirmed read — this is the literal implementation of "motion-based tracking holds identity until the number is readable again."
- If ByteTrack loses a track entirely and it reappears as a new track, that new track starts with `snail_code = None` and must re-earn identity via the same N-consecutive-read rule — conservative by design, matching "flag uncertainty rather than silently guess."
- Log a warning (feeds confidence flagging) if two live tracks ever hold the same code simultaneously, or a code reappears on a different track shortly after vanishing — both signal a possible identity swap.

**Confidence flagging** (`confidence.py`), per track per frame, one of:
- `CONFIRMED` — tag read within the last few frames, above threshold, matches held code. Green in overlay.
- `HELD` — no read for longer than a few frames but under a hard cap (tune once real fps is known), track continuously alive since last confirmed read. Amber — expected/tolerated during occlusion, not an error.
- `UNCERTAIN` — held past the hard cap, or track recently reborn and hasn't re-earned its code, or detector confidence is low for several frames, or a same-code/possible-swap condition is detected. Red.

Every CSV/JSON row carries `frame, track_id, snail_code, x, y, w, h, state, tag_confidence` — no silent gaps. `events.json` logs discrete `occlusion_start/end_confirmed/end_uncertain`, `identity_reassigned`, `possible_swap` events with frame numbers, which is what the client will scan first when spot-checking.

**Validation**: `C0023.MP4` is not opened, frame-extracted, or looked at until the pipeline is frozen (end of Day 5). Day 6 runs the frozen pipeline once, with zero tuning based on its output. `validation_summary.md` reports a manual spot-check table (~10-15 sampled timestamps: did the overlaid ID match the physical tag, was the flag state appropriate), occlusion-recovery counts, and an honest list of failure modes on unseen footage — plus a restatement of the closed-set upgrade path.

## Day-by-Day Timeline (7 business days)

1. **Env + inspection**: git init, move videos into `data\raw\`, `conda activate torch_env` (already has Python 3.12, torch 2.11+cu128, opencv, ultralytics 8.4.115), add `ffmpeg` (`conda install -c conda-forge ffmpeg`) and SAM3's dependencies (smallest checkpoint) into it, verify CUDA works on the 1660 for both YOLO and SAM3, run `probe.py` on all three clips for real fps/res/duration, extract the 1fps frame pool from `C0021.MP4`.
2. **Frame selection + SAM3 pre-labeling + CVAT correction start**: hand-pick ~150-200 diverse frames, run `sam3_prelabel.py` to generate candidate `snail`/`tag` boxes for all of them, set up the CVAT.ai task with the pre-annotations imported, begin correcting SAM3's output + adding code attributes — expect to get through most or all frames this day given SAM3 removes the from-scratch box-drawing work.
3. **Finish annotation + first detector run**: finish any remaining corrections, export from CVAT, `cvat_to_yolo.py` conversion, train `yolov8n`, check per-class recall (especially `tag`), build the tag-crop dataset for the digit classifier.
4. **Digit classifier + tracker wiring**: train the 6-class tag classifier with rotation/blur/brightness augmentation, wire ByteTrack, implement the identity-reconciliation state machine.
5. **Confidence flagging + full pipeline + freeze**: implement `confidence.py`, wire CSV/JSON export + color-coded video overlay, spot-check against `C0021.MP4` (in-sample), tune thresholds, then **freeze** — no more tuning after this point.
6. **Held-out validation**: run frozen pipeline once on `C0023.MP4`, spot-check manually, write `validation_summary.md`.
7. **Packaging + delivery**: write `README.md` (setup, run instructions, design rationale, known limitations), clean repo, verify weights load from a clean env, package all deliverables for handoff.

## Key Risks & Mitigations

- **Small dataset overfitting** → prioritize scenario diversity over frame count, heavy rotation augmentation on the classifier, small models (`yolov8n/s`), honest held-out reporting instead of tuning against it.
- **Tag illegible at edge-on angles** → explicit `unreadable` class; occlusion-hold logic covers this via motion continuity.
- **Lighting/reflection on wet shells** → deliberately include glare frames in annotation; brightness augmentation; ambiguous reads pushed to `UNCERTAIN` instead of a silent wrong answer.
- **Motion blur** → include blurred frames in annotation + augmentation; ByteTrack's low-confidence matching keeps boxes alive through blur.
- **Visually similar shells → ID swap on close pass** → N-consecutive-read re-confirmation + `possible_swap` event detection surfaces suspected swaps rather than trusting proximity alone.
- **6 GB VRAM limits** → conservative batch sizes (8-16 at 640px for yolov8n/s), monitor `nvidia-smi` on first training run; use the smallest available SAM3 checkpoint and run it as a separate one-off pre-labeling pass (not concurrently with YOLO training) so it doesn't compete for VRAM.
- **SAM3 mask quality on small/thin objects** (the tag disc is small, snails can be low-contrast against the textured background) → treat SAM3 output as a time-saving first draft only, not ground truth; human review in CVAT is still mandatory before export, budgeted explicitly into Day 2-3 rather than assumed to be zero-effort.
- **6/9 digit-bead ambiguity** (visually identical under 180° rotation on the current white beads) → print 6 and 9 in different colors on any future physical tag/bead revision so the two remain distinguishable at any on-screen rotation; doesn't help the existing white-bead video, which still needs a manual spot-check for 6/9 mislabels (see TODO.md).
- **Bead rotation labels currently unusable** → `manual_labels.csv` has `rotation_deg == 0` for all 2,742 digit-labeled rows (rotation was never adjusted during labeling), so the two-head classifier's rotation head is training against a constant target and learning nothing; needs a relabeling pass before that head is trusted (see TODO.md).

## Verification

- After Day 3 training run: inspect YOLO val mAP/recall per class (`snail`, `tag`) — tag recall is the one to watch closely since it gates the whole ID-reading pipeline.
- After Day 5 freeze: play back `outputs\C0021_annotated.mp4` alongside `C0021_tracks.csv`, manually confirm color-coded states line up with visible occlusion events for at least a few minutes of footage.
- Day 6: run `python src\validate.py` against `C0023.MP4` untouched, review `outputs\C0023_annotated.mp4` + `C0023_tracks.csv` + `C0023_events.json` against the success criteria (all 5 detected whenever visible, IDs correct outside occlusion, recovery within a couple frames post-occlusion, no silent wrong answers).
- Final check: `conda env export -n torch_env --no-builds > environment.yml` to snapshot the actual working env (for the client's reproducibility, since it's a reused/pre-existing env rather than one built from scratch for this project), confirm `pipeline.py` runs end-to-end in `torch_env` and loads `weights\*.pt` without errors, before handoff.