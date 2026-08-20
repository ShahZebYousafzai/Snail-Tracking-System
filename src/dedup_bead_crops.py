"""Collapse near-duplicate bead crops before labeling.

data/bead_crops/from_video_unlabeled was sampled at ~1fps -- snails barely
move between consecutive samples, so a tag observed across many seconds
produces a run of near-identical bead crops (same digit, same rotation,
same lighting). Labeling every one of those wastes effort without adding
information, and inflates the apparent per-class sample count.

This hashes every crop with a difference hash (dHash), clusters
near-duplicates (Hamming distance <= --threshold) within each source video,
and keeps only the highest-SAM3-confidence crop per cluster. Non-destructive:
original crops are untouched, this just writes a filtered manifest.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "bead_crops" / "from_video_unlabeled" / "manifest.csv"

HASH_SIZE = 8  # -> 64-bit hash


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def dhash(image_bgr: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = small[:, 1:] > small[:, :-1]  # HASH_SIZE x HASH_SIZE booleans
    bits = diff.flatten()
    weights = (1 << np.arange(bits.size, dtype=np.uint64))
    return np.uint64(np.sum(weights[bits], dtype=np.uint64))


def cluster_video_group(hashes: np.ndarray, frames: np.ndarray, threshold: int, max_frame_gap: int) -> np.ndarray:
    """Return a cluster-id array (same length as hashes) via "leader" clustering,
    processed in frame order -- ONLY between crops whose source frames are
    within `max_frame_gap` of each other.

    Deliberately NOT single-linkage/connected-components: an earlier version
    unioned any pair within `threshold`, and even gated by a time window that
    chain-merged transitively (frame 0 near frame 150 near frame 300 ...)
    into one cluster spanning the entire video.

    Each crop can only extend the SINGLE best-matching currently-active
    leader (nearest hash, not "first that matches") -- this is what stops
    the old chaining bug, since a crop can never simultaneously merge two
    leaders together the way transitive union-find could. The leader's
    anchor hash then ROLLS to the most recent match rather than staying
    fixed to the cluster's first frame: verified on real data that a fixed
    anchor fragments one continuous static run into several back-to-back
    clusters as the crop slowly drifts (compression noise, lighting) past
    the threshold relative to frame 0, even though each step barely changes
    -- e.g. one run split into a [0-50] cluster and a separate [75-575]
    cluster only 25 frames later, hash distance 8 vs a threshold of 6. A
    rolling anchor still can't bridge a real content change (that shows up
    as a sharp jump against the *previous* frame, not a gradual drift), so
    it fixes the fragmentation without reintroducing the whole-video merge.
    The time gate encodes the actual physical assumption -- crops are
    duplicates because the snail didn't move much in a few seconds (default
    ~6s, matching the OC-SORT max_age=150 convention used elsewhere in this
    repo, see src/tracker.py) -- not because they look generically similar.
    """
    n = len(hashes)
    order = np.argsort(frames, kind="stable")
    sorted_frames = frames[order]
    sorted_hashes = hashes[order]

    leaders: list[tuple[np.uint64, int, int]] = []  # (rolling_hash, last_frame, cluster_id)
    cluster_ids_sorted = np.full(n, -1, dtype=np.int64)
    next_cluster_id = 0

    for i in range(n):
        f = sorted_frames[i]
        h = sorted_hashes[i]
        leaders = [l for l in leaders if f - l[1] <= max_frame_gap]

        match_idx = None
        best_dist = threshold + 1
        for li, (rolling_hash, _last_frame, _cid) in enumerate(leaders):
            dist = int(np.bitwise_count(rolling_hash ^ h))
            if dist <= threshold and dist < best_dist:
                match_idx = li
                best_dist = dist

        if match_idx is None:
            cid = next_cluster_id
            next_cluster_id += 1
            leaders.append((h, f, cid))
        else:
            _, _, cid = leaders[match_idx]
            leaders[match_idx] = (h, f, cid)  # roll the anchor forward to this match
        cluster_ids_sorted[i] = cid

    cluster_ids = np.empty(n, dtype=np.int64)
    cluster_ids[order] = cluster_ids_sorted
    return cluster_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--threshold", type=int, default=6, help="max Hamming distance (0-64) to treat crops as duplicates")
    parser.add_argument("--max-frame-gap", type=int, default=150,
                         help="only compare crops within this many frames of each other (default 150 ~= 6s at 25fps)")
    parser.add_argument("--out", type=Path, default=None, help="default: <manifest_dir>/manifest_deduped.csv")
    parser.add_argument("--copy-dir", type=Path, default=None,
                         help="if set, copy each kept crop into this folder (original crops are left untouched)")
    args = parser.parse_args()

    if args.out:
        out_path = args.out
    elif args.copy_dir:
        out_path = args.copy_dir / "manifest.csv"
    else:
        out_path = args.manifest.parent / "manifest_deduped.csv"

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"loaded {len(rows)} crops from {args.manifest}", flush=True)

    hashes = np.empty(len(rows), dtype=np.uint64)
    for i, row in enumerate(rows):
        crop_path = PROJECT_ROOT / row["crop_path"]
        img = cv2.imread(str(crop_path))
        if img is None:
            hashes[i] = np.uint64(0)
            continue
        hashes[i] = dhash(img)
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(rows)}", flush=True)

    # Group by source video -- duplicates only make sense within one video's timeline.
    videos = sorted({row["video"] for row in rows})
    kept_rows = []
    total_clusters = 0
    for video in videos:
        idxs = [i for i, row in enumerate(rows) if row["video"] == video]
        group_hashes = hashes[idxs]
        group_frames = np.array([int(rows[i]["frame"]) for i in idxs], dtype=np.int64)
        cluster_ids = cluster_video_group(group_hashes, group_frames, args.threshold, args.max_frame_gap)
        n_clusters = cluster_ids.max() + 1 if len(cluster_ids) else 0
        total_clusters += n_clusters
        print(f"[{video}] {len(idxs)} crops -> {n_clusters} clusters", flush=True)

        clusters: dict[int, list[int]] = {}
        for local_i, cid in zip(idxs, cluster_ids):
            clusters.setdefault(int(cid), []).append(local_i)

        for cid, members in clusters.items():
            best = max(members, key=lambda i: float(rows[i]["sam3_score"]))
            row = dict(rows[best])
            row["cluster_size"] = len(members)
            kept_rows.append(row)

    if args.copy_dir:
        args.copy_dir.mkdir(parents=True, exist_ok=True)
        for row in kept_rows:
            src = PROJECT_ROOT / row["crop_path"]
            dst = args.copy_dir / src.name
            shutil.copy2(src, dst)
            row["crop_path"] = relative_to_project(dst)
        print(f"copied {len(kept_rows)} crops -> {args.copy_dir}", flush=True)

    with open(out_path, "w", newline="") as f:
        fieldnames = ["crop_path", "video", "frame", "tag_idx", "bead_idx", "sam3_score", "cluster_size"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"\nDONE. {len(rows)} crops -> {len(kept_rows)} diverse crops ({len(kept_rows) / len(rows):.1%} kept)")
    print(f"deduped manifest: {out_path}")


if __name__ == "__main__":
    main()
