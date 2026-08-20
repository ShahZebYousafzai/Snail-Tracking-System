"""Find and delete near-duplicate images within a single flat folder of bead
crops (in place), keeping one representative per duplicate group.

This is intentionally conservative, tuned against data/bead_crops/from_video_dedup:
- Only compares crops from the SAME source video. Verified on real data that
  comparing across videos produces false "duplicates" -- two different
  snails, filmed in entirely different sessions, whose bead just happens to
  show the same digit at a similar rotation (confirmed visually: a "0" from
  C0021 and a "0" from C0023, clearly different photos, hashed within
  distance 4 of each other).
- Uses a strict dHash Hamming-distance threshold (default 2/64 bits). Looser
  thresholds (4-6) were tested and also produced same-video false positives
  -- e.g. two different tags 40+ seconds apart, different tag_idx, that just
  happen to both show a similarly-rotated digit. Threshold 2 trades recall
  for precision deliberately: deleting a real unique training example is a
  worse outcome than leaving an occasional true duplicate in place.

Non-destructive to anything outside `--dir`: this only ever deletes files
inside the target folder, and only after they've been identified as the
non-representative member of a same-video, low-distance cluster.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "bead_crops" / "from_video_dedup"

HASH_SIZE = 8  # -> 64-bit hash


def dhash(image_bgr: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    weights = (1 << np.arange(bits.size, dtype=np.uint64))
    return np.uint64(np.sum(weights[bits], dtype=np.uint64))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="default: <dir>/manifest.csv")
    parser.add_argument("--threshold", type=int, default=2, help="max Hamming distance (0-64) to treat crops as duplicates")
    parser.add_argument("--dry-run", action="store_true", help="report what would be deleted without deleting")
    args = parser.parse_args()

    manifest_path = args.manifest or args.dir / "manifest.csv"
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"loaded {len(rows)} crops from {manifest_path}", flush=True)

    hashes = np.empty(len(rows), dtype=np.uint64)
    for i, row in enumerate(rows):
        img = cv2.imread(str(PROJECT_ROOT / row["crop_path"]))
        hashes[i] = dhash(img) if img is not None else np.uint64(0)
    videos = np.array([row["video"] for row in rows])

    n = len(rows)
    uf = UnionFind(n)
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = hashes[start:end]
        block_videos = videos[start:end]
        xor = block[:, None] ^ hashes[None, :]
        dist = np.bitwise_count(xor)
        same_video = block_videos[:, None] == videos[None, :]
        near = np.argwhere((dist <= args.threshold) & same_video)
        for i, j in near:
            gi = start + i
            if gi != j:
                uf.union(gi, j)

    roots = np.array([uf.find(i) for i in range(n)])
    _, cluster_ids = np.unique(roots, return_inverse=True)
    clusters: dict[int, list[int]] = {}
    for i, cid in zip(range(n), cluster_ids):
        clusters.setdefault(int(cid), []).append(i)

    kept_rows = []
    deleted_paths = []
    for cid, members in clusters.items():
        best = max(members, key=lambda i: float(rows[i]["sam3_score"]))
        kept_rows.append(rows[best])
        for i in members:
            if i != best:
                deleted_paths.append(PROJECT_ROOT / rows[i]["crop_path"])

    print(f"{len(clusters)} unique clusters, {len(deleted_paths)} duplicate files to remove "
          f"({len(deleted_paths) / n:.1%} of {n})")

    if args.dry_run:
        print("--dry-run: no files deleted, manifest not rewritten")
        return

    for p in deleted_paths:
        p.unlink(missing_ok=True)

    with open(manifest_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"DONE. deleted {len(deleted_paths)} files, {len(kept_rows)} unique crops remain in {args.dir}")
    print(f"manifest rewritten: {manifest_path}")


if __name__ == "__main__":
    main()
