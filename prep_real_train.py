"""Prepare data/real_train without an agent.

Run from the project root (D:\\Phishing). Default is a DRY RUN that only prints counts.

    python prep_real_train.py                 # dry run
    python prep_real_train.py --apply         # copy originals not in final -> real_train
    python prep_real_train.py --apply --move-half-final
        # also MOVE a seeded 50% of data/real_holdout/final into real_train
        # (final becomes smaller; back up data/ first)

What it does
1. Copies every image in data/real_holdout_originals/<channel>/ whose SHA-256 is NOT in
   data/real_holdout/final/<channel>/ into data/real_train/<channel>/legitimate/.
2. With --move-half-final, moves half of final (per channel, seed 42, ordered by hash)
   into the same training folders so they stop being test data.
Exact duplicates (same hash) are never copied twice.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CHANNELS = ["email", "sms", "whatsapp"]
ORIG = Path("data/real_holdout_originals")
FINAL = Path("data/real_holdout/final")
TRAIN = Path("data/real_train")

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="actually copy/move files")
ap.add_argument("--move-half-final", action="store_true")
args = ap.parse_args()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def images(d: Path):
    """All images under d, including any subfolders."""
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS) if d.exists() else []


def channel_images(root: Path, ch: str):
    """Images under any folder named <ch> found anywhere below root (handles nested zips)."""
    if not root.exists():
        return []
    dirs = [root / ch] if (root / ch).is_dir() else []
    dirs += [d for d in root.rglob(ch) if d.is_dir() and d not in dirs]
    seen, out = set(), []
    for d in dirs:
        for p in images(d):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


for ch in CHANNELS:
    dest = TRAIN / ch / "legitimate"
    if args.apply:
        dest.mkdir(parents=True, exist_ok=True)
    have = {sha(p) for p in images(dest)}
    final_files = channel_images(FINAL, ch)
    final_hashes = {sha(p) for p in final_files}

    # 1. originals that are not in final
    to_copy = []
    for p in channel_images(ORIG, ch):
        h = sha(p)
        if h not in final_hashes and h not in have:
            to_copy.append((p, h))
            have.add(h)

    # 2. optional: half of final goes to training
    to_move = []
    if args.move_half_final:
        keyed = sorted(((sha(p), p) for p in final_files), key=lambda x: x[0])
        random.Random(42).shuffle(keyed)
        for h, p in keyed[: len(keyed) // 2]:
            if h not in have:
                to_move.append((p, h))
                have.add(h)

    n_orig = len(channel_images(ORIG, ch))
    print(f"{ch}: found originals={n_orig}, final={len(final_files)}, already in train={len(images(dest))}")
    print(f"     originals->train {len(to_copy)}, final->train {len(to_move)}, "
          f"final stays {len(final_files) - len(to_move)}")

    if args.apply:
        for p, h in to_copy:
            shutil.copy2(p, dest / f"orig_{h[:10]}{p.suffix.lower()}")
        for p, h in to_move:
            shutil.move(str(p), dest / f"fin_{h[:10]}{p.suffix.lower()}")

for d in (ORIG, FINAL, TRAIN):
    print(f"{d}: {'exists' if d.exists() else 'NOT FOUND'}")

if not args.apply:
    print("\nDry run only. Re-run with --apply to make these changes.")