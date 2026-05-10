"""
One-time preprocessing: extract 2D axial slices from BraTS2020 NIfTI volumes
and save as compressed .npz files for fast training.

Each .npz contains:
  'image' : (4, H, W) float32  — z-score normalised 4 modalities (full slice)
  'label' : (H, W)    int32    — seg mask, label 4 remapped to 3

Output directory:
  data/brain/processed/slices/train/  (295 cases × N slices each)
  data/brain/processed/slices/val/    ( 73 cases × N slices each)

Usage:
  python training/preprocess_brats.py
  python training/preprocess_brats.py --slices-per-case 20 --workers 4
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

BRATS_BASE = Path(
    "data/brain/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
MODALITIES  = ["t1", "t1ce", "t2", "flair"]
OUT_DIR     = Path("data/brain/processed/slices")


# ---------------------------------------------------------------------------
# Helpers (importable by train_brain.py)
# ---------------------------------------------------------------------------

def _build_file_dicts(base: Path, val_ratio: float = 0.2, seed: int = 42):
    cases = sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("BraTS20_Training_")
    )
    all_dicts = []
    for case_dir in cases:
        name = case_dir.name
        d = {mod: case_dir / f"{name}_{mod}.nii" for mod in MODALITIES}
        d["seg"] = case_dir / f"{name}_seg.nii"
        if all(v.exists() for v in d.values()):
            all_dicts.append(d)
    if not all_dicts:
        raise RuntimeError(f"No complete BraTS2020 cases found in {base}")
    rng = random.Random(seed)
    rng.shuffle(all_dicts)
    n_val = max(1, int(len(all_dicts) * val_ratio))
    return all_dicts[n_val:], all_dicts[:n_val]


def _zscore(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mu, sigma = vol[mask].mean(), vol[mask].std()
    return (vol - mu) / max(sigma, 1e-8)


def _process_case(args):
    fd, out_dir, slices_per_case, seed = args
    import nibabel as nib

    name   = fd["t1"].parent.name
    rng    = random.Random(seed)
    saved  = []

    try:
        imgs = []
        for mod in MODALITIES:
            vol = nib.load(str(fd[mod])).get_fdata(dtype=np.float32)
            imgs.append(_zscore(vol))

        seg = nib.load(str(fd["seg"])).get_fdata(dtype=np.float32).astype(np.int32)
        seg[seg == 4] = 3

        D      = imgs[0].shape[2]
        margin = max(1, D // 6)
        z_candidates = list(range(margin, D - margin))
        rng.shuffle(z_candidates)
        chosen = z_candidates[:slices_per_case]

        for z in chosen:
            image = np.stack([img[:, :, z] for img in imgs], axis=0).astype(np.float32)
            label = seg[:, :, z].astype(np.int32)
            out_path = out_dir / f"{name}_z{z:03d}.npz"
            np.savez_compressed(str(out_path), image=image, label=label)
            saved.append(str(out_path))

    except Exception as exc:
        print(f"  [warn] {name}: {exc}", flush=True)

    return name, len(saved)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS2020 to .npz slices.")
    parser.add_argument("--slices-per-case", type=int, default=20)
    parser.add_argument("--workers",         type=int, default=1,
                        help="Parallel workers (1 = serial, safe on Windows).")
    parser.add_argument("--val-fraction",    type=float, default=0.2)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    brats_base = BRATS_BASE
    if not brats_base.is_dir():
        raise SystemExit(f"BraTS base not found: {brats_base}")

    train_dicts, val_dicts = _build_file_dicts(brats_base, args.val_fraction, args.seed)
    print(f"Cases: {len(train_dicts)} train / {len(val_dicts)} val", flush=True)
    print(f"Slices per case: {args.slices_per_case}  |  Workers: {args.workers}", flush=True)

    splits = [("train", train_dicts), ("val", val_dicts)]
    total_saved = 0
    t0 = time.time()

    for split_name, dicts in splits:
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Processing {split_name} ({len(dicts)} cases) ---", flush=True)

        tasks = [
            (fd, out_dir, args.slices_per_case, args.seed + i)
            for i, fd in enumerate(dicts)
        ]

        if args.workers == 1:
            for i, task in enumerate(tasks, 1):
                name, n = _process_case(task)
                elapsed = time.time() - t0
                eta = (elapsed / i) * (len(tasks) - i)
                print(
                    f"  [{i:3d}/{len(tasks)}] {name}  saved={n}"
                    f"  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                    flush=True,
                )
                total_saved += n
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_process_case, t): t for t in tasks}
                done = 0
                for fut in as_completed(futures):
                    name, n = fut.result()
                    done += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / done) * (len(tasks) - done)
                    print(
                        f"  [{done:3d}/{len(tasks)}] {name}  saved={n}"
                        f"  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                        flush=True,
                    )
                    total_saved += n

    elapsed = time.time() - t0
    print(f"\nDone. Total .npz files: {total_saved}  ({elapsed/60:.1f} min)", flush=True)
    print(f"Output: {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
