"""
BraTS2020 brain MRI segmentation training script.

Trains either the 2D or 3D U-Net on the BraTS2020 training set.
The 369 training cases are split 80/20 for train/val internally
(the official validation set has no segmentation masks).

BraTS2020 layout (actual on disk):
  data/brain/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
    BraTS20_Training_001/
      BraTS20_Training_001_t1.nii
      BraTS20_Training_001_t1ce.nii
      BraTS20_Training_001_t2.nii
      BraTS20_Training_001_flair.nii
      BraTS20_Training_001_seg.nii
    ...

Labels: 0=background, 1=necrosis/core, 2=edema, 4=enhancing tumor
Label 4 is remapped to 3 before training.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn as nn
from monai.data import CacheDataset as _MonaiCacheDataset
from monai.losses import DiceFocalLoss
from monai.transforms import (
    Compose, Rand2DElasticd, RandAdjustContrastd,
    RandFlipd, RandGaussianNoised, RandRotated,
    ConcatItemsd, EnsureChannelFirstd, EnsureTyped,
    Lambdad, LoadImaged, NormalizeIntensityd,
    Rand3DElasticd, RandCropByPosNegLabeld, SelectItemsd,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.brain.unet_2d import build_unet_2d
from models.brain.unet_3d import build_unet_3d, get_sliding_window_inferer
from utils.metrics import BrainMetricsCalculator, MetricsLogger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BRATS_BASE      = Path("data/brain/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
MODALITIES      = ["t1", "t1ce", "t2", "flair"]
PATCH_SIZE      = 128
SLICES_PER_CASE = 8          # random slices served per case per epoch (2D)
NUM_EPOCHS      = 75
BATCH_SIZE      = 8
ES_PATIENCE     = 10
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
VAL_FRACTION    = 0.2
NUM_WORKERS     = 4
RESULTS_DIR     = "results"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _build_file_dicts(
    base: Path = BRATS_BASE,
    val_ratio: float = VAL_FRACTION,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Scan BraTS2020 training dirs and return (train_dicts, val_dicts)."""
    cases = sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("BraTS20_Training_")
    )
    all_dicts: list[dict] = []
    for case_dir in cases:
        name = case_dir.name
        d: dict[str, Path] = {mod: case_dir / f"{name}_{mod}.nii" for mod in MODALITIES}
        d["seg"] = case_dir / f"{name}_seg.nii"
        if all(v.exists() for v in d.values()):
            all_dicts.append(d)

    if not all_dicts:
        raise RuntimeError(
            f"No complete BraTS2020 cases found in {base}.\n"
            "Expected dirs named BraTS20_Training_XXX containing "
            "t1/t1ce/t2/flair/seg .nii files."
        )

    rng = random.Random(seed)
    rng.shuffle(all_dicts)
    n_val = max(1, int(len(all_dicts) * val_ratio))
    return all_dicts[n_val:], all_dicts[:n_val]


def _load_volume(path: Path) -> np.ndarray:
    import nibabel as nib
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _zscore(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mu, sigma = vol[mask].mean(), vol[mask].std()
    return (vol - mu) / max(sigma, 1e-8)


class BraTS2DIterableDataset(IterableDataset):
    """Efficient 2D axial patch dataset for BraTS2020.

    Loads each 3D volume ONCE per epoch and yields SLICES_PER_CASE
    random axial patches from it, cutting NIfTI reads 8x vs a Map-style
    dataset that would reload the volume for every __getitem__ call.

    Supports multi-worker DataLoader: each worker handles a partition of
    cases, loading them in parallel.

    Returns:
        image : (4, P, P) float32 tensor — 4 modalities, z-score normalised
        label : (1, P, P) int64 tensor  — classes 0..3 (label 4 remapped to 3)
    """

    def __init__(
        self,
        file_dicts: list[dict],
        patch_size: int = PATCH_SIZE,
        slices_per_case: int = SLICES_PER_CASE,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.file_dicts      = file_dicts
        self.patch_size      = patch_size
        self.slices_per_case = slices_per_case
        self.augment         = augment
        self.seed            = seed
        self._epoch          = 0

    def __len__(self) -> int:
        return len(self.file_dicts) * self.slices_per_case

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_dicts  = self.file_dicts
        worker_seed = self.seed + self._epoch

        if worker_info is not None:
            per_worker = int(math.ceil(len(file_dicts) / worker_info.num_workers))
            start      = worker_info.id * per_worker
            end        = min(start + per_worker, len(file_dicts))
            file_dicts = file_dicts[start:end]
            worker_seed += worker_info.id

        rng      = random.Random(worker_seed)
        shuffled = list(file_dicts)
        rng.shuffle(shuffled)

        for fd in shuffled:
            try:
                imgs = [_zscore(_load_volume(fd[mod])) for mod in MODALITIES]
                seg  = _load_volume(fd["seg"]).astype(np.int32)
                seg[seg == 4] = 3

                D      = imgs[0].shape[2]
                margin = max(1, D // 6)
                P      = self.patch_size
                H, W   = imgs[0].shape[0], imgs[0].shape[1]

                for _ in range(self.slices_per_case):
                    z  = rng.randint(margin, D - margin - 1)
                    x0 = rng.randint(0, max(0, H - P))
                    y0 = rng.randint(0, max(0, W - P))

                    image = np.stack(
                        [img[x0:x0+P, y0:y0+P, z] for img in imgs], axis=0
                    )                                              # (4,P,P)
                    label = seg[x0:x0+P, y0:y0+P, z][np.newaxis] # (1,P,P)

                    if self.augment:
                        if rng.random() < 0.5:
                            image = image[:, :, ::-1].copy()
                            label = label[:, :, ::-1].copy()
                        if rng.random() < 0.5:
                            image = image[:, ::-1, :].copy()
                            label = label[:, ::-1, :].copy()
                        k = rng.randint(0, 3)
                        if k:
                            image = np.rot90(image, k, axes=(1, 2)).copy()
                            label = np.rot90(label, k, axes=(1, 2)).copy()

                    yield (
                        torch.from_numpy(image.copy()).float(),
                        torch.from_numpy(label.astype(np.int64)),
                    )
            except Exception as exc:
                print(f"[warn] skipping {fd.get('t1','?')}: {exc}", flush=True)


# Keep the old name as an alias so test imports still work
BraTS2DDataset = BraTS2DIterableDataset


# ---------------------------------------------------------------------------
# Fast Map-style dataset backed by pre-extracted .npz slices
# ---------------------------------------------------------------------------

from torch.utils.data import Dataset as _Dataset


class BraTS2DFastDataset(_Dataset):
    """Loads pre-extracted .npz slices with MONAI augmentation pipeline.

    Each .npz contains:
      'image' : (4, H, W) float32  (full-resolution slice, z-score normalised)
      'label' : (H, W)    int32

    Training augmentations (MONAI):
      - RandFlip  (both axes, p=0.5)
      - RandRotate (±0.3 rad, p=0.5)
      - RandElasticDeform (sigma=5-7, magnitude=50-150, p=0.3)
      - RandGaussianNoise (std=0.05, p=0.2)
      - RandAdjustContrast (gamma=0.7-1.3, p=0.3)
    """

    def __init__(
        self,
        npz_dir: Path,
        patch_size: int = PATCH_SIZE,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.files      = sorted(npz_dir.glob("*.npz"))
        self.patch_size = patch_size
        self._rng       = random.Random(seed)
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {npz_dir}")

        self._aug = self._build_aug() if augment else None

    @staticmethod
    def _build_aug() -> Compose:
        return Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandRotated(
                keys=["image", "label"], range_x=0.3, prob=0.5,
                mode=["bilinear", "nearest"], padding_mode="zeros",
            ),
            Rand2DElasticd(
                keys=["image", "label"],
                spacing=(5, 7),           # control-point spacing (maps to sigma_range concept)
                magnitude_range=(50, 150),
                prob=0.3,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data  = np.load(str(self.files[idx]))
        image = data["image"].astype(np.float32)   # (4, H, W)
        label = data["label"].astype(np.int32)      # (H, W)

        H, W = image.shape[1], image.shape[2]
        P    = self.patch_size
        x0   = self._rng.randint(0, max(0, H - P))
        y0   = self._rng.randint(0, max(0, W - P))

        image = image[:, x0:x0+P, y0:y0+P]
        label = label[x0:x0+P, y0:y0+P][np.newaxis].astype(np.float32)  # (1,P,P) float for MONAI

        img_t = torch.from_numpy(image)
        lbl_t = torch.from_numpy(label)

        if self._aug is not None:
            d     = self._aug({"image": img_t, "label": lbl_t})
            img_t = d["image"]
            lbl_t = d["label"]

        return img_t.float(), lbl_t.long()


# ---------------------------------------------------------------------------
# 3-D pipeline — direct NIfTI loading (memory-efficient, no MONAI CacheDataset)
# ---------------------------------------------------------------------------

class BraTS3DDataset(_Dataset):
    """Patch-based 3D training dataset using direct nibabel loading.

    Loads one BraTS case per __getitem__, extracts a single 128³ patch with
    50 % positive (tumor-centred) / 50 % random sampling, applies simple
    axis-aligned augmentation, then explicitly frees all intermediate arrays.

    Returns:
        image : (4, P, P, P) float32 — 4 modalities, z-score normalised
        label : (1, P, P, P) int64   — classes 0..3 (label 4 remapped to 3)
    """

    def __init__(
        self,
        file_dicts: list[dict],
        patch_size: int = 128,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.file_dicts = file_dicts
        self.patch_size = patch_size
        self.augment    = augment
        self._rng       = random.Random(seed)

    def __len__(self) -> int:
        return len(self.file_dicts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import gc
        fd = self.file_dicts[idx]

        imgs = [_zscore(_load_volume(fd[mod])) for mod in MODALITIES]  # each (H,W,D)
        seg  = _load_volume(fd["seg"]).astype(np.int32)
        seg[seg == 4] = 3

        image = np.stack(imgs, axis=0)   # (4, H, W, D) float32
        del imgs

        P        = self.patch_size
        H, W, D  = image.shape[1:]
        tumor_vx = np.argwhere(seg > 0)

        if len(tumor_vx) > 0 and self._rng.random() < 0.5:
            c  = tumor_vx[self._rng.randint(0, len(tumor_vx) - 1)]
            x0 = int(np.clip(c[0] - P // 2, 0, H - P))
            y0 = int(np.clip(c[1] - P // 2, 0, W - P))
            z0 = int(np.clip(c[2] - P // 2, 0, D - P))
        else:
            x0 = self._rng.randint(0, max(0, H - P))
            y0 = self._rng.randint(0, max(0, W - P))
            z0 = self._rng.randint(0, max(0, D - P))

        img_patch = image[:, x0:x0+P, y0:y0+P, z0:z0+P].copy()    # (4, P, P, P)
        lbl_patch = seg[x0:x0+P, y0:y0+P, z0:z0+P][np.newaxis]     # (1, P, P, P)
        del image, seg, tumor_vx
        gc.collect()

        if self.augment:
            for ax in (1, 2, 3):
                if self._rng.random() < 0.5:
                    img_patch = np.flip(img_patch, axis=ax).copy()
                    lbl_patch = np.flip(lbl_patch, axis=ax).copy()
            k = self._rng.randint(0, 3)
            if k:
                ax_pair   = self._rng.choice([(1, 2), (1, 3), (2, 3)])
                img_patch = np.rot90(img_patch, k, axes=ax_pair).copy()
                lbl_patch = np.rot90(lbl_patch, k, axes=ax_pair).copy()

        return (
            torch.from_numpy(img_patch.astype(np.float32)),
            torch.from_numpy(lbl_patch.astype(np.int64)),
        )


class BraTS3DFullDataset(_Dataset):
    """Returns full (4, H, W, D) volumes for SlidingWindowInferer validation.

    One NIfTI case per __getitem__; intermediate arrays are freed explicitly
    to keep RAM usage to ~175 MB peak per call (rather than accumulating).
    """

    def __init__(self, file_dicts: list[dict]) -> None:
        self.file_dicts = file_dicts

    def __len__(self) -> int:
        return len(self.file_dicts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import gc
        fd   = self.file_dicts[idx]
        imgs = [_zscore(_load_volume(fd[mod])) for mod in MODALITIES]
        seg  = _load_volume(fd["seg"]).astype(np.int32)
        seg[seg == 4] = 3
        image = np.stack(imgs, axis=0).astype(np.float32)   # (4, H, W, D)
        del imgs
        gc.collect()
        return (
            torch.from_numpy(image),
            torch.from_numpy(seg[np.newaxis].astype(np.int64)),  # (1, H, W, D)
        )


def _build_3d_loaders(
    brats_base: Path,
    batch_size: int,
    num_workers: int,
    cache_rate: float,
    val_fraction: float,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Build 3D loaders using direct nibabel loading (no MONAI CacheDataset)."""
    train_file_dicts, val_file_dicts = _build_file_dicts(brats_base, val_fraction)

    train_ds = BraTS3DDataset(train_file_dicts, patch_size=128, augment=True,  seed=42)
    val_ds   = BraTS3DFullDataset(val_file_dicts)  # full volumes for SlidingWindowInferer

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, len(train_file_dicts), len(val_file_dicts)


@torch.no_grad()
def _val_epoch_3d(
    model, loader, loss_fn, device, metrics_calc, inferer
) -> tuple[float, dict]:
    """Validate 3D model using SlidingWindowInferer; dice computed per-volume to cap RAM."""
    model.eval()
    total    = 0.0
    dice_acc: dict[str, float] = {}
    n_vols   = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        out  = inferer(imgs, model)
        total += loss_fn(out, lbls).item()
        pred = out.argmax(dim=1).cpu().numpy()    # (1, D, H, W)
        tgt  = lbls.squeeze(1).cpu().numpy()      # (1, D, H, W)
        vol_dice = metrics_calc.compute_dice(pred, tgt)
        for k, v in vol_dice.items():
            dice_acc[k] = dice_acc.get(k, 0.0) + v
        n_vols += 1
        del out, pred, tgt
        torch.cuda.empty_cache()
    avg_dice = {k: v / max(n_vols, 1) for k, v in dice_acc.items()}
    return total / max(len(loader), 1), avg_dice


# ---------------------------------------------------------------------------
# CLI  (kept for backward-compat with tests)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BraTS2020 brain MRI U-Net training (2D or 3D)."
    )
    parser.add_argument("--mode",         choices=["2d", "3d"], default="3d")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--data-dir",     type=str,   default="data/brain")
    parser.add_argument("--results-dir",  type=str,   default="results")
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--cache-rate",   type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def build_model(mode: str) -> nn.Module:
    if mode == "2d":
        return build_unet_2d(in_channels=4, num_classes=4)
    return build_unet_3d(in_channels=4, num_classes=4)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _val_epoch(model, loader, loss_fn, device, metrics_calc) -> tuple[float, dict]:
    model.eval()
    total, all_preds, all_targets = 0.0, [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        total += loss_fn(out, lbls).item()
        all_preds.append(out.argmax(dim=1).cpu().numpy())
        all_targets.append(lbls.squeeze(1).cpu().numpy())
    preds_arr   = np.concatenate(all_preds,   axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return total / max(len(loader), 1), metrics_calc.compute_dice(preds_arr, targets_arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = BATCH_SIZE if args.mode == "2d" else 2

    brats_base = BRATS_BASE
    if args.data_dir != "data/brain":
        brats_base = (
            Path(args.data_dir)
            / "raw"
            / "BraTS2020_TrainingData"
            / "MICCAI_BraTS2020_TrainingData"
        )

    train_ds = val_ds = None   # only set in 2D path; used for set_epoch()
    inferer  = None            # only set in 3D path

    if args.mode == "3d":
        train_loader, val_loader, n_train, n_val = _build_3d_loaders(
            brats_base, args.batch_size, args.num_workers,
            args.cache_rate, args.val_fraction,
        )
        inferer = get_sliding_window_inferer(sw_batch_size=2, overlap=0.5)
        print(f"Samples    : {n_train} train  /  {n_val} val")
    else:
        # Prefer pre-extracted .npz slices for fast I/O; fall back to live NIfTI loading
        npz_train = Path("data/brain/processed/slices/train")
        npz_val   = Path("data/brain/processed/slices/val")
        use_fast  = npz_train.is_dir() and any(npz_train.glob("*.npz"))

        if use_fast:
            print(f"Mode       : FAST (.npz cache found at {npz_train})")
            train_ds   = BraTS2DFastDataset(npz_train, augment=True,  seed=42)
            val_ds     = BraTS2DFastDataset(npz_val,   augment=False, seed=0)
            train_shuffle = True
        else:
            print(f"BraTS base : {brats_base}")
            train_dicts, val_dicts = _build_file_dicts(brats_base, args.val_fraction)
            print(f"Cases      : {len(train_dicts)} train / {len(val_dicts)} val")
            train_ds      = BraTS2DIterableDataset(train_dicts, augment=True,  seed=42)
            val_ds        = BraTS2DIterableDataset(val_dicts,   augment=False, seed=0)
            train_shuffle = False   # IterableDataset shuffles internally

        print(f"Samples    : {len(train_ds):,} train  /  {len(val_ds):,} val")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=train_shuffle,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )

    model = build_model(args.mode).to(DEVICE)

    # DiceFocalLoss: necrosis (class 1) gets 1.5x weight for small-structure focus
    class_weight = torch.tensor([1.0, 1.5, 1.0, 1.0]).to(DEVICE)
    loss_fn = DiceFocalLoss(
        to_onehot_y=True, softmax=True,
        gamma=2.0,
        weight=class_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_fn    = "attnunet2d_best.pth" if args.mode == "2d" else "attnunet3d_best.pth"
    model_save = str(Path("models") / "brain" / save_fn)
    Path(model_save).parent.mkdir(parents=True, exist_ok=True)

    metrics_calc = BrainMetricsCalculator(num_classes=4)
    logger       = MetricsLogger(task=f"brain_{args.mode}",
                                 results_dir=args.results_dir,
                                 model_save_path=model_save)
    early_stop   = EarlyStopping(patience=ES_PATIENCE)

    print(f"Device     : {DEVICE}")
    print(
        f"Mode       : {args.mode}  |  Epochs: {args.epochs}"
        f"  |  Batch: {args.batch_size}  |  LR: {args.lr:.0e}"
    )

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):   # IterableDataset only
            train_ds.set_epoch(epoch)
            val_ds.set_epoch(epoch)
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        if args.mode == "3d":
            val_loss, metrics = _val_epoch_3d(
                model, val_loader, loss_fn, DEVICE, metrics_calc, inferer
            )
        else:
            val_loss, metrics = _val_epoch(model, val_loader, loss_fn, DEVICE, metrics_calc)
        scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, metrics)

        dice_mean = metrics.get("dice_mean", 0.0)
        saved     = logger.save_best_model(model, dice_mean, epoch)
        marker    = "  *** BEST ***" if saved else ""
        es_info   = f"  [ES {early_stop.counter}/{ES_PATIENCE}]" if early_stop.counter > 0 else ""

        print(
            f"[{epoch:02d}/{args.epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice={dice_mean:.4f}  lr={current_lr:.2e}"
            f"{marker}{es_info}"
        )

        if early_stop.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    logger.generate_plots()
    logger.generate_summary()
    print(f"\nDone. Best dice_mean={logger.best_metric:.4f} @ epoch {logger.best_epoch}")
    print(f"Checkpoint : {model_save}")


if __name__ == "__main__":
    main()
