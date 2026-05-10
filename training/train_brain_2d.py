"""
2D U-Net brain MRI tumor segmentation training (MONAI).

Data layout expected under data/brain/:
  train/<case_id>/image.nii.gz   — multi-channel (T1/T1ce/T2/FLAIR) or single-channel
  train/<case_id>/label.nii.gz   — integer mask (0=background, 1..N=tumor regions)
  val/<case_id>/...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.metrics import BrainMetricsCalculator, MetricsLogger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CLASSES = 4          # background + WT/TC/ET (BraTS convention)
IN_CHANNELS = 4          # T1, T1ce, T2, FLAIR
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
DATA_DIR = Path("data/brain")
MODEL_SAVE = "models/brain/unet2d_best.pth"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _make_transforms(train: bool) -> Compose:
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        CropForegroundd(keys=keys, source_key="image"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys),
    ]
    aug = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]
    return Compose(base + aug if train else base)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    return UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _data_dicts(split: str) -> list[dict]:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return []
    cases = sorted(c for c in split_dir.iterdir() if c.is_dir())
    return [
        {"image": str(c / "image.nii.gz"), "label": str(c / "label.nii.gz")}
        for c in cases
        if (c / "image.nii.gz").exists() and (c / "label.nii.gz").exists()
    ]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    train_dicts = _data_dicts("train")
    val_dicts = _data_dicts("val")
    if not train_dicts:
        raise RuntimeError(
            f"No training cases found in {DATA_DIR / 'train'}. "
            "Place NIfTI files as data/brain/train/<case>/image.nii.gz + label.nii.gz"
        )

    train_loader = DataLoader(
        Dataset(train_dicts, transform=_make_transforms(train=True)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Dataset(val_dicts, transform=_make_transforms(train=False)),
        batch_size=1, shuffle=False, num_workers=2,
    )

    model = build_model()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    inferer = SimpleInferer()

    calc = BrainMetricsCalculator(num_classes=NUM_CLASSES)
    logger = MetricsLogger(
        task="brain_2d",
        results_dir=RESULTS_DIR,
        model_save_path=MODEL_SAVE,
    )

    print(f"Device: {DEVICE}  |  Train cases: {len(train_dicts)}  |  Val cases: {len(val_dicts)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- train ----
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            out = inferer(imgs, model)
            loss = loss_fn(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_losses: list[float] = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                out = inferer(imgs, model)
                val_losses.append(loss_fn(out, lbls).item())
                all_preds.append(torch.argmax(out, dim=1).cpu().numpy())
                all_targets.append(lbls.squeeze(1).cpu().numpy().astype(int))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        pred_arr = np.concatenate(all_preds, axis=0)
        tgt_arr = np.concatenate(all_targets, axis=0)
        metrics = calc.compute_all(pred_arr, tgt_arr)

        current_lr = float(scheduler.get_last_lr()[0])
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, metrics)
        scheduler.step()

        saved = logger.save_best_model(model, metrics["dice_mean"], epoch)
        marker = "  *** BEST ***" if saved else ""
        print(
            f"[{epoch:03d}/{NUM_EPOCHS}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice={metrics['dice_mean']:.4f}  "
            f"hd95={metrics.get('hd95_mean', float('nan')):.2f}  "
            f"lr={current_lr:.2e}"
            f"{marker}"
        )

    print("\nGenerating plots and summary…")
    logger.generate_plots()
    logger.generate_summary()
    print(f"Done. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    train()
