"""
DenseNet-121 4-class COVID-19 chest X-ray classification.

v3 changes (underfitting fix):
  - Simplified augmentation (no MixUp/CutMix)
  - LR warmup 1e-6→1e-4 over 5 epochs, then CosineAnnealingLR(T_max=45)
  - Dropout 0.3, weight decay 1e-4, label smoothing 0.1
  - All DenseNet layers trainable (no freezing)
  - 50 epochs, early stopping patience=7

Data layout:
  data/lung/raw/COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from data.dataset_lung import CLASS_NAMES, COVIDDataset
from models.lung.densenet121 import build_densenet121
from utils.metrics import MetricsLogger
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CLASSES    = 4
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LR             = 1e-4
LR_START       = 1e-6       # warmup starting LR
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
VAL_SPLIT      = 0.2
USE_CLAHE      = False
NUM_WORKERS    = 4
ES_PATIENCE    = 7
DATA_DIR       = Path("data/lung")
MODEL_SAVE     = "models/lung/chexnet_best.pth"
RESULTS_DIR    = "results"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    # -- datasets --
    train_ds = COVIDDataset(data_dir=DATA_DIR, split="train",
                            val_split=VAL_SPLIT, use_clahe=USE_CLAHE)
    train_ds.transform = _train_transform()

    val_ds = COVIDDataset(data_dir=DATA_DIR, split="val",
                          val_split=VAL_SPLIT, use_clahe=USE_CLAHE)
    val_ds.transform = _val_transform()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())

    # -- model: all layers trainable --
    model   = build_densenet121(num_classes=NUM_CLASSES, pretrained=True,
                                dropout=DROPOUT)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -- optimizer: warmup starts from LR_START --
    optimizer = AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR runs for the post-warmup portion
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    logger     = MetricsLogger(task="lung", results_dir=RESULTS_DIR,
                               model_save_path=MODEL_SAVE)
    early_stop = EarlyStopping(patience=ES_PATIENCE)

    Path(MODEL_SAVE).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device   : {DEVICE}\n"
        f"Classes  : {CLASS_NAMES}\n"
        f"Train    : {len(train_ds):,}  |  Val: {len(val_ds):,}\n"
        f"Epochs   : {NUM_EPOCHS} (ES patience={ES_PATIENCE})\n"
        f"LR       : warmup {LR_START:.0e}->{LR:.0e} ({WARMUP_EPOCHS} ep) "
        f"then CosineAnnealing(T_max={NUM_EPOCHS - WARMUP_EPOCHS}, eta_min=1e-6)\n"
        f"WD       : {WEIGHT_DECAY}  |  Dropout: {DROPOUT}  |  Label-smooth: 0.1"
    )

    for epoch in range(1, NUM_EPOCHS + 1):

        # ---------- LR schedule ----------
        if epoch <= WARMUP_EPOCHS:
            # linear warmup: 1e-6 → 1e-4
            warmup_lr = LR_START + (LR - LR_START) * (epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ---------- train ----------
        model.train()
        train_losses: list[float] = []

        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE)
            lbls = (lbls if isinstance(lbls, torch.Tensor)
                    else torch.tensor(lbls)).to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---------- validate ----------
        model.eval()
        val_losses:  list[float] = []
        all_preds:   list[int]   = []
        all_targets: list[int]   = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                lbls = (lbls if isinstance(lbls, torch.Tensor)
                        else torch.tensor(lbls)).to(DEVICE)
                out  = model(imgs)
                val_losses.append(loss_fn(out, lbls).item())
                all_preds.extend(out.argmax(dim=1).cpu().tolist())
                all_targets.extend(lbls.cpu().tolist())

        # CosineAnnealing steps only after warmup
        if epoch > WARMUP_EPOCHS:
            cosine_scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        preds_arr  = np.array(all_preds)
        tgts_arr   = np.array(all_targets)

        metrics = {
            "accuracy":    float(accuracy_score(tgts_arr, preds_arr)),
            "f1_macro":    float(f1_score(tgts_arr, preds_arr,
                                          average="macro",    zero_division=0)),
            "f1_weighted": float(f1_score(tgts_arr, preds_arr,
                                          average="weighted", zero_division=0)),
        }
        for cls_f1, cls_name in zip(
            f1_score(tgts_arr, preds_arr, average=None, zero_division=0),
            CLASS_NAMES,
        ):
            metrics[f"f1_{cls_name.replace(' ', '_')}"] = float(cls_f1)

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, metrics)

        saved  = logger.save_best_model(model, metrics["f1_macro"], epoch)
        marker = "  *** BEST ***" if saved else ""
        es_info = f"  [ES {early_stop.counter}/{ES_PATIENCE}]" \
                  if early_stop.counter > 0 else ""

        print(
            f"[{epoch:02d}/{NUM_EPOCHS}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  "
            f"lr={current_lr:.2e}"
            f"{marker}{es_info}"
        )

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val_loss improvement for {ES_PATIENCE} epochs).")
            break

    # -- final confusion matrix --
    print("\nComputing final confusion matrix…")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_targets.extend(
                lbls.tolist() if isinstance(lbls, torch.Tensor)
                else torch.tensor(lbls).tolist()
            )

    cm = confusion_matrix(all_targets, all_preds)
    logger.generate_plots(confusion_mat=cm, class_names=CLASS_NAMES)
    logger.generate_summary()
    print(f"\nDone.  Best f1_macro={logger.best_metric:.4f} @ epoch {logger.best_epoch}")
    print(f"Results   : {RESULTS_DIR}/\nCheckpoint: {MODEL_SAVE}")


if __name__ == "__main__":
    train()
