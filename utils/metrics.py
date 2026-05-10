"""
Unified metrics, CSV logging, checkpoint saving, and plot generation
for brain MRI segmentation and chest X-ray classification tasks.
"""

import csv
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# CheXNet 14-class labels (NIH ChestX-ray14)
LUNG_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Return directed surface distances d(gt→pred) and d(pred→gt)."""
    pred_bool = pred_bin.astype(bool)
    gt_bool = gt_bin.astype(bool)

    # edt of complement = distance to nearest foreground voxel
    pred_edt = distance_transform_edt(~pred_bool)
    gt_edt = distance_transform_edt(~gt_bool)

    # Surface = foreground pixels/voxels that border background
    pred_surface = pred_bool & ~binary_erosion(pred_bool)
    gt_surface = gt_bool & ~binary_erosion(gt_bool)

    if not pred_surface.any() or not gt_surface.any():
        return np.array([np.nan]), np.array([np.nan])

    d_gt_to_pred = pred_edt[gt_surface]
    d_pred_to_gt = gt_edt[pred_surface]
    return d_gt_to_pred, d_pred_to_gt


# ---------------------------------------------------------------------------
# Brain Segmentation Metrics
# ---------------------------------------------------------------------------

class BrainMetricsCalculator:
    """
    All metrics operate on integer-valued numpy arrays where 0 = background
    and 1..num_classes-1 are foreground classes.

    pred / target shapes: (N, H, W) for 2-D or (N, D, H, W) for 3-D.
    """

    def __init__(self, num_classes: int, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth

    # --- per-metric methods ---

    def compute_dice(self, pred: np.ndarray, target: np.ndarray) -> dict:
        out = {}
        for c in range(1, self.num_classes):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            intersection = (p * t).sum()
            out[f"dice_class_{c}"] = float(
                (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            )
        vals = list(out.values())
        out["dice_mean"] = float(np.mean(vals)) if vals else 0.0
        return out

    def compute_iou(self, pred: np.ndarray, target: np.ndarray) -> dict:
        out = {}
        for c in range(1, self.num_classes):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            intersection = (p * t).sum()
            union = p.sum() + t.sum() - intersection
            out[f"iou_class_{c}"] = float(
                (intersection + self.smooth) / (union + self.smooth)
            )
        vals = list(out.values())
        out["iou_mean"] = float(np.mean(vals)) if vals else 0.0
        return out

    def compute_hausdorff95(self, pred: np.ndarray, target: np.ndarray) -> dict:
        out = {}
        for c in range(1, self.num_classes):
            p = pred == c
            t = target == c
            if not p.any() or not t.any():
                out[f"hd95_class_{c}"] = float("nan")
                continue
            d1, d2 = _surface_distances(p, t)
            all_d = np.concatenate([d1, d2])
            valid = all_d[~np.isnan(all_d)]
            out[f"hd95_class_{c}"] = float(np.percentile(valid, 95)) if valid.size else float("nan")
        valid_vals = [v for v in out.values() if not math.isnan(v)]
        out["hd95_mean"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        return out

    def compute_sensitivity(self, pred: np.ndarray, target: np.ndarray) -> dict:
        out = {}
        for c in range(1, self.num_classes):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            tp = (p * t).sum()
            fn = ((1 - p) * t).sum()
            out[f"sensitivity_class_{c}"] = float(
                (tp + self.smooth) / (tp + fn + self.smooth)
            )
        vals = list(out.values())
        out["sensitivity_mean"] = float(np.mean(vals)) if vals else 0.0
        return out

    def compute_specificity(self, pred: np.ndarray, target: np.ndarray) -> dict:
        out = {}
        for c in range(1, self.num_classes):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            tn = ((1 - p) * (1 - t)).sum()
            fp = (p * (1 - t)).sum()
            out[f"specificity_class_{c}"] = float(
                (tn + self.smooth) / (tn + fp + self.smooth)
            )
        vals = list(out.values())
        out["specificity_mean"] = float(np.mean(vals)) if vals else 0.0
        return out

    def compute_all(self, pred: np.ndarray, target: np.ndarray) -> dict:
        metrics = {}
        metrics.update(self.compute_dice(pred, target))
        metrics.update(self.compute_iou(pred, target))
        metrics.update(self.compute_hausdorff95(pred, target))
        metrics.update(self.compute_sensitivity(pred, target))
        metrics.update(self.compute_specificity(pred, target))
        return metrics


# ---------------------------------------------------------------------------
# Lung Classification Metrics
# ---------------------------------------------------------------------------

class LungMetricsCalculator:
    """
    Multi-label binary classification metrics for chest X-ray.

    probs:   np.ndarray (N, C) — sigmoid probabilities in [0, 1]
    targets: np.ndarray (N, C) — binary ground-truth labels {0, 1}
    """

    def __init__(self, num_classes: int = 14, class_names: list | None = None):
        self.num_classes = num_classes
        self.class_names = (class_names or LUNG_CLASSES)[:num_classes]

    def compute_all(self, probs: np.ndarray, targets: np.ndarray) -> dict:
        preds = (probs >= 0.5).astype(int)
        metrics = {}

        # Per-class AUC-ROC
        auc_vals = []
        for i, name in enumerate(self.class_names):
            if targets[:, i].sum() > 0:
                val = float(roc_auc_score(targets[:, i], probs[:, i]))
            else:
                val = float("nan")
                warnings.warn(f"AUC skipped for class '{name}': no positive samples.")
            metrics[f"auc_{name}"] = val
            auc_vals.append(val)

        valid_aucs = [v for v in auc_vals if not math.isnan(v)]
        metrics["auc_macro"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        # Flat binary metrics (aggregate over all classes)
        flat_p = preds.flatten()
        flat_t = targets.flatten()
        metrics["accuracy"] = float(accuracy_score(flat_t, flat_p))
        metrics["precision"] = float(precision_score(flat_t, flat_p, zero_division=0))
        metrics["recall"] = float(recall_score(flat_t, flat_p, zero_division=0))
        metrics["f1"] = float(f1_score(flat_t, flat_p, zero_division=0))

        return metrics

    def compute_confusion_matrix(
        self, probs: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        preds = (probs >= 0.5).astype(int)
        return confusion_matrix(targets.flatten(), preds.flatten())

    def compute_roc_curves(
        self, probs: np.ndarray, targets: np.ndarray
    ) -> dict[str, tuple]:
        """Returns {class_name: (fpr, tpr, auc_val)} for classes with positive samples."""
        curves = {}
        for i, name in enumerate(self.class_names):
            if targets[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
            curves[name] = (fpr, tpr, float(auc(fpr, tpr)))
        return curves


# ---------------------------------------------------------------------------
# MetricsLogger — CSV + checkpoint + plots + summary
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Per-epoch logging, best-model checkpoint saving, plot generation,
    and training summary for a single training run.

    task: 'brain_2d' | 'brain_3d' | 'lung'
    """

    def __init__(
        self,
        task: str,
        results_dir: str = "results",
        model_save_path: str | None = None,
    ):
        self.task = task
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots" / task
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.results_dir / f"{task}_metrics.csv"
        self.summary_path = self.results_dir / f"{task}_training_summary.txt"
        self.model_save_path = Path(
            model_save_path or self.results_dir / f"{task}_best_model.pth"
        )
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        self.history: list[dict] = []
        self.best_metric: float = -np.inf
        self.best_epoch: int = 0
        self._header_written = False

    # --- logging ---

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        metrics: dict,
    ) -> None:
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
        }
        for k, v in metrics.items():
            row[k] = round(v, 6) if isinstance(v, float) and not math.isnan(v) else v
        self.history.append(row)
        self._write_csv_row(row)

    def _write_csv_row(self, row: dict) -> None:
        mode = "a" if self._header_written else "w"
        with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    # --- checkpoint ---

    def save_best_model(
        self, model: nn.Module, metric_value: float, epoch: int
    ) -> bool:
        """Save state_dict if metric_value improves. Returns True when saved."""
        if math.isnan(metric_value) or metric_value <= self.best_metric:
            return False
        self.best_metric = metric_value
        self.best_epoch = epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_metric": metric_value,
            },
            self.model_save_path,
        )
        return True

    # --- plots ---

    def generate_plots(
        self,
        roc_curves: dict | None = None,
        confusion_mat: np.ndarray | None = None,
        class_names: list | None = None,
    ) -> None:
        if not self.history:
            return
        self._plot_loss_curve()
        self._plot_lr_curve()
        if self.task in ("brain_2d", "brain_3d"):
            self._plot_dice_curve()
        else:
            self._plot_accuracy_curve()
        if roc_curves:
            self._plot_roc_curves(roc_curves)
        if confusion_mat is not None:
            self._plot_confusion_matrix(confusion_mat, class_names)
        plt.close("all")

    def _epochs(self):
        return [r["epoch"] for r in self.history]

    def _col(self, key):
        return [r.get(key) for r in self.history]

    def _plot_loss_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self._epochs(), self._col("train_loss"), label="Train Loss", lw=2)
        ax.plot(self._epochs(), self._col("val_loss"), label="Val Loss", lw=2)
        if self.best_epoch:
            ax.axvline(
                self.best_epoch, color="red", ls="--", alpha=0.7,
                label=f"Best epoch {self.best_epoch}",
            )
        ax.set(xlabel="Epoch", ylabel="Loss",
               title=f"{self.task.upper()} — Train vs Validation Loss")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "loss_curve.png", dpi=150)
        plt.close(fig)

    def _plot_lr_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(self._epochs(), self._col("lr"), lw=2, color="darkorange")
        ax.set(xlabel="Epoch", ylabel="Learning Rate (log)",
               title=f"{self.task.upper()} — Learning Rate Schedule")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "lr_curve.png", dpi=150)
        plt.close(fig)

    def _plot_dice_curve(self) -> None:
        dice_keys = [k for k in self.history[0] if k.startswith("dice_class_")]
        fig, ax = plt.subplots(figsize=(10, 5))
        for key in dice_keys:
            ax.plot(self._epochs(), self._col(key), label=key, lw=1.5)
        if "dice_mean" in self.history[0]:
            ax.plot(
                self._epochs(), self._col("dice_mean"),
                label="Dice Mean", lw=2.5, ls="--", color="black",
            )
        if self.best_epoch:
            ax.axvline(self.best_epoch, color="red", ls="--", alpha=0.7,
                       label=f"Best epoch {self.best_epoch}")
        ax.set(xlabel="Epoch", ylabel="Dice Score",
               title=f"{self.task.upper()} — Dice Score per Epoch")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "dice_curve.png", dpi=150)
        plt.close(fig)

    def _plot_accuracy_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, label in [("accuracy", "Accuracy"), ("f1", "F1"), ("auc_macro", "AUC Macro")]:
            if key in self.history[0]:
                vals = [r.get(key) or 0 for r in self.history]
                ax.plot(self._epochs(), vals, label=label, lw=2)
        if self.best_epoch:
            ax.axvline(self.best_epoch, color="red", ls="--", alpha=0.7,
                       label=f"Best epoch {self.best_epoch}")
        ax.set(xlabel="Epoch", ylabel="Score",
               title=f"{self.task.upper()} — Classification Metrics per Epoch")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "accuracy_curve.png", dpi=150)
        plt.close(fig)

    def _plot_roc_curves(self, roc_curves: dict) -> None:
        n = len(roc_curves)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for ax, (name, (fpr, tpr, auc_val)) in zip(axes_flat, roc_curves.items()):
            ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(f"{self.task.upper()} — ROC Curves", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "roc_curves.png", dpi=150)
        plt.close(fig)

    def _plot_confusion_matrix(
        self, cm: np.ndarray, class_names: list | None = None
    ) -> None:
        labels = class_names or ["Negative", "Positive"]
        size = max(6, len(labels))
        fig, ax = plt.subplots(figsize=(size, size - 1))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set(xlabel="Predicted", ylabel="Actual",
               title=f"{self.task.upper()} — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    # --- summary ---

    def generate_summary(self) -> None:
        if not self.history:
            return
        best_row = next(
            (r for r in self.history if r["epoch"] == self.best_epoch),
            self.history[-1],
        )
        last_row = self.history[-1]

        sep = "=" * 62
        lines = [
            sep,
            f"  Training Summary — {self.task.upper()}",
            f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
            f"  Total epochs   : {last_row['epoch']}",
            f"  Best epoch     : {self.best_epoch}",
            f"  Best metric    : {self.best_metric:.6f}",
            f"  Checkpoint     : {self.model_save_path}",
            "",
            "  --- Best Epoch Metrics ---",
        ]
        for k, v in best_row.items():
            if k != "epoch":
                lines.append(f"    {k:<32} {v}")
        lines += ["", "  --- Final Epoch Metrics ---"]
        for k, v in last_row.items():
            if k != "epoch":
                lines.append(f"    {k:<32} {v}")
        lines += [
            "",
            f"  CSV log   : {self.csv_path}",
            f"  Plots dir : {self.plots_dir}",
            sep,
        ]

        report = "\n".join(lines)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
