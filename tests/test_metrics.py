"""
Unit tests for utils/metrics.py — uses synthetic tensors, no real data needed.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np
import pytest
import torch

from utils.metrics import (
    BrainMetricsCalculator,
    LungMetricsCalculator,
    MetricsLogger,
    LUNG_CLASSES,
)


# ---------------------------------------------------------------------------
# BrainMetricsCalculator
# ---------------------------------------------------------------------------

class TestBrainMetrics:
    def setup_method(self):
        self.calc = BrainMetricsCalculator(num_classes=4)
        rng = np.random.default_rng(0)
        N, H, W = 2, 64, 64
        self.pred = rng.integers(0, 4, size=(N, H, W))
        self.target = rng.integers(0, 4, size=(N, H, W))

    def test_dice_keys(self):
        d = self.calc.compute_dice(self.pred, self.target)
        for c in range(1, 4):
            assert f"dice_class_{c}" in d
        assert "dice_mean" in d

    def test_dice_perfect(self):
        arr = np.ones((1, 32, 32), dtype=int)
        d = self.calc.compute_dice(arr, arr.copy())
        assert d["dice_class_1"] == pytest.approx(1.0, abs=1e-4)

    def test_dice_zero(self):
        pred = np.zeros((1, 32, 32), dtype=int)
        tgt = np.ones((1, 32, 32), dtype=int)
        d = self.calc.compute_dice(pred, tgt)
        assert d["dice_class_1"] < 0.01

    def test_iou_keys(self):
        d = self.calc.compute_iou(self.pred, self.target)
        assert "iou_mean" in d

    def test_hd95_returns_nan_for_empty_class(self):
        pred = np.zeros((1, 32, 32), dtype=int)
        tgt = np.zeros((1, 32, 32), dtype=int)
        d = self.calc.compute_hausdorff95(pred, tgt)
        assert math.isnan(d["hd95_class_1"])

    def test_sensitivity_range(self):
        d = self.calc.compute_sensitivity(self.pred, self.target)
        for c in range(1, 4):
            v = d[f"sensitivity_class_{c}"]
            assert 0.0 <= v <= 1.0

    def test_specificity_range(self):
        d = self.calc.compute_specificity(self.pred, self.target)
        for c in range(1, 4):
            v = d[f"specificity_class_{c}"]
            assert 0.0 <= v <= 1.0

    def test_compute_all_has_all_keys(self):
        d = self.calc.compute_all(self.pred, self.target)
        required = ["dice_mean", "iou_mean", "hd95_mean", "sensitivity_mean", "specificity_mean"]
        for key in required:
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# LungMetricsCalculator
# ---------------------------------------------------------------------------

class TestLungMetrics:
    def setup_method(self):
        self.calc = LungMetricsCalculator(num_classes=14)
        rng = np.random.default_rng(1)
        N = 100
        self.probs = rng.random((N, 14)).astype(np.float32)
        self.targets = rng.integers(0, 2, size=(N, 14))
        # ensure every class has at least one positive sample
        for i in range(14):
            self.targets[i, i] = 1

    def test_metrics_keys(self):
        m = self.calc.compute_all(self.probs, self.targets)
        for key in ["accuracy", "precision", "recall", "f1", "auc_macro"]:
            assert key in m, f"Missing: {key}"
        for name in LUNG_CLASSES:
            assert f"auc_{name}" in m

    def test_auc_in_range(self):
        m = self.calc.compute_all(self.probs, self.targets)
        assert 0.0 <= m["auc_macro"] <= 1.0

    def test_confusion_matrix_shape(self):
        cm = self.calc.compute_confusion_matrix(self.probs, self.targets)
        assert cm.shape == (2, 2)

    def test_roc_curves_structure(self):
        curves = self.calc.compute_roc_curves(self.probs, self.targets)
        for name, (fpr, tpr, auc_val) in curves.items():
            assert len(fpr) == len(tpr)
            assert 0.0 <= auc_val <= 1.0


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class TestMetricsLogger:
    def _make_dummy_model(self):
        return torch.nn.Linear(4, 2)

    def test_csv_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricsLogger(task="brain_2d", results_dir=tmp)
            logger.log_epoch(1, 0.5, 0.4, 1e-4, {"dice_mean": 0.7})
            logger.log_epoch(2, 0.4, 0.35, 9e-5, {"dice_mean": 0.75})
            csv_path = Path(tmp) / "brain_2d_metrics.csv"
            assert csv_path.exists()
            rows = csv_path.read_text().strip().split("\n")
            assert len(rows) == 3  # header + 2 data rows

    def test_best_model_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricsLogger(
                task="brain_2d",
                results_dir=tmp,
                model_save_path=str(Path(tmp) / "best.pth"),
            )
            model = self._make_dummy_model()
            logger.log_epoch(1, 0.5, 0.4, 1e-4, {"dice_mean": 0.6})
            saved = logger.save_best_model(model, 0.6, epoch=1)
            assert saved
            assert Path(tmp, "best.pth").exists()

    def test_second_save_only_when_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricsLogger(task="lung", results_dir=tmp,
                                   model_save_path=str(Path(tmp) / "best.pth"))
            model = self._make_dummy_model()
            logger.save_best_model(model, 0.8, epoch=1)
            assert not logger.save_best_model(model, 0.75, epoch=2)  # no improvement
            assert logger.save_best_model(model, 0.85, epoch=3)      # improvement

    def test_generate_plots_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricsLogger(task="brain_2d", results_dir=tmp)
            for ep in range(1, 4):
                logger.log_epoch(ep, 0.5, 0.4, 1e-4,
                                 {"dice_class_1": 0.6, "dice_class_2": 0.5, "dice_mean": 0.55})
            logger.best_epoch = 2
            logger.generate_plots()  # should not raise
            plots = list(Path(tmp, "plots", "brain_2d").glob("*.png"))
            assert len(plots) >= 3  # loss, dice, lr

    def test_summary_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricsLogger(task="lung", results_dir=tmp)
            logger.log_epoch(1, 0.3, 0.35, 1e-4,
                             {"accuracy": 0.8, "f1": 0.75, "auc_macro": 0.85})
            logger.best_epoch = 1
            logger.best_metric = 0.85
            logger.generate_summary()
            summary = Path(tmp) / "lung_training_summary.txt"
            assert summary.exists()
            assert "lung" in summary.read_text(encoding="utf-8").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
