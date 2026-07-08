"""
tests/test_evaluation.py
========================
Tests for src/models/evaluation.py
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    compute_fraud_metrics,
    compare_models,
    plot_pr_curve,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_threshold_analysis,
    plot_calibration_curve,
)


@pytest.fixture
def imbalanced_labels():
    """Imbalanced y_true/y_prob resembling real fraud data (~5% fraud)."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 950 + [1] * 50)
    y_prob = np.where(
        y_true == 1,
        rng.beta(7, 2, 1000),
        rng.beta(1, 8, 1000),
    )
    return y_true, y_prob


class TestComputeFraudMetrics:
    def test_all_keys_present(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        required = [
            "pr_auc", "roc_auc", "f2_score", "f1_score",
            "fraud_recall", "fraud_precision", "accuracy",
            "mcc", "log_loss", "tp", "tn", "fp", "fn",
            "n_samples", "n_fraud", "fraud_rate", "total_cost_usd",
        ]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_pr_auc_in_valid_range(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert 0.0 <= m["pr_auc"] <= 1.0

    def test_roc_auc_in_valid_range(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_f2_in_valid_range(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert 0.0 <= m["f2_score"] <= 1.0

    def test_recall_in_valid_range(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert 0.0 <= m["fraud_recall"] <= 1.0

    def test_n_fraud_correct(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert m["n_fraud"] == int(y_true.sum())

    def test_n_samples_correct(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert m["n_samples"] == len(y_true)

    def test_confusion_counts_sum_to_n(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["n_samples"]

    def test_perfect_classifier(self):
        y_true = np.array([0] * 100 + [1] * 10)
        y_prob = np.array([0.01] * 100 + [0.99] * 10)
        m = compute_fraud_metrics(y_true, y_prob, threshold=0.5)
        assert m["fraud_recall"]    == pytest.approx(1.0)
        assert m["fraud_precision"] == pytest.approx(1.0)
        assert m["fn"] == 0
        assert m["fp"] == 0

    def test_total_cost_is_non_negative(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(y_true, y_prob)
        assert m["total_cost_usd"] >= 0.0

    def test_lower_threshold_increases_recall(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m_low  = compute_fraud_metrics(y_true, y_prob, threshold=0.1)
        m_high = compute_fraud_metrics(y_true, y_prob, threshold=0.9)
        assert m_low["fraud_recall"] >= m_high["fraud_recall"]

    def test_higher_threshold_increases_precision(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m_low  = compute_fraud_metrics(y_true, y_prob, threshold=0.1)
        m_high = compute_fraud_metrics(y_true, y_prob, threshold=0.9)
        assert m_high["fraud_precision"] >= m_low["fraud_precision"]

    def test_accepts_series_input(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m = compute_fraud_metrics(pd.Series(y_true), y_prob)
        assert "pr_auc" in m


class TestCompareModels:
    def test_returns_dataframe(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m1 = compute_fraud_metrics(y_true, y_prob, threshold=0.3)
        m2 = compute_fraud_metrics(y_true, y_prob, threshold=0.5)
        results = {
            "model_a": {"val_metrics": m1, "cv_pr_auc_mean": 0.75},
            "model_b": {"val_metrics": m2, "cv_pr_auc_mean": 0.70},
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        assert "pr_auc" in df.columns

    def test_sorted_by_pr_auc_descending(self, imbalanced_labels):
        y_true, y_prob = imbalanced_labels
        m_base = compute_fraud_metrics(y_true, y_prob)
        results = {
            "model_a": {"val_metrics": {**m_base, "pr_auc": 0.90}},
            "model_b": {"val_metrics": {**m_base, "pr_auc": 0.60}},
            "model_c": {"val_metrics": {**m_base, "pr_auc": 0.75}},
        }
        df = compare_models(results)
        pr_aucs = df["pr_auc"].tolist()
        assert pr_aucs == sorted(pr_aucs, reverse=True)


class TestPlotFunctions:
    """Smoke tests — verify plots return Figures without errors."""

    def test_pr_curve_returns_figure(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        fig = plot_pr_curve(y_true, {"model": y_prob})
        assert fig is not None
        plt.close(fig)

    def test_roc_curve_returns_figure(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        fig = plot_roc_curve(y_true, {"model": y_prob})
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_returns_figure(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        y_pred = (y_prob >= 0.5).astype(int)
        fig = plot_confusion_matrix(y_true, y_pred, model_name="test")
        assert fig is not None
        plt.close(fig)

    def test_threshold_analysis_returns_figure(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        fig = plot_threshold_analysis(y_true, y_prob)
        assert fig is not None
        plt.close(fig)

    def test_calibration_curve_returns_figure(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        fig = plot_calibration_curve(y_true, {"model": y_prob})
        assert fig is not None
        plt.close(fig)

    def test_multi_model_pr_curve(self, imbalanced_labels):
        import matplotlib.pyplot as plt
        y_true, y_prob = imbalanced_labels
        fig = plot_pr_curve(y_true, {
            "model_a": y_prob,
            "model_b": y_prob * 0.85 + 0.05,
        })
        assert fig is not None
        plt.close(fig)
