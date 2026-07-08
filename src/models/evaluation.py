"""
src/models/evaluation.py
========================
Fraud-specific evaluation metrics and plots.

Fraud detection requires different metrics than balanced classification:
  - PR-AUC  : Primary metric — handles severe class imbalance
  - F2-score : Business metric — missing fraud costs more than false alarm
  - Fraud Recall    : % of actual fraud transactions caught
  - Fraud Precision : % of fraud alerts that are real fraud
  - ROC-AUC : General discriminability reference

Plots
-----
  - Precision-Recall curve (primary)
  - ROC curve (secondary reference)
  - Confusion matrix at tuned threshold
  - Threshold analysis: F2 + Recall vs threshold
  - Calibration curve
  - Cost curve (FP/FN costs vs threshold)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_fraud_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    beta: float = 2.0,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud-specific metrics.

    Parameters
    ----------
    y_true : array-like  — true binary labels
    y_prob : np.ndarray  — predicted fraud probability
    threshold : float    — classification threshold
    beta : float         — fbeta beta (2.0 = F2)
    cost_fp, cost_fn : float — $ cost per FP / FN

    Returns
    -------
    dict[str, float | int]
    """
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = float(fp) * cost_fp + float(fn) * cost_fn

    metrics = {
        "threshold":        round(float(threshold), 4),
        "pr_auc":           round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc":          round(float(roc_auc_score(y_true, y_prob)), 4),
        "f2_score":         round(float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)), 4),
        "f1_score":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fraud_recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "fraud_precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "mcc":              round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "log_loss":         round(float(log_loss(y_true, y_prob)), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_samples":  int(len(y_true)),
        "n_fraud":    int(y_true.sum()),
        "fraud_rate": round(float(y_true.mean()), 6),
        "n_predicted_fraud": int(y_pred.sum()),
        "total_cost_usd":  round(total_cost, 2),
    }

    logger.info(
        f"Metrics — PR-AUC={metrics['pr_auc']:.4f} | "
        f"F2={metrics['f2_score']:.4f} | "
        f"Recall={metrics['fraud_recall']:.4f} | "
        f"Precision={metrics['fraud_precision']:.4f} | "
        f"Cost=${total_cost:,.0f}"
    )
    return metrics


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """Ranked comparison DataFrame from multiple model results."""
    rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update(res.get("val_metrics", {}))
        row["cv_pr_auc_mean"] = res.get("cv_pr_auc_mean", np.nan)
        row["cv_pr_auc_std"]  = res.get("cv_pr_auc_std",  np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)
    return df.reset_index(drop=True)


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    models_probs: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (8, 6),
):
    """Precision-Recall curves for one or multiple models."""
    import matplotlib.pyplot as plt
    COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, y_prob) in enumerate(models_probs.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, lw=2, color=COLORS[i % len(COLORS)],
                label=f"{name} (AP={ap:.4f})")

    ax.axhline(baseline, color="grey", ls="--", lw=1.5,
               label=f"Baseline (fraud rate={baseline:.4%})")
    ax.set_xlabel("Recall (Fraud Detected)", fontsize=12)
    ax.set_ylabel("Precision (Alert Accuracy)", fontsize=12)
    ax.set_title("Precision-Recall Curve\n(Primary metric for imbalanced fraud data)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    models_probs: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (8, 6),
):
    """ROC curves (secondary reference metric for fraud)."""
    import matplotlib.pyplot as plt
    COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, y_prob) in enumerate(models_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=COLORS[i % len(COLORS)],
                label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (reference — less informative for imbalanced data)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    figsize: Tuple[int, int] = (12, 5),
):
    """Absolute + normalized confusion matrices side by side."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = ["Legitimate", "Fraud"]
    for ax, (data, title, fmt) in zip(
        axes,
        [(cm, "Absolute Counts", "d"), (cm_norm, "Normalized (Row %)", ".2%")],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title}\n{model_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    figsize: Tuple[int, int] = (16, 5),
):
    """F2, Recall, Precision + cost curve vs threshold."""
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.01, 0.99, 150)
    f2s, recalls, precisions, costs = [], [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2s.append(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        costs.append(fp * cost_fp + fn * cost_fn)

    best_f2_thresh = thresholds[np.argmax(f2s)]
    best_cost_thresh = thresholds[np.argmin(costs)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(thresholds, f2s,       lw=2, color="#2196F3", label="F2-Score")
    axes[0].plot(thresholds, recalls,   lw=2, color="#4CAF50", label="Fraud Recall")
    axes[0].plot(thresholds, precisions,lw=2, color="#FF9800", label="Fraud Precision")
    axes[0].axvline(best_f2_thresh, color="#2196F3", ls="--", lw=1.5,
                    label=f"Best F2 @ {best_f2_thresh:.2f}")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("F2 / Recall / Precision vs Threshold", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(thresholds, [c / 1000 for c in costs], lw=2, color="#F44336")
    axes[1].axvline(best_cost_thresh, color="#9C27B0", ls="--", lw=1.5,
                    label=f"Min cost @ {best_cost_thresh:.2f}")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Total Cost ($K)")
    axes[1].set_title(f"Cost Curve (FP=${cost_fp}, FN=${cost_fn})", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Threshold Analysis for Fraud Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    models_probs: Dict[str, np.ndarray],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 6),
):
    """Reliability diagram (calibration curve)."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")

    for i, (name, y_prob) in enumerate(models_probs.items()):
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                color=COLORS[i % len(COLORS)], label=name)

    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def print_fraud_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str = "Model",
) -> str:
    """Print a fraud-specific classification summary."""
    from sklearn.metrics import classification_report
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraud"],
        digits=4,
    )
    metrics = compute_fraud_metrics(y_true, y_prob, threshold)
    header = (
        f"\n{'='*60}\n  {model_name} — Fraud Detection Report\n{'='*60}\n"
        f"  PR-AUC:    {metrics['pr_auc']:.4f}\n"
        f"  F2-Score:  {metrics['f2_score']:.4f}\n"
        f"  Recall:    {metrics['fraud_recall']:.4f}\n"
        f"  Precision: {metrics['fraud_precision']:.4f}\n"
        f"  Threshold: {threshold:.3f}\n"
        f"  Cost ($):  ${metrics['total_cost_usd']:,.0f}\n"
        f"{'='*60}\n"
    )
    print(header + report)
    return report
