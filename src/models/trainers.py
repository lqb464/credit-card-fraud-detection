"""
src/models/trainers.py
======================
Unified training interface for credit card fraud detection.

Supported algorithms
--------------------
hist_gradient_boosting : sklearn HistGradientBoostingClassifier
                         Native class_weight='balanced' support
xgboost                : XGBClassifier with scale_pos_weight
lightgbm               : LGBMClassifier with is_unbalance=True
logistic_regression    : Balanced baseline

Threshold Optimization
-----------------------
For fraud detection, the default 0.5 threshold is almost always wrong
due to class imbalance. We optimize for:
  - F2-score  : Recall weighted 2× (miss fraud = worse than false alarm)
  - Cost-min  : Minimize (FP × cost_fp + FN × cost_fn)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ── Classifier factory ────────────────────────────────────────────────────────

def _build_classifier(
    name: str,
    params: Dict[str, Any],
    scale_pos_weight: float = 1.0,
) -> Any:
    """Instantiate classifier by name with imbalance-handling params."""
    rs = params.get("random_state", 42)

    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight=params.get("class_weight", "balanced"),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("max_iter", 200),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.1),
            random_state=rs,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,   # handles imbalance
            random_state=rs,
            n_jobs=-1,
            eval_metric="aucpr",
            verbosity=0,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            is_unbalance=params.get("is_unbalance", True),
            random_state=rs,
            n_jobs=-1,
            verbose=-1,
        )
    elif name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 0.1),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 500),
            solver=params.get("solver", "lbfgs"),
            random_state=rs,
        )
    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            "Choose from: hist_gradient_boosting, xgboost, lightgbm, logistic_regression"
        )


def build_sklearn_pipeline(
    model_name: str,
    model_params: Dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Build full sklearn Pipeline: ColumnTransformer → Classifier.

    HistGradientBoosting natively handles NaN, but we still standardize
    for consistency and LogisticRegression compatibility.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", cat_pipe,     categorical_features),
    ], remainder="drop")

    classifier = _build_classifier(model_name, model_params, scale_pos_weight)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ── Threshold optimizers ──────────────────────────────────────────────────────

def optimize_threshold_f2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 100,
) -> float:
    """Find threshold maximizing F-beta score (default β=2 for fraud)."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score, best_thresh = 0.0, 0.5
    for t in thresholds:
        score = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, float(t)
    logger.info(f"Optimal threshold (F{beta}): {best_thresh:.3f} → score={best_score:.4f}")
    return best_thresh


def optimize_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    n_thresholds: int = 100,
) -> float:
    """Find threshold minimizing expected cost = FP×cost_fp + FN×cost_fn."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_cost, best_thresh = float("inf"), 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_thresh = cost, float(t)
    logger.info(f"Optimal threshold (cost-min): {best_thresh:.3f} → cost=${best_cost:,.0f}")
    return best_thresh


# ── Training entry point ──────────────────────────────────────────────────────

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: Dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    threshold_method: str = "f2",
    beta: float = 2.0,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    cv_sample_size: int = 30_000,
    run_name: Optional[str] = None,
    log_to_mlflow: bool = True,
    save_path: Optional[Path] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train fraud detection model with threshold optimization and MLflow logging.

    Parameters
    ----------
    model_name : str
    X_train, y_train : training data
    X_val, y_val : validation / test data
    model_params : dict of hyperparameters
    numeric_features, categorical_features : feature name lists
    threshold_method : "f2" | "cost" | "f1" | "fixed"
    beta : fbeta beta for F2 optimization
    cost_fp, cost_fn : $ cost per FP/FN (for cost method)
    cv_sample_size : subsample size for 3-fold CV
    run_name : MLflow run name
    log_to_mlflow : bool
    save_path : Path to save model artifact
    random_state : int

    Returns
    -------
    dict: pipeline, threshold, val_metrics, cv_pr_auc_mean, cv_pr_auc_std
    """
    logger.info(f"Training [{model_name}] | train={len(X_train):,} | fraud={int(y_train.sum()):,} ({y_train.mean():.4%})")

    # Compute scale_pos_weight for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    pipeline = build_sklearn_pipeline(
        model_name, model_params,
        numeric_features, categorical_features,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline.fit(X_train, y_train)

    # Predict on val
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]

    # Threshold optimization
    if threshold_method == "f2":
        threshold = optimize_threshold_f2(y_val.values, y_prob_val, beta=beta)
    elif threshold_method == "cost":
        threshold = optimize_threshold_cost(y_val.values, y_prob_val, cost_fp, cost_fn)
    elif threshold_method == "f1":
        threshold = optimize_threshold_f2(y_val.values, y_prob_val, beta=1.0)
    elif threshold_method == "fixed":
        threshold = model_params.get("fixed_threshold", 0.5)
    else:
        threshold = 0.5

    # Cross-validation (PR-AUC on subsample)
    idx = np.random.RandomState(random_state).choice(len(X_train), min(cv_sample_size, len(X_train)), replace=False)
    cv_scores = cross_val_score(
        pipeline, X_train.iloc[idx], y_train.iloc[idx],
        cv=3, scoring="average_precision", n_jobs=-1,
    )

    from src.models.evaluation import compute_fraud_metrics
    val_metrics = compute_fraud_metrics(y_val.values, y_prob_val, threshold=threshold, beta=beta)
    val_metrics["cv_pr_auc_mean"] = round(float(cv_scores.mean()), 4)
    val_metrics["cv_pr_auc_std"]  = round(float(cv_scores.std()),  4)

    logger.info(
        f"[{model_name}] PR-AUC={val_metrics['pr_auc']:.4f} "
        f"F2={val_metrics['f2_score']:.4f} Recall={val_metrics['fraud_recall']:.4f} "
        f"threshold={threshold:.3f}"
    )

    if log_to_mlflow:
        with mlflow.start_run(run_name=run_name or model_name):
            mlflow.log_params({
                "model": model_name,
                "threshold_method": threshold_method,
                "threshold": threshold,
                "scale_pos_weight": round(scale_pos_weight, 2),
                **{k: v for k, v in model_params.items()
                   if isinstance(v, (int, float, str, bool))},
            })
            mlflow.log_metrics({
                k: v for k, v in val_metrics.items() if isinstance(v, (int, float))
            })
            mlflow.sklearn.log_model(
                pipeline, "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "threshold": threshold}, save_path)
        logger.info(f"Model saved: {save_path}")

    return {
        "pipeline":        pipeline,
        "threshold":       threshold,
        "val_metrics":     val_metrics,
        "cv_pr_auc_mean":  float(cv_scores.mean()),
        "cv_pr_auc_std":   float(cv_scores.std()),
        "scale_pos_weight":scale_pos_weight,
    }
