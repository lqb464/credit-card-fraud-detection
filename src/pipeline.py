"""
src/pipeline.py
===============
End-to-end fraud detection pipeline.

Steps
-----
1. Load train + test data (with stratified subsampling)
2. Feature engineering (Haversine, datetime, log transforms)
3. Split features / target
4. Train model (F2-optimized threshold) + MLflow tracking
5. Evaluate on full test set
6. Save model artifact + metrics report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import yaml

from src.data.loader import load_train_test
from src.data.features import (
    engineer_features,
    split_features_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
)
from src.models.evaluation import compute_fraud_metrics, print_fraud_report
from src.models.trainers import train_model

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
    model_params: Optional[Dict] = None,
    smoke_test: bool = False,
    save_outputs: bool = True,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full fraud detection pipeline.

    Parameters
    ----------
    config_path : YAML config path
    model_name : override default model from config
    model_params : override model hyperparameters
    smoke_test : use small subsample for fast iteration
    save_outputs : save model + metrics to disk
    run_name : MLflow run name

    Returns
    -------
    dict: pipeline, threshold, test_metrics, feature_names
    """
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)

    algo     = model_name or cfg["models"]["default_model"]
    algo_cfg = model_params or cfg["models"].get(algo, {})
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["data"]

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_train_raw, df_test_raw = load_train_test(
        train_path=data_cfg["train_path"],
        test_path=data_cfg["test_path"],
        train_sample_size=data_cfg["train_sample_size"],
        smoke_test=smoke_test,
        smoke_train=data_cfg["smoke_train_size"],
        smoke_test_n=data_cfg["smoke_test_size"],
        random_state=data_cfg["random_state"],
    )

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    df_train = engineer_features(df_train_raw)
    df_test  = engineer_features(df_test_raw)

    # ── 3. Split X / y ────────────────────────────────────────────────────────
    feat_names = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X_train, y_train = split_features_target(df_train, feature_names=feat_names)
    X_test,  y_test  = split_features_target(df_test,  feature_names=feat_names)

    num_feats = cfg["features"]["numeric"]
    cat_feats = cfg["features"]["categorical"]

    # ── 4. Train ──────────────────────────────────────────────────────────────
    save_path = None
    if save_outputs:
        out_dir = Path(cfg["outputs"]["models_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{algo}_model.joblib"

    result = train_model(
        model_name=algo,
        X_train=X_train, y_train=y_train,
        X_val=X_test,    y_val=y_test,
        model_params=algo_cfg,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        threshold_method=eval_cfg["threshold_method"],
        beta=eval_cfg["beta"],
        cost_fp=eval_cfg["cost_fp"],
        cost_fn=eval_cfg["cost_fn"],
        cv_sample_size=eval_cfg["cv_sample_size"],
        run_name=run_name or f"{algo}_run",
        log_to_mlflow=True,
        save_path=save_path,
        random_state=data_cfg["random_state"],
    )

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    pipeline  = result["pipeline"]
    threshold = result["threshold"]
    y_prob    = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_fraud_metrics(
        y_test.values, y_prob,
        threshold=threshold,
        beta=eval_cfg["beta"],
        cost_fp=eval_cfg["cost_fp"],
        cost_fn=eval_cfg["cost_fn"],
    )
    print_fraud_report(y_test.values, y_prob, threshold, model_name=algo)

    # ── 6. Save metrics ───────────────────────────────────────────────────────
    if save_outputs:
        rep_dir = Path(cfg["outputs"]["reports_dir"])
        rep_dir.mkdir(parents=True, exist_ok=True)
        report_path = rep_dir / f"{algo}_metrics.json"
        with open(report_path, "w") as f:
            json.dump(
                {**test_metrics,
                 "cv_pr_auc_mean": result["cv_pr_auc_mean"],
                 "cv_pr_auc_std":  result["cv_pr_auc_std"],
                 "model": algo,
                 "n_features": len(feat_names)},
                f, indent=2,
            )
        logger.info(f"Metrics saved: {report_path}")

    return {
        "pipeline":      pipeline,
        "threshold":     threshold,
        "test_metrics":  test_metrics,
        "feature_names": feat_names,
    }
