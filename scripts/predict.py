"""
scripts/predict.py
==================
Score new transactions and assign fraud risk tiers.

Usage
-----
    python scripts/predict.py \
        --model-path outputs/models/hist_gradient_boosting_model.joblib

    python scripts/predict.py \
        --model-path outputs/models/xgboost_model.joblib \
        --input data/new_transactions.csv \
        --output outputs/reports/scored_transactions.csv \
        --threshold 0.4 \
        --top-n 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_train_test, load_single
from src.data.features import engineer_features, split_features_target
from src.pipeline import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transactions for fraud")
    p.add_argument("--model-path", required=True, help="Saved model .joblib path")
    p.add_argument("--input", type=str, default=None,
                   help="Input CSV (default: test set from config)")
    p.add_argument("--output", type=str,
                   default="outputs/reports/scored_transactions.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override saved threshold")
    p.add_argument("--top-n", type=int, default=None,
                   help="Print top N highest-risk transactions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}"); sys.exit(1)

    artifact  = joblib.load(model_path)
    pipeline  = artifact["pipeline"]
    threshold = args.threshold if args.threshold is not None else artifact["threshold"]
    logger.info(f"Loaded: {model_path.name} | threshold={threshold:.3f}")

    # Load data
    if args.input:
        df_raw = load_single(args.input)
    else:
        _, df_raw = load_train_test(
            cfg["data"]["train_path"],
            cfg["data"]["test_path"],
        )
        logger.info(f"Using test set: {len(df_raw):,} transactions")

    df = engineer_features(df_raw)
    feat_names = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    avail = [f for f in feat_names if f in df.columns]
    X = df[avail]

    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    out = df_raw.copy()
    out["fraud_probability"] = y_prob.round(4)
    out["fraud_prediction"]  = y_pred
    out["risk_tier"] = pd.cut(
        y_prob,
        bins=[0, 0.2, 0.5, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    logger.info(
        f"\n📊 Scoring Summary:"
        f"\n   Total transactions   : {len(out):,}"
        f"\n   Flagged as fraud     : {y_pred.sum():,} ({y_pred.mean():.4%})"
        f"\n   High risk (>0.5)     : {(y_prob > 0.5).sum():,}"
        f"\n   Medium risk (0.2-0.5): {((y_prob >= 0.2) & (y_prob <= 0.5)).sum():,}"
    )

    if args.top_n:
        top = out.nlargest(args.top_n, "fraud_probability")[[
            "fraud_probability", "fraud_prediction", "risk_tier",
            *[c for c in ["amt", "category", "state", "is_fraud"] if c in out.columns],
        ]]
        print(f"\nTop {args.top_n} Highest-Risk Transactions:")
        print(top.to_string())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"\nScored transactions saved: {out_path}")


if __name__ == "__main__":
    main()
