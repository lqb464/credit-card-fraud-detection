"""
scripts/train.py
================
CLI for training a fraud detection model.

Usage
-----
    python scripts/train.py                          # Default (HistGradientBoosting)
    python scripts/train.py --model xgboost
    python scripts/train.py --all                    # All 4 models
    python scripts/train.py --model lightgbm --smoke-test
    python scripts/train.py --model xgboost --threshold cost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_MODELS = [
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "logistic_regression",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a credit card fraud detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", "-m", choices=ALL_MODELS, default=None,
                   help="Model to train (default: from config.yaml)")
    p.add_argument("--all", action="store_true", help="Train all 4 models")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick run on 10K sample")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving model artifacts")
    return p.parse_args()


def train_one(args: argparse.Namespace, model_name: str) -> None:
    if model_name is None:
        cfg = run_pipeline.__globals__.get("load_config", lambda: {"models": {"default_model": "hist_gradient_boosting"}})()
        model_name = cfg["models"]["default_model"]

    logger.info("=" * 65)
    logger.info(f"  Training: {model_name.upper()}")
    logger.info("=" * 65)

    result = run_pipeline(
        config_path=args.config,
        model_name=model_name,
        smoke_test=args.smoke_test,
        save_outputs=not args.no_save,
        run_name=args.run_name,
    )
    m = result["test_metrics"]
    logger.info(
        f"\n📊 Results [{model_name}]:\n"
        f"   PR-AUC          : {m['pr_auc']:.4f}\n"
        f"   ROC-AUC         : {m['roc_auc']:.4f}\n"
        f"   F2-Score        : {m['f2_score']:.4f}\n"
        f"   Fraud Recall    : {m['fraud_recall']:.4f}\n"
        f"   Fraud Precision : {m['fraud_precision']:.4f}\n"
        f"   Threshold       : {m['threshold']:.3f}\n"
        f"   Total Cost      : ${m['total_cost_usd']:,.0f}"
    )


def main() -> None:
    args = parse_args()
    models = ALL_MODELS if args.all else [args.model]
    for m in models:
        try:
            train_one(args, m)
        except Exception as e:
            logger.error(f"Failed [{m}]: {e}", exc_info=True)

    logger.info("\n✅ Training complete!")
    logger.info("   MLflow UI: mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db")


if __name__ == "__main__":
    main()
