"""
src/data/loader.py
==================
Load and validate the Sparkov credit card fraud dataset.

Dataset schema (Sparkov synthetic fraud dataset):
    trans_date_trans_time, cc_num, merchant, category, amt,
    first, last, gender, street, city, state, zip,
    lat, long, city_pop, job, dob, trans_num,
    unix_time, merch_lat, merch_long, is_fraud
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL: str = "is_fraud"

# Raw fields we actually need (avoid loading all 23 columns)
REQUIRED_RAW_FIELDS: list[str] = [
    "amt", "category", "gender", "state", "city_pop",
    "lat", "long", "merch_lat", "merch_long",
    "trans_date_trans_time", "dob", "is_fraud",
]

# Optional — used for transaction velocity if present
OPTIONAL_RAW_FIELDS: list[str] = ["cc_num", "unix_time"]


def load_train_test(
    train_path: str | Path,
    test_path:  str | Path,
    train_sample_size: Optional[int] = 200_000,
    smoke_test: bool = False,
    smoke_train: int = 10_000,
    smoke_test_n: int = 3_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw Sparkov train and test CSVs with optional stratified subsampling.

    Parameters
    ----------
    train_path, test_path : str | Path
    train_sample_size : int, optional
        Subsample training set size (stratified). None = full dataset (~1.3M).
    smoke_test : bool
        Fast iteration mode — small sample.
    smoke_train, smoke_test_n : int
        Sample sizes in smoke mode.
    random_state : int

    Returns
    -------
    (df_train_raw, df_test_raw) : raw DataFrames ready for feature engineering.

    Raises
    ------
    FileNotFoundError  if CSV files are missing.
    ValueError         if required columns are absent.
    """
    train_path = Path(train_path)
    test_path  = Path(test_path)

    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    if smoke_test:
        train_sample_size = smoke_train
        test_n = smoke_test_n
    else:
        test_n = None  # full test set

    logger.info(f"Loading train: {train_path.name}")
    df_train = _load_csv(train_path, nrows=None)
    logger.info(f"Loading test:  {test_path.name}")
    df_test  = _load_csv(test_path,  nrows=test_n)

    _validate(df_train, "train")
    _validate(df_test,  "test")

    if train_sample_size and len(df_train) > train_sample_size:
        df_train = _stratified_sample(df_train, train_sample_size, random_state)
        logger.info(f"Subsampled train → {len(df_train):,} rows (stratified)")

    _log_summary(df_train, df_test)
    return df_train, df_test


def load_single(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a single CSV for inference (no target required)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return _load_csv(path, nrows=nrows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    """Read CSV, keeping only necessary columns to reduce memory."""
    # Peek to find available columns
    available = pd.read_csv(path, nrows=1).columns.tolist()
    use_cols  = [c for c in REQUIRED_RAW_FIELDS + OPTIONAL_RAW_FIELDS if c in available]
    df = pd.read_csv(path, usecols=use_cols, nrows=nrows, low_memory=False)
    return df.reset_index(drop=True)


def _validate(df: pd.DataFrame, split: str) -> None:
    """Check that all required raw fields are present."""
    missing = set(REQUIRED_RAW_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"[{split}] Missing columns: {missing}")
    logger.info(f"[{split}] Schema OK — {len(df):,} rows × {df.shape[1]} cols")


def _stratified_sample(
    df: pd.DataFrame, n: int, random_state: int
) -> pd.DataFrame:
    """Stratified subsample preserving fraud/legit ratio."""
    frac = min(1.0, n / len(df))
    return (
        df.groupby(TARGET_COL, group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )


def _log_summary(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    fraud_train = df_train[TARGET_COL].mean() if TARGET_COL in df_train else float("nan")
    fraud_test  = df_test[TARGET_COL].mean()  if TARGET_COL in df_test  else float("nan")
    logger.info(f"Train: {len(df_train):,} rows | Fraud rate: {fraud_train:.4%}")
    logger.info(f"Test:  {len(df_test):,} rows  | Fraud rate: {fraud_test:.4%}")
    n_fraud = int(df_train[TARGET_COL].sum()) if TARGET_COL in df_train else 0
    logger.info(f"Train fraud count: {n_fraud:,} | Imbalance ratio ≈ 1:{int(1/fraud_train)}")
