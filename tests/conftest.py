"""
tests/conftest.py
=================
Shared fixtures for the fraud detection test suite.
Generates synthetic Sparkov-schema DataFrames in memory.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import generate_synthetic_raw, engineer_features, split_features_target


def make_fraud_df(n: int = 500, fraud_ratio: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Synthetic raw Sparkov DataFrame."""
    return generate_synthetic_raw(n_samples=n, fraud_ratio=fraud_ratio, random_state=seed)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small raw Sparkov-schema DataFrame (500 rows, 5% fraud)."""
    return make_fraud_df(n=500, fraud_ratio=0.05)


@pytest.fixture
def engineered_df(raw_df) -> pd.DataFrame:
    """Feature-engineered DataFrame from raw_df."""
    return engineer_features(raw_df)


@pytest.fixture
def X_y(engineered_df):
    """Feature matrix + target from engineered_df."""
    return split_features_target(engineered_df)


@pytest.fixture
def large_raw_df() -> pd.DataFrame:
    """Larger raw DataFrame for integration tests."""
    return make_fraud_df(n=1500, fraud_ratio=0.03, seed=99)
