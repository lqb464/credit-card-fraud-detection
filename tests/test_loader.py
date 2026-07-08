"""
tests/test_loader.py
====================
Tests for src/data/loader.py
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.loader import _validate, _stratified_sample, TARGET_COL, REQUIRED_RAW_FIELDS
from tests.conftest import make_fraud_df


class TestValidate:
    def test_passes_on_valid_df(self):
        df = make_fraud_df(n=50)
        _validate(df, "test")  # Should not raise

    def test_raises_on_missing_required_columns(self):
        df = make_fraud_df(n=50).drop(columns=["amt", "category"])
        with pytest.raises(ValueError, match="Missing columns"):
            _validate(df, "test")

    def test_all_required_fields_in_fixture(self):
        df = make_fraud_df(n=20)
        missing = set(REQUIRED_RAW_FIELDS) - set(df.columns)
        assert not missing, f"Fixture missing: {missing}"


class TestStratifiedSample:
    def test_returns_requested_size(self):
        df = make_fraud_df(n=1000, fraud_ratio=0.05)
        result = _stratified_sample(df, n=200, random_state=42)
        # Allow ±5% tolerance
        assert abs(len(result) - 200) <= 10

    def test_preserves_fraud_ratio_approximately(self):
        df = make_fraud_df(n=2000, fraud_ratio=0.05)
        original_rate = df[TARGET_COL].mean()
        result = _stratified_sample(df, n=400, random_state=42)
        sampled_rate = result[TARGET_COL].mean()
        # Should be within ±3% of original
        assert abs(sampled_rate - original_rate) < 0.03

    def test_resets_index(self):
        df = make_fraud_df(n=500)
        result = _stratified_sample(df, n=100, random_state=42)
        assert list(result.index) == list(range(len(result)))

    def test_no_sample_when_smaller_than_n(self):
        df = make_fraud_df(n=100)
        result = _stratified_sample(df, n=5000, random_state=42)
        # Can't get more than available
        assert len(result) <= len(df)

    def test_both_classes_present_in_sample(self):
        df = make_fraud_df(n=1000, fraud_ratio=0.05)
        result = _stratified_sample(df, n=100, random_state=42)
        assert set(result[TARGET_COL].unique()) == {0, 1}
