"""
tests/test_features.py
======================
Tests for src/data/features.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    haversine_km,
    engineer_features,
    split_features_target,
    generate_synthetic_raw,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_COL,
)


class TestHaversineKm:
    def test_same_point_zero_distance(self):
        d = haversine_km(
            np.array([40.0]), np.array([-74.0]),
            np.array([40.0]), np.array([-74.0]),
        )
        assert np.isclose(d, 0.0, atol=1e-6)

    def test_nyc_to_london_approx(self):
        """NYC (40.7, -74.0) → London (51.5, -0.1) ≈ 5570 km."""
        d = haversine_km(
            np.array([40.7128]), np.array([-74.0060]),
            np.array([51.5074]), np.array([-0.1278]),
        )
        assert 5400 < d[0] < 5700

    def test_vectorized_output(self):
        lat = np.array([40.0, 35.0, 50.0])
        lon = np.array([-74.0, -118.0, 8.0])
        d = haversine_km(lat, lon, lat + 1.0, lon + 1.0)
        assert d.shape == (3,)
        assert (d > 0).all()

    def test_returns_non_negative(self):
        rng = np.random.default_rng(0)
        lat = rng.uniform(25, 50, 100)
        lon = rng.uniform(-120, -70, 100)
        d = haversine_km(lat, lon, lat + rng.normal(0, 2, 100), lon + rng.normal(0, 2, 100))
        assert (d >= 0).all()


class TestEngineerFeatures:
    def test_returns_dataframe(self, raw_df):
        result = engineer_features(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_numeric_features_present(self, raw_df):
        result = engineer_features(raw_df)
        for feat in NUMERIC_FEATURES:
            assert feat in result.columns, f"Missing: {feat}"

    def test_all_categorical_features_present(self, raw_df):
        result = engineer_features(raw_df)
        for feat in CATEGORICAL_FEATURES:
            assert feat in result.columns, f"Missing: {feat}"

    def test_target_preserved(self, raw_df):
        result = engineer_features(raw_df)
        assert TARGET_COL in result.columns

    def test_row_count_preserved(self, raw_df):
        result = engineer_features(raw_df)
        assert len(result) == len(raw_df)

    def test_log_amt_non_negative(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["log_amt"] >= 0).all()

    def test_log_distance_non_negative(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["log_distance_km"] >= 0).all()

    def test_is_night_binary(self, raw_df):
        result = engineer_features(raw_df)
        assert set(result["is_night"].unique()).issubset({0, 1})

    def test_is_weekend_binary(self, raw_df):
        result = engineer_features(raw_df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_hour_in_range(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["hour"] >= 0).all()
        assert (result["hour"] <= 23).all()

    def test_day_of_week_in_range(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["day_of_week"] >= 0).all()
        assert (result["day_of_week"] <= 6).all()

    def test_age_in_range(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["age"] >= 0).all()
        assert (result["age"] <= 120).all()

    def test_no_nulls_in_numeric_features(self, raw_df):
        result = engineer_features(raw_df)
        for feat in NUMERIC_FEATURES:
            assert result[feat].notna().all(), f"NaN found in {feat}"

    def test_log_city_pop_positive(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["log_city_pop"] > 0).all()

    def test_amt_to_pop_ratio_non_negative(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["amt_to_pop_ratio"] >= 0).all()

    def test_fraud_transactions_higher_avg_distance(self):
        """Fraud synthetic data is generated with larger geo offsets."""
        df_raw = generate_synthetic_raw(n_samples=2000, fraud_ratio=0.1, random_state=0)
        df_feat = engineer_features(df_raw)
        fraud_dist   = df_feat[df_feat[TARGET_COL] == 1]["log_distance_km"].mean()
        legit_dist   = df_feat[df_feat[TARGET_COL] == 0]["log_distance_km"].mean()
        assert fraud_dist > legit_dist  # Fraud should be farther on average

    def test_fraud_transactions_higher_avg_amount(self):
        """Fraud synthetic data is generated with higher amounts."""
        df_raw = generate_synthetic_raw(n_samples=2000, fraud_ratio=0.1, random_state=0)
        df_feat = engineer_features(df_raw)
        fraud_amt = df_feat[df_feat[TARGET_COL] == 1]["log_amt"].mean()
        legit_amt = df_feat[df_feat[TARGET_COL] == 0]["log_amt"].mean()
        assert fraud_amt > legit_amt


class TestSplitFeaturesTarget:
    def test_returns_dataframe_and_series(self, engineered_df):
        X, y = split_features_target(engineered_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, engineered_df):
        X, _ = split_features_target(engineered_df)
        assert TARGET_COL not in X.columns

    def test_y_is_binary(self, engineered_df):
        _, y = split_features_target(engineered_df)
        assert set(y.unique()).issubset({0, 1})

    def test_lengths_match(self, engineered_df):
        X, y = split_features_target(engineered_df)
        assert len(X) == len(y)

    def test_all_feature_names_present(self, engineered_df):
        X, _ = split_features_target(engineered_df)
        for feat in FEATURE_NAMES:
            assert feat in X.columns, f"Missing: {feat}"


class TestGenerateSyntheticRaw:
    def test_returns_dataframe(self):
        df = generate_synthetic_raw(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_synthetic_raw(n_samples=200, fraud_ratio=0.05)
        assert len(df) == 200

    def test_fraud_column_present(self):
        df = generate_synthetic_raw(n_samples=100)
        assert TARGET_COL in df.columns

    def test_fraud_ratio_approximate(self):
        df = generate_synthetic_raw(n_samples=2000, fraud_ratio=0.05)
        ratio = df[TARGET_COL].mean()
        assert 0.02 < ratio < 0.10  # allow generous tolerance

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_raw(n_samples=100, random_state=0)
        df2 = generate_synthetic_raw(n_samples=100, random_state=0)
        assert df1["amt"].equals(df2["amt"])
