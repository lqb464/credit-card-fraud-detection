"""
tests/test_trainers.py
======================
Tests for src/models/trainers.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.trainers import (
    _build_classifier,
    build_sklearn_pipeline,
    optimize_threshold_f2,
    optimize_threshold_cost,
)
from src.data.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def X_y_small():
    """Small balanced X, y for fast model tests."""
    from src.data.features import generate_synthetic_raw, engineer_features, split_features_target
    df = generate_synthetic_raw(n_samples=600, fraud_ratio=0.15, random_state=7)
    df = engineer_features(df)
    return split_features_target(df)


class TestBuildClassifier:
    @pytest.mark.parametrize("name", [
        "hist_gradient_boosting", "logistic_regression",
    ])
    def test_builds_valid_classifier(self, name):
        clf = _build_classifier(name, {"random_state": 42})
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict_proba")

    def test_xgboost_uses_scale_pos_weight(self):
        pytest.importorskip("xgboost")
        clf = _build_classifier("xgboost", {}, scale_pos_weight=100.0)
        assert clf.scale_pos_weight == 100.0

    def test_lightgbm_is_unbalance(self):
        pytest.importorskip("lightgbm")
        clf = _build_classifier("lightgbm", {})
        assert clf.is_unbalance is True

    def test_raises_on_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            _build_classifier("mystery_algo", {})

    def test_logistic_balanced_class_weight(self):
        clf = _build_classifier("logistic_regression", {"class_weight": "balanced"})
        assert clf.class_weight == "balanced"


class TestBuildSklearnPipeline:
    def test_returns_pipeline(self):
        pipe = build_sklearn_pipeline(
            "logistic_regression", {},
            NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        )
        assert isinstance(pipe, Pipeline)

    def test_has_preprocessor_and_classifier(self):
        pipe = build_sklearn_pipeline(
            "hist_gradient_boosting", {},
            NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        )
        assert "preprocessor" in pipe.named_steps
        assert "classifier"   in pipe.named_steps

    def test_fits_and_predicts(self, X_y_small):
        X, y = X_y_small
        pipe = build_sklearn_pipeline(
            "logistic_regression", {"max_iter": 200},
            NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        )
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_probabilities_sum_to_one(self, X_y_small):
        X, y = X_y_small
        pipe = build_sklearn_pipeline(
            "hist_gradient_boosting",
            {"max_iter": 30},
            NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        )
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestOptimizeThresholdF2:
    @pytest.fixture
    def y_true_prob(self):
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 950 + [1] * 50)
        y_prob = np.where(
            y_true == 1,
            rng.beta(7, 2, 1000),
            rng.beta(2, 7, 1000),
        )
        return y_true, y_prob

    def test_returns_float_in_range(self, y_true_prob):
        y_true, y_prob = y_true_prob
        t = optimize_threshold_f2(y_true, y_prob)
        assert 0.0 <= t <= 1.0

    def test_better_f2_than_default(self, y_true_prob):
        from sklearn.metrics import fbeta_score
        y_true, y_prob = y_true_prob
        t_opt = optimize_threshold_f2(y_true, y_prob, beta=2.0)
        f2_opt = fbeta_score(y_true, (y_prob >= t_opt).astype(int), beta=2.0, zero_division=0)
        f2_default = fbeta_score(y_true, (y_prob >= 0.5).astype(int), beta=2.0, zero_division=0)
        assert f2_opt >= f2_default - 0.01  # tolerance

    def test_beta_1_gives_f1_threshold(self, y_true_prob):
        y_true, y_prob = y_true_prob
        t = optimize_threshold_f2(y_true, y_prob, beta=1.0)
        assert 0.0 <= t <= 1.0


class TestOptimizeThresholdCost:
    @pytest.fixture
    def y_true_prob(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 950 + [1] * 50)
        y_prob = np.where(
            y_true == 1,
            rng.beta(6, 2, 1000),
            rng.beta(2, 6, 1000),
        )
        return y_true, y_prob

    def test_returns_float_in_range(self, y_true_prob):
        y_true, y_prob = y_true_prob
        t = optimize_threshold_cost(y_true, y_prob)
        assert 0.0 <= t <= 1.0

    def test_high_fn_cost_pushes_threshold_down(self, y_true_prob):
        """High cost of missing fraud → lower threshold (catch more fraud)."""
        y_true, y_prob = y_true_prob
        t_low_fn  = optimize_threshold_cost(y_true, y_prob, cost_fp=10, cost_fn=50)
        t_high_fn = optimize_threshold_cost(y_true, y_prob, cost_fp=10, cost_fn=5000)
        # Very high FN cost should push threshold lower (be more aggressive)
        assert t_high_fn <= t_low_fn + 0.1  # allow small tolerance
