"""
src/data/features.py
====================
Feature engineering for the Sparkov credit card fraud dataset.

Transforms raw Sparkov fields into a structured, model-ready feature set.

Engineered Features
-------------------
Numeric:
    log_amt             : log1p(transaction amount) — handles heavy right skew
    log_distance_km     : log1p(Haversine distance customer ↔ merchant)
    city_pop            : city population (as-is)
    log_city_pop        : log1p(city_pop)
    age                 : customer age in years at transaction time
    hour                : transaction hour (0-23)
    day_of_week         : 0=Monday … 6=Sunday
    is_night            : 1 if hour in [23, 0-5]
    is_weekend          : 1 if day_of_week in {5, 6}
    amt_to_pop_ratio    : amt / city_pop — large spend in small city is suspicious

Categorical:
    category            : merchant category (grocery_pos, etc.)
    gender              : M / F
    state               : US state code
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

TARGET_COL: str = "is_fraud"

NUMERIC_FEATURES: list[str] = [
    "log_amt",
    "log_distance_km",
    "city_pop",
    "log_city_pop",
    "age",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    "amt_to_pop_ratio",
]

CATEGORICAL_FEATURES: list[str] = ["category", "gender", "state"]

FEATURE_NAMES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Haversine distance in kilometres.

    Parameters
    ----------
    lat1, lon1 : customer coordinates
    lat2, lon2 : merchant coordinates

    Returns
    -------
    np.ndarray of distances in km.
    """
    R = 6_371.0  # Earth radius km
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a  = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── Main feature engineering function ────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Sparkov dataframe into engineered feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from loader (contains Sparkov columns).

    Returns
    -------
    pd.DataFrame
        Feature matrix with FEATURE_NAMES + optionally TARGET_COL.
    """
    out = pd.DataFrame(index=df.index)

    # ── Amount ────────────────────────────────────────────────────────────────
    amt      = pd.to_numeric(df.get("amt",      pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    city_pop = pd.to_numeric(df.get("city_pop", pd.Series(10_000, index=df.index)), errors="coerce").fillna(10_000)

    out["log_amt"]          = np.log1p(np.maximum(0.0, amt.values))
    out["city_pop"]         = city_pop.values
    out["log_city_pop"]     = np.log1p(np.maximum(1.0, city_pop.values))
    out["amt_to_pop_ratio"] = (amt / (city_pop + 1.0)).values

    # ── Geospatial ────────────────────────────────────────────────────────────
    lat      = _safe_numeric(df, "lat",        default=37.0)
    lon      = _safe_numeric(df, "long",       default=-96.0)
    merch_lat= _safe_numeric(df, "merch_lat",  default=37.0)
    merch_lon= _safe_numeric(df, "merch_long", default=-96.0)

    dist                  = haversine_km(lat, lon, merch_lat, merch_lon)
    out["log_distance_km"] = np.log1p(np.maximum(0.0, dist))

    # ── Datetime ──────────────────────────────────────────────────────────────
    if "trans_date_trans_time" in df.columns:
        ts          = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        hour        = ts.dt.hour.fillna(12).astype(int).values
        day_of_week = ts.dt.dayofweek.fillna(0).astype(int).values
    else:
        hour        = _safe_int(df, "hour",        default=12)
        day_of_week = _safe_int(df, "day_of_week", default=0)

    out["hour"]       = hour
    out["day_of_week"]= day_of_week
    out["is_night"]   = ((hour >= 23) | (hour <= 5)).astype(int)
    out["is_weekend"] = np.isin(day_of_week, [5, 6]).astype(int)

    # ── Age ───────────────────────────────────────────────────────────────────
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        dob = pd.to_datetime(df["dob"], errors="coerce")
        ts  = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        age = ((ts - dob).dt.days / 365.25).fillna(35.0)
        out["age"] = age.values.clip(0, 120)
    else:
        out["age"] = _safe_numeric(df, "age", default=35.0).clip(0, 120)

    # ── Categoricals ──────────────────────────────────────────────────────────
    out["category"] = df["category"].astype(str).fillna("misc_net").values if "category" in df.columns else "misc_net"
    out["gender"]   = df["gender"].astype(str).fillna("F").values           if "gender"   in df.columns else "F"
    out["state"]    = df["state"].astype(str).fillna("CA").values            if "state"    in df.columns else "CA"

    # ── Target (preserve if present) ─────────────────────────────────────────
    if TARGET_COL in df.columns:
        out[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int).values

    logger.debug(f"Feature engineering complete: {out.shape}")
    return out


def split_features_target(
    df: pd.DataFrame,
    feature_names: list[str] = FEATURE_NAMES,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split engineered DataFrame into X and y."""
    available = [f for f in feature_names if f in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].astype(int)
    logger.info(
        f"Feature matrix: {X.shape} | "
        f"Fraud: {int(y.sum()):,} ({y.mean():.4%})"
    )
    return X, y


# ── Synthetic data generator (for tests) ─────────────────────────────────────

def generate_synthetic_raw(
    n_samples: int = 2_000,
    fraud_ratio: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic raw Sparkov-schema DataFrame for testing.

    Fraud transactions have higher amounts and larger geographic distances.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(2, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    categories = ["grocery_pos", "entertainment", "gas_transport", "misc_net", "shopping_net"]
    states     = ["CA", "NY", "TX", "FL", "IL"]
    genders    = ["F", "M"]

    amts = np.concatenate([
        rng.exponential(scale=50.0,  size=n_legit),
        rng.exponential(scale=300.0, size=n_fraud),
    ])

    base_lat = rng.uniform(30.0, 45.0, n_samples)
    base_lon = rng.uniform(-120.0, -70.0, n_samples)
    # Fraud → merchant far from customer
    offset_lat = np.concatenate([rng.normal(0, 0.05, n_legit), rng.normal(5.0, 2.0, n_fraud)])
    offset_lon = np.concatenate([rng.normal(0, 0.05, n_legit), rng.normal(5.0, 2.0, n_fraud)])

    ts = pd.date_range("2019-01-01", periods=n_samples, freq="min")

    df = pd.DataFrame({
        "amt":                  amts,
        "category":             rng.choice(categories, n_samples),
        "gender":               rng.choice(genders, n_samples),
        "state":                rng.choice(states, n_samples),
        "city_pop":             rng.integers(1_000, 500_000, n_samples),
        "lat":                  base_lat,
        "long":                 base_lon,
        "merch_lat":            base_lat + offset_lat,
        "merch_long":           base_lon + offset_lon,
        "trans_date_trans_time":ts.strftime("%Y-%m-%d %H:%M:%S"),
        "dob":                  "1985-05-15",
        TARGET_COL:             np.array([0] * n_legit + [1] * n_fraud),
    })
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_numeric(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).values
    return np.full(len(df), default, dtype=float)


def _safe_int(df: pd.DataFrame, col: str, default: int) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int).values
    return np.full(len(df), default, dtype=int)
