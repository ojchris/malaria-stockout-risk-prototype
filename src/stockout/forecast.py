"""Forecast weekly demand a few weeks ahead, and judge the forecast honestly.

The question: standing at the end of week t, how many treatment courses (or
tests) will a facility need in week t + HORIZON_WEEKS? Four weeks matches the
typical delivery lead time, so it is the horizon an ordering decision needs.

Three forecasters are compared on the same held-out weeks:

* seasonal-naive baseline: "the same week last year" (needs no model)
* moving-average baseline: "the average of the last 4 weeks"
* gradient boosting (LightGBM): learns from recent demand, calendar position,
  facility traits and (optionally) the rain index

Gradient boosting also gives a prediction interval from two extra models that
estimate the 10th and 90th percentiles, so the interval should hold the true
value about 80% of the time. The evaluation checks whether it really does.

Rules this module follows so the results mean something:

* Features at week t use only data up to week t. Nothing from the future leaks in.
* The train/test split is by date: train on earlier weeks, test on later ones.
* Demand is modelled per 1,000 people, so big and small facilities share one model.

All data here is synthetic, so these results show the method works on
simulated data. They say nothing about real-world accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from stockout.data_gen import week_of_year

HORIZON_WEEKS = 4
SPLIT_WEEK = 104  # target weeks before this are train, this week onward are test
GROUP = ["facility_id", "commodity"]

FEATURES_NO_RAIN = [
    "rate_lag0",
    "rate_lag1",
    "rate_lag2",
    "rate_lag3",
    "rate_mean4",
    "rate_mean8",
    "target_woy_sin",
    "target_woy_cos",
    "pop_k",
    "rural",
    "is_rdt",
]
FEATURES_WITH_RAIN = FEATURES_NO_RAIN + ["rain_index"]


def build_dataset(
    weekly: pd.DataFrame, facilities: pd.DataFrame, horizon: int = HORIZON_WEEKS
) -> pd.DataFrame:
    """One row per facility, commodity and origin week, with features, target and baselines.

    Every feature column looks only backwards from the origin week. Columns
    starting with `target_` look `horizon` weeks forward and are answers, not inputs.
    """
    df = weekly.merge(
        facilities[["facility_id", "catchment_population", "setting"]], on="facility_id"
    )
    df = df.sort_values(GROUP + ["week_start"]).reset_index(drop=True)
    grouped = df.groupby(GROUP, sort=False)

    df["week_idx"] = grouped.cumcount()
    df["pop_k"] = df["catchment_population"] / 1000.0
    df["rate"] = df["demand"] / df["pop_k"]  # demand per 1,000 people
    rate = df.groupby(GROUP, sort=False)["rate"]

    # Backward-looking features
    for k in range(4):
        df[f"rate_lag{k}"] = rate.shift(k)
    df["rate_mean4"] = rate.transform(lambda s: s.rolling(4).mean())
    df["rate_mean8"] = rate.transform(lambda s: s.rolling(8).mean())
    df["rural"] = (df["setting"] == "rural").astype(int)
    df["is_rdt"] = (df["commodity"] == "RDT").astype(int)

    # Forward-looking answers
    df["target_rate"] = rate.shift(-horizon)
    df["target_demand"] = df.groupby(GROUP, sort=False)["demand"].shift(-horizon)
    df["target_week_idx"] = df["week_idx"] + horizon
    # The target week's place in the year is pure calendar arithmetic from the origin
    # date, known in advance. It does not read any future row of the table.
    df["target_week_start"] = df["week_start"] + pd.to_timedelta(7 * horizon, unit="D")
    target_woy = week_of_year(pd.DatetimeIndex(df["target_week_start"]))
    df["target_woy_sin"] = np.sin(2 * np.pi * target_woy / 52)
    df["target_woy_cos"] = np.cos(2 * np.pi * target_woy / 52)

    # Baselines, both computed from data up to the origin week
    df["pred_seasonal_naive"] = df.groupby(GROUP, sort=False)["demand"].shift(52 - horizon)
    df["pred_moving_average"] = df["rate_mean4"] * df["pop_k"]
    return df


def split_by_date(
    df: pd.DataFrame, features: list[str], split_week: int = SPLIT_WEEK
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on rows whose target week is before `split_week`, test on the rest.

    Test rows must also have both baselines, so every method is scored on the
    same rows.
    """
    needed = features + ["target_rate", "target_demand"]
    usable = df.dropna(subset=needed)
    train = usable[usable["target_week_idx"] < split_week]
    test = usable[usable["target_week_idx"] >= split_week].dropna(
        subset=["pred_seasonal_naive", "pred_moving_average"]
    )
    return train.copy(), test.copy()


@dataclass
class ForecastModels:
    mean: LGBMRegressor
    lower: LGBMRegressor
    upper: LGBMRegressor
    features: list[str]


def fit_models(
    train: pd.DataFrame,
    features: list[str],
    quantiles: tuple[float, float] = (0.1, 0.9),
    seed: int = 0,
) -> ForecastModels:
    """Fit the mean model and the lower and upper quantile models."""
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    X, y = train[features], train["target_rate"]
    return ForecastModels(
        mean=LGBMRegressor(objective="regression", **params).fit(X, y),
        lower=LGBMRegressor(objective="quantile", alpha=quantiles[0], **params).fit(X, y),
        upper=LGBMRegressor(objective="quantile", alpha=quantiles[1], **params).fit(X, y),
        features=features,
    )


def predict(models: ForecastModels, df: pd.DataFrame) -> pd.DataFrame:
    """Forecast and interval, in demand units (courses or tests per week)."""
    X = df[models.features]
    scale = df["pop_k"].to_numpy()
    pred = np.clip(models.mean.predict(X), 0, None) * scale
    lo = np.clip(models.lower.predict(X), 0, None) * scale
    hi = np.clip(models.upper.predict(X), 0, None) * scale
    return pd.DataFrame(
        {
            "pred": pred,
            "lower": np.minimum(lo, hi),  # quantile models can cross; keep lower <= upper
            "upper": np.maximum(lo, hi),
        },
        index=df.index,
    )


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error: the average miss."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error: like MAE but punishes big misses more."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    """Average of (forecast - actual). Positive means the forecast runs high."""
    return float(np.mean(np.asarray(pred) - np.asarray(y)))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actual values that fall inside [lower, upper]."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def score_forecasts(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per method with error measures on the same test rows."""
    y = test["target_demand"].to_numpy()
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "method": name,
                "MAE": mae(y, p),
                "RMSE": rmse(y, p),
                "bias": bias(y, p),
                "MAE_pct_of_mean_demand": 100 * mae(y, p) / float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows)
