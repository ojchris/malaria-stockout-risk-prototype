import numpy as np
import pandas as pd
import pytest

from stockout.data_gen import GeneratorConfig, generate_dataset
from stockout.forecast import (
    FEATURES_NO_RAIN,
    FEATURES_WITH_RAIN,
    HORIZON_WEEKS,
    SPLIT_WEEK,
    bias,
    build_dataset,
    fit_models,
    interval_coverage,
    mae,
    predict,
    rmse,
    score_forecasts,
    split_by_date,
)


@pytest.fixture(scope="module")
def raw():
    return generate_dataset(GeneratorConfig())


@pytest.fixture(scope="module")
def dataset(raw):
    facilities, weekly = raw
    return build_dataset(weekly, facilities)


@pytest.fixture(scope="module")
def fitted(dataset):
    train, test = split_by_date(dataset, FEATURES_WITH_RAIN)
    models = fit_models(train, FEATURES_WITH_RAIN)
    return train, test, models, predict(models, test)


# ---- features and targets -------------------------------------------------


def test_target_is_demand_four_weeks_later(raw, dataset):
    _, weekly = raw
    one = weekly[(weekly["facility_id"] == "FAC-001") & (weekly["commodity"] == "AL")]
    one = one.sort_values("week_start").reset_index(drop=True)
    ds = dataset[(dataset["facility_id"] == "FAC-001") & (dataset["commodity"] == "AL")]
    ds = ds.sort_values("week_idx").reset_index(drop=True)
    t = 20
    assert ds.loc[t, "target_demand"] == one.loc[t + HORIZON_WEEKS, "demand"]
    assert ds.loc[t, "target_week_start"] == one.loc[t + HORIZON_WEEKS, "week_start"]


def test_lag_features_look_backwards(dataset):
    ds = dataset[(dataset["facility_id"] == "FAC-002") & (dataset["commodity"] == "RDT")]
    ds = ds.sort_values("week_idx").reset_index(drop=True)
    t = 30
    assert ds.loc[t, "rate_lag1"] == ds.loc[t - 1, "rate"]
    assert ds.loc[t, "rate_lag3"] == ds.loc[t - 3, "rate"]
    expected_mean4 = ds.loc[t - 3 : t, "rate"].mean()
    assert ds.loc[t, "rate_mean4"] == pytest.approx(expected_mean4)


def test_features_never_use_the_future(raw):
    """Cutting the data off at a date must not change any feature on or before that date."""
    facilities, weekly = raw
    cutoff = pd.Timestamp("2024-06-03")
    full = build_dataset(weekly, facilities)
    cut = build_dataset(weekly[weekly["week_start"] <= cutoff], facilities)

    keys = GROUP_KEYS = ["facility_id", "commodity", "week_start"]
    feature_cols = FEATURES_WITH_RAIN + ["pred_moving_average", "pred_seasonal_naive"]
    a = full[full["week_start"] <= cutoff].set_index(keys)[feature_cols].sort_index()
    b = cut.set_index(keys)[feature_cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


# ---- split ----------------------------------------------------------------


def test_split_is_by_date_with_no_overlap(dataset):
    train, test = split_by_date(dataset, FEATURES_WITH_RAIN)
    assert train["target_week_idx"].max() < SPLIT_WEEK
    assert test["target_week_idx"].min() >= SPLIT_WEEK
    assert len(train) > 0 and len(test) > 0


def test_no_missing_values_in_train_or_test(dataset):
    train, test = split_by_date(dataset, FEATURES_WITH_RAIN)
    cols = FEATURES_WITH_RAIN + ["target_rate", "target_demand"]
    assert not train[cols].isna().any().any()
    assert not test[cols + ["pred_seasonal_naive", "pred_moving_average"]].isna().any().any()


# ---- baselines ------------------------------------------------------------


def test_seasonal_naive_is_demand_from_the_same_week_last_year(raw, dataset):
    _, weekly = raw
    one = weekly[(weekly["facility_id"] == "FAC-003") & (weekly["commodity"] == "AL")]
    one = one.sort_values("week_start").reset_index(drop=True)
    ds = dataset[(dataset["facility_id"] == "FAC-003") & (dataset["commodity"] == "AL")]
    ds = ds.sort_values("week_idx").reset_index(drop=True)
    t = 110  # origin; target week is 114, a year earlier is 62
    assert ds.loc[t, "pred_seasonal_naive"] == one.loc[t + HORIZON_WEEKS - 52, "demand"]


def test_moving_average_is_mean_of_last_four_weeks(raw, dataset):
    _, weekly = raw
    one = weekly[(weekly["facility_id"] == "FAC-004") & (weekly["commodity"] == "AL")]
    one = one.sort_values("week_start").reset_index(drop=True)
    ds = dataset[(dataset["facility_id"] == "FAC-004") & (dataset["commodity"] == "AL")]
    ds = ds.sort_values("week_idx").reset_index(drop=True)
    t = 60
    assert ds.loc[t, "pred_moving_average"] == pytest.approx(one.loc[t - 3 : t, "demand"].mean())


# ---- metrics --------------------------------------------------------------


def test_metrics_on_a_small_example():
    y = np.array([10, 20, 30])
    p = np.array([12, 18, 36])
    assert mae(y, p) == pytest.approx((2 + 2 + 6) / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 36) / 3))
    assert bias(y, p) == pytest.approx((2 - 2 + 6) / 3)


def test_interval_coverage_counts_values_inside():
    y = np.array([1, 5, 9, 12])
    lower = np.array([0, 6, 8, 10])
    upper = np.array([2, 8, 10, 14])
    # inside: 1 in [0,2] yes, 5 in [6,8] no, 9 in [8,10] yes, 12 in [10,14] yes
    assert interval_coverage(y, lower, upper) == pytest.approx(0.75)


def test_score_forecasts_uses_the_same_rows_for_every_method():
    test = pd.DataFrame({"target_demand": [10.0, 20.0, 30.0]})
    scores = score_forecasts(
        test, {"perfect": np.array([10.0, 20.0, 30.0]), "off": np.array([12.0, 18.0, 36.0])}
    )
    assert scores.set_index("method").loc["perfect", "MAE"] == 0
    assert scores.set_index("method").loc["off", "MAE"] == pytest.approx(10 / 3)


# ---- the model itself -----------------------------------------------------


def test_predictions_are_non_negative_and_intervals_are_ordered(fitted):
    _, _, _, out = fitted
    assert (out["pred"] >= 0).all()
    assert (out["lower"] >= 0).all()
    assert (out["lower"] <= out["upper"]).all()


def test_model_beats_both_baselines_on_held_out_weeks(fitted):
    _, test, _, out = fitted
    scores = score_forecasts(
        test,
        {
            "model": out["pred"].to_numpy(),
            "seasonal_naive": test["pred_seasonal_naive"].to_numpy(),
            "moving_average": test["pred_moving_average"].to_numpy(),
        },
    ).set_index("method")
    assert scores.loc["model", "MAE"] < scores.loc["seasonal_naive", "MAE"]
    assert scores.loc["model", "MAE"] < scores.loc["moving_average", "MAE"]


def test_interval_coverage_is_reasonably_close_to_the_target(fitted):
    """An 80% interval should hold roughly 80% of actuals. Allow a wide band and report the rest."""
    _, test, _, out = fitted
    cover = interval_coverage(test["target_demand"], out["lower"], out["upper"])
    assert 0.55 < cover < 0.95


def test_rain_index_is_the_only_difference_between_the_two_feature_sets():
    assert set(FEATURES_WITH_RAIN) - set(FEATURES_NO_RAIN) == {"rain_index"}
