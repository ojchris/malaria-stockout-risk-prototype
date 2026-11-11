import numpy as np
import pandas as pd
import pytest

from stockout.data_gen import (
    SIZE_CLASSES,
    GeneratorConfig,
    generate_dataset,
    seasonal_factor,
    summarise,
)


@pytest.fixture(scope="module")
def data():
    return generate_dataset(GeneratorConfig())


def test_same_seed_gives_identical_data():
    cfg = GeneratorConfig(n_facilities=6, n_weeks=60, seed=7)
    fac_a, weekly_a = generate_dataset(cfg)
    fac_b, weekly_b = generate_dataset(cfg)
    pd.testing.assert_frame_equal(fac_a, fac_b)
    pd.testing.assert_frame_equal(weekly_a, weekly_b)


def test_different_seed_gives_different_data():
    _, a = generate_dataset(GeneratorConfig(n_facilities=6, n_weeks=60, seed=1))
    _, b = generate_dataset(GeneratorConfig(n_facilities=6, n_weeks=60, seed=2))
    assert not a["demand"].equals(b["demand"])


def test_table_shapes(data):
    facilities, weekly = data
    cfg = GeneratorConfig()
    assert len(facilities) == cfg.n_facilities
    assert len(weekly) == cfg.n_facilities * len(cfg.commodities) * cfg.n_weeks
    assert facilities["facility_id"].is_unique


def test_facility_populations_fit_their_size_class(data):
    facilities, _ = data
    for _, row in facilities.iterrows():
        low, high = SIZE_CLASSES[row["size_class"]][0]
        assert low <= row["catchment_population"] < high


def test_stock_is_conserved_every_week(data):
    """closing stock = opening + received - expired - consumed, with no stock appearing or vanishing."""
    _, weekly = data
    expected = (
        weekly["opening_stock"] + weekly["received"] - weekly["expired"] - weekly["consumption"]
    )
    assert (weekly["stock_end"] == expected).all()


def test_opening_stock_is_previous_closing_stock(data):
    _, weekly = data
    weekly = weekly.sort_values(["facility_id", "commodity", "week_start"])
    prev_close = weekly.groupby(["facility_id", "commodity"])["stock_end"].shift(1)
    later = prev_close.notna()
    assert (weekly.loc[later, "opening_stock"] == prev_close[later]).all()


def test_no_negative_quantities(data):
    _, weekly = data
    for col in ["demand", "consumption", "unmet_demand", "received", "expired", "stock_end"]:
        assert (weekly[col] >= 0).all(), col


def test_consumption_never_exceeds_demand_or_available_stock(data):
    _, weekly = data
    assert (weekly["consumption"] <= weekly["demand"]).all()
    assert (weekly["unmet_demand"] == weekly["demand"] - weekly["consumption"]).all()
    available = weekly["opening_stock"] + weekly["received"] - weekly["expired"]
    assert (weekly["consumption"] <= available).all()


def test_stockout_flag_matches_unmet_demand(data):
    _, weekly = data
    assert (weekly["stockout"] == (weekly["unmet_demand"] > 0)).all()


def test_seasonal_factor_averages_one_and_peaks_where_told():
    weeks = np.arange(1, 53)
    factor = seasonal_factor(weeks, peak_week=29, strength=0.7)
    assert factor.mean() == pytest.approx(1.0)
    assert weeks[factor.argmax()] == 29
    assert factor.max() > 2 * factor.min()


def test_demand_peaks_after_rain_with_the_set_lag(data):
    """Average demand across facilities peaks case_lag_weeks after the rain peak."""
    _, weekly = data
    cfg = GeneratorConfig()
    by_week = weekly[weekly["commodity"] == "AL"].groupby("week_of_year")["demand"].mean()
    # smooth over 3 weeks so Poisson noise does not move the peak
    smooth = by_week.rolling(3, center=True, min_periods=1).mean()
    peak = int(smooth.idxmax())
    assert abs(peak - (cfg.rain_peak_week + cfg.case_lag_weeks)) <= 2


def test_wet_season_demand_is_much_higher_than_dry_season(data):
    _, weekly = data
    al = weekly[weekly["commodity"] == "AL"]
    wet = al.loc[al["week_of_year"].between(30, 36), "demand"].mean()
    dry = al.loc[al["week_of_year"].between(2, 8), "demand"].mean()
    assert wet > 2 * dry


def test_stockouts_happen_but_are_not_the_norm(data):
    _, weekly = data
    share = summarise(weekly)["stockout_week_share"]
    assert 0.03 < share < 0.35


def test_stockouts_are_worse_in_the_ramp_up_than_the_dry_season(data):
    _, weekly = data
    by_week = weekly.groupby("week_of_year")["stockout"].mean()
    assert by_week.loc[18:30].mean() > 3 * by_week.loc[1:8].mean()


def test_some_stock_expires_but_only_a_small_share(data):
    _, weekly = data
    share = summarise(weekly)["expired_share_of_received"]
    assert 0 < share < 0.05


def test_rural_facilities_have_more_stockouts_than_urban(data):
    facilities, weekly = data
    merged = weekly.merge(facilities[["facility_id", "setting"]], on="facility_id")
    rates = merged.groupby("setting")["stockout"].mean()
    assert rates["rural"] > rates["urban"]
