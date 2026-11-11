"""Score the forecasters on held-out weeks, print the results and draw a figure.

Usage:  python scripts/run_forecast_eval.py

Writes docs/results/phase2_metrics.csv and docs/figures/forecast_eval.png.
All data is synthetic, so the numbers show the method works on simulated data,
not how it would perform on real facilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockout.data_gen import GeneratorConfig, generate_dataset
from stockout.forecast import (
    FEATURES_NO_RAIN,
    FEATURES_WITH_RAIN,
    HORIZON_WEEKS,
    SPLIT_WEEK,
    build_dataset,
    fit_models,
    interval_coverage,
    predict,
    score_forecasts,
    split_by_date,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    facilities, weekly = generate_dataset(GeneratorConfig())
    df = build_dataset(weekly, facilities)

    # Same test rows for every method (rain features are a superset of the no-rain ones)
    train, test = split_by_date(df, FEATURES_WITH_RAIN)
    print(f"Horizon: {HORIZON_WEEKS} weeks ahead. Split at week {SPLIT_WEEK}.")
    print(f"Train rows: {len(train):,}  Test rows: {len(test):,}")
    print(
        f"Train targets: {train['target_week_start'].min().date()} to {train['target_week_start'].max().date()}"
    )
    print(
        f"Test targets:  {test['target_week_start'].min().date()} to {test['target_week_start'].max().date()}\n"
    )

    with_rain = predict(fit_models(train, FEATURES_WITH_RAIN), test)
    no_rain = predict(fit_models(train, FEATURES_NO_RAIN), test)

    preds = {
        "Seasonal naive (same week last year)": test["pred_seasonal_naive"].to_numpy(),
        "Moving average (last 4 weeks)": test["pred_moving_average"].to_numpy(),
        "LightGBM without rain index": no_rain["pred"].to_numpy(),
        "LightGBM with rain index": with_rain["pred"].to_numpy(),
    }
    scores = score_forecasts(test, preds)

    y = test["target_demand"].to_numpy()
    print("Error on held-out weeks (units: courses or tests per facility-week)")
    print(scores.round(2).to_string(index=False))

    # Where do the methods differ? The rising season is the hard part.
    woy = test["target_week_start"].dt.dayofyear.to_numpy() // 7 + 1
    rising = (woy >= 18) & (woy <= 33)
    rows = []
    for name, p in preds.items():
        err = np.abs(y - p)
        rows.append(
            {
                "method": name,
                "MAE_rising_season_wk18_33": err[rising].mean(),
                "MAE_other_weeks": err[~rising].mean(),
            }
        )
    by_season = pd.DataFrame(rows)
    print("\nMean absolute error by season of the target week")
    print(by_season.round(2).to_string(index=False))

    # Prediction intervals
    interval_rows = []
    for label, out in [("LightGBM without rain index", no_rain), ("LightGBM with rain index", with_rain)]:
        for scope, mask in [("all", np.ones(len(test), bool))] + [
            (c, (test["commodity"] == c).to_numpy()) for c in ["AL", "RDT"]
        ]:
            interval_rows.append(
                {
                    "method": label,
                    "scope": scope,
                    "target_coverage": 0.80,
                    "actual_coverage": interval_coverage(
                        y[mask], out["lower"].to_numpy()[mask], out["upper"].to_numpy()[mask]
                    ),
                    "mean_interval_width": float(
                        (out["upper"] - out["lower"]).to_numpy()[mask].mean()
                    ),
                }
            )
    intervals = pd.DataFrame(interval_rows)
    print("\nPrediction intervals (an 80% interval should hold about 80% of actual values)")
    print(intervals.round(3).to_string(index=False))

    results_dir = ROOT / "docs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores.merge(by_season, on="method").to_csv(results_dir / "phase2_metrics.csv", index=False)
    intervals.to_csv(results_dir / "phase2_intervals.csv", index=False)

    # Figure
    test = test.assign(
        model=with_rain["pred"], lower=with_rain["lower"], upper=with_rain["upper"]
    )
    al = test[test["commodity"] == "AL"]
    total = al.groupby("target_week_start")[
        ["target_demand", "model", "pred_seasonal_naive", "pred_moving_average"]
    ].sum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    ax = axes[0]
    ax.plot(total.index, total["target_demand"], color="black", linewidth=2, label="Actual")
    ax.plot(total.index, total["model"], color="#1f4e79", label="LightGBM")
    ax.plot(total.index, total["pred_moving_average"], color="#c0504d", linestyle="--", label="Moving average")
    ax.plot(total.index, total["pred_seasonal_naive"], color="#2e8b57", linestyle=":", label="Seasonal naive")
    ax.set_title(f"AL demand across all facilities, {HORIZON_WEEKS} weeks ahead (held-out year, synthetic data)")
    ax.set_ylabel("Treatment courses / week")
    ax.legend(loc="upper left", ncol=2)

    # One mid-sized facility with its 80% interval
    fid = facilities.sort_values("catchment_population").iloc[len(facilities) // 2]["facility_id"]
    one = al[al["facility_id"] == fid].sort_values("target_week_start")
    ax = axes[1]
    ax.fill_between(
        one["target_week_start"], one["lower"], one["upper"], color="#7aa6c2", alpha=0.45, label="80% interval"
    )
    ax.plot(one["target_week_start"], one["model"], color="#1f4e79", label="LightGBM forecast")
    ax.plot(one["target_week_start"], one["target_demand"], color="black", marker="o", markersize=3, linewidth=1, label="Actual")
    ax.set_title(f"One facility ({fid}): forecast with 80% prediction interval")
    ax.set_ylabel("Treatment courses / week")
    ax.legend(loc="upper left", ncol=3)

    fig.tight_layout()
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "forecast_eval.png", dpi=110)
    print("\nSaved docs/results/phase2_metrics.csv, phase2_intervals.csv and docs/figures/forecast_eval.png")


if __name__ == "__main__":
    main()
