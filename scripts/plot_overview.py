"""Draw an overview of the synthetic data so you can check it looks plausible.

Usage:  python scripts/plot_overview.py [--out docs/figures/synthetic_overview.png]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stockout.data_gen import GeneratorConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/figures/synthetic_overview.png")
    args = parser.parse_args()

    facilities, weekly = generate_dataset(GeneratorConfig())
    al = weekly[weekly["commodity"] == "AL"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    # 1. Demand across all facilities, with the rain driver behind it
    total = al.groupby("week_start")["demand"].sum()
    rain = al.groupby("week_start")["rain_index"].first()
    ax = axes[0]
    ax.plot(total.index, total.values, color="#1f4e79", label="Total AL demand (all facilities)")
    ax.set_ylabel("Treatment courses / week")
    ax2 = ax.twinx()
    ax2.plot(rain.index, rain.values, color="#7aa6c2", linestyle="--", label="Rain index")
    ax2.set_ylabel("Rain index (1 = average)")
    ax.set_ylim(0, total.max() * 1.3)
    ax2.set_ylim(0, rain.max() * 1.3)
    ax.set_title("Demand follows the rains with a lag (synthetic data)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 2. Share of facilities stocked out each week
    share = al.groupby("week_start")["stockout"].mean() * 100
    axes[1].fill_between(share.index, share.values, color="#c0504d", alpha=0.6)
    axes[1].set_ylabel("% of facilities stocked out")
    axes[1].set_title("Stockouts peak while demand is ramping up")

    # 3. One facility: stock, demand and deliveries
    fid = facilities.sort_values("catchment_population").iloc[len(facilities) // 2]["facility_id"]
    one = al[al["facility_id"] == fid]
    ax = axes[2]
    ax.plot(one["week_start"], one["stock_end"], color="#1f4e79", label="Stock at end of week")
    ax.bar(one["week_start"], one["demand"], width=5, color="#bbbbbb", label="Weekly demand")
    ax.scatter(
        one.loc[one["received"] > 0, "week_start"],
        one.loc[one["received"] > 0, "received"],
        color="#2e8b57",
        s=18,
        label="Delivery received",
        zorder=3,
    )
    out_weeks = one[one["stockout"]]
    ax.scatter(out_weeks["week_start"], [0] * len(out_weeks), marker="x", color="#c0504d", label="Stockout week")
    ax.set_ylabel("Treatment courses")
    ax.set_title(f"One facility ({fid}): stock, demand and deliveries")
    ax.legend(loc="upper left", ncol=2)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
