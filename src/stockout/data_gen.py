"""Synthetic facility-level stock data for malaria commodities.

Everything here is SIMULATED. Facility-level logistics data for Nigeria is not
openly available, so this module invents it. It is meant to exercise the
forecasting and risk-flag methods, not to describe any real facility.

What is simulated, per facility, commodity and week:

* Demand follows a yearly seasonal pattern. Rain peaks first and cases peak a
  few weeks later (the lag), scaled by the facility's catchment population,
  with random noise.
* Facilities review stock every few weeks and order using a simple rule based
  on recent consumption. Deliveries arrive late at random (more often in rural
  facilities) and are sometimes only partly filled.
* Each delivery is a batch with its own expiry date. Batches are used
  first-expired-first-out (FEFO). Expired stock is written off.
* Occasionally an unrequested, oversized, short-dated shipment is pushed to a
  facility. This creates surplus stock that may expire.
* A stockout is any week in which some demand could not be met.

Everything is driven by a fixed seed, so the same config always produces the
same data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommoditySpec:
    """One tracked commodity."""

    code: str
    name: str
    unit: str
    demand_multiplier: float  # units needed per treated case
    shelf_weeks: tuple[int, int]  # remaining shelf life at delivery: (min, max)


COMMODITIES: tuple[CommoditySpec, ...] = (
    CommoditySpec("AL", "Artemether-lumefantrine", "treatment course", 1.0, (30, 104)),
    CommoditySpec("RDT", "Malaria rapid diagnostic test", "test", 1.5, (20, 78)),
)

# size class -> ((min, max catchment population), share of facilities)
SIZE_CLASSES: dict[str, tuple[tuple[int, int], float]] = {
    "small": ((3_000, 8_000), 0.50),
    "medium": ((8_000, 20_000), 0.35),
    "large": ((20_000, 45_000), 0.15),
}


@dataclass(frozen=True)
class GeneratorConfig:
    """All the knobs. Defaults give about 45 facilities over 3 years."""

    n_facilities: int = 45
    n_weeks: int = 156
    start_date: str = "2023-01-02"  # a Monday
    seed: int = 42
    annual_cases_per_person: float = 0.08  # treated cases per person per year
    rain_peak_week: int = 29  # week of year when rainfall peaks
    case_lag_weeks: int = 4  # cases peak this many weeks after rain
    seasonality_strength: float = 0.7  # larger = sharper wet/dry contrast
    demand_noise_sd: float = 0.15  # log-scale noise on weekly demand
    review_weeks: int = 4  # facilities review stock this often
    safety_weeks: float = 2.5  # extra weeks of cover the ordering rule adds
    history_weeks: int = 4  # recent weeks of demand used for ordering
    full_fill_prob: float = 0.75  # chance an order is delivered in full
    push_shipment_prob: float = 0.05  # chance of an unrequested oversized shipment
    urban_share: float = 0.4
    commodities: tuple[CommoditySpec, ...] = COMMODITIES


def week_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """Week of year as 1..52 (the last days of a year fold into week 52)."""
    return np.minimum((dates.dayofyear.to_numpy() - 1) // 7 + 1, 52)


def seasonal_factor(woy: np.ndarray, peak_week: int, strength: float) -> np.ndarray:
    """Smooth yearly cycle that peaks at `peak_week` and averages 1.0 over a year."""

    def raw(w: np.ndarray) -> np.ndarray:
        return np.exp(strength * np.cos(2 * np.pi * (np.asarray(w) - peak_week) / 52))

    return raw(woy) / raw(np.arange(1, 53)).mean()


def make_facilities(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Invent the facilities: size, catchment population, setting, supply traits."""
    n = cfg.n_facilities
    names = list(SIZE_CLASSES)
    probs = [SIZE_CLASSES[k][1] for k in names]
    size_class = rng.choice(names, size=n, p=probs)
    population = np.array([int(rng.integers(*SIZE_CLASSES[s][0])) for s in size_class])
    urban = rng.random(n) < cfg.urban_share
    base_lead = np.where(urban, rng.integers(2, 4, n), rng.integers(3, 7, n))
    delay_prob = np.where(urban, 0.12, 0.28)
    review_offset = rng.integers(0, cfg.review_weeks, n)
    return pd.DataFrame(
        {
            "facility_id": [f"FAC-{i + 1:03d}" for i in range(n)],
            "size_class": size_class,
            "catchment_population": population,
            "setting": np.where(urban, "urban", "rural"),
            "base_lead_time_weeks": base_lead.astype(int),
            "delay_prob": delay_prob,
            "review_offset": review_offset.astype(int),
        }
    )


def _simulate_stock(
    demand: np.ndarray,
    fac: pd.Series,
    spec: CommoditySpec,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run one facility and commodity forward week by week."""
    n = len(demand)
    lo, hi = spec.shelf_weeks
    base_weekly = (
        fac.catchment_population * cfg.annual_cases_per_person / 52 * spec.demand_multiplier
    )
    lead = int(fac.base_lead_time_weeks)
    offset = int(fac.review_offset)

    # Each batch is [expiry_week, quantity]. Opening stock is 8 weeks of average demand.
    initial = int(round(base_weekly * 8))
    batches: list[list[int]] = [[int(rng.integers(lo, hi + 1)), initial]]
    pipeline: list[list[int]] = []  # [arrival_week, quantity, remaining shelf weeks]

    opening = np.zeros(n, dtype=int)
    received = np.zeros(n, dtype=int)
    expired = np.zeros(n, dtype=int)
    consumption = np.zeros(n, dtype=int)
    stock_end = np.zeros(n, dtype=int)

    for t in range(n):
        opening[t] = initial if t == 0 else stock_end[t - 1]

        # 1. Deliveries due this week arrive as new batches.
        arriving = [p for p in pipeline if p[0] == t]
        pipeline = [p for p in pipeline if p[0] != t]
        for _, qty, shelf in arriving:
            batches.append([t + shelf, qty])
        received[t] = sum(p[1] for p in arriving)

        # 2. Expired batches are written off.
        expired[t] = sum(q for e, q in batches if e <= t)
        batches = [b for b in batches if b[0] > t]

        # 3. Patients are served first-expired-first-out until stock runs out.
        batches.sort(key=lambda b: b[0])
        need = int(demand[t])
        served = 0
        for b in batches:
            take = min(b[1], need - served)
            b[1] -= take
            served += take
            if served == need:
                break
        batches = [b for b in batches if b[1] > 0]
        consumption[t] = served
        stock_end[t] = sum(b[1] for b in batches)

        # 4. On review weeks the facility places an order from recent consumption.
        if (t - offset) % cfg.review_weeks == 0:
            # Plan from demand (served + turned-away patients, as recorded in registers),
            # not from consumption, which stockouts would understate.
            recent = demand[max(0, t + 1 - cfg.history_weeks) : t + 1].mean()
            weekly = max(recent, 0.25 * base_weekly)  # never plan for near-zero demand
            target = weekly * (lead + cfg.review_weeks + cfg.safety_weeks)
            on_order = sum(p[1] for p in pipeline)
            order = target - stock_end[t] - on_order
            if order >= 1:
                extra = int(rng.integers(2, 9)) if rng.random() < fac.delay_prob else 0
                fill = 1.0 if rng.random() < cfg.full_fill_prob else rng.uniform(0.6, 0.95)
                qty = int(round(order * fill))
                arrival = t + 1 + lead + extra
                shelf = int(rng.integers(lo, hi + 1))
                if qty > 0 and arrival < n:
                    pipeline.append([arrival, qty, shelf])
            # Occasionally a programme pushes an unrequested, oversized, short-dated
            # shipment. These create the surplus that later risks expiring.
            if rng.random() < cfg.push_shipment_prob:
                push_qty = int(round(weekly * rng.uniform(6, 12)))
                push_arrival = t + 1 + lead
                push_shelf = int(rng.integers(10, 41))
                if push_qty > 0 and push_arrival < n:
                    pipeline.append([push_arrival, push_qty, push_shelf])

    return {
        "opening_stock": opening,
        "received": received,
        "expired": expired,
        "consumption": consumption,
        "stock_end": stock_end,
    }


def generate_dataset(cfg: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (facilities, weekly) tables for the given config."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    facilities = make_facilities(cfg, rng)

    dates = pd.date_range(cfg.start_date, periods=cfg.n_weeks, freq="7D")
    woy = week_of_year(dates)
    rain_index = seasonal_factor(woy, cfg.rain_peak_week, cfg.seasonality_strength)
    # Cases follow rainfall with a delay.
    case_factor = seasonal_factor(
        woy - cfg.case_lag_weeks, cfg.rain_peak_week, cfg.seasonality_strength
    )

    frames = []
    for _, fac in facilities.iterrows():
        expected_cases = fac.catchment_population * cfg.annual_cases_per_person / 52 * case_factor
        # One noise draw per week is shared by all commodities (same underlying cases).
        case_noise = rng.lognormal(0.0, cfg.demand_noise_sd, cfg.n_weeks)
        for spec in cfg.commodities:
            demand = rng.poisson(expected_cases * case_noise * spec.demand_multiplier)
            sim = _simulate_stock(demand, fac, spec, cfg, rng)
            unmet = demand - sim["consumption"]
            frames.append(
                pd.DataFrame(
                    {
                        "facility_id": fac.facility_id,
                        "commodity": spec.code,
                        "week_start": dates,
                        "week_of_year": woy,
                        "rain_index": rain_index,
                        "demand": demand,
                        "consumption": sim["consumption"],
                        "unmet_demand": unmet,
                        "opening_stock": sim["opening_stock"],
                        "received": sim["received"],
                        "expired": sim["expired"],
                        "stock_end": sim["stock_end"],
                        "stockout": unmet > 0,
                    }
                )
            )
    weekly = pd.concat(frames, ignore_index=True)
    return facilities, weekly


def summarise(weekly: pd.DataFrame) -> dict[str, float]:
    """A few headline numbers for a quick sanity check."""
    return {
        "rows": float(len(weekly)),
        "stockout_week_share": float(weekly["stockout"].mean()),
        "unmet_demand_share": float(weekly["unmet_demand"].sum() / weekly["demand"].sum()),
        "expired_share_of_received": float(weekly["expired"].sum() / weekly["received"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic malaria stock data.")
    parser.add_argument("--out", default="data/synthetic", help="output folder")
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--facilities", type=int, default=GeneratorConfig.n_facilities)
    parser.add_argument("--weeks", type=int, default=GeneratorConfig.n_weeks)
    args = parser.parse_args()

    cfg = GeneratorConfig(seed=args.seed, n_facilities=args.facilities, n_weeks=args.weeks)
    facilities, weekly = generate_dataset(cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facilities.to_csv(out / "facilities.csv", index=False)
    weekly.to_csv(out / "weekly_stock.csv", index=False)
    print(f"Wrote {len(facilities)} facilities and {len(weekly)} weekly rows to {out}/")
    for key, value in summarise(weekly).items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
