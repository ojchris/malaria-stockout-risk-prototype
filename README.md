# Malaria Stockout-Risk Prototype

A small decision-support prototype that estimates which health facilities are likely to run out of essential malaria medicines, and (planned) where surplus stock near expiry could be redistributed.

**Status: v0.1, in progress. Started 10 November 2026.**

## Read this first

- **Built in response to a job posting.** This prototype was started in November 2026 in response to the third sample project in the Gates Foundation AI Fellow (Africa Regional Offices) posting: forecasting stockout risk and flagging redistribution opportunities before expiry. It is a learning and demonstration project, not production work and not done for any client.
- **The data is synthetic.** Facility-level stock and consumption data for Nigeria is not openly available. The generator in `src/stockout/` simulates seasonality, delivery delays and expiry. Results show that the method works on simulated data; they say nothing about real-world performance.
- **AI tools were used.** Much of the code was written with Claude (Anthropic) and reviewed and run by the repository owner. Commits made through Claude carry a `Co-Authored-By` line.

## Planned scope

| Stage | What | Status |
|---|---|---|
| v0.1 | Synthetic data generator, seasonal-naive baseline, gradient-boosting forecast with intervals, stockout-risk flag, Streamlit page, model card | Not started |
| Next | Real open data for facility locations and malaria seasonality (GRID3, Malaria Atlas Project, rainfall) | Planned |
| Next | Expiry-aware redistribution suggestions | Planned |
| Next | LLM explanation layer over public guideline documents, with a small evaluation set | Planned |
| Next | Fairness and robustness checks, tests in CI, hosted demo | Planned |

## Project layout

```
src/stockout/   Python package: data generation, models, risk flags
tests/          Unit tests
app/            Streamlit interface (added later)
docs/           Model card and notes (added later)
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Limitations

To be completed with the model card. In short: synthetic data, a small number of facilities and medicines, and no validation against real logistics records.

## License

MIT (see `LICENSE`).
