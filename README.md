# SeerCast — Drift-Aware Multi-Horizon Retail Demand Forecasting

SeerCast is a portfolio-grade forecasting system built on the M5 Forecasting
dataset. It forecasts daily unit sales at the `item_id × store_id × day` level,
starting with the `CA_1` store subset, using a 28-day horizon.

This is a **predictive decision-support system**, not a causal system. Scenario
simulation answers *"what does the model predict if we change this input?"* —
not *"this input caused the demand change."*

## Business problem

Retailers need accurate short-horizon demand forecasts to avoid stockouts,
overstock, lost sales, and poor replenishment planning. SeerCast answers:

- How many units should we expect to sell?
- Which products are at risk of demand spikes?
- How uncertain is the forecast?
- What might happen if price changes?
- What might happen during events / SNAP days?
- What happens if recent demand momentum increases or decreases?

## Methodology (high level)

1. **Honest baselines** — naive, seasonal-naive, moving-average. Anything
   beyond this must beat them on rolling-origin backtests.
2. **Direct multi-horizon LightGBM** — point forecasts for horizons
   `[1, 7, 14, 28]`, then full `1–28` for scenario work.
3. **Quantile LightGBM** — `p10 / p50 / p90` for uncertainty-aware decisions.
4. **Rolling-origin backtesting** — never random splits. Train up to an origin,
   forecast the next 28 days, slide the origin forward, repeat.
5. **Scenario simulator** — predictive what-ifs over price, events, SNAP, and
   demand momentum.

## Important limitations

- **Not causal.** We do not claim that changing price *causes* a demand change.
  We only show what the trained model predicts under that input change.
- **Historical regime only.** The model has only seen the M5 historical period.
  Out-of-distribution scenarios (e.g. very large price moves, novel events)
  should be treated as extrapolation.
- **Subset first.** Initial work is `store_id == "CA_1"` to keep iteration
  cheap. Generalization to other stores/states is a later step.

## Repo layout

```
seercast/
  data/{raw, interim, processed}/    # raw M5 inputs, joined base table, supervised tables
  notebooks/                          # thin demo notebooks calling modules
  src/seercast/                       # library code (config, data, features, models, evaluation, scenario, viz)
  outputs/{figures, reports, models}/ # backtest scores, predictions, trained model artifacts
```

See `src/seercast/` for the actual implementation. Notebooks are intentionally
thin — they orchestrate, the modules do the work.

## Getting the data

The raw M5 files (`calendar.csv`, `sales_train_validation.csv`,
`sales_train_evaluation.csv`, `sell_prices.csv`, `sample_submission.csv`)
should be placed in `data/raw/`. They are not bundled with this repo. Source:
[M5 Forecasting — Accuracy on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

## Quick start

```bash
pip install -e .
# place M5 csvs in data/raw/
python -c "from seercast.data import load_m5_raw; print(load_m5_raw('data').sales.shape)"
```

## Phase status

- [x] Phase 1 — Project charter (this README)
- [x] Phase 2 — Data understanding & project setup
- [ ] Phase 3 — Baseline forecasting
- [ ] Phase 4 — Feature engineering
- [ ] Phase 5 — LightGBM point model
- [ ] Phase 6 — Probabilistic forecasting (quantile LightGBM)
- [ ] Phase 7 — Scenario simulation
