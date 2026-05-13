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
  dataset/                            # raw M5 CSVs (not bundled; you place them here)
  data/{interim, processed}/          # joined base table, supervised feature tables
  notebooks/                          # thin demo notebooks calling modules
  src/seercast/                       # library code (config, data, features, models, evaluation, scenario, viz)
  outputs/{figures, reports, models}/ # backtest scores, predictions, trained model artifacts
```

See `src/seercast/` for the actual implementation. Notebooks are intentionally
thin — they orchestrate, the modules do the work.

## Getting the data

The raw M5 files (`calendar.csv`, `sales_train_validation.csv`,
`sales_train_evaluation.csv`, `sell_prices.csv`, `sample_submission.csv`)
should be placed in the `dataset/` folder at the repo root. They are not
bundled with this repo. Source:
[M5 Forecasting — Accuracy on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

## Quick start

```bash
pip install -e .
# place M5 csvs in dataset/
python -c "from seercast.data import load_m5_raw; print(load_m5_raw().sales.shape)"
```

## End-to-end run order

Each script validates its inputs strictly and writes its outputs in
``data/`` or ``outputs/``. Run them in this order:

```bash
python -m seercast.training.build_base_table        # Phase 2: dataset/*.csv   -> data/interim/m5_base_ca1.parquet
python -m seercast.training.train_baselines         # Phase 3: base table       -> outputs/reports/baseline_*.{parquet,csv}
python -m seercast.training.build_features          # Phase 4: base table       -> data/processed/train_features_ca1{,_full_horizon}.parquet
python -m seercast.training.train_lightgbm          # Phase 5: features (+ Phase 3 baselines, optional) -> outputs/{models,reports}/lightgbm_*
python -m seercast.training.train_quantile_lightgbm # Phase 6: features (+ Phase 5 point preds, optional) -> outputs/{models,reports}/lightgbm_quantile_* + uncertainty_diagnostics
python -m seercast.training.run_scenarios           # Phase 7: full-horizon features + Phase 6 quantile bundle -> outputs/reports/scenario_*
```

**Dependency notes.**

* Phase 3 (`train_baselines`) and Phase 4 (`build_features`) both consume
  the Phase 2 base table; you can run them in either order or in parallel.
* Phase 5 (`train_lightgbm`) reads the Phase 4 supervised table to train.
  It will *also* read `outputs/reports/baseline_predictions_ca1.parquet`
  (the Phase 3 deliverable) to write `outputs/reports/model_comparison_ca1.csv`.
  If that baseline file is missing the script still writes the LightGBM
  predictions / scores / model artifact and just skips the comparison
  with a `NOTE`. Run `train_baselines` first if you want the comparison.
* Phase 6 (`train_quantile_lightgbm`) reads the Phase 4 features. If the
  Phase 5 point-prediction parquet exists, it adds a `p50 vs lightgbm_point`
  comparison; otherwise it skips that comparison cleanly.
* Phase 7 (`run_scenarios`) reads the *full-horizon* feature table from
  Phase 4 and the quantile model bundle from Phase 6.

## Phase status

- [x] Phase 1 — Project charter (this README)
- [x] Phase 2 — Data understanding & project setup
- [x] Phase 3 — Baseline forecasting
- [x] Phase 4 — Feature engineering

## Phase 4 note: training origins vs. backtest origins

The supervised feature table includes **both** regular weekly training
origins (from `default_training_origins`, dense enough for LightGBM to
learn from) **and** the fixed backtest origins from `config.BACKTEST.origins`
(currently `(1500, 1556, 1612)`). Backtest origins are coerced from M5
`d_N` integers/strings into actual dates via
`seercast.evaluation.backtesting.origin_to_date`, then unioned, deduped,
and sorted with the training origins. This guarantees the Phase 5 LightGBM
backtest evaluates on the same days as the Phase 3 baselines, so
`outputs/reports/model_comparison_ca1.csv` is a fair comparison.

- [x] Phase 5 — LightGBM point model
- [x] Phase 6 — Probabilistic forecasting (quantile LightGBM)
- [x] Phase 7 — Scenario simulation (predictive, not causal)
