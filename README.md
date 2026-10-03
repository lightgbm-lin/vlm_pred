# Daily volume prediction (S&P 500)

This project forecasts each stock's next-day volume relative to its recent average. The data is a daily
panel of 497 S&P 500 stocks from 2000 to 2019.

- **Target**: `y = volume / vlm_pred_naive - 1`, clipped to [-1, 10]. `vlm_pred_naive` is a 60-day-halflife
  EWMA of past volume, so `y` is the volume surprise against a naive forecast.
- **Information set**: the forecast for date T uses data through T-1's close, plus calendar facts
  derivable from dates up to T (e.g. weekday, gap since the previous session). No look-ahead: nothing
  that needs dates after T, such as the future holiday schedule.
- **Model**: LightGBM with a gamma objective on `y + 1`, weighted by `sp_weight`, fit walk-forward
  (expanding window, yearly refit).
- **Metric**: R² of a pooled, `sp_weight`-weighted regression of `y` on the prediction (`vlm_pred/metric.py`).
- **Split**: dates through 2014 are in-sample and are used for all research. 2015–2019 is held out.

## Layout

```
data/sp500.h5            input panel, indexed by (date, uspn)
vlm_pred/                the model package: data loading, features, walk-forward, metric
vlm_pred/research/       agent-driven feature search: harness, log, candidates, results
notebooks/               exploration and model comparison (OLS vs LightGBM MSE vs gamma)
.claude/skills/          instructions for the research agent (Claude Code skill)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`data/sp500.h5` must be present.

## Run

Score the current feature set. Run it from the repo root. The first run builds the features and takes
about 1 minute; after that, the result is cached:

```bash
python -m vlm_pred.research.run --baseline
```

Expected: walk-forward R² ≈ **0.2735** over 2002–2014 (vs 0.2309 for the `vlm_1_ratio`-only baseline).

To explore interactively, open the notebook:

```bash
jupyter notebook notebooks/VolumePrediction.ipynb
```

## Research

The features beyond the baseline were found by an agent loop. In each iteration, the agent proposes an
economically motivated factor, the harness checks it for look-ahead and measures its walk-forward
ΔR², and the agent logs the outcome. See [vlm_pred/research/README.md](vlm_pred/research/README.md) for how it
works and [vlm_pred/research/hypotheses.md](vlm_pred/research/hypotheses.md) for every hypothesis tested,
including rejected ones.
