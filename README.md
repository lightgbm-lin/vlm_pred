# Daily volume prediction (S&P 500)

This project forecasts each stock's next-day volume relative to its recent average. The data is a daily
panel of S&P 500 stocks from 2000 to 2019.

## Getting started

### 1. Set up the environment first

Create the conda environment from the provided [`environment.yml`](environment.yml) before running anything:

```bash
conda env create -f environment.yml
conda activate vlm_pred
```

### 2. Put `sp500.h5` in `data/`

> **The data file must be at `data/sp500.h5`.** It isn't in the repo, and nothing runs without it.

### 3. Read `vlm_pred.pdf`

Start with [`vlm_pred.pdf`](vlm_pred.pdf) for an overview of the project: the problem, the approach,
and the results.

### 4. View the charts and results in the notebook

```bash
jupyter notebook notebooks/VolumePrediction.ipynb
```

### 5. Reproduce the results with `main.py`

Run from the repo root. To reproduce from scratch, tune the hyperparameters first, then train and score:

```bash
python -m vlm_pred.main --tune --n-trials 100
```

```bash
python -m vlm_pred.main
```

- `--tune` searches the LightGBM hyperparameters on the in-sample dates (through 2014) and writes
  `trials/best.json` and `trials/trials.csv`, overwriting the saved ones.
- With no flag, it fits the model walk-forward over the full sample, then prints R² for the default and
  tuned LightGBM, in-sample (through 2014) and out-of-sample (2015–2019). The tuned parameters are read
  from `trials/best.json`, so you can skip tuning and use the saved parameters.

### 6. Research new factors with Claude

The [`volume-researcher`](.claude/skills/volume-researcher/SKILL.md) skill runs an agent loop in
[Claude Code](https://claude.com/claude-code). It proposes one economically motivated factor at a time,
checks it for look-ahead, scores its walk-forward ΔR² against the baseline, and logs the outcome. Open
Claude Code in this repo and ask, for example:

```
run the volume researcher for 5 iterations
```

Results are logged in [`vlm_pred/research/hypotheses.md`](vlm_pred/research/hypotheses.md). See
[`vlm_pred/research/README.md`](vlm_pred/research/README.md) for how the harness works.

## Model

- **Target**: `y = volume / vlm_pred_naive - 1`, clipped to [-1, 10]. `vlm_pred_naive` is a 60-day-halflife
  EWMA of past volume, so `y` is the volume surprise against a naive forecast.
- **Information set**: the forecast for date T uses data through T-1's close, plus calendar facts
  derivable from dates up to T (e.g. weekday, gap since the previous session). No look-ahead: nothing
  that needs dates after T, such as the future holiday schedule.
- **Features**: `FEATURE_FUNCS` in [`vlm_pred/main.py`](vlm_pred/main.py), implemented in
  [`vlm_pred/feature.py`](vlm_pred/feature.py).
- **Model**: LightGBM with a gamma objective on `y + 1`, weighted by `sp_weight`, fit walk-forward
  (expanding window, yearly refit).
- **Metric**: R² of a pooled, `sp_weight`-weighted regression of `y` on the prediction (`vlm_pred/metric.py`).
- **Split**: feature research uses only dates through 2009. Hyperparameter tuning uses dates through 2014.
  2015–2019 is held out.

## Layout

```
vlm_pred.pdf             project overview
environment.yml          conda environment
data/sp500.h5            input panel, indexed by (date, uspn); not in the repo
notebooks/               charts and results
trials/                  saved hyperparameter search (best.json is used by main.py)
vlm_pred/                the model package: data loading, features, walk-forward, metric, main.py
vlm_pred/research/       agent-driven feature search: harness, log, candidates, results
.claude/skills/          the volume-researcher skill for Claude Code
```
