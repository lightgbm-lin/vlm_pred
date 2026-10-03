---
name: volume-researcher
description: Iterative research loop for daily stock-volume prediction factors. Reads vlm_pred/feature.py for the baseline factors. Then reads vlm_pred/research/hypotheses.md, proposes one economically motivated factor, implements it as vlm_pred/research/candidates/hNNN_*.py, evaluates it with vlm_pred/research/run.py (walk-forward LightGBM gamma, pooled R² vs the baseline, with an automatic look-ahead check), logs the result in vlm_pred/research/hypotheses.md, and repeats. Use when asked to research, propose, test, or iterate on volume-prediction features/factors.
---

# Volume factor researcher

You add factors, one at a time, that improve out-of-sample forecasts of next-day volume surprise, against the baseline features set implemented in `vlm_pred/feature.py`. Every
hypothesis you test, whether it passes or fails, gets logged in `vlm_pred/research/hypotheses.md` so no old idea is tested again.

## Problem setup

- **Spine**: a DataFrame indexed by unique `(date, uspn)` (`uspn` = stock), sorted by date then uspn, with
  columns `price_adj, price_close, price_open, price_high, price_low, volume, sp_weight, ret_raw,
  vlm_pred_naive, y, y_ratio` and a set of baseline features that are the outputs of the default arg of the enrich* 
  functions in `vlm_pred/feature.py`.
  - `vlm_pred_naive`: EWMA of *past* volume, 60-day halflife (shifted, so it's known before `date`).
  - `y = volume / vlm_pred_naive - 1`, clipped to [-1, 10]. It's heavily right-skewed. `y_ratio = y + 1`.
  - Baseline features (all point-in-time, i.e. built from data through t-1):
    - `vlm_{1,5,10,21}_ratio` (`enrich_vlm_ratio`): EWMA of past volume at that halflife / `vlm_pred_naive` - 1.
    - `vol_ewm_{1,5,10,21}` (`enrich_vol_ewm`): EWMA std of past returns.
    - `ret_{1..5}`, `y_{1..5}` (`enrich_lagged_ret`, `enrich_lagged_targets`): lagged returns and targets.
    - `max_y_{5,10,20}` (`enrich_max_targets`): rolling max of past `y`.
    - `earnings_day_prob` (`enrich_earnings_day`): chance T is an earnings day, from quarterly max-`y` spikes
      (`y` > 1) 52 weeks earlier, spread ±2 business days.
    - Calendar (`enrich_calendar_features`): `dow, month, week_of_month, is_month_start, is_month_end,
      is_quarter_end, is_msci_review, is_opex, is_quad_witch, is_post_holiday`.
- **Information set**: the prediction for `date` T is made **before T's open**, using data through T-1's
  close. Same-day `price_*`, `volume`, `ret_raw`, `sp_weight`, `y`, and `y_ratio` at T are off-limits.
- **Model**: `LGBMRegressor(random_state=42, verbose=-1, objective='gamma')` fit on
  `ratio_target = y + 1` (rows where it's ≤ 0 are dropped), weighted by `sp_weight`.
- **Evaluation**: `WalkForward(..., train_window=None)` uses an expanding window, refits yearly, and needs
  252 dates of history before the first fit, so predictions cover 2002–2009. The score is the R² of a
  pooled, `sp_weight`-weighted regression of `y` on the prediction (`vlm_pred.metric.evaluate`).
- **Data split**: research uses only dates ≤ `VAL_CUTOFF` (2010-01-01). Never load, compute on, or look
  at the VAL or OOS periods (`val_df` `oos_df`).

## Files

| Path | Role |
|---|---|
| `vlm_pred/research/hypotheses.md` | Research log: the current model, every tested hypothesis, and an idea backlog. Read it first. |
| `vlm_pred/research/candidates/hNNN_short_name.py` | One file per hypothesis, exposing `compute(df)`. |
| `vlm_pred/research/candidates/selected.txt` | Accepted candidates in order, one module name per line. This is the source of truth for the feature set. |
| `vlm_pred/research/run.py` | The evaluation harness. Don't edit it during research, or scores stop being comparable. |
| `vlm_pred/research/results/hNNN_short_name.json` | Full output of each run. |
| `vlm_pred/feature.py` | The baseline feature helpers (see Problem setup). They're always in the model. Reuse their code in candidates, but don't retest them as hypotheses. |

`vlm_pred/research/run.py` commands. Run them from the repo root: `-m` is what makes the `vlm_pred` package importable.

```bash
python -m vlm_pred.research.run --baseline                                  # score of the current feature set (cached)
python -m vlm_pred.research.run vlm_pred/research/candidates/h007_foo.py    # evaluate a candidate vs the current feature set
python -m vlm_pred.research.run --accept h007_foo                           # append to candidates/selected.txt
```

A candidate run takes about 1 minute. `vlm_pred/research/run.py` builds the enriched frame (the training spine, plus
the baseline features, plus each selected candidate's columns applied in order), then:
1. calls your `compute`
2. checks coverage
3. runs a **look-ahead check**: it recomputes on data truncated at a date T, with T's same-day columns
   scrambled, and requires identical values at T
4. walk-forwards the baseline (cached) and baseline + new columns
5. prints overall and per-year ΔR², LightGBM gain shares, and a suggested decision.

## The loop

Run the number of iterations the user asks for. If they don't say, run 3, then stop and summarize.

1. **Review.** Read `vlm_pred/research/hypotheses.md`: the current feature set, the current R², every tested hypothesis
   (accepted *and* rejected), and the backlog. If the file is empty, initialize it from the template below
   and fill in the baseline numbers with `python -m vlm_pred.research.run --baseline`. Glance at recent `vlm_pred/research/results/*.json`
   when you need detail.
2. **Propose.** Pick one hypothesis that is not a restatement of anything already tested (see
   "What counts as new"). Write down:
   - the economic mechanism: *why* volume tomorrow should deviate from its 60-day EWMA
   - the expected sign or shape
   - why the current features don't already capture it

   Prefer ideas the backlog or earlier results point to. For example, a rejected feature with a high gain
   share but no ΔR² may be redundant with something already in the model, which suggests testing a
   residualized version.
3. **Implement.** Create `vlm_pred/research/candidates/hNNN_short_name.py`, using the next free NNN. Follow the contract
   below. Sanity-check it quickly on a slice before the full run: shape, NaN rate, a `describe()`, and
   its correlation with `y`.
4. **Evaluate.** Run `python -m vlm_pred.research.run vlm_pred/research/candidates/hNNN_short_name.py`. If leakage is flagged, fix the
   feature and rerun. Never exempt a column to get around the check.
5. **Decide.** The default rule, also printed by `vlm_pred/research/run.py`, is: no leakage, **ΔR² ≥ 0.0005**, **and** R²
   improves in **≥ 2/3 of years**. You may overrule it, but give the reason in the log. Reasons to
   overrule include: the gain is concentrated in one crisis year, or the gain is a near-duplicate of an
   accepted feature. When a candidate is accepted, run `python -m vlm_pred.research.run --accept hNNN_short_name`. It
   becomes part of the enriched frame for every later run, and its ΔR² becomes the new baseline.
6. **Log.** Update `vlm_pred/research/hypotheses.md`: add the entry, update the "Current model" block if the candidate was
   accepted, and adjust the backlog, including follow-ups the result suggests. Log rejected and leaky
   ideas too. A negative result is information.
7. Repeat.

At the end, report to the user:
- each hypothesis tested, with its ΔR², years improved, and decision
- the new overall R²
- the most promising next ideas

## Candidate file contract

```python
"""H007: <one-line hypothesis>.

Mechanism: <why this predicts volume / vlm_pred_naive at t, economically>.
Expected: <sign/shape, e.g. higher |ret_{t-1}| -> higher y_t, saturating>.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level='uspn')
    absret = g['ret_raw'].shift(1).abs()            # shift(1): only data through t-1
    return pd.DataFrame({'h007_absret_1': absret}, index=df.index)
```

- `compute(df)` receives the enriched frame (a shallow copy) and returns a DataFrame (or Series) of
  **new columns only**, on the same index. Prefix column names with `hNNN_`. Keep it to about 1–5
  columns per hypothesis, so the gain can be attributed to one idea.
- Put the hypothesis in the module docstring. `vlm_pred/research/run.py` stores it in the results JSON.
- Import project helpers from the `vlm_pred` package, e.g. `from vlm_pred.feature import
  enrich_vlm_ratio`. Never use bare `from feature import ...`.
- **Per-stock time series**: use `groupby(level='uspn')` and `.shift(1)` on anything built from same-day
  columns. Rolling and EWMA windows must end at t-1, for example `g[col].transform(lambda x:
  x.ewm(halflife=h).mean().shift(1))` or `vlm_pred.util.calc_ewm(df, halflife, column)`.
- **Cross-sectional or market-wide features**: aggregate with `groupby(level='date')` on *lagged*
  inputs, or aggregate the raw inputs and then shift by date, not by row. For example, compute the
  market mean of `y` per date, shift it one date, and map it back onto the (date, uspn) index.
- **No full-sample statistics**: no global mean or std normalization, no quantiles fit on all dates, and
  nothing else that uses the future. The truncation check catches these.
- **No look-ahead, no exemptions**: a feature for date T may use only data on dates before T, plus
  facts about T derivable from dates up to T (e.g. weekday, gap since the previous session). Anything
  that needs the future trading calendar, such as "tomorrow is a holiday", is not allowed. Every column
  must pass the truncation check.
- Return floats or bools. Inf is converted to NaN, and LightGBM handles NaN natively, so leave missing
  values as NaN rather than filling them with 0.
- Keep it vectorized. The frame has about 1.6M rows, and `groupby(...).transform(lambda ...)` is fine.

## What counts as new

- **Monotone transforms of a single existing feature are not new.** Trees are invariant to them. log,
  rank, clip, square, sign-preserving scaling, and the same statistic at a halflife that's nearly the
  same all count as already tested.
- **New information is new**: a different input column, a different horizon regime (1 day vs
  1 quarter), cross-sectional context, event detection, or calendar/schedule information.
- **Some combinations are new** when a tree would find them hard to build from existing splits:
  ratios or differences of two features, residuals versus the market, and interactions with a
  conditioning variable.
- **Make features comparable across stocks.** The model is pooled, so use ratios, returns, z-scores
  against the stock's own history, or cross-sectional ranks, not raw volume, dollar, or price levels.
  Size effects should come from something like `sp_weight` lagged one day.

## hypotheses.md template

````markdown
# Volume factor research log

## Current model
- Features: baseline features (`vlm_pred/feature.py`) + <selected candidates' columns>
- Selected candidates (see candidates/selected.txt): <list>
- Walk-forward R² (2002–2009, train only): 0.xxxxx  (t = …)

## Tested hypotheses

### H001 — <short name> — ACCEPTED | REJECTED | LEAKY
- File: `candidates/h001_short_name.py` · columns: `h001_…`
- Hypothesis / mechanism: …
- Result: R² 0.xxxxx → 0.xxxxx (Δ +0.00xxx), years improved k/8, gain share x.xx
- Notes: per-year pattern, surprises, why it (did not) work, follow-ups.

## Backlog
- <idea> — <why it's promising / what result motivated it>
````

Keep entries short, and keep the log append-only. Don't delete history. If a later ablation removes an
accepted feature, add a note to its entry and update `selected.txt`.

## Pitfalls

- **Selection bias**: after many tries, some candidates pass by luck. Be suspicious of gains that
  appear in only a few years or come from a feature with a tiny gain share, and prefer simpler versions.
- **Parameter sweeps**: don't sweep a parameter (a halflife, a window) across several candidates and keep
  the best one. Pick the value from the mechanism, test it once, and log any sweep as a single
  hypothesis.
- **Protected files**: never modify `vlm_pred/research/run.py`, `vlm_pred/walkforward.py`, `vlm_pred/metric.py`, `vlm_pred/data.py`, or the model
  settings to make a candidate look better. If the harness seems wrong, stop and tell the user.
- **Accepting a candidate invalidates earlier comparisons**: earlier results were measured against a
  smaller feature set. A previously rejected idea may be worth retesting only if the new feature set
  plausibly changes its value. Log it as a new hypothesis that references the old one.
