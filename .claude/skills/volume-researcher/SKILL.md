---
name: volume-researcher
description: Research loop for daily stock-volume prediction factors. Proposes one factor at a time as vlm_pred/research/candidates/hNNN_*.py, scores it against the baseline with vlm_pred/research/run.py, and logs the result in vlm_pred/research/hypotheses.md. Use when asked to research, propose, test, or iterate on volume-prediction features.
---

# Volume factor researcher

Find features that improve the forecast of `y = volume / vlm_pred_naive - 1` (next-day volume surprise),
on top of the baseline: `FEATURE_FUNCS` in `vlm_pred/main.py`, implemented in `vlm_pred/feature.py`.

## Files

- `vlm_pred/research/hypotheses.md`: the log of the current model, every tested hypothesis, and a backlog. Read it first.
- `vlm_pred/research/candidates/hNNN_name.py`: one file per hypothesis.
- `vlm_pred/research/candidates/selected.txt`: accepted candidates, added on top of the baseline.

Run these commands from the repo root:

```bash
python -m vlm_pred.research.run --baseline                                     # score the current model
python -m vlm_pred.research.run vlm_pred/research/candidates/hNNN_name.py     # score a candidate (~1 min)
python -m vlm_pred.research.run --accept hNNN_name                             # add it to selected.txt
```

## Loop

Run the number of iterations the user asks for (default 3), then report each hypothesis's ΔR² and
decision, plus the new R².

1. Read `hypotheses.md` and pick one idea that hasn't been tested. Write down its economic mechanism.
2. Write `candidates/hNNN_name.py`, using the next free NNN.
3. Run it. The harness checks for look-ahead and prints ΔR², the years improved, and a suggested decision.
4. Accept it if there's no leakage, ΔR² ≥ 0.0005, and R² improves in at least 2/3 of years. Run `--accept`.
5. Log it in `hypotheses.md`, whatever the outcome, as name, mechanism, ΔR², years improved and a note.
   Update "Current model" if you accepted it.

## Candidate file

```python
"""H001: <one-line hypothesis and mechanism>."""
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    absret = df.groupby(level='uspn')['ret_raw'].shift(1).abs()
    return pd.DataFrame({'h001_absret_1': absret}, index=df.index)
```

`df` is indexed by `(date, uspn)` and holds the raw data plus the baseline features. Return only new
columns, prefixed `hNNN_`. Use 1–5 columns.

## Rules

- **Point-in-time.** A feature for date T may use only data through T-1 (`shift(1)` per stock), plus
  facts about T derivable from past dates (e.g. weekday). No full-sample statistics. Never work around
  the look-ahead check.
- **No exchange-schedule knowledge**, such as early closes or holiday eves.
- **Use only dates through 2009.** The harness enforces this; don't load later data yourself.
- **Make features comparable across stocks**: use ratios, z-scores or ranks, not raw levels.
- **Monotone transforms of existing features aren't new**, because trees ignore them.
- **Don't edit** `run.py`, `main.py`, `walkforward.py`, `metric.py` or `data.py`. If the harness looks
  wrong, stop and tell the user.
