# Research loop

The features in the model were found one hypothesis at a time by a Claude Code agent, following the
skill in [`.claude/skills/volume-researcher/SKILL.md`](../../.claude/skills/volume-researcher/SKILL.md).
The agent writes candidate features and the log. The harness, `run.py`, is fixed: the agent may not
edit it, so scores stay comparable across iterations.

| Path | Role |
|---|---|
| `run.py` | Evaluation harness |
| `hypotheses.md` | Research log: current model, every tested hypothesis, and the backlog |
| `candidates/hNNN_*.py` | One factor per file, exposing `compute(df)`. The module docstring states the hypothesis. |
| `candidates/selected.txt` | Accepted candidates, in order. This is the source of truth for the feature set. |
| `results/hNNN_*.json` | Full output of each evaluation |

## What `run.py` does for a candidate

1. Builds the in-sample frame (dates through 2014; the holdout is never touched), plus `vlm_1_ratio`,
   plus the columns of every selected candidate.
2. Calls the candidate's `compute(df)` and checks its coverage.
3. **Look-ahead check**: recomputes the feature on data truncated at a date T, with T's same-day market
   data scrambled, and requires identical values at T. No column is exempt: look-ahead bias is not
   allowed, even for scheduled information such as future holidays.
4. Walk-forwards the current feature set (cached) and the feature set with the candidate added.
5. Reports overall and per-year ΔR², plus LightGBM gain shares.

**Accept rule**: no leakage, ΔR² ≥ 0.0005, and R² improves in at least 2/3 of years. The agent may
overrule the rule, but must give the reason in the log.

## Results so far

| | Factor | ΔR² | Years improved | Decision |
|---|---|---|---|---|
| | baseline `vlm_1_ratio` | 0.23088 | | |
| H001 | volume term structure (5d/21d) | +0.00169 | 12/13 | accepted |
| H002 | scheduled calendar events | +0.02777 | 13/13 | accepted |
| H003 | market-wide volume shock | +0.00095 | 6/13 | rejected |
| H004 | earnings seasonality | +0.00549 | 12/13 | accepted |
| H005 | T-1 price shock vs own volatility | +0.00637 | 13/13 | accepted |
| H006 | idiosyncratic volume ratio | +0.00134 | 10/13 | accepted |
| | **current model** | **0.27353** | | |

## Commands

Run these from the repo root:

```bash
python -m vlm_pred.research.run --baseline                                  # score the current feature set
python -m vlm_pred.research.run vlm_pred/research/candidates/h007_foo.py    # evaluate a new candidate
python -m vlm_pred.research.run --accept h007_foo                           # add it to selected.txt
```

To resume the agent loop, ask Claude Code in this repo to "run the volume researcher for N iterations".
