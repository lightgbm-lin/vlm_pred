"""Permutation importance by refitting: shuffle a group of feature columns, rerun the walk-forward,
and compare the score with the unshuffled run."""
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from vlm_pred.metric import evaluate
from vlm_pred.walkforward import WalkForward


def permutation_test(wf: WalkForward, df: pd.DataFrame, columns: Sequence[str], target: str, weight: str,
                     seeds: Iterable[int] = range(1, 21), path=None) -> pd.DataFrame:
    """Score `wf` on `df` once unchanged and once per seed with `columns` permuted.

    The columns are shuffled together (one row permutation per seed), so their joint distribution is
    kept and only their link to the target is broken. Every run refits from scratch.

    `target` and `weight` are the columns the predictions are evaluated against, which can differ from
    `wf.target` (e.g. fit on y + 1, evaluate on y).

    Returns one row per run, with the baseline first (seed <NA>): r-squared, t-stat, and delta_r2 vs
    the baseline. If `path` is given, the table is also written there as CSV.
    """
    columns = list(columns)
    missing = set(columns) - set(wf.features)
    if missing:
        raise ValueError(f'not in wf.features, so permuting them would change nothing: {sorted(missing)}')

    def score(data):
        return evaluate(wf.run(data), data[target], data[weight])

    rows = [{'seed': None, **score(df)}]
    permuted = df.copy()
    for seed in seeds:
        idx = np.random.default_rng(seed).permutation(len(df))
        for col in columns:
            permuted[col] = df[col].to_numpy()[idx]
        rows.append({'seed': seed, **score(permuted)})

    res = pd.DataFrame(rows).astype({'seed': 'Int64'})
    res['delta_r2'] = res['r-squared'] - res['r-squared'].iloc[0]
    if path is not None:
        res.to_csv(path, index=False)
    return res
