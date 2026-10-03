"""H003: earnings-day probability from volume spikes 13, 26 and 39 weeks earlier (the baseline only uses 52 weeks).

Mechanism: firms report quarterly on a roughly 13-week cycle, so last quarter's earnings reaction (a stock's
quarterly max-y spike) is a timelier guide to the next one than last year's, follows schedule drift, and is
available in a stock's first year. Agreement across several past quarters raises confidence. Earnings-type
spikes (y > 1.5 on 2.6% of days) carry ~58% of the weighted y variance, and the baseline flags only ~12% of them.
Expected: higher prob_13w and higher multi-quarter consensus -> higher y_t, convex.
"""
import numpy as np
import pandas as pd

LAGS_WEEKS = (13, 26, 39)


def _eday_prob(df: pd.DataFrame, lag_days: int, tol_days: int = 2, min_y: float = 1.0) -> pd.Series:
    """As vlm_pred.feature.enrich_earnings_day, but drops window dates inside the spike's own quarter.

    A spike is a quarter's max y, which is only known once that quarter ends; with a 13-week lag and the
    tolerance window, the target date can fall before then.
    """
    y = df['y']
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')

    q_max = y.groupby([uspn, dates.to_period('Q')]).transform('max')
    spikes = df.index[(y == q_max) & (y > min_y)]

    s_dates = spikes.get_level_values('date')
    s_uspn = spikes.get_level_values('uspn')
    q_end = s_dates.to_period('Q').end_time.normalize()
    centre = s_dates + pd.Timedelta(days=lag_days)
    offsets = np.arange(-tol_days, tol_days + 1)
    weights = (tol_days + 1 - np.abs(offsets)) / (tol_days + 1) ** 2
    parts = []
    for o, w in zip(offsets, weights):
        tgt = centre + pd.offsets.BDay(o)
        known = tgt > q_end
        parts.append(pd.Series(w, index=pd.MultiIndex.from_arrays([tgt[known], s_uspn[known]])))
    expected = pd.concat(parts)
    return expected.groupby(level=[0, 1]).max().reindex(df.index, fill_value=0.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    probs = {lag: _eday_prob(df, lag_days=7 * lag) for lag in LAGS_WEEKS}
    return pd.DataFrame({
        'h003_eday_prob_13w': probs[13].to_numpy(),
        # Consensus over the three quarterly lags and the baseline's 52-week lag
        'h003_eday_prob_sum': (sum(probs.values()) + df['earnings_day_prob']).to_numpy(),
    }, index=df.index)
