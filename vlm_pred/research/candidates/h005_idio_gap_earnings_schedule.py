"""H005: H003's earnings schedule with cleaner spikes: idiosyncratic volume picks the quarter's spike, and an
overnight-gap-confirmed consensus is added as its own column.

Mechanism: earnings news is firm-specific and released outside trading hours, so a true earnings day shows a
volume spike that is not market-wide and an overnight price gap. H003 takes each quarter's raw max-y day, which
is sometimes a market-wide frenzy day (index events, crashes) rather than earnings. Here the quarter's spike is
the max of y minus that day's cross-sectional median y (still requiring > 1), and spikes whose overnight
return is >= 2 sigma of the stock's daily vol (a conventional abnormal-move cutoff) feed a separate
"confirmed" consensus, so the model can trust them more without losing the unconfirmed ones.
Diagnostic on H003's spikes: 13w/52w recurrence 14%/15% for gaps < 1 sigma vs 22-25%/25-29% for gaps >= 2 sigma;
11%/9% for spikes on days with market median y >= 0.5 vs 18%/20% on normal days.
Expected: as H003 (higher prob -> higher y_t, convex), with sharper separation; confirmed sum steeper.
"""
import numpy as np
import pandas as pd

LAGS_WEEKS = (13, 26, 39, 52)
MIN_Y = 1.0
GAP_Z = 2.0


def _eday_prob(spikes: pd.MultiIndex, index: pd.MultiIndex, lag_days: int, tol_days: int = 2) -> pd.Series:
    """Triangular +/- tol_days pmf centred lag_days after each spike, as in H003.

    Spikes are quarter maxima, known only at quarter end, so window dates inside the spike's quarter are dropped.
    """
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
    return expected.groupby(level=[0, 1]).max().reindex(index, fill_value=0.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')

    idio = df['y'] - df['y'].groupby(dates).transform('median')
    q_max = idio.groupby([uspn, dates.to_period('Q')]).transform('max')
    is_spike = (idio == q_max) & (idio > MIN_Y)

    # Overnight return, split-safe: close-to-close total return net of the same-day intraday move
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1
    gap_z = gap.abs() / df['vol_ewm_21'].where(df['vol_ewm_21'] > 0)
    is_confirmed = is_spike & (gap_z >= GAP_Z)

    spikes, confirmed = df.index[is_spike], df.index[is_confirmed]
    probs = {lag: _eday_prob(spikes, df.index, 7 * lag) for lag in LAGS_WEEKS}
    probs_conf = {lag: _eday_prob(confirmed, df.index, 7 * lag) for lag in LAGS_WEEKS}
    return pd.DataFrame({
        'h005_eday_prob_13w': probs[13].to_numpy(),
        'h005_eday_prob_sum': sum(probs.values()).to_numpy(),
        'h005_eday_prob_gap_sum': sum(probs_conf.values()).to_numpy(),
    }, index=df.index)
