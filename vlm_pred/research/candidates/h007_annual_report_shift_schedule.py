"""H007: H005's earnings schedule with a one-week split for projections that cross the annual report.

Mechanism: Dec-year-end firms report Q4 together with (or ahead of) the audited 10-K, in calendar Q1, and that
report tends to come about a week later in the cycle than the interim ones. A spike projected *into* calendar Q1
from another quarter therefore often lands one week late, and one projected *out of* Q1 one week early.
Same-quarter projections (the 52w lag) are unaffected. Only some firms shift, so each crossing projection puts
half its mass on the usual +/- 2 day triangle and half on the same triangle shifted one week (same weekday).
Diagnostic on H005's spikes (offset of the next spike from k*91 days, all quarterly lags): unaffected pairs peak
at 0 (21%); pairs into Q1 peak at 0 (10%) and +7 (10%); pairs out of Q1 peak at 0 (12%) and -7 (9%); the days in
between stay at ~2-3%, so a shift fits better than a wider window.
Expected: as H005 (higher prob -> higher y_t, convex), with more hits on Q1 and Q2 earnings days.
"""
import numpy as np
import pandas as pd

LAGS_WEEKS = (13, 26, 39, 52)
MIN_Y = 1.0
GAP_Z = 2.0
SHIFT_DAYS = 7
SHIFT_SHARE = 0.5


def _triangle(dates: pd.DatetimeIndex, uspn, q_end, tol_days: int, scale) -> list:
    """Triangular +/- tol_days pmf around each date, scaled per spike; drops dates inside the spike's quarter."""
    offsets = np.arange(-tol_days, tol_days + 1)
    weights = (tol_days + 1 - np.abs(offsets)) / (tol_days + 1) ** 2
    parts = []
    for o, w in zip(offsets, weights):
        tgt = dates + pd.offsets.BDay(o)
        known = np.asarray(tgt > q_end) & (scale > 0)
        parts.append(pd.Series(w * scale[known], index=pd.MultiIndex.from_arrays([tgt[known], uspn[known]])))
    return parts


def _eday_prob(spikes: pd.MultiIndex, index: pd.MultiIndex, lag_days: int, tol_days: int = 2) -> pd.Series:
    """H005's pmf, split between the usual centre and a one-week shift when the projection crosses calendar Q1.

    Spikes are quarter maxima, known only at quarter end, so window dates inside the spike's quarter are dropped.
    """
    s_dates = spikes.get_level_values('date')
    s_uspn = spikes.get_level_values('uspn')
    q_end = s_dates.to_period('Q').end_time.normalize()
    centre = s_dates + pd.Timedelta(days=lag_days)
    shift = SHIFT_DAYS * ((centre.quarter == 1).astype(int) - (s_dates.quarter == 1).astype(int))
    crosses = shift != 0

    main_share = np.where(crosses, 1 - SHIFT_SHARE, 1.0)
    parts = _triangle(centre, s_uspn, q_end, tol_days, main_share)
    parts += _triangle(centre + pd.to_timedelta(shift, unit='D'), s_uspn, q_end, tol_days,
                       np.where(crosses, SHIFT_SHARE, 0.0))
    expected = pd.concat(parts)
    # A spike's two triangles are a week apart and never overlap; across spikes, keep the higher probability
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
        'h007_eday_prob_13w': probs[13].to_numpy(),
        'h007_eday_prob_sum': sum(probs.values()).to_numpy(),
        'h007_eday_prob_gap_sum': sum(probs_conf.values()).to_numpy(),
    }, index=df.index)
