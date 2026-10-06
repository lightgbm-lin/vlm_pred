"""H016: earnings schedule centred by last year's gap between the same two quarters (replaces H007's Q1 rule).

Mechanism: the annual report comes about a week later in the reporting cycle than interim reports, but in the
firm's fiscal Q4, which is calendar Q1 only for Dec/Jan year-ends (June, Sept and May year-ends are common in the
S&P 500). Instead of H007's calendar-Q1 rule, a projection from a spike in quarter q to quarter q+k (lag 13k
weeks) is centred at spike + last year's gap between the stock's q-4 and q-4+k spikes, rounded to whole weeks
(firms keep their reporting weekday). This handles any fiscal year-end without knowing which quarter is the
annual one. The gap is only used within +/- 14 days of 13k weeks (beyond that one of last year's spikes was likely
not an earnings day); otherwise, or when last year's pair is missing, the fixed lag is used. 52w lags are unchanged.
Pre-test (pairs with both years' offsets within 14d): last year's week offset predicts this year's only weakly
(last year +7 -> this year +7 30%, 0 35%); mean pmf mass on the realised date: fixed .097, H007 .096, H016 .084.
Expected: as H007 (higher prob -> higher y_t). Columns replace eday_prob_*; compare as a replacement.
"""
import numpy as np
import pandas as pd

from vlm_pred.feature import _daily_vol, _overnight_and_intraday

LAGS_WEEKS = (13, 26, 39, 52)
TOL_DAYS = 2
MIN_Y = 1.0
GAP_Z = 2.0
MAX_SHIFT_DAYS = 14


def _centres(spikes: pd.DataFrame, last_year: pd.Series, lag_weeks: int) -> pd.Series:
    """spike + lag, shifted by last year's week offset between the same two quarters when it is known and plausible."""
    fixed = spikes['date'] + pd.Timedelta(weeks=lag_weeks)
    if lag_weeks % 52 == 0:
        return fixed

    k = lag_weeks // 13
    def last_year_date(quarter):
        return pd.Series(last_year.reindex(pd.MultiIndex.from_arrays([spikes['uspn'], quarter])).to_numpy())

    gap = (last_year_date(spikes['q'] + k - 4) - last_year_date(spikes['q'] - 4)).dt.days.to_numpy()
    offset = gap - 7 * lag_weeks
    shift = np.where(np.abs(offset) <= MAX_SHIFT_DAYS, 7 * np.round(offset / 7), 0)
    return fixed + pd.to_timedelta(np.nan_to_num(shift), unit='D')


def _eday_prob(spikes: pd.DataFrame, last_year: pd.Series, index: pd.MultiIndex, lag_weeks: int) -> pd.Series:
    """Triangular +/- TOL_DAYS pmf around each centre; window dates inside the spike's own quarter are dropped."""
    centre = pd.DatetimeIndex(_centres(spikes, last_year, lag_weeks))
    known_after = spikes['q'].dt.end_time.dt.normalize().to_numpy()
    offsets = np.arange(-TOL_DAYS, TOL_DAYS + 1)
    weights = (TOL_DAYS + 1 - np.abs(offsets)) / (TOL_DAYS + 1) ** 2
    parts = []
    for o, w in zip(offsets, weights):
        date = centre + pd.offsets.BDay(o)
        known = date > known_after
        parts.append(pd.Series(w, index=pd.MultiIndex.from_arrays([date[known], spikes['uspn'][known]])))
    expected = pd.concat(parts)
    return expected.groupby(level=[0, 1]).max().reindex(index, fill_value=0.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')

    idio = df['y'] - df['y'].groupby(dates).transform('median')
    q_max = idio.groupby([uspn, dates.to_period('Q')]).transform('max')
    is_spike = (idio == q_max) & (idio > MIN_Y)
    gap, _ = _overnight_and_intraday(df)
    is_confirmed = is_spike & (gap.abs() / _daily_vol(df) >= GAP_Z)

    def table(mask):
        t = df.index[mask].to_frame(index=False)[['uspn', 'date']]
        return t.assign(q=t['date'].dt.to_period('Q')).reset_index(drop=True)

    spikes, confirmed = table(is_spike), table(is_confirmed)
    # Last year's timing comes from all spikes, also for the confirmed column (ties: first spike of the quarter)
    last_year = spikes.drop_duplicates(['uspn', 'q']).set_index(['uspn', 'q'])['date']

    probs = {lag: _eday_prob(spikes, last_year, df.index, lag) for lag in LAGS_WEEKS}
    probs_conf = {lag: _eday_prob(confirmed, last_year, df.index, lag) for lag in LAGS_WEEKS}
    return pd.DataFrame({
        'h016_eday_prob_13w': probs[13].to_numpy(),
        'h016_eday_prob_sum': sum(probs.values()).to_numpy(),
        'h016_eday_prob_gap_sum': sum(probs_conf.values()).to_numpy(),
    }, index=df.index)
