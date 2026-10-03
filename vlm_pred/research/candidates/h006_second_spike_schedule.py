"""H006: earnings-day probability from each quarter's second-largest idiosyncratic volume spike (low-confidence column).

Mechanism: H005 takes one spike per quarter, the largest idiosyncratic y. When that day was not the earnings day
(merger news, guidance warning, investor day), the earnings day is usually the runner-up, and it recurs on the
quarterly cycle like any earnings day. The second spike must be > 5 trading days from the quarter's top day
(about the time one event's volume takes to decay), so it is a separate event rather than the top spike's
follow-through day, which the +/- 2 day tolerance already covers. It gets its own consensus column (13/26/39/52w
lags) so the model can weight it below H005's top-spike consensus.
Pre-test diagnostic: second spikes recur on-cycle (13w/52w) 14%/14% vs 11%/11% at off-cycle placebo lags
(6w/19w), compared with 17%/19% vs 7%/9% for top spikes: a weak schedule signal, but ~1.7k big days are reached
only by second spikes.
Expected: higher prob2_sum -> higher y_t, much flatter than H005's consensus; small incremental gain.
"""
import numpy as np
import pandas as pd

LAGS_WEEKS = (13, 26, 39, 52)
MIN_Y = 1.0
SAME_EVENT_DAYS = 5


def _eday_prob(spikes: pd.MultiIndex, index: pd.MultiIndex, lag_days: int, tol_days: int = 2) -> pd.Series:
    """Triangular +/- tol_days pmf centred lag_days after each spike, as in H003/H005.

    Spikes depend on the whole quarter, known only at quarter end, so window dates inside the spike's quarter
    are dropped.
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
    quarter = dates.to_period('Q')

    idio = df['y'] - df['y'].groupby(dates).transform('median')
    q_max = idio.groupby([uspn, quarter]).transform('max')

    # Trading-day position within each stock, to exclude the top spike's own event window
    pos = pd.Series(df.groupby(level='uspn').cumcount().to_numpy(), index=df.index)
    pos_top = pos.where(idio == q_max).groupby([uspn, quarter]).transform('min')
    idio2 = idio.mask((pos - pos_top).abs() <= SAME_EVENT_DAYS)
    q_max2 = idio2.groupby([uspn, quarter]).transform('max')
    second = df.index[(idio2 == q_max2) & (idio2 > MIN_Y)]

    probs = [_eday_prob(second, df.index, 7 * lag) for lag in LAGS_WEEKS]
    return pd.DataFrame({'h006_eday_prob2_sum': sum(probs).to_numpy()}, index=df.index)
