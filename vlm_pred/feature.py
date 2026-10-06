import pandas as pd
import numpy as np
from collections.abc import Sequence
from vlm_pred.util import calc_ewm


def enrich_vlm_ratio(df: pd.DataFrame, hls: Sequence[int] = (1, 5, 10, 21)) -> pd.DataFrame:
    vlm_ewm = pd.concat([calc_ewm(df, halflife=hl, column='volume') for hl in hls], axis=1)
    vlm_pred_naive = df['vlm_pred_naive'].mask(df['vlm_pred_naive']<=0)

    vlm_ratio = vlm_ewm.divide(vlm_pred_naive, axis=0) - 1
    vlm_ratio.columns = [f'vlm_{hl}_ratio' for hl in hls]

    return vlm_ratio


def enrich_vol_ewm(df: pd.DataFrame, hls: Sequence[int] = (1, 5, 10, 21)) -> pd.DataFrame:
    vol_ewm_df = pd.concat([df.groupby('uspn')['ret_raw'].transform(lambda x: x.ewm(halflife=hl).std().shift(1)) for hl in hls], axis=1)
    vol_ewm_df.columns = [f'vol_ewm_{hl}' for hl in hls]

    return vol_ewm_df


def enrich_lagged_ret(df: pd.DataFrame, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    lagged_rets = pd.concat(
        [df.groupby('uspn')['ret_raw'].shift(lag) for lag in lags], axis=1)
    lagged_rets.columns = [f'ret_{lag}' for lag in lags]

    return lagged_rets


def enrich_lagged_targets(df: pd.DataFrame, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    lagged_targets = pd.concat(
        [df.groupby('uspn')['y'].shift(lag) for lag in lags], axis=1)
    lagged_targets.columns = [f'y_{lag}' for lag in lags]

    return lagged_targets


def enrich_max_targets(df: pd.DataFrame, lags: Sequence[int] = (5, 10, 20)) -> pd.DataFrame:
    max_targets = pd.concat(
        [df.groupby('uspn')['y'].transform(lambda x: x.rolling(lag).max().shift(1)) for lag in lags], axis=1)
    max_targets.columns = [f'max_y_{lag}' for lag in lags]

    return max_targets


def enrich_earnings_day(df: pd.DataFrame, lag_days: int = 364, tol_days: int = 2, min_y: float = 1.0) -> pd.DataFrame:
    """Probability that a date is an earnings day, from last year's volume spikes, point-in-time.

    A stock's earnings reaction day in a calendar quarter is taken to be its biggest y day, if volume was at least
    (1 + min_y)x normal. Firms report in the same week each year, so such a day lag_days (52 weeks, which keeps the
    weekday) before t predicts an earnings day at t. The prediction is spread over +/- tol_days weekdays as a
    triangular pmf that peaks at exactly lag_days and falls linearly towards the ends (holidays keep their share,
    so a window with a holiday sums to a little under 1). Finding a quarter's max needs the whole
    quarter, which ends well before the date the spike is used for, so there is no look-ahead.
    """
    y = df['y']
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')

    q_max = y.groupby([uspn, dates.to_period('Q')]).transform('max')
    spikes = df.index[(y == q_max) & (y > min_y)]

    s_dates = spikes.get_level_values('date')
    s_uspn = spikes.get_level_values('uspn')
    # Spikes are on trading days, so the 52-week-later centre is a weekday too
    centre = s_dates + pd.Timedelta(days=lag_days)
    offsets = np.arange(-tol_days, tol_days + 1)
    # Triangular pmf: weights tol+1-|o| scaled to sum to 1, so the ends get 1/(tol+1)^2 of the mass
    weights = (tol_days + 1 - np.abs(offsets)) / (tol_days + 1) ** 2
    expected = pd.concat([pd.Series(w, index=pd.MultiIndex.from_arrays([centre + pd.offsets.BDay(o), s_uspn]))
                          for o, w in zip(offsets, weights)])
    # Windows of spikes that tie for a quarter's max can overlap; keep the higher probability
    prob = expected.groupby(level=['date', 'uspn']).max().reindex(df.index, fill_value=0.0)

    return pd.DataFrame({'earnings_day_prob': prob.to_numpy()}, index=df.index)


def enrich_earnings_schedule(df: pd.DataFrame, lags_weeks: Sequence[int] = (13, 26, 39, 52), tol_days: int = 2,
                             min_y: float = 1.0, gap_z: float = 2.0, shift_days: int = 7,
                             shift_share: float = 0.5) -> pd.DataFrame:
    """Probability that a date is an earnings day, projected from the last four quarters' volume spikes.

    A spike is a stock's biggest idiosyncratic y (y net of the day's cross-sectional median) in a calendar quarter,
    if above min_y. A confirmed spike also gapped overnight by at least gap_z daily vols. Firms report on a
    quarterly cycle, so a spike lag weeks (which keeps the weekday) before t predicts an earnings day at t, spread
    over +/- tol_days business days as in enrich_earnings_day.

    Dec-year-end firms report Q4 with the annual report, in calendar Q1, about a week later in the cycle than the
    interim reports. A projection into calendar Q1 from another quarter therefore puts shift_share of its mass one
    week later, and one out of Q1 shift_share of its mass one week earlier.

    Columns: the nearest lag's probability, the sum over lags, and the sum over lags from confirmed spikes only.
    """
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')

    idio = df['y'] - df['y'].groupby(dates).transform('median')
    q_max = idio.groupby([uspn, dates.to_period('Q')]).transform('max')
    is_spike = (idio == q_max) & (idio > min_y)

    gap, _ = _overnight_and_intraday(df)
    is_confirmed = is_spike & (gap.abs() / _daily_vol(df) >= gap_z)

    def schedule(spikes: pd.MultiIndex) -> dict[int, pd.Series]:
        return {lag: _earnings_schedule_prob(spikes, df.index, 7 * lag, tol_days, shift_days, shift_share)
                for lag in lags_weeks}

    probs = schedule(df.index[is_spike])
    probs_confirmed = schedule(df.index[is_confirmed])

    return pd.DataFrame({
        f'eday_prob_{lags_weeks[0]}w': probs[lags_weeks[0]].to_numpy(),
        'eday_prob_sum': sum(probs.values()).to_numpy(),
        'eday_prob_gap_sum': sum(probs_confirmed.values()).to_numpy(),
    }, index=df.index)


def _earnings_schedule_prob(spikes: pd.MultiIndex, index: pd.MultiIndex, lag_days: int, tol_days: int,
                            shift_days: int, shift_share: float) -> pd.Series:
    """Triangular pmf around each spike + lag_days, with shift_share moved a week when the projection crosses Q1."""
    s_dates = spikes.get_level_values('date')
    centre = s_dates + pd.Timedelta(days=lag_days)
    proj = pd.DataFrame({
        'uspn': spikes.get_level_values('uspn'),
        'centre': centre,
        # A quarter's max is only known once the quarter ends
        'known_after': s_dates.to_period('Q').end_time.normalize(),
        'mass': 1.0,
    })

    # +shift_days into calendar Q1, -shift_days out of it, 0 otherwise
    shift = shift_days * ((centre.quarter == 1).astype(int) - (s_dates.quarter == 1).astype(int))
    crosses = shift != 0
    shifted = proj[crosses].assign(centre=proj['centre'][crosses] + pd.to_timedelta(shift[crosses], unit='D'),
                                   mass=shift_share)
    proj.loc[crosses, 'mass'] = 1 - shift_share
    proj = pd.concat([proj, shifted], ignore_index=True)

    offsets = np.arange(-tol_days, tol_days + 1)
    weights = (tol_days + 1 - np.abs(offsets)) / (tol_days + 1) ** 2
    parts = []
    for o, w in zip(offsets, weights):
        date = pd.DatetimeIndex(proj['centre']) + pd.offsets.BDay(o)
        known = date > proj['known_after'].to_numpy()
        parts.append(pd.Series(w * proj['mass'][known].to_numpy(),
                               index=pd.MultiIndex.from_arrays([date[known], proj['uspn'][known]])))
    expected = pd.concat(parts)
    # A spike's two triangles are a week apart and never overlap; across spikes, keep the higher probability
    return expected.groupby(level=[0, 1]).max().reindex(index, fill_value=0.0)


def enrich_price_path(df: pd.DataFrame, ret_days: Sequence[int] = (20, 60), range_days: int = 252) -> pd.DataFrame:
    """Longer-horizon price path through t-1: cumulative returns and position within the trailing 52-week range.

    Turnover rises after large past returns (disposition effect, attention trading), and stocks near their 52-week
    high or low draw breakout trading, distress selling and anchoring on those levels. pos_52w is 0 at the range's
    low and 1 at its high, NaN with fewer than 60 days of history or a flat range.
    """
    price = df.groupby(level='uspn')['price_adj']
    last_close = price.shift(1)
    hi = price.transform(lambda x: x.rolling(range_days, min_periods=60).max().shift(1))
    lo = price.transform(lambda x: x.rolling(range_days, min_periods=60).min().shift(1))

    feats = pd.DataFrame({f'ret_{n}d': last_close / price.shift(n + 1) - 1 for n in ret_days}, index=df.index)
    feats['pos_52w'] = (last_close - lo) / (hi - lo).where(hi > lo)
    return feats


def enrich_overnight_gap(df: pd.DataFrame) -> pd.DataFrame:
    """t-1's return split into the overnight gap and the intraday move, each in units of the stock's daily vol.

    A big overnight gap marks news released outside trading hours, and the volume that follows stays elevated as
    traders re-price; a large intraday move without a gap is more often order-flow driven, with less follow-through.
    """
    gap, intraday = _overnight_and_intraday(df)
    lagged = pd.DataFrame({'gap': gap, 'intraday': intraday}).groupby(level='uspn').shift(1)
    vol = _daily_vol(df)

    return pd.DataFrame({
        'gap_z_1': lagged['gap'] / vol,
        'intra_z_1': lagged['intraday'] / vol,
    }, index=df.index)


def _overnight_and_intraday(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Same-day overnight and intraday returns. The gap is split-safe: close-to-close total return net of intraday."""
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1
    return gap, intraday - 1


def _daily_vol(df: pd.DataFrame) -> pd.Series:
    """Daily return std through t-1 (vol_ewm_21), NaN where it is not positive."""
    vol = enrich_vol_ewm(df, hls=(21,))['vol_ewm_21']
    return vol.where(vol > 0)


def _to_day(x) -> np.ndarray:
    return np.asarray(x, dtype="datetime64[D]")


def enrich_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features per date, all point-in-time.

    Features from df's dates only look at the current and earlier dates, because a date's presence in df
    is not known in advance; month end and option expiry treat every weekday as a session instead.
    """

    d = pd.DatetimeIndex(df.index.get_level_values('date')).normalize()
    cal = d.unique().sort_values()
    prev_td = pd.Series(cal, index=cal).shift(1)

    f = pd.DataFrame(index=cal)
    f["dow"] = cal.weekday
    f["month"] = cal.month
    f["is_month_start"] = cal.to_period("M") != prev_td.dt.to_period("M")
    # Nth occurrence of this weekday in the month (1-5); with dow and month this locates e.g. the 3rd Friday
    f["week_of_month"] = (cal.day - 1) // 7 + 1
    # Holidays inferred from gaps since the previous date
    f["is_post_holiday"] = np.busday_count(_to_day(prev_td.fillna(cal[0])), _to_day(cal)) > 1

    # Month end assumes the next weekday is a session, so it misses the few months whose last weekday is a holiday
    per = cal.to_period("M")
    next_wd = pd.DatetimeIndex(np.busday_offset(_to_day(cal), 1, roll="forward"))
    f["is_month_end"] = per != next_wd.to_period("M")
    f["is_quarter_end"] = f["is_month_end"] & cal.month.isin([3, 6, 9, 12])
    f["is_msci_review"] = f["is_month_end"] & cal.month.isin([2, 5, 8, 11])

    # Monthly expiry: third Friday (when it's a holiday, expiry moves to Thursday, which this misses)
    f["is_opex"] = (cal.weekday == 4) & (f["week_of_month"] == 3)
    f["is_quad_witch"] = f["is_opex"] & cal.month.isin([3, 6, 9, 12])

    feats = f.reindex(d).astype(int)
    feats.index = df.index
    return feats
