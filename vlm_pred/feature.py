import pandas as pd
import numpy as np
from collections.abc import Sequence
from vlm_pred import config
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


def _to_day(x) -> np.ndarray:
    return np.asarray(x, dtype="datetime64[D]")


def enrich_calendar_features(df: pd.DataFrame, use_exchange_calendar: bool | None = None) -> pd.DataFrame:
    """Calendar features per date, all point-in-time.

    Features from df's dates only look at the current and earlier dates, because a date's presence in df
    is not known in advance; month end and option expiry treat every weekday as a session instead. Features
    that need the holiday schedule come from the XNYS calendar in exchange_calendars and are skipped when
    use_exchange_calendar is False (default: config.USE_EXCHANGE_CALENDAR).
    """
    if use_exchange_calendar is None:
        use_exchange_calendar = config.USE_EXCHANGE_CALENDAR

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

    if use_exchange_calendar:
        f = f.join(_exchange_calendar_features(cal))

    feats = f.reindex(d).astype(int)
    feats.index = df.index
    return feats


def _exchange_calendar_features(cal: pd.DatetimeIndex) -> pd.DataFrame:
    import exchange_calendars as xcals  # imported here so the package is only needed when it is used

    nyse = xcals.get_calendar("XNYS", start=cal[0] - pd.Timedelta(days=31), end=cal[-1] + pd.Timedelta(days=60))
    # Ad hoc closures (9/11, Hurricane Sandy, state funerals) were not known well in advance, so they count
    # as sessions when looking ahead
    adhoc = pd.DatetimeIndex(nyse.adhoc_holidays)
    sched = nyse.sessions.union(adhoc[(adhoc >= nyse.sessions[0]) & (adhoc <= nyse.sessions[-1])])
    next_td = sched[sched.searchsorted(cal, side="right")]

    f = pd.DataFrame(index=cal)
    f["is_pre_holiday"] = np.busday_count(_to_day(cal), _to_day(next_td)) > 1
    # Early closes (scheduled and announced ad hoc ones such as 2003-12-26)
    f["is_half_day"] = cal.isin(nyse.early_closes)
    return f
