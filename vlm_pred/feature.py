import pandas as pd
import numpy as np
from vlm_pred.util import calc_ewm


def enrich_vlm_ratio(df: pd.DataFrame, hls: list[int]) -> pd.DataFrame:
    vlm_ewm = pd.concat([calc_ewm(df, halflife=hl, column='volume') for hl in hls], axis=1)
    vlm_pred_naive = df['vlm_pred_naive'].mask(df['vlm_pred_naive']<=0)

    vlm_ratio = vlm_ewm.divide(vlm_pred_naive, axis=0) - 1
    vlm_ratio.columns = [f'vlm_{hl}_ratio' for hl in hls]

    return vlm_ratio


def enrich_vol_ewm(df: pd.DataFrame, hls: list[int]) -> pd.DataFrame:
    vol_ewm_df = pd.concat([df.groupby('uspn')['ret_raw'].transform(lambda x: x.ewm(halflife=hl).std().shift(1)) for hl in hls], axis=1)
    vol_ewm_df.columns = [f'vol_ewm_{hl}' for hl in hls]

    return vol_ewm_df


def enrich_lagged_ret(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    lagged_rets = pd.concat(
        [df.groupby('uspn')['ret_raw'].shift(lag) for lag in lags], axis=1)
    lagged_rets.columns = [f'ret_{lag}' for lag in lags]

    return lagged_rets


def enrich_lagged_targets(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    lagged_targets = pd.concat(
        [df.groupby('uspn')['y'].shift(lag) for lag in lags], axis=1)
    lagged_targets.columns = [f'y_{lag}' for lag in lags]

    return lagged_targets


def enrich_max_targets(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    max_targets = pd.concat(
        [df.groupby('uspn')['y'].transform(lambda x: x.rolling(lag).max().shift(1)) for lag in lags], axis=1)
    max_targets.columns = [f'max_y_{lag}' for lag in lags]

    return max_targets


def enrich_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    level = 'date'
    d = pd.DatetimeIndex(df.index.get_level_values(level)).normalize()
    cal = d.unique().sort_values()  # trading calendar from df itself
    s = pd.Series(cal, index=cal)
    prev_td, next_td = s.shift(1), s.shift(-1)
    per = cal.to_period("M")

    f = pd.DataFrame(index=cal)
    f["dow"] = cal.weekday
    f["month"] = cal.month
    f["is_month_end"] = per != next_td.dt.to_period("M")
    f["is_month_start"] = per != prev_td.dt.to_period("M")
    f["is_quarter_end"] = f["is_month_end"] & cal.month.isin([3, 6, 9, 12])
    f["is_msci_review"] = f["is_month_end"] & cal.month.isin([2, 5, 8, 11])

    # Monthly expiry: third Friday, rolled back to the prior trading day if it's a holiday
    fifteenth = per.to_timestamp() + pd.Timedelta(days=14)
    third_fri = fifteenth + pd.to_timedelta((4 - fifteenth.weekday) % 7, unit="D")
    pos = cal.searchsorted(third_fri, side="right") - 1
    f["is_opex"] = (pos >= 0) & (cal == cal[np.clip(pos, 0, None)])
    f["is_quad_witch"] = f["is_opex"] & cal.month.isin([3, 6, 9, 12])
    f["is_russell_recon"] = (cal.month == 6) & (cal.weekday == 4) & (cal.day >= 22) & (cal.day <= 28)

    # Holidays inferred from gaps in the calendar
    D = lambda x: x.values.astype("datetime64[D]")
    f["is_post_holiday"] = np.busday_count(D(prev_td.fillna(cal[0])), D(s)) > 1
    f["is_pre_holiday"] = np.busday_count(D(s), D(next_td.fillna(cal[-1]))) > 1

    # Half days
    f["is_half_day"] = (
            ((cal.month == 11) & (cal.weekday == 4) & (cal.day >= 23) & (cal.day <= 29))
            | ((cal.month == 12) & (cal.day == 24))
            | ((cal.month == 7) & (cal.day == 3))
    )

    feats = f.reindex(d).replace(np.nan, False).astype(int)
    feats.index = df.index
    return feats
