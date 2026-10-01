"""H003: The market-wide volume surprise at T-1, and the stock's deviation from it, predict y.

Mechanism: much of the daily volume surprise is common (macro news, FOMC, index flows, volatility
spikes). Common and idiosyncratic volume shocks decay differently: market-wide shocks are driven by
persistent volatility regimes, while an idiosyncratic spike (news on one name) tends to revert
faster. vlm_1_ratio mixes the two; splitting it lets the model apply a different persistence to each.
The market's own y at T-1 also carries the freshest reading of the common factor.
Expected: higher market y_{t-1} / market ratio -> higher y_t; for a given vlm_1_ratio, a larger
idiosyncratic share -> lower y_t (more reversion).
"""
import numpy as np
import pandas as pd


def _wmean_by_date(x: pd.Series, w: pd.Series) -> pd.Series:
    ok = x.notna() & w.notna()
    num = (x.where(ok) * w.where(ok)).groupby(level='date').sum()
    den = w.where(ok).groupby(level='date').sum()
    return num / den.replace(0, np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    w_lag = df.groupby(level='uspn')['sp_weight'].shift(1)  # weights known at T-1

    # market y on date t-1: aggregate same-day y at each date, then shift by one date
    mkt_y = _wmean_by_date(df['y'], df['sp_weight']).shift(1)
    # market level of the (already lagged) 1-day volume ratio at T
    mkt_r1 = _wmean_by_date(df['vlm_1_ratio'], w_lag)

    out = pd.DataFrame(index=df.index)
    out['h003_mkt_y_1'] = mkt_y.reindex(dates).to_numpy()
    out['h003_mkt_vlm_1_ratio'] = mkt_r1.reindex(dates).to_numpy()
    out['h003_idio_vlm_1_ratio'] = df['vlm_1_ratio'] - out['h003_mkt_vlm_1_ratio']
    return out
