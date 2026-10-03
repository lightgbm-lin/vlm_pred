"""H001: market-wide volume surprise (sp_weight-weighted mean y across stocks) at t-1, and its idiosyncratic residual.

Mechanism: a stock's volume surprise mixes a common component (macro news, volatility regime, index-level
flows) with a stock-specific one (company news). The common part is persistent (volatility regimes last weeks)
while idiosyncratic spikes decay within days, so splitting y_1 into market and residual lets the model apply
different persistence to each.
Expected: higher market y_{t-1} (and its 5-day EWMA) -> higher y_t; idiosyncratic y_1 mean-reverts faster
than the market part.
"""
import numpy as np
import pandas as pd


def _by_date_shifted(s: pd.Series) -> pd.Series:
    """Shift a per-date series by one date (not row), so date t only sees dates < t."""
    return s.sort_index().shift(1)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    w = df['sp_weight'].where(df['y'].notna())
    num = (df['y'] * w).groupby(dates).sum()
    den = w.groupby(dates).sum()
    mkt_y = (num / den.mask(den <= 0))                      # market y on date t (uses same-day data)

    mkt_y_1 = _by_date_shifted(mkt_y)                       # known before t's open
    mkt_y_ewm5 = mkt_y.ewm(halflife=5).mean().shift(1)

    out = pd.DataFrame(index=df.index)
    out['h001_mkt_y_1'] = mkt_y_1.reindex(dates).to_numpy()
    out['h001_mkt_y_ewm5'] = mkt_y_ewm5.reindex(dates).to_numpy()
    out['h001_idio_y_1'] = df['y_1'] - out['h001_mkt_y_1']
    return out
