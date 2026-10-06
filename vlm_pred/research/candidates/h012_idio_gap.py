"""H012: market-relative overnight gap at t-1: stock gap minus the sp_weight-weighted market gap, in vol units.

Mechanism: an overnight move shared with the whole market reflects macro or foreign news (futures moving
overnight). It lifts trading broadly and comparatively little for any single stock. The part of the gap beyond
the market's is company news, which drives the stock's own volume surprise. H010's raw gap mixes the two.
This is a stock-level column (no date-level market column) to avoid H001's date-memorization problem; beta is
fixed at 1.
Expected: larger |idio gap z| -> higher y_t, beyond gap_z_1.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1                  # split-safe overnight return on day t
    w = df['sp_weight'].where(gap.notna())
    mkt_gap = (gap * w).groupby(dates).sum() / w.groupby(dates).sum().replace(0, np.nan)
    idio = (gap - mkt_gap.reindex(dates).to_numpy()).rename('idio')   # same-day; lagged below
    idio_1 = idio.groupby(level='uspn').shift(1)
    vol = df['vol_ewm_21'].where(df['vol_ewm_21'] > 0)       # return std through t-1
    return pd.DataFrame({'h012_gap_idio_z_1': idio_1 / vol}, index=df.index)
