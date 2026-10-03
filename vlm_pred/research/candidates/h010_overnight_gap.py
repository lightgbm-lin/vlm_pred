"""H010: split of t-1's return into overnight gap and intraday move, each in units of the stock's daily vol.

Mechanism: the overnight part of a return reflects news released outside trading hours (company announcements,
analyst actions, macro or foreign news), so a big gap marks a discrete information event, and the volume that
follows stays elevated as traders re-price. A large intraday move without a gap is more often order-flow driven,
with less follow-through. ret_1 mixes the two, and price_open isn't used by any feature so far. (H002's high-low
range was redundant, but a gap measures when the move happened, not the day's range.)
Residual screen of the H007 model: |gap_1|/vol explained .0011 of residual variance.
Expected: larger |gap_z_1| -> higher y_t, beyond what |ret_1| implies; intraday moves weaker.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1                  # split-safe overnight return
    parts = pd.DataFrame({'gap': gap, 'intra': intraday - 1}, index=df.index)
    lagged = parts.groupby(level='uspn').shift(1)
    vol = df['vol_ewm_21'].where(df['vol_ewm_21'] > 0)       # return std through t-1

    return pd.DataFrame({
        'h010_gap_z_1': lagged['gap'] / vol,
        'h010_intra_z_1': lagged['intra'] / vol,
    }, index=df.index)
