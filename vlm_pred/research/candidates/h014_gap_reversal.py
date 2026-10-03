"""H014: gap reversal at t-1: signed geometric mean of the overnight gap and the intraday move, in vol units.

Mechanism: a stock that gaps one way overnight and then trades back the other way during the day shows the
market disagreeing about what the news means. The price hasn't settled, so trading continues the next day. A gap
that keeps going in the same direction is news absorbed and confirmed. H010's signed gap and intraday columns
let trees cut out the reversal quadrant but not the amount reversed, which is the product of the two.
Column: sign(gap * intra) * sqrt(|gap * intra|), both at t-1 in vol_ewm_21 units. Negative means reversal,
positive means continuation; zero when either move is zero.
Expected: more negative (bigger reversal) -> higher y_t, beyond h010_gap_z_1 / h010_intra_z_1.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1                  # split-safe overnight return
    prod = (gap * (intraday - 1)).groupby(level='uspn').shift(1)
    vol = df['vol_ewm_21'].where(df['vol_ewm_21'] > 0)       # return std through t-1
    geo = np.sign(prod) * np.sqrt(prod.abs()) / vol
    return pd.DataFrame({'h014_gap_intra_geo': geo}, index=df.index)
