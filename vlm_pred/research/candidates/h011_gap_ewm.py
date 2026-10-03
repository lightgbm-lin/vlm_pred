"""H011: overnight news flow over about a week: EWMA (halflife 5d) of |overnight gap| in units of daily vol.

Mechanism: a stock that keeps gapping overnight is "in the news" (an ongoing M&A story, litigation, guidance
revisions, a sector story). Each new item draws traders back, so volume stays elevated while the story runs.
H010 showed the overnight/intraday split of ret_1 carries information beyond ret_1; one day's gap is a noisy
measure of the news state, and an EWMA over about a week (halflife 5, the life of a running story) measures it
better.
Data check: no row has its open outside [low, high], and only 17 rows have |gap| > 20 sigma, so no cleaning.
Expected: higher gap_absz_ewm5 -> higher y_t, beyond h010_gap_z_1.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    intraday = df['price_close'] / df['price_open'].where(df['price_open'] > 0)
    gap = (1 + df['ret_raw']) / intraday - 1                  # split-safe overnight return on day t
    vol = df['vol_ewm_21'].where(df['vol_ewm_21'] > 0)       # return std through t-1
    absz = (gap.abs() / vol).rename('absz')                   # day-t gap in units of vol known before t
    ewm = absz.groupby(level='uspn').transform(lambda x: x.ewm(halflife=5).mean().shift(1))
    return pd.DataFrame({'h011_gap_absz_ewm5': ewm}, index=df.index)
