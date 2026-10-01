"""H005: The size of T-1's price move relative to the stock's own volatility predicts y.

Mechanism (mixture of distributions): volume and |price change| are both driven by information
arrival. vlm_1_ratio says how much traded at T-1 but not whether it moved prices. Volume that came
with a large move (return, intraday range, or overnight gap relative to the stock's normal
volatility) signals genuine news that keeps being digested and traded; heavy volume without a price
move is more likely liquidity/flow trading that reverts.
Expected: higher shock z-scores -> higher y_t, especially conditional on high vlm_1_ratio;
the overnight gap flags news released outside trading hours.
"""
import numpy as np
import pandas as pd

HL = 21  # one month of history for the stock's normal volatility


def compute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level='uspn')
    prev_close = g['price_close'].shift(1)
    log_range = np.log(df['price_high'] / df['price_low'])
    log_gap = np.log(df['price_open'] / prev_close).abs()
    absret = df['ret_raw'].abs()

    # same-day quantities on date t; normalizers are EWMAs through t-1. Shift both by one day below.
    vol = g['ret_raw'].transform(lambda x: x.ewm(halflife=HL, min_periods=10).std().shift(1))
    rng_norm = log_range.groupby(level='uspn').transform(
        lambda x: x.ewm(halflife=HL, min_periods=10).mean().shift(1))

    z = pd.DataFrame({
        'h005_absret_z': absret / vol,
        'h005_range_z': log_range / rng_norm,
        'h005_gap_z': log_gap / vol,
    }, index=df.index).replace([np.inf, -np.inf], np.nan)
    return z.groupby(level='uspn').shift(1)  # only data through t-1
