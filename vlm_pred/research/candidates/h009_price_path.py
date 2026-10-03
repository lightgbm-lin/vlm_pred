"""H009: longer-horizon price path: 20d and 60d cumulative returns and position within the trailing 52-week range.

Mechanism: turnover rises after large past returns (disposition effect and attention trading; Statman, Thorley
& Vorkink 2006 find turnover follows returns over the past weeks to months), and stocks near their 52-week high or
low draw breakout trading, distress selling and anchoring on those levels. The baseline only sees 5 days of
returns, so a one- to three-month horizon and position in the yearly range are new information.
Residual screen of the H007 model: 52w-low proximity and 20d return were the top stock-level signals.
Expected: U-shaped in returns and in 52w position (extremes -> higher y_t), stronger near the 52w low.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level='uspn')['price_adj']
    p1 = g.shift(1)                                     # last close known before t's open
    hi = g.transform(lambda x: x.rolling(252, min_periods=60).max().shift(1))
    lo = g.transform(lambda x: x.rolling(252, min_periods=60).min().shift(1))
    span = (hi - lo).where(hi > lo)

    return pd.DataFrame({
        'h009_ret_20d': p1 / g.shift(21) - 1,
        'h009_ret_60d': p1 / g.shift(61) - 1,
        'h009_pos_52w': (p1 - lo) / span,
    }, index=df.index)
