"""H002: intraday high-low range at t-1 (and its 5-day EWMA) relative to the stock's own 60-day range norm.

Mechanism: under the mixture-of-distributions hypothesis volume and volatility both scale with the rate of
information arrival. The high-low range is a far more efficient volatility estimator than close-to-close
returns, and it registers days with large intraday disagreement or reversals that leave |ret_1| small.
Such activity is persistent, so an abnormally wide range at t-1 signals above-normal volume at t.
Expected: higher range_1_rel / range_5_rel -> higher y_t, saturating.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    hi, lo = df['price_high'], df['price_low']
    rng = np.log(hi.where(hi > 0) / lo.where(lo > 0))       # same-day range, split-invariant
    g = rng.groupby(level='uspn')
    range_1 = g.shift(1)
    range_ewm5 = g.transform(lambda x: x.ewm(halflife=5).mean().shift(1))
    range_ewm60 = g.transform(lambda x: x.ewm(halflife=60).mean().shift(1))
    norm = range_ewm60.where(range_ewm60 > 0)

    return pd.DataFrame({
        'h002_range_1_rel': range_1 / norm - 1,
        'h002_range_5_rel': range_ewm5 / norm - 1,
    }, index=df.index)
