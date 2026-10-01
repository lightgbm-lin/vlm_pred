"""H006: The stock's volume ratios relative to the market's (idiosyncratic part only) predict y.

Follow-up to H003 (rejected): the market-level columns there had high gain share but inconsistent
year-by-year gains, likely because a date-level feature has only ~250 distinct values per year and
the tree fits date-specific noise. Here only per-stock differences are added.

Mechanism: an idiosyncratic volume shock (stock-specific news, a block trade) tends to revert faster
than a common shock (macro news, volatility regime), which persists with market-wide volatility.
vlm_1_ratio / h001_vlm_5_ratio mix the two; the stock-minus-market difference lets the tree apply a
different persistence to the idiosyncratic part.
Expected: for a given vlm_1_ratio, a larger idiosyncratic component -> lower y_t (more reversion).
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
    out = pd.DataFrame(index=df.index)
    for src, name in [('vlm_1_ratio', 'h006_idio_vlm_1_ratio'), ('h001_vlm_5_ratio', 'h006_idio_vlm_5_ratio')]:
        mkt = _wmean_by_date(df[src], w_lag)  # inputs are already lagged ratios -> known before T
        out[name] = df[src] - mkt.reindex(dates).to_numpy()
    return out
