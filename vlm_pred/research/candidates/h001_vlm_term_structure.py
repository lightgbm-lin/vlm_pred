"""H001: Medium-horizon volume ratios (5- and 21-day halflife EWMA vs the 60-day EWMA) predict y.

Mechanism: volume has a slow-moving regime component (news cycles, volatility regimes, index/fund
flows) that persists for weeks. The 1-day-halflife ratio mostly reflects the last 1-3 days and is
dominated by transient spikes that mean-revert; the 60-day EWMA lags a regime shift. A weekly and a
monthly ratio tell the model whether an elevated last day sits on top of an elevated level (persists)
or is an isolated spike (reverts).
Expected: higher ratios -> higher y, roughly monotone; weekly/monthly ratio should be more persistent
per unit than vlm_1_ratio.
"""
import pandas as pd

from vlm_pred.feature import enrich_vlm_ratio


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out = enrich_vlm_ratio(df, [5, 21])  # calc_vlm_ewma shifts by 1 -> data through t-1
    out.columns = ['h001_vlm_5_ratio', 'h001_vlm_21_ratio']
    return out
