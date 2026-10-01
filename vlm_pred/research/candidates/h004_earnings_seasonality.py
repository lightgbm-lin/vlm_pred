"""H004: A volume spike ~1 quarter or ~1 year before T predicts a spike at T (earnings seasonality).

Mechanism: earnings announcements are the largest recurring idiosyncratic volume events and are
spaced ~63 trading days apart; firms report in nearly the same calendar week each year, so the
~252-day echo is the most precise, the ~63-day echo the most recent. Neither the EWMA ratios nor the
calendar features can anticipate a stock-specific scheduled event.
Expected: large y around t-63 / t-252 -> higher y_t; the tight (+-1 day) window sharper than the wide
(+-6 day) window, which captures drift in the reporting date.
"""
import pandas as pd


def _lag_window_max(g, lo: int, hi: int) -> pd.Series:
    """max of y over lags lo..hi (inclusive), per stock; lo >= 1 so only data through t-1."""
    return g.transform(lambda x: x.shift(lo).rolling(hi - lo + 1, min_periods=1).max())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level='uspn')['y']
    return pd.DataFrame({
        'h004_y_q_tight': _lag_window_max(g, 62, 64),
        'h004_y_q_wide': _lag_window_max(g, 57, 69),
        'h004_y_yr_tight': _lag_window_max(g, 251, 253),
        'h004_y_yr_wide': _lag_window_max(g, 245, 259),
    }, index=df.index)
