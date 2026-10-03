"""H004: H003's quarterly earnings schedule with spikes defined by a trailing max (known on the spike day).

Mechanism: as H003 — firms report on a ~13-week cycle, so a volume spike 13/26/39 weeks ago predicts an
earnings day at t. H003 defines a spike as a calendar quarter's max y, which is only known at quarter end, so
it must drop 13-week windows landing in the spike's own quarter. Here a spike is a day with y > 1 that is the
highest y of the trailing 63 trading days (about a quarter), which is known on the spike day itself: no window
needs dropping, and spikes are not tied to calendar-quarter boundaries (an early-quarter earnings day still
counts when a bigger event follows later in the same quarter).
Expected: as H003 — higher prob_13w and consensus -> higher y_t, convex; similar or slightly better than H003.
"""
import numpy as np
import pandas as pd

LAGS_WEEKS = (13, 26, 39)
WINDOW = 63          # trading days, about one quarter


def _eday_prob(df: pd.DataFrame, lag_days: int, tol_days: int = 2, min_y: float = 1.0) -> pd.Series:
    """Triangular +/- tol_days pmf centred lag_days after each trailing-max spike, as in enrich_earnings_day."""
    y = df['y']
    trail_max = y.groupby(level='uspn').transform(lambda x: x.rolling(WINDOW, min_periods=1).max())
    spikes = df.index[(y == trail_max) & (y > min_y)]

    s_dates = spikes.get_level_values('date')
    s_uspn = spikes.get_level_values('uspn')
    centre = s_dates + pd.Timedelta(days=lag_days)
    offsets = np.arange(-tol_days, tol_days + 1)
    weights = (tol_days + 1 - np.abs(offsets)) / (tol_days + 1) ** 2
    expected = pd.concat([pd.Series(w, index=pd.MultiIndex.from_arrays([centre + pd.offsets.BDay(o), s_uspn]))
                          for o, w in zip(offsets, weights)])
    return expected.groupby(level=[0, 1]).max().reindex(df.index, fill_value=0.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    probs = {lag: _eday_prob(df, lag_days=7 * lag) for lag in LAGS_WEEKS}
    return pd.DataFrame({
        'h004_eday_prob_13w': probs[13].to_numpy(),
        # Consensus over the three quarterly lags and the baseline's 52-week lag
        'h004_eday_prob_sum': (sum(probs.values()) + df['earnings_day_prob']).to_numpy(),
    }, index=df.index)
