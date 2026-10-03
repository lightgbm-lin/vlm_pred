"""H013: peer spillovers: return-correlation peers' volume surprise yesterday and earnings probability today.

Mechanism: stocks that move together share news. A competitor's event or a sector story lifts trading in its
peers (sector rotation, pair and relative-value trading), and when a peer reports earnings investors re-price
the rest of the industry (intra-industry information transfer; Foster 1981). There is no industry data in the
spine, so peers are each stock's 10 most correlated stocks by daily returns over the prior 252 trading days,
recomputed at the start of each quarter from returns before that quarter (point-in-time). Peer features vary
by stock, unlike H001's date-level market columns.
Pre-test (residual of the H010 model, top vs bottom quintile within y_1 terciles): peer y_1 +.01 to +.03,
peer earnings prob today +.01 to +.04. Small but survives the y_1 control.
Expected: higher peer_y_1 and peer_eday_prob_0 -> higher y_t; small gain (~+.001).
"""
import numpy as np
import pandas as pd

K = 10
WINDOW = 252
MIN_OBS = 120


def _peer_matrices(df: pd.DataFrame) -> dict:
    """{quarter: (uspns, K-nearest peer matrix)} from returns strictly before the quarter starts."""
    R = df['ret_raw'].unstack('uspn')
    dates = R.index
    out = {}
    for q in pd.PeriodIndex(dates, freq='Q').unique():
        start = dates.searchsorted(q.start_time)
        hist = R.iloc[max(0, start - WINDOW):start]
        ok = hist.notna().sum() >= MIN_OBS
        if start < MIN_OBS or ok.sum() < K + 1:
            continue
        C = hist.loc[:, ok].corr(min_periods=MIN_OBS).to_numpy().copy()
        np.fill_diagonal(C, -np.inf)
        C = np.nan_to_num(C, nan=-np.inf)
        idx = np.argsort(-C, axis=1, kind='stable')[:, :K]
        P = np.zeros_like(C)
        np.put_along_axis(P, idx, 1.0, axis=1)
        P[~np.isfinite(np.take_along_axis(C, idx, axis=1)).all(axis=1)] = 0
        out[q] = (hist.columns[ok], P)
    return out


def _peer_mean(index: pd.MultiIndex, wide: pd.DataFrame, mats: dict) -> pd.Series:
    """Mean of `wide` (date x uspn) over each stock's peers for that date's quarter, ignoring missing peers."""
    dates = wide.index
    parts = []
    for q, (cols, P) in mats.items():
        dq = dates[(dates >= q.start_time) & (dates <= q.end_time)]
        if len(dq) == 0:
            continue
        V = wide.reindex(index=dq, columns=cols).to_numpy()
        num = np.nan_to_num(V) @ P.T
        den = (~np.isnan(V)).astype(float) @ P.T
        parts.append(pd.DataFrame(num / np.where(den > 0, den, np.nan), index=dq, columns=cols).stack())
    s = pd.concat(parts)
    s.index.names = ['date', 'uspn']
    return s.reindex(index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    mats = _peer_matrices(df)
    y_prev = df['y'].unstack('uspn').shift(1)                 # peers' y on the previous date
    eday = df['h007_eday_prob_sum'].unstack('uspn')            # peers' earnings prob today, known pre-open
    return pd.DataFrame({
        'h013_peer_y_1': _peer_mean(df.index, y_prev, mats).to_numpy(),
        'h013_peer_eday_prob_0': _peer_mean(df.index, eday, mats).to_numpy(),
    }, index=df.index)
