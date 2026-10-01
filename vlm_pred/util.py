import numpy as np
import pandas as pd


def to_2d_array(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:,None]
    return a


def to_x_y_w(pred, y, w):
    pred = to_2d_array(pred)
    y = to_2d_array(y)
    w = to_2d_array(w)
    constant = np.ones(len(pred))[:, None]
    x = np.hstack([constant, pred])
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1) & np.isfinite(w).all(axis=1) & (w >= 0).all(axis=1)

    return x[mask], y[mask], w[mask]


def calc_ewm(df: pd.DataFrame, halflife: int, column: str):
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()

    g = df.groupby(level='uspn')[column]

    return g.transform(lambda x: x.ewm(halflife=halflife).mean().shift(1))
