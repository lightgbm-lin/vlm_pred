import numpy as np


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
