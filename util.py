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
    w = np.nan_to_num(w, True, 0.0)
    w = np.clip(w, min=0)
    constant = np.ones(len(pred))[:, None]
    x = np.hstack([constant, pred])

    return x, y, w
