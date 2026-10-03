import statsmodels.api as sm
from vlm_pred.util import to_x_y_w


def evaluate(pred, y, w, mask=None):
    if mask is not None:
        pred = pred[mask]
        y = y[mask]
        w = w[mask]

    x, y, w = to_x_y_w(pred, y, w)

    model = sm.WLS(y, x, weights=w).fit()

    return {'r-squared': model.rsquared.item(), 't-stat': model.tvalues[1].item()}
