import statsmodels.api as sm
from util import to_x_y_w


def evaluate(pred, y, w):
    x, y, w = to_x_y_w(pred, y, w)

    model = sm.WLS(y, x, weights=w).fit()

    return {'r-squared': model.rsquared.item(), 't-stat': model.tvalues[1].item()}
