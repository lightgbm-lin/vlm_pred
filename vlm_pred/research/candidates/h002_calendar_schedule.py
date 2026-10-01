"""H002: Scheduled calendar events on T (and whether T-1 was a distorted session) predict y.

Mechanism: several days have volume set by the trading schedule rather than by information flow.
Option expiry / quad witching (hedge unwinds, S&P quarterly rebalance) and the Russell reconstitution
concentrate index/derivative flows into the close; half days and holiday-adjacent days have short
sessions or thin participation; turn of month brings fund flows; Mondays are quiet. The EWMA
features cannot anticipate these, and when T-1 was such a day its volume distorts vlm_1_ratio in a
way that does not persist into T.
Expected: quad witch >> opex > normal; Russell recon > normal; half day << pre/post-holiday < normal;
month-end slightly >; Monday <. After a special T-1 session, vlm_1_ratio should be discounted.
"""
import numpy as np
import pandas as pd

from vlm_pred.feature import enrich_calendar_features

KNOWN_IN_ADVANCE = {
    'h002_opex': 'monthly option expiry (3rd Friday, rolled back for exchange holidays) is scheduled',
    'h002_holiday_adj': 'exchange holiday schedule is published in advance',
    'h002_month_end': 'the trading calendar (last trading day of the month) is published in advance',
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    cal = dates.unique().sort_values()
    # one row per trading date; enrich_calendar_features only uses the date level
    c = enrich_calendar_features(pd.DataFrame(index=pd.MultiIndex.from_arrays(
        [cal, np.zeros(len(cal))], names=['date', 'uspn'])))
    c.index = cal

    f = pd.DataFrame(index=cal)
    f['h002_opex'] = c['is_opex'] + c['is_quad_witch']                 # 0 / 1 opex / 2 quad witch
    f['h002_russell'] = c['is_russell_recon']
    f['h002_half_day'] = c['is_half_day']
    f['h002_holiday_adj'] = c['is_pre_holiday'] | c['is_post_holiday']
    f['h002_month_end'] = c['is_month_end']
    f['h002_dow'] = c['dow']
    # T-1 was a schedule-distorted session (its volume says little about T)
    special = (c['is_opex'] | c['is_russell_recon'] | c['is_half_day']).astype(float)
    f['h002_prev_special'] = special.shift(1)

    out = f.reindex(dates).astype(float)
    out.index = df.index
    return out
