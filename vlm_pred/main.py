import pandas as pd
from vlm_pred.data import load_data_df, train_val_oos_split
from vlm_pred.feature import enrich_vlm_ratio, enrich_vol_ewm, enrich_lagged_ret, enrich_calendar_features, \
    enrich_lagged_targets, enrich_max_targets


def load_all_data():
    data_df = load_data_df()

    vlm_ratio_df = enrich_vlm_ratio(data_df, [1, 5, 10, 21])
    vol_ewm_df = enrich_vol_ewm(data_df, [1, 5, 10, 21])
    lagged_ret_df = enrich_lagged_ret(data_df, [1, 2, 3, 4, 5])
    lagged_targets_df = enrich_lagged_targets(data_df, [1, 2, 3, 4, 5])
    max_targets = enrich_max_targets(data_df, [5, 10, 20])
    cal_df = enrich_calendar_features(data_df)

    full_df = pd.concat([data_df, vlm_ratio_df, vol_ewm_df, lagged_ret_df, max_targets, lagged_targets_df, cal_df], axis=1)

    ins_df, oos_df = train_val_oos_split(full_df)

    return ins_df, oos_df
