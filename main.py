import pandas as pd
from data import load_data_df, ins_oos_split
from feature import enrich_vlm_ratio, enrich_vol_ewm


def load_all_data():
    data_df = load_data_df()

    vlm_ratio_df = enrich_vlm_ratio(data_df, [1, 21])
    vol_ewm_df = enrich_vol_ewm(data_df, [1, 21])

    full_df = pd.concat([data_df, vlm_ratio_df, vol_ewm_df], axis=1)

    ins_df, oos_df = ins_oos_split(full_df)

    return ins_df, oos_df
