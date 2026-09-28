import pandas as pd
from data import calc_vlm_ewma


def enrich_vlm_ratio(df: pd.DataFrame, hls: list[int]) -> pd.DataFrame:
    vlm_ewm = pd.concat([calc_vlm_ewma(df, halflife=hl) for hl in hls], axis=1)
    vlm_pred_naive = df['vlm_pred_naive'].mask(df['vlm_pred_naive']<=0)

    vlm_ratio = vlm_ewm.divide(vlm_pred_naive, axis=0) - 1
    vlm_ratio.columns = [f'vlm_{hl}_ratio' for hl in hls]

    return vlm_ratio


def enrich_vol_ewm(df: pd.DataFrame, hls: list[int]) -> pd.DataFrame:
    vol_ewm_df = pd.concat([df.groupby('uspn')['ret_raw'].transform(lambda x: x.ewm(halflife=hl).std().shift(1)) for hl in hls], axis=1)
    vol_ewm_df.columns = [f'vol_ewm_{hl}' for hl in hls]

    return vol_ewm_df
