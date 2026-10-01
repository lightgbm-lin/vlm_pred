import os
import pandas as pd

from vlm_pred.config import DATA_FILENAME, OOS_CUTOFF
from vlm_pred.util import calc_ewm


def load_data_df():
    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(f'{DATA_FILENAME} cannot be found. Hint: put sp500.h5 in the data/ folder at the project root')

    df = pd.DataFrame(pd.read_hdf(DATA_FILENAME))

    df['vlm_pred_naive'] = calc_ewm(df, halflife=60, column='volume')
    df['y'] = (df['volume'] / df['vlm_pred_naive'].mask(df['vlm_pred_naive'] <= 0) - 1).clip(lower=-1, upper=10)
    df['y_ratio'] = (df['y'] + 1).clip(lower=1e-6)

    return df


def ins_oos_split(df: pd.DataFrame):
    ins_df = df[df.index.get_level_values('date') <= OOS_CUTOFF].copy()
    oos_df = df[df.index.get_level_values('date') > OOS_CUTOFF].copy()

    return ins_df, oos_df


def main():
    data_df = load_data_df()
    ins_df, oos_df = ins_oos_split(data_df)
    print(f"found ins_df {ins_df.shape}")
    print(f"found oos_df {oos_df.shape}")


if __name__ == '__main__':
    main()