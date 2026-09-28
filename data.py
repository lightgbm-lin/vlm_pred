import os
import pandas as pd

from config import DATA_FILENAME, OOS_CUTOFF


def calc_vlm_ewma(df, halflife):
    # TODO: should ewm be using time arg?
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()

    g = df.groupby(level='uspn')['volume']

    return g.transform(lambda x: x.ewm(halflife=halflife).mean().shift(1))


def load_data_df():
    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(f'{DATA_FILENAME} cannot be found. Hint: ensure that sp500.h5 is in a folder data/, and the Python src code is in src/')

    df = pd.DataFrame(pd.read_hdf(DATA_FILENAME))

    df['vlm_pred_naive'] = calc_vlm_ewma(df, halflife=60)
    df['y'] = (df['volume'] / df['vlm_pred_naive'].mask(df['vlm_pred_naive'] <= 0) - 1).clip(lower=-1, upper=10)

    return df


def ins_oos_split(df):
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