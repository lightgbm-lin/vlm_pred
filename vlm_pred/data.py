import os
import pandas as pd

from vlm_pred.config import DATA_FILENAME, OOS_CUTOFF, VAL_CUTOFF
from vlm_pred.util import calc_ewm


def load_data_df():
    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(f'{DATA_FILENAME} cannot be found. Hint: put sp500.h5 in the data/ folder at the project root')

    df = pd.DataFrame(pd.read_hdf(DATA_FILENAME))
    print(f"loaded df: {df.shape} from {DATA_FILENAME}")

    df['vlm_pred_naive'] = calc_ewm(df, halflife=60, column='volume')
    df['y'] = (df['volume'] / df['vlm_pred_naive'].mask(df['vlm_pred_naive'] <= 0) - 1).clip(lower=-1, upper=10)
    df['y_ratio'] = (df['y'] + 1).where(lambda s: s > 0)

    return df


def train_test_split(df: pd.DataFrame):
    dates = df.index.get_level_values('date')

    train_df = df[dates <= VAL_CUTOFF].copy()
    val_df = df[(dates > VAL_CUTOFF) & (dates <= OOS_CUTOFF)].copy()
    oos_df = df[dates > OOS_CUTOFF].copy()

    return train_df, val_df, oos_df


def main():
    data_df = load_data_df()
    train_df, val_df, oos_df = train_test_split(data_df)
    print(f"found train_df: {train_df.shape}, val_df: {val_df.shape}, oos_df: {oos_df.shape}")


if __name__ == '__main__':
    main()