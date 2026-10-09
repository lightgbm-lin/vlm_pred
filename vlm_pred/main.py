"""Fit the volume model walk-forward and score it, or tune its hyperparameters.

    python -m vlm_pred.main                  # score the default and tuned models, in-sample and out-of-sample
    python -m vlm_pred.main --tune           # search hyperparameters on the in-sample dates
    python -m vlm_pred.main --tune --n-trials 20

Tuning writes trials/best.json, and scoring reads the tuned parameters from it.
"""
import argparse
import json

import pandas as pd
from lightgbm import LGBMRegressor

from vlm_pred import feature
from vlm_pred.config import OOS_CUTOFF, ROOT
from vlm_pred.data import load_data_df
from vlm_pred.metric import evaluate
from vlm_pred.tune import tune
from vlm_pred.walkforward import WalkForward

TARGET = 'y'
MODEL_TARGET = 'y_ratio'  # y + 1, since the gamma objective needs a positive target
WEIGHT = 'sp_weight'
TRIALS_DIR = ROOT / 'trials'

FEATURE_FUNCS = [
    feature.enrich_vlm_ratio,
    feature.enrich_vol_ewm,
    feature.enrich_lagged_ret,
    feature.enrich_lagged_targets,
    feature.enrich_max_targets,
    feature.enrich_calendar_features,
    feature.enrich_earnings_schedule,
    feature.enrich_price_path,
    feature.enrich_overnight_gap,
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the baseline features to df. Returns the enriched df and the feature names."""
    features = pd.concat([func(df) for func in FEATURE_FUNCS], axis=1)
    return pd.concat([df, features], axis=1), features.columns.tolist()


def make_walkforward(features: list[str], **params) -> WalkForward:
    """Gamma LightGBM on y + 1, refit yearly on an expanding window. params override the LightGBM defaults."""
    model = LGBMRegressor(random_state=42, verbose=-1, objective='gamma', deterministic=True,
                          force_col_wise=True, **params)
    return WalkForward(model=model, features=features, target=MODEL_TARGET, weight=WEIGHT, train_window=None)


def is_out_of_sample(df: pd.DataFrame):
    return df.index.get_level_values('date') > OOS_CUTOFF


def report(name: str, preds: pd.Series, df: pd.DataFrame) -> None:
    """Print the R-squared and t-stat of preds against the target, in-sample and out-of-sample."""
    oos = is_out_of_sample(df)
    for label, mask in [('in-sample', ~oos), ('out-of-sample', oos)]:
        res = evaluate(preds, df[TARGET], df[WEIGHT], mask=mask)
        print(f"{name:<8} {label:<14} R² = {res['r-squared']:.4f}   t = {res['t-stat']:.1f}")


def run_score(df: pd.DataFrame, features: list[str]) -> None:
    tuned_params = json.loads((TRIALS_DIR / 'best.json').read_text())['params']
    for name, params in [('default', {}), ('tuned', tuned_params)]:
        preds = make_walkforward(features, **params).run(df)
        report(name, preds, df)


def run_tune(df: pd.DataFrame, features: list[str], n_trials: int) -> None:
    in_sample = df[~is_out_of_sample(df)]
    study = tune(make_walkforward(features), in_sample, target=TARGET, weight=WEIGHT, n_trials=n_trials,
                 out_dir=TRIALS_DIR)
    print(f'best in-sample R² = {study.best_value:.4f} with {study.best_params}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tune', action='store_true', help='tune hyperparameters instead of scoring')
    parser.add_argument('--n-trials', type=int, default=100, help='number of tuning trials (default: 100)')
    args = parser.parse_args()

    df, features = build_features(load_data_df())
    if args.tune:
        run_tune(df, features, args.n_trials)
    else:
        run_score(df, features)


if __name__ == '__main__':
    main()
