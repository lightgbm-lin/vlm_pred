import pandas as pd
from lightgbm import LGBMRegressor

from vlm_pred.config import OOS_CUTOFF
from vlm_pred.metric import evaluate
from vlm_pred.walkforward import WalkForward
from vlm_pred.data import load_data_df
from vlm_pred.feature import enrich_vlm_ratio, enrich_vol_ewm, enrich_lagged_ret, enrich_lagged_targets, \
    enrich_max_targets, enrich_calendar_features, enrich_earnings_schedule, enrich_price_path, \
    enrich_overnight_gap



def main():
    data_df = load_data_df()

    baseline_funcs = [enrich_vlm_ratio, enrich_vol_ewm, enrich_lagged_ret, enrich_lagged_targets, enrich_max_targets,
                      enrich_calendar_features, enrich_earnings_schedule, enrich_price_path, enrich_overnight_gap]

    feature_dfs = []
    for func in baseline_funcs:
        feature_dfs.append(func(data_df))

    enriched_df = pd.concat([data_df] + feature_dfs, axis=1)

    from vlm_pred.data import train_test_split

    train_df, val_df, test_df = train_test_split(enriched_df)

    features = [c for f in feature_dfs for c in f.columns]
    target = 'y'
    target_ratio = 'y_ratio'
    weight = 'sp_weight'

    default_lgbm_gamma_model = LGBMRegressor(random_state=42, verbose=-1, objective='gamma', deterministic=True,
                                     force_col_wise=True)
    lgbm_gamma_wf = WalkForward(model=default_lgbm_gamma_model, features=features, target=target_ratio, weight=weight,
                                train_window=None)
    lgbm_gamma_preds = lgbm_gamma_wf.run(train_df)

    ins_result = evaluate(lgbm_gamma_preds, train_df[target], train_df[weight])
    print(ins_result)

    lgbm_gamma_params = {'n_estimators': 963,
                         'learning_rate': 0.01025674056791372,
                         'num_leaves': 190,
                         'min_child_samples': 35,
                         'colsample_bytree': 0.5201068463749413,
                         'reg_lambda': 0.01957113058068085}
    lgbm_gamma_model = LGBMRegressor(random_state=42, verbose=-1, objective='gamma', deterministic=True,
                                     force_col_wise=True, **lgbm_gamma_params)

    lgbm_gamma_wf = WalkForward(model=lgbm_gamma_model, features=features, target=target_ratio, weight=weight,
                                train_window=None)
    lgbm_gamma_preds = lgbm_gamma_wf.run(enriched_df)

    ins_mask = enriched_df.index.get_level_values('date') <= OOS_CUTOFF
    ins_result = evaluate(lgbm_gamma_preds, enriched_df[target], enriched_df[weight], mask=ins_mask)
    print(ins_result)

    oos_mask = enriched_df.index.get_level_values('date') > OOS_CUTOFF
    oos_result = evaluate(lgbm_gamma_preds, enriched_df[target], enriched_df[weight], mask=oos_mask)

    print(oos_result)


if __name__ == '__main__':
    main()
