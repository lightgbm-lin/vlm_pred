"""Hyperparameter tuning with optuna: each trial reruns the walk-forward with new model parameters and is
scored by the R-squared of its predictions."""
import dataclasses
from collections.abc import Callable

import optuna
import pandas as pd
from sklearn.base import clone

from vlm_pred.metric import evaluate
from vlm_pred.walkforward import WalkForward


def lgbm_space(trial: optuna.Trial) -> dict:
    """Default search space for LGBMRegressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 5000, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100, log=True),
    }


def tune(wf: WalkForward, df: pd.DataFrame, target: str, weight: str, n_trials: int = 50,
         space: Callable[[optuna.Trial], dict] = lgbm_space, seed: int = 42) -> optuna.Study:
    """Search `space` for the model parameters that maximizes the walk-forward R-squared of `wf` on `df`.

    Each trial sets the sampled parameters on a clone of `wf.model`, keeping its other settings (e.g.
    objective, random_state). `target` and `weight` are the columns the predictions are evaluated
    against, which can differ from `wf.target` (e.g. fit on y + 1, evaluate on y).

    Returns the study: `study.best_params`, `study.best_value`, and `study.trials_dataframe()`.
    """
    def objective(trial):
        model = clone(wf.model).set_params(**space(trial))
        preds = dataclasses.replace(wf, model=model).run(df)
        return evaluate(preds, df[target], df[weight])['r-squared']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials)
    return study
