from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm


@dataclass
class WalkForward:
    """Walk-forward fit/predict over a (date, uspn) indexed dataframe.

    At the start of each `refit_every` period (e.g. each month), a fresh clone
    of `model` is fit on the previous `train_window` dates (all history if None)
    and used to predict every date in that period. `gap` dates are skipped
    between train and predict to avoid leakage when the target looks ahead.
    """
    model: object                   # any sklearn estimator/Pipeline, incl. lightgbm
    features: list[str]
    target: str
    weight: str | None = None       # column of sample weights
    train_window: int | None = 252  # number of dates; None = expanding window
    refit_every: str = 'Y'          # refit period: 'M' month, 'Q' quarter, 'Y' year
    min_train: int = 252            # dates of history required before first fit
    gap: int = 0

    def run(self, df: pd.DataFrame) -> pd.Series:
        dates = df.index.get_level_values('date')
        uniq = dates.unique().sort_values()
        periods = uniq.to_period(self.refit_every)
        self.models_ = {}
        preds = []

        for period in tqdm(periods.unique(), total=len(periods.unique()), leave=False):
            test_dates = uniq[periods == period]
            train_end = uniq.get_loc(test_dates[0]) - self.gap
            if train_end < self.min_train:
                continue
            train_start = 0 if self.train_window is None else max(0, train_end - self.train_window)
            train = df[dates.isin(uniq[train_start:train_end])]
            test = df[dates.isin(test_dates)]

            model = self._fit(train)
            self.models_[period] = model
            preds.append(pd.Series(model.predict(test[self.features]), index=test.index))

        return pd.concat(preds).reindex(df.index).rename('pred')

    def _fit(self, train: pd.DataFrame):
        train = train.dropna(subset=[self.target] + ([self.weight] if self.weight else []))
        model = clone(self.model)
        kwargs = {}
        if self.weight:
            key = f'{model.steps[-1][0]}__sample_weight' if isinstance(model, Pipeline) else 'sample_weight'
            kwargs[key] = train[self.weight].to_numpy()
        return model.fit(train[self.features], train[self.target], **kwargs)
