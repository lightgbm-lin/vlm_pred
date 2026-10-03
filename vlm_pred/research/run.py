"""Evaluate a candidate volume-prediction factor against the current feature set.

Run from the repo root:

    python -m vlm_pred.research.run vlm_pred/research/candidates/h001_name.py   # evaluate a candidate vs the current model
    python -m vlm_pred.research.run --baseline                                  # print the current model's score only
    python -m vlm_pred.research.run --accept h001_name                          # add a candidate to the selected feature set

The enriched frame is the training spine (date <= VAL_CUTOFF) plus the baseline features
from vlm_pred/feature.py plus the columns of every candidate listed in candidates/selected.txt, applied in order.
The VAL and OOS periods are never used here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from vlm_pred.config import ROOT as REPO_ROOT
from vlm_pred import feature
from vlm_pred.data import load_data_df, train_test_split
from vlm_pred.metric import evaluate
from vlm_pred.walkforward import WalkForward

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / 'candidates'
SELECTED = CANDIDATES / 'selected.txt'
RESULTS = ROOT / 'results'
CACHE = ROOT / '.cache'

TARGET = 'y'
WEIGHT = 'sp_weight'
MODEL_TARGET = 'y_ratio'  # y + 1, NaN (so dropped from training) where it's <= 0; see vlm_pred/data.py
# Market data observed on date t itself; a feature for date t must not depend on these at t.
SAME_DAY_COLS = ['price_adj', 'price_close', 'price_open', 'price_high', 'price_low',
                 'volume', 'sp_weight', 'ret_raw', 'y', 'y_ratio']

MIN_DELTA_R2 = 0.0005
MIN_YEAR_HIT_RATE = 2 / 3


def make_model():
    return LGBMRegressor(random_state=42, verbose=-1, objective='gamma')


def load_module(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_names():
    if not SELECTED.exists():
        return []
    lines = (line.strip() for line in SELECTED.read_text().splitlines())
    return [line for line in lines if line and not line.startswith('#')]


def compute_features(module, df):
    """Run module.compute on a shallow copy of df and return only its new, float columns."""
    out = module.compute(df.copy(deep=False))
    if isinstance(out, pd.Series):
        out = out.to_frame()
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f'compute must return a DataFrame or Series, got {type(out)}')
    out = out[[c for c in out.columns if c not in df.columns]]
    if out.empty:
        raise ValueError('compute returned no new columns')
    if out.columns.duplicated().any():
        raise ValueError(f'duplicate column names: {out.columns[out.columns.duplicated()].tolist()}')
    if not out.index.equals(df.index):
        out = out.reindex(df.index)
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def leakage_check(module, df, full_out, quantiles=(0.25, 0.5, 0.9), seed=0):
    """Recompute on data truncated at date T with T's same-day market data scrambled.

    A leak-free feature for date T is unchanged by both. Returns {column: mismatch fraction}
    for offending columns. No column is exempt.
    """
    rng = np.random.default_rng(seed)
    dates = df.index.get_level_values('date')
    uniq = dates.unique().sort_values()
    bad = {}
    for q in quantiles:
        t = uniq[int(q * (len(uniq) - 1))]
        sub = df[dates <= t].copy()
        at_t = sub.index.get_level_values('date') == t
        for col in SAME_DAY_COLS:
            vals = sub.loc[at_t, col].to_numpy()
            sub.loc[at_t, col] = rng.permutation(vals) * rng.uniform(0.5, 2.0, len(vals))
        got = compute_features(module, sub).loc[at_t]
        want = full_out.loc[got.index]
        for col in got.columns:
            a, b = got[col].to_numpy(), want[col].to_numpy()
            ok = np.isclose(a, b, rtol=1e-6, atol=1e-9) | (np.isnan(a) & np.isnan(b))
            if not ok.all():
                bad[col] = max(bad.get(col, 0.0), float(1 - ok.mean()))
    return bad


def file_hash(*parts):
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode() if isinstance(part, str) else part)
    return h.hexdigest()[:12]


def baseline_features(df):
    """The baseline feature set from vlm_pred/feature.py, cached by that file's source and df's date range."""
    dates = df.index.get_level_values('date')
    key = file_hash(Path(feature.__file__).read_text(), str(dates.min()), str(dates.max()), str(len(df)))
    cache_file = CACHE / f'base_{key}.parquet'
    if cache_file.exists():
        return pd.read_parquet(cache_file), key
    base = pd.concat([
        feature.enrich_vlm_ratio(df),
        feature.enrich_vol_ewm(df),
        feature.enrich_lagged_ret(df),
        feature.enrich_lagged_targets(df),
        feature.enrich_max_targets(df),
        feature.enrich_earnings_day(df),
        feature.enrich_calendar_features(df),
    ], axis=1).astype(float)
    CACHE.mkdir(exist_ok=True)
    base.to_parquet(cache_file)
    return base, key


def build_enriched():
    """In-sample spine + baseline + selected candidate features. Returns (df, feature list, cache key)."""
    df, _, _ = train_test_split(load_data_df())
    base, base_key = baseline_features(df)
    df = pd.concat([df, base], axis=1)
    features = base.columns.tolist()

    names = selected_names()
    paths = [CANDIDATES / f'{name}.py' for name in names]
    key = file_hash(base_key, *[p.name + p.read_text() for p in paths])
    if not names:
        return df, features, key

    cache_file = CACHE / f'selected_{key}.parquet'
    if cache_file.exists():
        added = pd.read_parquet(cache_file)
        df = pd.concat([df, added], axis=1)
        return df, features + added.columns.tolist(), key

    added = []
    for path in paths:
        out = compute_features(load_module(path), df)
        df = pd.concat([df, out], axis=1)
        added.append(out)
    added = pd.concat(added, axis=1)
    CACHE.mkdir(exist_ok=True)
    added.to_parquet(cache_file)
    return df, features + added.columns.tolist(), key


def prepare(df):
    df = df.copy()
    df[WEIGHT] = df[WEIGHT].fillna(0)
    return df


def run_wf(df, features):
    wf = WalkForward(model=make_model(), features=features, target=MODEL_TARGET,
                     weight=WEIGHT, train_window=None)
    preds = wf.run(df)
    return preds, wf


def score(preds, df):
    res = {'overall': evaluate(preds, df[TARGET], df[WEIGHT]), 'by_year': {}}
    years = df.index.get_level_values('date').year
    for year in sorted(set(years[preds.notna().to_numpy()])):
        m = years == year
        res['by_year'][str(year)] = evaluate(preds[m], df.loc[m, TARGET], df.loc[m, WEIGHT])['r-squared']
    return res


def gain_shares(wf, features):
    gains = np.zeros(len(features))
    for model in wf.models_.values():
        g = model.booster_.feature_importance(importance_type='gain')
        gains += g / g.sum() if g.sum() > 0 else 0
    gains /= max(len(wf.models_), 1)
    return {f: round(float(g), 4) for f, g in sorted(zip(features, gains), key=lambda x: -x[1])}


def baseline_score(df, features, key):
    cache_file = CACHE / f'baseline_{file_hash(key, json.dumps(features), repr(make_model().get_params()))}.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    preds, wf = run_wf(df, features)
    res = score(preds, df) | {'features': features, 'gain_share': gain_shares(wf, features)}
    CACHE.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(res, indent=2))
    return res


def evaluate_candidate(path):
    path = Path(path)
    module = load_module(path)
    df, base_features, key = build_enriched()

    out = compute_features(module, df)
    coverage = {c: round(float(out[c].notna().mean()), 4) for c in out.columns}
    leaks = leakage_check(module, df, out)

    df = prepare(pd.concat([df, out], axis=1))
    base = baseline_score(df, base_features, key)
    features = base_features + out.columns.tolist()
    preds, wf = run_wf(df, features)
    cand = score(preds, df)

    years = sorted(base['by_year'])
    year_delta = {y: cand['by_year'][y] - base['by_year'][y] for y in years}
    hits = sum(d > 0 for d in year_delta.values())
    delta = cand['overall']['r-squared'] - base['overall']['r-squared']
    passes = not leaks and delta >= MIN_DELTA_R2 and hits >= MIN_YEAR_HIT_RATE * len(years)

    result = {
        'candidate': path.stem,
        'hypothesis': (module.__doc__ or '').strip(),
        'new_columns': out.columns.tolist(),
        'coverage': coverage,
        'leakage': leaks or 'passed',
        'base_features': base_features,
        'base_r2': base['overall']['r-squared'],
        'cand_r2': cand['overall']['r-squared'],
        'delta_r2': delta,
        'cand_tstat': cand['overall']['t-stat'],
        'years_improved': f'{hits}/{len(years)}',
        'by_year': {y: {'base': base['by_year'][y], 'cand': cand['by_year'][y],
                        'delta': year_delta[y]} for y in years},
        'gain_share': gain_shares(wf, features),
        'suggested_decision': 'ACCEPT' if passes else 'REJECT',
        'rule': f'no leakage, delta_r2 >= {MIN_DELTA_R2}, years improved >= {MIN_YEAR_HIT_RATE:.0%}',
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'{path.stem}.json').write_text(json.dumps(result, indent=2))
    return result


def print_result(r):
    print(f"\n=== {r['candidate']} ===")
    print(f"new columns : {r['new_columns']}")
    print(f"coverage    : {r['coverage']}")
    print(f"leakage     : {r['leakage']}")
    print(f"base R2     : {r['base_r2']:.5f}  ({len(r['base_features'])} features)")
    print(f"cand R2     : {r['cand_r2']:.5f}")
    print(f"delta R2    : {r['delta_r2']:+.5f}   years improved {r['years_improved']}")
    print('by year     : ' + '  '.join(f"{y}:{v['delta']:+.4f}" for y, v in r['by_year'].items()))
    print('gain share  : ' + ', '.join(f'{k}={v:.3f}' for k, v in r['gain_share'].items()))
    print(f"suggested   : {r['suggested_decision']}  ({r['rule']})")
    print(f"saved       : {(RESULTS / (r['candidate'] + '.json')).relative_to(REPO_ROOT)}")


def accept(name):
    path = CANDIDATES / f'{name}.py'
    if not path.exists():
        raise FileNotFoundError(path)
    names = selected_names()
    if name in names:
        print(f'{name} already selected')
        return
    with SELECTED.open('a') as f:
        f.write(f'{name}\n')
    print(f'selected: {names + [name]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('candidate', nargs='?', help='path to a candidate .py with compute(df)')
    parser.add_argument('--baseline', action='store_true', help='score the current feature set only')
    parser.add_argument('--accept', metavar='NAME', help='append candidates/NAME.py to selected.txt')
    args = parser.parse_args()

    if args.accept:
        accept(args.accept)
    elif args.baseline:
        df, features, key = build_enriched()
        res = baseline_score(prepare(df), features, key)
        print(json.dumps(res, indent=2))
    elif args.candidate:
        print_result(evaluate_candidate(args.candidate))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
