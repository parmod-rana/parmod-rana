from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

# Reuse only the frozen causal state/target construction and generic accounting
# from R3E. R3F's learning/selection logic below is distinct and preregistered.
import sys
from pathlib import Path
R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

FEATURES = list(base.FEATURES)
HORIZONS = (15, 30, 60)
Q_GRID = (0.95, 0.975, 0.99)
FAST_DATES = 63
SLOW_DATES = 126
FAST_MIN_DATES = 60
SLOW_MIN_DATES = 120
FAST_HALF_LIFE = 21.0
SLOW_HALF_LIFE = 42.0
MODEL_PARAMS = dict(base.MODEL_PARAMS)
assert len(FEATURES) == 42
assert MODEL_PARAMS == {
    'n_estimators': 100,
    'learning_rate': 0.04,
    'num_leaves': 7,
    'max_depth': 3,
    'min_child_samples': 200,
    'reg_lambda': 30.0,
    'random_state': 7584,
    'n_jobs': 1,
    'verbosity': -1,
}


def _last_dates(x: pd.DataFrame, month_start: pd.Timestamp, n: int) -> list:
    d = pd.Series(x.loc[x.timestamp < month_start, 'date'].drop_duplicates().sort_values().to_list())
    return d.tail(n).to_list()


def _weights_by_date(train: pd.DataFrame, half_life: float) -> np.ndarray:
    dates = sorted(pd.unique(train.date))
    rank = {d: i for i, d in enumerate(dates)}
    last = len(dates) - 1
    age = train.date.map(lambda d: last - rank[d]).to_numpy(float)
    return np.power(2.0, -age / half_life)


def _fit_binary(X: pd.DataFrame, target: np.ndarray, sample_weight: np.ndarray):
    u = np.unique(target[np.isfinite(target)])
    if len(u) < 2:
        p = float(u[0]) if len(u) else 0.0
        return ('constant', p)
    m = LGBMClassifier(objective='binary', **MODEL_PARAMS)
    m.fit(X, target.astype(int), sample_weight=sample_weight)
    return ('model', m)


def _pred_binary(obj, X: pd.DataFrame) -> np.ndarray:
    kind, payload = obj
    if kind == 'constant':
        return np.full(len(X), float(payload))
    return payload.predict_proba(X)[:, 1]


def _pct(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    a = np.sort(train_scores[np.isfinite(train_scores)])
    if len(a) == 0:
        return np.full(len(scores), np.nan)
    return np.searchsorted(a, scores, side='right') / len(a)


@dataclass
class WindowPred:
    side: str
    horizon: int
    raw_eval: np.ndarray
    pct_eval: np.ndarray
    threshold: float


def _fit_window(train: pd.DataFrame, ev: pd.DataFrame, h: int, half_life: float) -> dict[str, WindowPred]:
    tr = train[np.isfinite(train[f'y_{h}'])].copy()
    if tr.empty:
        return {}
    Xtr = tr[FEATURES]
    Xev = ev[FEATURES]
    y = tr[f'y_{h}'].to_numpy(float)
    w = _weights_by_date(tr, half_life)
    T = float(np.nanquantile(np.abs(y), 0.75))

    reg = LGBMRegressor(objective='huber', **MODEL_PARAMS)
    reg.fit(Xtr, y, sample_weight=w)
    up = _fit_binary(Xtr, (y >= T).astype(int), w)
    dn = _fit_binary(Xtr, (y <= -T).astype(int), w)

    mu_tr = reg.predict(Xtr)
    pu_tr = _pred_binary(up, Xtr)
    pd_tr = _pred_binary(dn, Xtr)
    mu_ev = reg.predict(Xev)
    pu_ev = _pred_binary(up, Xev)
    pd_ev = _pred_binary(dn, Xev)

    out = {}
    for side, sign in (('CE', 1.0), ('PE', -1.0)):
        raw_tr = sign * mu_tr + T * sign * (pu_tr - pd_tr)
        raw_ev = sign * mu_ev + T * sign * (pu_ev - pd_ev)
        out[side] = WindowPred(side, h, raw_ev, _pct(raw_tr, raw_ev), T)
    return out


def learn_month(x: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame | None:
    fast_dates = _last_dates(x, month_start, FAST_DATES)
    slow_dates = _last_dates(x, month_start, SLOW_DATES)
    if len(fast_dates) < FAST_MIN_DATES or len(slow_dates) < SLOW_MIN_DATES:
        return None
    train_fast = x[x.date.isin(fast_dates)].copy()
    train_slow = x[x.date.isin(slow_dates)].copy()
    eval_end = month_start + pd.offsets.MonthBegin(1)
    ev = x[(x.timestamp >= month_start) & (x.timestamp < eval_end)].copy()
    if ev.empty:
        return None

    rows = []
    for h in HORIZONS:
        fp = _fit_window(train_fast, ev, h, FAST_HALF_LIFE)
        sp = _fit_window(train_slow, ev, h, SLOW_HALF_LIFE)
        if not fp or not sp:
            continue
        for side, sign in (('CE', 1.0), ('PE', -1.0)):
            f = fp[side]; s = sp[side]
            z = ev[['timestamp', 'date', 'month', f'y_{h}', f'exit_ts_{h}']].copy()
            z.columns = ['timestamp', 'date', 'month', 'gross_underlying', 'exit_timestamp']
            z['gross_underlying'] = sign * z.gross_underlying
            z['side'] = side
            z['horizon'] = h
            z['raw_fast'] = f.raw_eval
            z['raw_slow'] = s.raw_eval
            z['pct_fast'] = f.pct_eval
            z['pct_slow'] = s.pct_eval
            z['eligible_consensus'] = (z.raw_fast > 0.0) & (z.raw_slow > 0.0)
            z['percentile'] = np.minimum(z.pct_fast, z.pct_slow)
            z['raw_score'] = np.minimum(z.raw_fast, z.raw_slow)
            z['T_fast'] = f.threshold
            z['T_slow'] = s.threshold
            rows.append(z)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3F evaluation year must be 2023 or 2024')
    packs = []
    for m in range(1, 13):
        ms = pd.Timestamp(year=year, month=m, day=1, tz='Asia/Kolkata')
        p = learn_month(x, ms)
        if p is not None:
            packs.append(p)
    if not packs:
        return pd.DataFrame()
    return pd.concat(packs, ignore_index=True)


def simulate(candidates: pd.DataFrame, q: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    c = candidates[
        np.isfinite(candidates.gross_underlying)
        & np.isfinite(candidates.percentile)
        & candidates.eligible_consensus
        & (candidates.percentile >= q)
    ].copy()
    if c.empty:
        return c
    c['side_tie'] = (c.side == 'CE').astype(int)
    c = c.sort_values(
        ['timestamp', 'percentile', 'raw_score', 'horizon', 'side_tie'],
        ascending=[True, False, False, True, False],
    )
    c = c.drop_duplicates('timestamp', keep='first').sort_values('timestamp')
    rows = []
    busy_until = None
    for r in c.itertuples(index=False):
        if busy_until is not None and r.timestamp < busy_until:
            continue
        rows.append(r._asdict())
        busy_until = r.exit_timestamp
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['net3'] = out.gross_underlying - 3.0
    out['net5'] = out.gross_underlying - 5.0
    out['half'] = np.where(pd.to_datetime(out.timestamp).dt.month <= 6, 'H1', 'H2')
    return out


def summarize(trades: pd.DataFrame) -> dict:
    return base.summarize(trades)


def worst_half_mean(s: dict) -> float:
    return min(s['net3']['H1']['mean'], s['net3']['H2']['mean'])


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    if len(FEATURES) != 42 or any(c not in x.columns for c in FEATURES):
        raise RuntimeError('FAIL_CLOSED: exact 42-feature state unavailable')
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: state year boundary invalid')
    return x
