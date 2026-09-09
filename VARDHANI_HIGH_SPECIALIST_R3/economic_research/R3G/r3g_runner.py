from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

FEATURES = list(base.FEATURES)
HORIZONS = (15, 30, 60)
Q_GRID = (0.90, 0.95, 0.975)
TRAIN_DATES = 126
MIN_TRAIN_DATES = 120
HEALTH_LOOKBACK = 40
HEALTH_MIN = 20
MODEL_PARAMS = dict(base.MODEL_PARAMS)
assert len(FEATURES) == 42
assert MODEL_PARAMS['n_jobs'] == 1


def _last_dates(x: pd.DataFrame, before: pd.Timestamp, n: int) -> list:
    d = x.loc[x.timestamp < before, 'date'].drop_duplicates().sort_values()
    return d.tail(n).to_list()


def _pct(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    a = np.sort(train_scores[np.isfinite(train_scores)])
    if len(a) == 0:
        return np.full(len(scores), np.nan)
    return np.searchsorted(a, scores, side='right') / len(a)


def learn_month(x: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame | None:
    dates = _last_dates(x, month_start, TRAIN_DATES)
    if len(dates) < MIN_TRAIN_DATES:
        return None
    tr0 = x[x.date.isin(dates)].copy()
    eval_end = month_start + pd.offsets.MonthBegin(1)
    ev0 = x[(x.timestamp >= month_start) & (x.timestamp < eval_end)].copy()
    if ev0.empty:
        return None

    out = []
    for h in HORIZONS:
        tr = tr0[np.isfinite(tr0[f'y_{h}'])].copy()
        ev = ev0.copy()
        if tr.empty:
            continue
        Xtr = tr[FEATURES]
        Xev = ev[FEATURES]
        y = tr[f'y_{h}'].to_numpy(float)
        T = float(np.nanquantile(np.abs(y), 0.75))
        reg = LGBMRegressor(objective='huber', **MODEL_PARAMS)
        reg.fit(Xtr, y)
        up = base.fit_binary(Xtr, (y >= T).astype(int))
        dn = base.fit_binary(Xtr, (y <= -T).astype(int))
        mu_tr = reg.predict(Xtr)
        pu_tr = base.pred_binary(up, Xtr)
        pd_tr = base.pred_binary(dn, Xtr)
        mu_ev = reg.predict(Xev)
        pu_ev = base.pred_binary(up, Xev)
        pd_ev = base.pred_binary(dn, Xev)

        for side, sign in (('CE', 1.0), ('PE', -1.0)):
            raw_tr = sign * mu_tr + T * sign * (pu_tr - pd_tr)
            raw_ev = sign * mu_ev + T * sign * (pu_ev - pd_ev)
            z = ev[['timestamp', 'date', 'month', f'y_{h}', f'exit_ts_{h}']].copy()
            z.columns = ['timestamp', 'date', 'month', 'gross_underlying', 'exit_timestamp']
            z['gross_underlying'] = sign * z.gross_underlying
            z['side'] = side
            z['horizon'] = h
            z['raw_score'] = raw_ev
            z['percentile'] = _pct(raw_tr, raw_ev)
            z['T_h'] = T
            out.append(z)
    return pd.concat(out, ignore_index=True) if out else None


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3G evaluation year must be 2023 or 2024')
    parts = []
    for m in range(1, 13):
        ms = pd.Timestamp(year=year, month=m, day=1, tz='Asia/Kolkata')
        p = learn_month(x, ms)
        if p is not None:
            parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _health_active_for_specialist(current: pd.DataFrame, history: pd.DataFrame, q: float) -> np.ndarray:
    hist = history[
        np.isfinite(history.percentile)
        & np.isfinite(history.gross_underlying)
        & (history.percentile >= q)
        & history.exit_timestamp.notna()
    ].copy()
    if hist.empty:
        return np.zeros(len(current), dtype=bool)
    hist = hist.sort_values('exit_timestamp').reset_index(drop=True)
    exit_ns = hist.exit_timestamp.astype('int64').to_numpy()
    hv = (hist.gross_underlying.to_numpy(float) - 3.0)
    pos = np.where(hv > 0, hv, 0.0)
    neg = np.where(hv < 0, -hv, 0.0)
    cs = np.r_[0.0, np.cumsum(hv)]
    cp = np.r_[0.0, np.cumsum(pos)]
    cn = np.r_[0.0, np.cumsum(neg)]

    ts_ns = current.timestamp.astype('int64').to_numpy()
    k = np.searchsorted(exit_ns, ts_ns, side='right')
    start = np.maximum(0, k - HEALTH_LOOKBACK)
    n = k - start
    total = cs[k] - cs[start]
    gp = cp[k] - cp[start]
    gl = cn[k] - cn[start]
    mean = np.divide(total, n, out=np.full(len(n), np.nan), where=n > 0)
    pf = np.divide(gp, gl, out=np.full(len(n), np.inf), where=gl > 0)
    return (n >= HEALTH_MIN) & (mean > 0.0) & (pf > 1.0)


def apply_health(candidates_eval: pd.DataFrame, history_all: pd.DataFrame, q: float) -> pd.DataFrame:
    if candidates_eval.empty:
        return candidates_eval
    c = candidates_eval.copy()
    c['health_active'] = False
    for side in ('CE', 'PE'):
        for h in HORIZONS:
            mask = (c.side == side) & (c.horizon == h)
            if not mask.any():
                continue
            cur = c.loc[mask].copy()
            hist = history_all[(history_all.side == side) & (history_all.horizon == h)].copy()
            c.loc[mask, 'health_active'] = _health_active_for_specialist(cur, hist, q)
    return c


def simulate(candidates_eval: pd.DataFrame, history_all: pd.DataFrame, q: float) -> pd.DataFrame:
    c = apply_health(candidates_eval, history_all, q)
    if c.empty:
        return c
    c = c[
        np.isfinite(c.gross_underlying)
        & np.isfinite(c.percentile)
        & np.isfinite(c.raw_score)
        & (c.raw_score > 0.0)
        & (c.percentile >= q)
        & c.health_active
    ].copy()
    if c.empty:
        return c
    c['side_tie'] = (c.side == 'CE').astype(int)
    c = c.sort_values(
        ['timestamp', 'percentile', 'raw_score', 'horizon', 'side_tie'],
        ascending=[True, False, False, True, False],
    ).drop_duplicates('timestamp', keep='first').sort_values('timestamp')
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


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    if len(FEATURES) != 42 or any(c not in x.columns for c in FEATURES):
        raise RuntimeError('FAIL_CLOSED: exact 42-feature state unavailable')
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: state year boundary invalid')
    return x


def summarize(trades: pd.DataFrame) -> dict:
    return base.summarize(trades)


def worst_half_mean(s: dict) -> float:
    return min(s['net3']['H1']['mean'], s['net3']['H2']['mean'])
