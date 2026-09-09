from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

CHECKPOINTS = tuple(
    (pd.Timestamp('2000-01-01 10:00') + pd.Timedelta(minutes=10*i)).strftime('%H:%M')
    for i in range(28)
)
LOOKBACK_DATES = 126
MIN_HISTORY = 120
DISLOCATION_Q = 0.65
MAX_TRADES_PER_DATE = 2
HORIZONS = (15, 30)


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    x['div15'] = x.n_ret_15 - x.s_ret_15
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: R3O state year boundary invalid')
    return x


def checkpoint_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    cols = [
        'timestamp','date','year','month','div15','n_ret_5','s_ret_5','vix_ret_5',
        'n_session_pos','y_15','exit_ts_15','y_30','exit_ts_30'
    ]
    c = x.loc[cp.isin(CHECKPOINTS), cols].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['dislocation_abs'] = c.div15.abs()
    c['strength_abs'] = c.n_ret_5.abs()
    c['dislocation_thr'] = np.nan
    c['strength_med'] = np.nan
    for name in CHECKPOINTS:
        z = c.loc[c.checkpoint == name].sort_values('timestamp')
        c.loc[z.index, 'dislocation_thr'] = (
            z.dislocation_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(DISLOCATION_Q).shift(1).to_numpy()
        )
        c.loc[z.index, 'strength_med'] = (
            z.strength_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .median().shift(1).to_numpy()
        )
    return c.reset_index(drop=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3O year must be 2023 or 2024')
    c = checkpoint_table(x)
    z = c[c.year == year].copy()
    req = [
        'div15','n_ret_5','s_ret_5','vix_ret_5','n_session_pos',
        'dislocation_abs','dislocation_thr','strength_abs','strength_med'
    ]
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z = z[z.div15 != 0.0].copy()
    z['dir_sign'] = -np.sign(z.div15).astype(int)
    z = z[z.dislocation_abs >= z.dislocation_thr]
    z = z[(z.dir_sign * z.n_ret_5) > 0.0]
    z = z[(z.dir_sign * z.s_ret_5) > 0.0]
    z = z[(z.dir_sign * z.vix_ret_5) < 0.0]
    z = z[
        ((z.dir_sign < 0) & (z.n_session_pos >= 0.55)) |
        ((z.dir_sign > 0) & (z.n_session_pos <= 0.45))
    ].copy()
    if z.empty:
        return z

    z['side'] = np.where(z.dir_sign > 0, 'CE', 'PE')
    z['horizon'] = np.where(z.strength_abs >= z.strength_med, 30, 15).astype(int)
    z['gross_underlying'] = np.where(
        z.horizon == 30,
        z.dir_sign * z.y_30,
        z.dir_sign * z.y_15,
    )
    z['exit_timestamp'] = z.exit_ts_15
    m30 = z.horizon == 30
    z.loc[m30, 'exit_timestamp'] = z.loc[m30, 'exit_ts_30']
    keep = [
        'timestamp','date','month','checkpoint','side','horizon','gross_underlying',
        'exit_timestamp','div15','dislocation_abs','dislocation_thr','n_ret_5',
        's_ret_5','vix_ret_5','n_session_pos','strength_abs','strength_med'
    ]
    return z[keep].sort_values('timestamp').reset_index(drop=True)


def simulate(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    z = candidates[np.isfinite(candidates.gross_underlying) & candidates.exit_timestamp.notna()].copy()
    z = z.sort_values('timestamp')
    rows = []
    busy_until = None
    date_counts: dict[object, int] = {}
    for r in z.itertuples(index=False):
        n = date_counts.get(r.date, 0)
        if n >= MAX_TRADES_PER_DATE:
            continue
        if busy_until is not None and r.timestamp < busy_until:
            continue
        rows.append(r._asdict())
        date_counts[r.date] = n + 1
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
