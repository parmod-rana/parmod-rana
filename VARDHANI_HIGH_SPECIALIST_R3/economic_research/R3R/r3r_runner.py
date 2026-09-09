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
    for i in range(25)
)
LOOKBACK_DATES = 126
MIN_HISTORY = 120
VIX_SHOCK_Q = 0.75
VIX_EXTREME_Q = 0.90
MAX_TRADES_PER_DATE = 2
HORIZONS = (30, 60)


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    x['vix15_abs'] = x.vix_ret_15.abs()
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: R3R state year boundary invalid')
    return x


def checkpoint_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    cols = [
        'timestamp','date','year','month','vix_ret_15','vix15_abs','n_ret_5','n_ret_15',
        's_ret_15','n_session_pos','y_30','exit_ts_30','y_60','exit_ts_60'
    ]
    c = x.loc[cp.isin(CHECKPOINTS), cols].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['vix_q75'] = np.nan
    c['vix_q90'] = np.nan
    for name in CHECKPOINTS:
        z = c.loc[c.checkpoint == name].sort_values('timestamp')
        c.loc[z.index, 'vix_q75'] = (
            z.vix15_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(VIX_SHOCK_Q).shift(1).to_numpy()
        )
        c.loc[z.index, 'vix_q90'] = (
            z.vix15_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(VIX_EXTREME_Q).shift(1).to_numpy()
        )
    return c.reset_index(drop=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3R year must be 2023 or 2024')
    c = checkpoint_table(x)
    z = c[c.year == year].copy()
    req = ['vix_ret_15','vix15_abs','n_ret_5','n_ret_15','s_ret_15','n_session_pos','vix_q75','vix_q90']
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z = z[z.vix_ret_15 != 0.0].copy()
    z['dir_sign'] = -np.sign(z.vix_ret_15).astype(int)
    z = z[z.vix15_abs >= z.vix_q75]
    z = z[(z.dir_sign * z.n_ret_15) > 0.0]
    z = z[(z.dir_sign * z.n_ret_5) > 0.0]
    z = z[(z.dir_sign * z.s_ret_15) > 0.0]
    z = z[
        ((z.dir_sign < 0) & (z.n_session_pos <= 0.45)) |
        ((z.dir_sign > 0) & (z.n_session_pos >= 0.55))
    ].copy()
    if z.empty:
        return z

    z['horizon'] = np.where(z.vix15_abs >= z.vix_q90, 60, 30).astype(int)
    z['side'] = np.where(z.dir_sign > 0, 'CE', 'PE')
    z['gross_underlying'] = np.where(
        z.horizon == 60,
        z.dir_sign * z.y_60,
        z.dir_sign * z.y_30,
    )
    z['exit_timestamp'] = z.exit_ts_30
    m60 = z.horizon == 60
    z.loc[m60, 'exit_timestamp'] = z.loc[m60, 'exit_ts_60']
    keep = [
        'timestamp','date','month','checkpoint','side','horizon','gross_underlying',
        'exit_timestamp','vix_ret_15','vix15_abs','vix_q75','vix_q90','n_ret_5',
        'n_ret_15','s_ret_15','n_session_pos'
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
