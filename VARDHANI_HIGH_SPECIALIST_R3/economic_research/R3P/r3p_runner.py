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
RV15_Q = 0.55
MOM15_Q = 0.65
MAX_TRADES_PER_DATE = 2
HORIZONS = (30, 60)


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    x['mom15_abs'] = x.n_ret_15.abs()
    x['mom30_abs'] = x.n_ret_30.abs()
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: R3P state year boundary invalid')
    return x


def checkpoint_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    cols = [
        'timestamp','date','year','month','n_rv_15','n_ret_5','n_ret_15','n_ret_30',
        's_ret_15','vix_ret_15','n_trend_eff_15','n_trend_eff_30','n_session_pos',
        'mom15_abs','mom30_abs','y_30','exit_ts_30','y_60','exit_ts_60'
    ]
    c = x.loc[cp.isin(CHECKPOINTS), cols].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['rv15_thr'] = np.nan
    c['mom15_thr'] = np.nan
    c['trend15_med'] = np.nan
    c['trend30_med'] = np.nan
    c['mom30_med'] = np.nan
    for name in CHECKPOINTS:
        z = c.loc[c.checkpoint == name].sort_values('timestamp')
        c.loc[z.index, 'rv15_thr'] = (
            z.n_rv_15.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(RV15_Q).shift(1).to_numpy()
        )
        c.loc[z.index, 'mom15_thr'] = (
            z.mom15_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(MOM15_Q).shift(1).to_numpy()
        )
        c.loc[z.index, 'trend15_med'] = (
            z.n_trend_eff_15.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .median().shift(1).to_numpy()
        )
        c.loc[z.index, 'trend30_med'] = (
            z.n_trend_eff_30.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .median().shift(1).to_numpy()
        )
        c.loc[z.index, 'mom30_med'] = (
            z.mom30_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .median().shift(1).to_numpy()
        )
    return c.reset_index(drop=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3P year must be 2023 or 2024')
    c = checkpoint_table(x)
    z = c[c.year == year].copy()
    req = [
        'n_rv_15','n_ret_5','n_ret_15','n_ret_30','s_ret_15','vix_ret_15',
        'n_trend_eff_15','n_trend_eff_30','n_session_pos','mom15_abs','mom30_abs',
        'rv15_thr','mom15_thr','trend15_med','trend30_med','mom30_med'
    ]
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z['dir_sign'] = np.sign(z.n_ret_15).astype(int)
    z = z[z.dir_sign != 0]
    z = z[z.n_rv_15 >= z.rv15_thr]
    z = z[z.mom15_abs >= z.mom15_thr]
    z = z[z.n_trend_eff_15 >= z.trend15_med]
    z = z[(z.dir_sign * z.s_ret_15) > 0.0]
    z = z[(z.dir_sign * z.vix_ret_15) < 0.0]
    z = z[(z.dir_sign * z.n_ret_5) > 0.0]
    z = z[
        ((z.dir_sign > 0) & (z.n_session_pos >= 0.60)) |
        ((z.dir_sign < 0) & (z.n_session_pos <= 0.40))
    ].copy()
    if z.empty:
        return z

    long_h = (z.n_trend_eff_30 >= z.trend30_med) & (z.mom30_abs >= z.mom30_med)
    z['horizon'] = np.where(long_h, 60, 30).astype(int)
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
        'exit_timestamp','n_rv_15','rv15_thr','n_ret_5','n_ret_15','n_ret_30',
        'mom15_abs','mom15_thr','mom30_abs','mom30_med','s_ret_15','vix_ret_15',
        'n_trend_eff_15','trend15_med','n_trend_eff_30','trend30_med','n_session_pos'
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
