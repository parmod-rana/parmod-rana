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
SENSEX_IMPULSE_Q = 0.70
LEAD_GAP_Q = 0.60
EXTREME_LEAD_GAP_Q = 0.85
MAX_TRADES_PER_DATE = 2
HORIZONS = (30, 60)


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    x['s15_abs'] = x.s_ret_15.abs()
    x['n15_abs'] = x.n_ret_15.abs()
    x['n5_abs'] = x.n_ret_5.abs()
    x['lead_gap'] = x.s15_abs - x.n15_abs
    if set(x.year.unique()) != {2022, 2023, 2024}:
        raise RuntimeError('FAIL_CLOSED: R3S state year boundary invalid')
    return x


def _positive_gap_thresholds(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q60 = np.full(len(z), np.nan, dtype=float)
    q85 = np.full(len(z), np.nan, dtype=float)
    vals = z.lead_gap.to_numpy(dtype=float)
    for i in range(len(z)):
        lo = max(0, i - LOOKBACK_DATES)
        hist = vals[lo:i]
        if hist.size < MIN_HISTORY:
            continue
        pos = hist[np.isfinite(hist) & (hist > 0.0)]
        if pos.size == 0:
            continue
        q60[i] = float(np.quantile(pos, LEAD_GAP_Q))
        q85[i] = float(np.quantile(pos, EXTREME_LEAD_GAP_Q))
    return q60, q85


def checkpoint_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    cols = [
        'timestamp','date','year','month','s_ret_15','n_ret_15','n_ret_5','vix_ret_5',
        'n_session_pos','s15_abs','n15_abs','n5_abs','lead_gap',
        'y_30','exit_ts_30','y_60','exit_ts_60'
    ]
    c = x.loc[cp.isin(CHECKPOINTS), cols].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['s_impulse_thr'] = np.nan
    c['n5_med'] = np.nan
    c['lead_gap_thr'] = np.nan
    c['lead_gap_q85'] = np.nan
    for name in CHECKPOINTS:
        z = c.loc[c.checkpoint == name].sort_values('timestamp')
        c.loc[z.index, 's_impulse_thr'] = (
            z.s15_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .quantile(SENSEX_IMPULSE_Q).shift(1).to_numpy()
        )
        c.loc[z.index, 'n5_med'] = (
            z.n5_abs.rolling(LOOKBACK_DATES, min_periods=MIN_HISTORY)
            .median().shift(1).to_numpy()
        )
        q60, q85 = _positive_gap_thresholds(z)
        c.loc[z.index, 'lead_gap_thr'] = q60
        c.loc[z.index, 'lead_gap_q85'] = q85
    return c.reset_index(drop=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError('FAIL_CLOSED: R3S year must be 2023 or 2024')
    c = checkpoint_table(x)
    z = c[c.year == year].copy()
    req = [
        's_ret_15','n_ret_15','n_ret_5','vix_ret_5','n_session_pos',
        's15_abs','n15_abs','n5_abs','lead_gap','s_impulse_thr','n5_med',
        'lead_gap_thr','lead_gap_q85'
    ]
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z['dir_sign'] = np.sign(z.s_ret_15).astype(int)
    z = z[z.dir_sign != 0]
    z = z[z.s15_abs >= z.s_impulse_thr]
    z = z[(z.dir_sign * z.n_ret_15) > 0.0]
    z = z[z.s15_abs > z.n15_abs]
    z = z[z.lead_gap >= z.lead_gap_thr]
    z = z[(z.dir_sign * z.n_ret_5) > 0.0]
    z = z[z.n5_abs >= z.n5_med]
    z = z[(z.dir_sign * z.vix_ret_5) < 0.0]
    z = z[
        ((z.dir_sign > 0) & (z.n_session_pos >= 0.55)) |
        ((z.dir_sign < 0) & (z.n_session_pos <= 0.45))
    ].copy()
    if z.empty:
        return z

    z['horizon'] = np.where(z.lead_gap >= z.lead_gap_q85, 60, 30).astype(int)
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
        'exit_timestamp','s_ret_15','n_ret_15','n_ret_5','vix_ret_5','s15_abs',
        'n15_abs','lead_gap','s_impulse_thr','lead_gap_thr','lead_gap_q85','n5_abs',
        'n5_med','n_session_pos'
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
