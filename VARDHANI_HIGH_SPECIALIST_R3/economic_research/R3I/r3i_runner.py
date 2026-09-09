from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

CHECKPOINT_STRINGS = (
    '09:45','10:00','10:15','10:30','10:45','11:00','11:15',
    '11:30','11:45','12:00','12:15','12:30','12:45','13:00'
)
FIXED_Q = 0.70
LOOKBACK_DATES = 126
MIN_HISTORY = 120
HORIZONS = (60,120)


def build_state(frames: dict[str,pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    entry = x.groupby('date').n_open.shift(-1)
    exit120 = x.groupby('date').n_close.shift(-120)
    x['y_120'] = (exit120 / entry - 1.0) * 10000.0
    x['exit_ts_120'] = x.groupby('date').timestamp.shift(-120)
    if set(x.year.unique()) != {2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED: R3I state year boundary invalid')
    return x


def checkpoint_frame(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    c = x.loc[cp.isin(CHECKPOINT_STRINGS)].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['gap_abs'] = c.n_opening_gap_pct.abs()
    c['mom_abs'] = c.n_ret_30.abs()
    c['gap_thr'] = np.nan
    c['mom_thr'] = np.nan
    c['trend_med'] = np.nan
    for name in CHECKPOINT_STRINGS:
        z = c.loc[c.checkpoint == name].sort_values('timestamp')
        c.loc[z.index,'gap_thr'] = z.gap_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(FIXED_Q).shift(1).to_numpy()
        c.loc[z.index,'mom_thr'] = z.mom_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(FIXED_Q).shift(1).to_numpy()
        c.loc[z.index,'trend_med'] = z.n_trend_eff_30.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
    return c.sort_values('timestamp').reset_index(drop=True)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023,2024):
        raise RuntimeError('FAIL_CLOSED: R3I year must be 2023 or 2024')
    c = checkpoint_frame(x)
    c = c[c.timestamp.dt.year == year].copy()
    finite = (
        np.isfinite(c.gap_abs)&np.isfinite(c.mom_abs)&np.isfinite(c.gap_thr)&np.isfinite(c.mom_thr)
        &np.isfinite(c.n_ret_30)&np.isfinite(c.s_ret_15)&np.isfinite(c.vix_ret_15)
        &np.isfinite(c.n_session_pos)&np.isfinite(c.n_trend_eff_30)&np.isfinite(c.trend_med)
    )
    c = c[finite].copy()
    c['sign'] = np.sign(c.n_ret_30).astype(int)
    c = c[c.sign != 0]
    c = c[(c.gap_abs >= c.gap_thr) & (c.mom_abs >= c.mom_thr)]
    c = c[(c.sign*c.s_ret_15) > 0.0]
    c = c[(c.sign*c.vix_ret_15) < 0.0]
    c = c[((c.sign>0)&(c.n_session_pos>=0.65)) | ((c.sign<0)&(c.n_session_pos<=0.35))].copy()
    c['horizon'] = np.where(c.n_trend_eff_30 >= c.trend_med,120,60).astype(int)
    c['side'] = np.where(c.sign>0,'CE','PE')
    c['gross_underlying'] = np.where(c.horizon==120,c.sign*c.y_120,c.sign*c.y_60)
    c['exit_timestamp'] = c.exit_ts_60
    m120 = c.horizon==120
    c.loc[m120,'exit_timestamp'] = c.loc[m120,'exit_ts_120']
    return c[[
        'timestamp','date','month','checkpoint','side','horizon','gross_underlying','exit_timestamp',
        'n_opening_gap_pct','n_ret_30','s_ret_15','vix_ret_15','n_session_pos','n_trend_eff_30',
        'gap_thr','mom_thr','trend_med'
    ]].sort_values('timestamp').reset_index(drop=True)


def simulate(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty: return c
    c = c[np.isfinite(c.gross_underlying) & c.exit_timestamp.notna()].sort_values('timestamp').copy()
    rows=[]; busy=None
    for r in c.itertuples(index=False):
        if busy is not None and r.timestamp < busy: continue
        rows.append(r._asdict()); busy=r.exit_timestamp
    out=pd.DataFrame(rows)
    if out.empty: return out
    out['net3']=out.gross_underlying-3.0
    out['net5']=out.gross_underlying-5.0
    out['half']=np.where(pd.to_datetime(out.timestamp).dt.month<=6,'H1','H2')
    return out


def summarize(trades: pd.DataFrame)->dict:
    return base.summarize(trades)
