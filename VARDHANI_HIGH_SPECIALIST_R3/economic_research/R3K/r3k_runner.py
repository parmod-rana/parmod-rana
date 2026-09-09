from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

INITIAL_CHECKPOINTS = ('09:45','09:50','09:55','10:00','10:05','10:10','10:15','10:20','10:25')
FIXED_Q = 0.70
LOOKBACK_DATES = 126
MIN_HISTORY = 120
HORIZONS = (60,120)
CONFIRM_DELAY_MIN = 5


def build_state(frames: dict[str,pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    entry = x.groupby('date').n_open.shift(-1)
    exit120 = x.groupby('date').n_close.shift(-120)
    x['y_120'] = (exit120 / entry - 1.0) * 10000.0
    x['exit_ts_120'] = x.groupby('date').timestamp.shift(-120)
    if set(x.year.unique()) != {2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED: R3K state year boundary invalid')
    return x


def threshold_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    init = x.loc[cp.isin(INITIAL_CHECKPOINTS), ['timestamp','date','year','month','n_opening_gap_pct','n_ret_30','s_ret_15','vix_ret_15','n_session_pos']].copy()
    init['checkpoint'] = init.timestamp.dt.strftime('%H:%M')
    init['gap_abs'] = init.n_opening_gap_pct.abs(); init['mom_abs'] = init.n_ret_30.abs()
    init['gap_thr'] = np.nan; init['mom_thr'] = np.nan
    for name in INITIAL_CHECKPOINTS:
        z = init.loc[init.checkpoint==name].sort_values('timestamp')
        init.loc[z.index,'gap_thr'] = z.gap_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(FIXED_Q).shift(1).to_numpy()
        init.loc[z.index,'mom_thr'] = z.mom_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(FIXED_Q).shift(1).to_numpy()

    confirms = tuple((pd.Timestamp('2000-01-01 '+s)+pd.Timedelta(minutes=CONFIRM_DELAY_MIN)).strftime('%H:%M') for s in INITIAL_CHECKPOINTS)
    cf = x.loc[cp.isin(confirms), ['timestamp','n_ret_5','s_ret_5','vix_ret_5','n_session_pos','n_trend_eff_30','y_60','exit_ts_60','y_120','exit_ts_120']].copy()
    cf['confirm_checkpoint'] = cf.timestamp.dt.strftime('%H:%M')
    cf['trend_med'] = np.nan
    for name in confirms:
        z = cf.loc[cf.confirm_checkpoint==name].sort_values('timestamp')
        cf.loc[z.index,'trend_med'] = z.n_trend_eff_30.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
    cf = cf.rename(columns={c:'c_'+c for c in ['n_ret_5','s_ret_5','vix_ret_5','n_session_pos','n_trend_eff_30','y_60','exit_ts_60','y_120','exit_ts_120','trend_med','confirm_checkpoint']})
    init['confirm_timestamp'] = init.timestamp + pd.Timedelta(minutes=CONFIRM_DELAY_MIN)
    return init.merge(cf, left_on='confirm_timestamp', right_on='timestamp', how='left', suffixes=('','_confirm')).drop(columns=['timestamp_confirm'])


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023,2024):
        raise RuntimeError('FAIL_CLOSED: R3K year must be 2023 or 2024')
    c = threshold_table(x)
    c = c[c.year==year].copy()
    req = ['gap_abs','mom_abs','gap_thr','mom_thr','n_ret_30','s_ret_15','vix_ret_15','n_session_pos','c_n_ret_5','c_s_ret_5','c_vix_ret_5','c_n_session_pos','c_n_trend_eff_30','c_trend_med']
    c = c[np.logical_and.reduce([np.isfinite(c[k]) for k in req])].copy()
    c['sign'] = np.sign(c.n_ret_30).astype(int)
    c = c[c.sign != 0]
    c = c[(c.gap_abs>=c.gap_thr) & (c.mom_abs>=c.mom_thr)]
    c = c[(c.sign*c.s_ret_15)>0.0]
    c = c[(c.sign*c.vix_ret_15)<0.0]
    c = c[((c.sign>0)&(c.n_session_pos>=0.65)) | ((c.sign<0)&(c.n_session_pos<=0.35))]
    c = c[(c.sign*c.c_n_ret_5)>0.0]
    c = c[(c.sign*c.c_s_ret_5)>0.0]
    c = c[(c.sign*c.c_vix_ret_5)<0.0]
    c = c[((c.sign>0)&(c.c_n_session_pos>=0.65)) | ((c.sign<0)&(c.c_n_session_pos<=0.35))].copy()
    c = c.sort_values('confirm_timestamp').drop_duplicates('date', keep='first').copy()
    c['horizon'] = np.where(c.c_n_trend_eff_30>=c.c_trend_med,120,60).astype(int)
    c['side'] = np.where(c.sign>0,'CE','PE')
    c['gross_underlying'] = np.where(c.horizon==120,c.sign*c.c_y_120,c.sign*c.c_y_60)
    c['exit_timestamp'] = c.c_exit_ts_60
    m120=c.horizon==120; c.loc[m120,'exit_timestamp']=c.loc[m120,'c_exit_ts_120']
    return c[['timestamp','confirm_timestamp','date','month','checkpoint','c_confirm_checkpoint','side','horizon','gross_underlying','exit_timestamp','n_opening_gap_pct','n_ret_30','s_ret_15','vix_ret_15','n_session_pos','c_n_ret_5','c_s_ret_5','c_vix_ret_5','c_n_session_pos','c_n_trend_eff_30','gap_thr','mom_thr','c_trend_med']].sort_values('confirm_timestamp').reset_index(drop=True)


def simulate(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty: return c
    c=c[np.isfinite(c.gross_underlying)&c.exit_timestamp.notna()].sort_values('confirm_timestamp').copy()
    rows=[]; busy=None
    for r in c.itertuples(index=False):
        if busy is not None and r.confirm_timestamp < busy: continue
        rows.append(r._asdict()); busy=r.exit_timestamp
    out=pd.DataFrame(rows)
    if out.empty: return out
    out['net3']=out.gross_underlying-3.0; out['net5']=out.gross_underlying-5.0
    out['half']=np.where(pd.to_datetime(out.confirm_timestamp).dt.month<=6,'H1','H2')
    return out


def summarize(trades: pd.DataFrame)->dict:
    return base.summarize(trades)
