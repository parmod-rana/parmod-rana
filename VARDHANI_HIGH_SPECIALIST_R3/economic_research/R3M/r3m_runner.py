from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3E_DIR=Path(__file__).resolve().parents[1]/'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0,str(R3E_DIR))
import r3e_runner as base

CHECKPOINTS=('10:00','10:15','10:30','10:45','11:00','11:15','11:30','11:45','12:00','12:15','12:30','12:45','13:00','13:15','13:30','13:45','14:00')
LOOKBACK_DATES=126
MIN_HISTORY=120
RV_Q=0.35
ATR_Q=0.50
IMPULSE_Q=0.75
MAX_TRADES_PER_DATE=2


def build_state(frames:dict[str,pd.DataFrame])->pd.DataFrame:
    x=base.add_targets(base.build_state(frames))
    x['pre_rv15']=x.groupby('date').n_rv_15.shift(5)
    x['pre_atr14_pct']=x.groupby('date').n_atr14_pct.shift(5)
    if set(x.year.unique())!={2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED: R3M state year boundary invalid')
    return x


def checkpoint_table(x:pd.DataFrame)->pd.DataFrame:
    cp=x.timestamp.dt.strftime('%H:%M')
    cols=['timestamp','date','year','month','pre_rv15','pre_atr14_pct','n_ret_5','n_ret_15','s_ret_5','vix_ret_5','n_trend_eff_5','n_trend_eff_15','n_session_pos','y_30','exit_ts_30','y_60','exit_ts_60']
    c=x.loc[cp.isin(CHECKPOINTS),cols].copy().sort_values('timestamp')
    c['checkpoint']=c.timestamp.dt.strftime('%H:%M')
    c['impulse_abs']=c.n_ret_5.abs()
    for k in ['rv_thr','atr_thr','impulse_thr','trend5_med','trend15_med']:
        c[k]=np.nan
    for name in CHECKPOINTS:
        z=c.loc[c.checkpoint==name].sort_values('timestamp')
        c.loc[z.index,'rv_thr']=z.pre_rv15.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(RV_Q).shift(1).to_numpy()
        c.loc[z.index,'atr_thr']=z.pre_atr14_pct.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(ATR_Q).shift(1).to_numpy()
        c.loc[z.index,'impulse_thr']=z.impulse_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(IMPULSE_Q).shift(1).to_numpy()
        c.loc[z.index,'trend5_med']=z.n_trend_eff_5.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
        c.loc[z.index,'trend15_med']=z.n_trend_eff_15.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
    return c.reset_index(drop=True)


def build_year_candidates(x:pd.DataFrame,year:int)->pd.DataFrame:
    if year not in (2023,2024):
        raise RuntimeError('FAIL_CLOSED: R3M year must be 2023 or 2024')
    c=checkpoint_table(x)
    z=c[c.year==year].copy()
    req=['pre_rv15','pre_atr14_pct','rv_thr','atr_thr','impulse_abs','impulse_thr','n_ret_5','n_ret_15','s_ret_5','vix_ret_5','n_trend_eff_5','n_trend_eff_15','trend5_med','trend15_med','n_session_pos']
    z=z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z=z[(z.pre_rv15<=z.rv_thr)&(z.pre_atr14_pct<=z.atr_thr)&(z.impulse_abs>=z.impulse_thr)].copy()
    z['dir_sign']=np.sign(z.n_ret_5).astype(int)
    z=z[z.dir_sign!=0]
    z=z[(z.dir_sign*z.n_ret_15)>0]
    z=z[(z.dir_sign*z.s_ret_5)>0]
    z=z[(z.dir_sign*z.vix_ret_5)<0]
    z=z[z.n_trend_eff_5>=z.trend5_med]
    z=z[((z.dir_sign>0)&(z.n_session_pos>=0.80))|((z.dir_sign<0)&(z.n_session_pos<=0.20))].copy()
    z['horizon']=np.where(z.n_trend_eff_15>=z.trend15_med,60,30).astype(int)
    z['side']=np.where(z.dir_sign>0,'CE','PE')
    z['gross_underlying']=np.where(z.horizon==60,z.dir_sign*z.y_60,z.dir_sign*z.y_30)
    z['exit_timestamp']=z.exit_ts_30
    m=z.horizon==60; z.loc[m,'exit_timestamp']=z.loc[m,'exit_ts_60']
    keep=['timestamp','date','month','checkpoint','side','horizon','gross_underlying','exit_timestamp','pre_rv15','pre_atr14_pct','n_ret_5','n_ret_15','s_ret_5','vix_ret_5','n_trend_eff_5','n_trend_eff_15','n_session_pos','rv_thr','atr_thr','impulse_thr','trend5_med','trend15_med']
    return z[keep].sort_values('timestamp').reset_index(drop=True)


def simulate(c:pd.DataFrame)->pd.DataFrame:
    if c.empty: return c
    z=c[np.isfinite(c.gross_underlying)&c.exit_timestamp.notna()].sort_values('timestamp').copy()
    rows=[]; busy=None; counts={}
    for r in z.itertuples(index=False):
        d=r.date; n=counts.get(d,0)
        if n>=MAX_TRADES_PER_DATE: continue
        if busy is not None and r.timestamp<busy: continue
        rows.append(r._asdict()); counts[d]=n+1; busy=r.exit_timestamp
    out=pd.DataFrame(rows)
    if out.empty: return out
    out['net3']=out.gross_underlying-3.0; out['net5']=out.gross_underlying-5.0
    out['half']=np.where(pd.to_datetime(out.timestamp).dt.month<=6,'H1','H2')
    return out


def summarize(trades:pd.DataFrame)->dict:
    return base.summarize(trades)
