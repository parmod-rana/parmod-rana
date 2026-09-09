from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3E_DIR = Path(__file__).resolve().parents[1] / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_runner as base

CHECKPOINTS = ('09:45','10:00','10:15','10:30','10:45','11:00')
LOOKBACK_DATES = 126
MIN_HISTORY = 120
DISLOCATION_Q = 0.60
WICK_Q = 0.60


def build_state(frames: dict[str,pd.DataFrame]) -> pd.DataFrame:
    x = base.add_targets(base.build_state(frames))
    entry = x.groupby('date').n_open.shift(-1)
    exit120 = x.groupby('date').n_close.shift(-120)
    x['y_120'] = (exit120/entry - 1.0)*10000.0
    x['exit_ts_120'] = x.groupby('date').timestamp.shift(-120)
    if set(x.year.unique()) != {2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED: R3L state year boundary invalid')
    return x


def checkpoint_table(x: pd.DataFrame) -> pd.DataFrame:
    cp = x.timestamp.dt.strftime('%H:%M')
    cols = [
        'timestamp','date','year','month','n_opening_gap_pct','n_ret_5','n_ret_15','n_ret_30',
        's_ret_5','s_ret_15','vix_ret_5','vix_ret_15','n_session_pos','n_body_frac',
        'n_upper_wick_frac','n_lower_wick_frac','n_trend_eff_15','n_trend_eff_30',
        'y_30','exit_ts_30','y_60','exit_ts_60','y_120','exit_ts_120'
    ]
    c = x.loc[cp.isin(CHECKPOINTS), cols].copy().sort_values('timestamp')
    c['checkpoint'] = c.timestamp.dt.strftime('%H:%M')
    c['gap_abs'] = c.n_opening_gap_pct.abs(); c['mom_abs'] = c.n_ret_30.abs()
    for name in ['gap_thr','mom_thr','trend15_med','trend30_med','upper_wick_thr','lower_wick_thr']:
        c[name] = np.nan
    for name in CHECKPOINTS:
        z = c.loc[c.checkpoint==name].sort_values('timestamp')
        c.loc[z.index,'gap_thr'] = z.gap_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(DISLOCATION_Q).shift(1).to_numpy()
        c.loc[z.index,'mom_thr'] = z.mom_abs.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(DISLOCATION_Q).shift(1).to_numpy()
        c.loc[z.index,'trend15_med'] = z.n_trend_eff_15.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
        c.loc[z.index,'trend30_med'] = z.n_trend_eff_30.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).median().shift(1).to_numpy()
        c.loc[z.index,'upper_wick_thr'] = z.n_upper_wick_frac.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(WICK_Q).shift(1).to_numpy()
        c.loc[z.index,'lower_wick_thr'] = z.n_lower_wick_frac.rolling(LOOKBACK_DATES,min_periods=MIN_HISTORY).quantile(WICK_Q).shift(1).to_numpy()
    return c.reset_index(drop=True)


def _common(c: pd.DataFrame, year: int) -> pd.DataFrame:
    z = c[c.year==year].copy()
    req = ['gap_abs','mom_abs','gap_thr','mom_thr','n_ret_30','n_session_pos','n_trend_eff_15','n_trend_eff_30','trend15_med','trend30_med','upper_wick_thr','lower_wick_thr']
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z['disp_sign'] = np.sign(z.n_ret_30).astype(int)
    z = z[z.disp_sign != 0]
    z = z[(z.gap_abs>=z.gap_thr)&(z.mom_abs>=z.mom_thr)].copy()
    return z


def continuation_candidates(c: pd.DataFrame, year: int) -> pd.DataFrame:
    z = _common(c,year)
    req = ['n_ret_5','n_ret_15','s_ret_15','vix_ret_15']
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z = z[(z.disp_sign*z.n_ret_5)>0]
    z = z[(z.disp_sign*z.n_ret_15)>0]
    z = z[(z.disp_sign*z.s_ret_15)>0]
    z = z[(z.disp_sign*z.vix_ret_15)<0]
    z = z[z.n_trend_eff_15>=z.trend15_med]
    z = z[((z.disp_sign>0)&(z.n_session_pos>=0.65))|((z.disp_sign<0)&(z.n_session_pos<=0.35))].copy()
    z = z.sort_values('timestamp').drop_duplicates('date',keep='first').copy()
    z['specialist']='CONT'; z['dir_sign']=z.disp_sign
    z['horizon']=np.where(z.n_trend_eff_30>=z.trend30_med,120,60).astype(int)
    z['gross_underlying']=np.where(z.horizon==120,z.dir_sign*z.y_120,z.dir_sign*z.y_60)
    z['exit_timestamp']=z.exit_ts_60
    m=z.horizon==120; z.loc[m,'exit_timestamp']=z.loc[m,'exit_ts_120']
    return z


def rejection_candidates(c: pd.DataFrame, year: int) -> pd.DataFrame:
    z = _common(c,year)
    req = ['n_ret_5','s_ret_5','vix_ret_5','n_body_frac','n_upper_wick_frac','n_lower_wick_frac']
    z = z[np.logical_and.reduce([np.isfinite(z[k]) for k in req])].copy()
    z = z[(z.disp_sign*z.n_ret_5)<0]
    z = z[(z.disp_sign*z.n_body_frac)<0]
    wick_ok = ((z.disp_sign>0)&(z.n_upper_wick_frac>=z.upper_wick_thr)) | ((z.disp_sign<0)&(z.n_lower_wick_frac>=z.lower_wick_thr))
    z = z[wick_ok]
    z = z[(z.disp_sign*z.s_ret_5)<=0]
    z = z[(z.disp_sign*z.vix_ret_5)>=0].copy()
    z = z.sort_values('timestamp').drop_duplicates('date',keep='first').copy()
    z['specialist']='REJECT'; z['dir_sign']=-z.disp_sign
    z['horizon']=np.where(z.n_trend_eff_15>=z.trend15_med,60,30).astype(int)
    z['gross_underlying']=np.where(z.horizon==60,z.dir_sign*z.y_60,z.dir_sign*z.y_30)
    z['exit_timestamp']=z.exit_ts_30
    m=z.horizon==60; z.loc[m,'exit_timestamp']=z.loc[m,'exit_ts_60']
    return z


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023,2024):
        raise RuntimeError('FAIL_CLOSED: R3L year must be 2023 or 2024')
    c=checkpoint_table(x)
    a=continuation_candidates(c,year); b=rejection_candidates(c,year)
    allc=pd.concat([a,b],ignore_index=True) if (not a.empty or not b.empty) else pd.DataFrame()
    if allc.empty: return allc
    same=allc.groupby(['timestamp','date']).specialist.nunique()
    conflicts=set(same[same>1].index)
    if conflicts:
        allc=allc[~allc.set_index(['timestamp','date']).index.isin(conflicts)].copy()
    allc['side']=np.where(allc.dir_sign>0,'CE','PE')
    keep=['timestamp','date','month','checkpoint','specialist','side','horizon','gross_underlying','exit_timestamp','n_opening_gap_pct','n_ret_5','n_ret_15','n_ret_30','s_ret_5','s_ret_15','vix_ret_5','vix_ret_15','n_session_pos','n_body_frac','n_upper_wick_frac','n_lower_wick_frac','n_trend_eff_15','n_trend_eff_30','gap_thr','mom_thr','trend15_med','trend30_med','upper_wick_thr','lower_wick_thr']
    return allc[keep].sort_values('timestamp').reset_index(drop=True)


def simulate(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty: return c
    z=c[np.isfinite(c.gross_underlying)&c.exit_timestamp.notna()].sort_values(['timestamp','specialist']).copy()
    rows=[]; busy=None
    for r in z.itertuples(index=False):
        if busy is not None and r.timestamp<busy: continue
        rows.append(r._asdict()); busy=r.exit_timestamp
    out=pd.DataFrame(rows)
    if out.empty: return out
    out['net3']=out.gross_underlying-3.0; out['net5']=out.gross_underlying-5.0
    out['half']=np.where(pd.to_datetime(out.timestamp).dt.month<=6,'H1','H2')
    return out


def summarize(trades: pd.DataFrame)->dict:
    out=base.summarize(trades)
    if not trades.empty:
        out['specialists']={}
        for name,g in trades.groupby('specialist'):
            out['specialists'][str(name)]={'n':int(len(g)),'net3':base.stats(g.net3),'net5':base.stats(g.net5)}
    return out
