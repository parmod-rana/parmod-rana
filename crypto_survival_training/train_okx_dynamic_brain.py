from __future__ import annotations
import json, math, os, time, hashlib, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests, joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# GLOBAL CRYPTO HUNTER R1F / GENERATION 3
# Dynamic policy brain: NO fixed trend/breakout/mean-reversion trading authority.
# The neural policy learns action values directly for LONG/SHORT/WAIT,
# holding horizon and exposure from causal market state and realized outcomes.

BASE='https://www.okx.com'; BAR='4H'; BAR_MS=14_400_000
START_MS=int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()*1000)
VAL_START=int(pd.Timestamp('2026-07-01',tz='UTC').timestamp()*1000)
FINAL_START=int(pd.Timestamp('2026-08-01',tz='UTC').timestamp()*1000)
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','72'))
COST_BPS=float(os.getenv('ROUNDTRIP_COST_BPS','30'))
OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen3_dynamic')); OUT.mkdir(parents=True,exist_ok=True)
HORIZONS=(1,2,6)                 # 4h, 8h, 24h
SIZES=(0.5,1.0)                  # learned action includes exposure
MAX_H=max(HORIZONS)
RISK_PENALTIES=(0.12,0.28)       # validation chooses objective calibration; not a trading strategy
UNCERTAINTY_PENALTIES=(0.0,0.5,1.0)
CAPACITIES=(2.0,3.0,5.0)         # hard portfolio safety capacities evaluated only on validation
SEEDS=(17,43)
MAX_TRAIN_ROWS=int(os.getenv('MAX_TRAIN_ROWS','180000'))
_tls=threading.local(); _rate_lock=threading.Lock(); _next_req=0.0

def session():
    if not hasattr(_tls,'s'):
        _tls.s=requests.Session(); _tls.s.headers['User-Agent']='GLOBAL-CRYPTO-HUNTER-R1F-DYNAMIC/1.0'
    return _tls.s

def throttle():
    global _next_req
    with _rate_lock:
        now=time.monotonic(); target=max(now,_next_req); _next_req=target+0.115
    if target>now: time.sleep(target-now)

def get(path,params=None,retries=8):
    last=None
    for i in range(retries):
        try:
            throttle(); r=session().get(BASE+path,params=params,timeout=30); last=r
            if r.status_code==200:
                j=r.json()
                if str(j.get('code','0'))=='0': return j.get('data') or []
        except Exception as e: last=e
        time.sleep(min(6,0.35*(2**i)))
    raise RuntimeError(f'OKX failed {path} {params}: {last}')

def discover():
    inst=get('/api/v5/public/instruments',{'instType':'SWAP'})
    # We intentionally do not use future price performance for universe selection.
    # Long-lived currently-live crypto swaps are used for broad behavioral training;
    # the live app still scans the full executable universe dynamically.
    rows=[]
    for x in inst:
        if x.get('state')!='live' or x.get('settleCcy')!='USDT': continue
        if str(x.get('groupId') or '') in {'6','7'}: continue   # OKX RWA groups, not crypto
        iid=x.get('instId','')
        if not iid.endswith('-USDT-SWAP'): continue
        try: list_time=int(x.get('listTime') or 0)
        except: list_time=0
        if not list_time: continue
        rows.append((list_time,iid))
    rows.sort(key=lambda z:(z[0],z[1]))
    return [iid for _,iid in rows[:MAX_SYMBOLS]]

def fetch_history(inst):
    rows=[]; after=None; calls=0
    for _ in range(120):
        p={'instId':inst,'bar':BAR,'limit':'100'}
        if after is not None: p['after']=str(after)
        data=get('/api/v5/market/history-candles',p); calls+=1
        if not data: break
        for z in data:
            if len(z)>=6 and (len(z)<9 or str(z[8])=='1'):
                rows.append([int(z[0]),float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5])])
        oldest=min(int(z[0]) for z in data)
        if oldest<=START_MS or (after is not None and oldest>=after): break
        after=oldest
    if not rows: return inst,pd.DataFrame(),calls
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts')
    df=df[df.ts>=START_MS].reset_index(drop=True)
    return inst,df,calls

def individual_frame(raw, symbol):
    d=raw.copy(); c=d.close.astype(float); o=d.open.astype(float); h=d.high.astype(float); l=d.low.astype(float); v=d.volume.astype(float)
    for k in (1,2,3,6,12,24,42): d[f'ret{k}']=(c/c.shift(k)-1.0)*10000.0
    r1=d.ret1
    for k in (6,12,24,42): d[f'rv{k}']=r1.rolling(k,min_periods=k).std()
    d['range_bps']=(h-l)/c.replace(0,np.nan)*10000.0
    d['body_bps']=(c-o)/o.replace(0,np.nan)*10000.0
    d['close_pos']=(c-l)/(h-l).replace(0,np.nan)
    lv=np.log1p(v.clip(lower=0)); d['vol_z24']=(lv-lv.rolling(24,min_periods=12).mean())/lv.rolling(24,min_periods=12).std().replace(0,np.nan)
    d['vol_chg6']=(v/v.shift(6).replace(0,np.nan)-1.0).clip(-10,10)
    d['mean_dev12']=(c/c.rolling(12,min_periods=12).mean()-1.0)*10000.0
    d['mean_dev42']=(c/c.rolling(42,min_periods=42).mean()-1.0)*10000.0
    # Crypto is 24/7. Cyclical clock features let the network learn recurring liquidity/session behavior itself.
    dt=pd.to_datetime(d.ts,unit='ms',utc=True); how=(dt.dt.dayofweek*24+dt.dt.hour).astype(float)
    d['week_sin']=np.sin(2*np.pi*how/168.0); d['week_cos']=np.cos(2*np.pi*how/168.0)
    d['day_sin']=np.sin(2*np.pi*dt.dt.hour.astype(float)/24.0); d['day_cos']=np.cos(2*np.pi*dt.dt.hour.astype(float)/24.0)
    # Outcome tensors. These are never included in FEATURES.
    for hh in HORIZONS:
        future_close=c.shift(-hh); d[f'fret_{hh}']=(future_close/c-1.0)*10000.0
        lows=pd.concat([l.shift(-j) for j in range(1,hh+1)],axis=1).min(axis=1)
        highs=pd.concat([h.shift(-j) for j in range(1,hh+1)],axis=1).max(axis=1)
        d[f'long_mae_{hh}']=((c-lows)/c*10000.0).clip(lower=0)
        d[f'short_mae_{hh}']=((highs-c)/c*10000.0).clip(lower=0)
    d['symbol']=symbol
    return d

BASE_FEATURES=['ret1','ret2','ret3','ret6','ret12','ret24','ret42','rv6','rv12','rv24','rv42','range_bps','body_bps','close_pos','vol_z24','vol_chg6','mean_dev12','mean_dev42','week_sin','week_cos','day_sin','day_cos']
MARKET_FEATURES=['mkt_ret1','mkt_ret6','mkt_ret24','breadth1','breadth6','dispersion1','dispersion6','rs1','rs6','rs24','rank_ret1','rank_ret6','rank_ret24','rank_rv24','rank_volume']
FEATURES=BASE_FEATURES+MARKET_FEATURES

def add_market_state(ds):
    g=ds.groupby('ts',sort=False)
    ds['mkt_ret1']=g.ret1.transform('median'); ds['mkt_ret6']=g.ret6.transform('median'); ds['mkt_ret24']=g.ret24.transform('median')
    ds['breadth1']=g.ret1.transform(lambda x:(x>0).mean()); ds['breadth6']=g.ret6.transform(lambda x:(x>0).mean())
    ds['dispersion1']=g.ret1.transform('std'); ds['dispersion6']=g.ret6.transform('std')
    ds['rs1']=ds.ret1-ds.mkt_ret1; ds['rs6']=ds.ret6-ds.mkt_ret6; ds['rs24']=ds.ret24-ds.mkt_ret24
    ds['rank_ret1']=g.ret1.rank(pct=True); ds['rank_ret6']=g.ret6.rank(pct=True); ds['rank_ret24']=g.ret24.rank(pct=True)
    ds['rank_rv24']=g.rv24.rank(pct=True); ds['rank_volume']=g.vol_z24.rank(pct=True)
    return ds

def action_specs():
    out=[]
    for hh in HORIZONS:
        for size in SIZES:
            out.append({'name':f'LONG_{size:g}_{hh*4}H','dir':1,'size':float(size),'h':hh})
            out.append({'name':f'SHORT_{size:g}_{hh*4}H','dir':-1,'size':float(size),'h':hh})
    return out
ACTIONS=action_specs()

def reward_matrix(df, risk_penalty):
    ys=[]
    for a in ACTIONS:
        f=df[f'fret_{a["h"]}'].to_numpy(float); mae=df[('long_mae_' if a['dir']>0 else 'short_mae_')+str(a['h'])].to_numpy(float)
        # Risk grows super-linearly with exposure, allowing the learned policy to prefer 0.5x in unstable states.
        raw=a['size']*a['dir']*f - COST_BPS*a['size']
        risk=risk_penalty*(a['size']**1.55)*mae
        ys.append(raw-risk)
    return np.column_stack(ys)

def actual_net(df, action_idx):
    a=ACTIONS[int(action_idx)]
    return a['size']*a['dir']*float(df[f'fret_{a["h"]}']) - COST_BPS*a['size']

def fit_ensemble(train, risk_penalty):
    use=train.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+[f'fret_{h}' for h in HORIZONS]+[f'long_mae_{h}' for h in HORIZONS]+[f'short_mae_{h}' for h in HORIZONS]).copy()
    if len(use)>MAX_TRAIN_ROWS:
        # Deterministic time-spread sampling, after all causal features/targets are computed.
        idx=np.linspace(0,len(use)-1,MAX_TRAIN_ROWS,dtype=int); use=use.iloc[idx]
    X=use[FEATURES].to_numpy(float); Y=reward_matrix(use,risk_penalty)
    xs=StandardScaler().fit(X); ys=StandardScaler().fit(Y); Xs=xs.transform(X); Ys=ys.transform(Y)
    models=[]
    for seed in SEEDS:
        print('FIT_POLICY',risk_penalty,'seed',seed,'rows',len(use),flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m=MLPRegressor(hidden_layer_sizes=(96,64,32),activation='relu',solver='adam',alpha=0.0015,batch_size=1024,learning_rate_init=0.0008,max_iter=90,early_stopping=True,validation_fraction=0.08,n_iter_no_change=8,random_state=seed,verbose=False)
            m.fit(Xs,Ys)
        models.append(m)
    return {'x_scaler':xs,'y_scaler':ys,'models':models,'risk_penalty':risk_penalty,'rows':len(use)}

def predict_policy(bundle, df):
    X=bundle['x_scaler'].transform(df[FEATURES].to_numpy(float)); preds=[]
    for m in bundle['models']:
        preds.append(bundle['y_scaler'].inverse_transform(m.predict(X)))
    p=np.stack(preds,axis=0); return p.mean(axis=0),p.std(axis=0)

def simulate(df, mean_q, std_q, uncertainty_penalty, capacity):
    q=mean_q-uncertainty_penalty*std_q; best=np.argmax(q,axis=1); bestq=q[np.arange(len(q)),best]
    data=df[['ts','symbol']].copy(); data['i']=np.arange(len(data)); data['a']=best; data['q']=bestq
    data=data[data.q>0].sort_values(['ts','q'],ascending=[True,False])
    active=[]; trades=[]; pnl_by_exit={}
    for ts,g in data.groupby('ts',sort=True):
        # Release capital only after the policy's chosen holding horizon ends.
        active=[p for p in active if p['exit']>ts]
        used=sum(p['size'] for p in active); avail=max(0.0,capacity-used)
        active_sy={p['symbol'] for p in active}
        if avail<0.49: continue
        for row in g.itertuples(index=False):
            if row.symbol in active_sy: continue
            a=ACTIONS[int(row.a)]
            if a['size']>avail+1e-9: continue
            src=df.iloc[int(row.i)]; net=actual_net(src,row.a); exit_ts=int(row.ts+a['h']*BAR_MS)
            active.append({'symbol':row.symbol,'size':a['size'],'exit':exit_ts}); active_sy.add(row.symbol); avail-=a['size']
            trades.append({'entry_ts':int(row.ts),'exit_ts':exit_ts,'symbol':row.symbol,'action':a['name'],'size':a['size'],'pred_q':float(row.q),'net_bps':float(net)})
            pnl_by_exit[exit_ts]=pnl_by_exit.get(exit_ts,0.0)+float(net)/capacity
            if avail<0.49: break
    if not trades:
        return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_week_fraction':0.,'active_actions':0,'portfolio_return_bps':0.},[]
    t=pd.DataFrame(trades); net=t.net_bps.to_numpy(float); pos=net[net>0].sum(); neg=-net[net<0].sum()
    pnl=pd.Series(pnl_by_exit).sort_index(); curve=pnl.cumsum().to_numpy(); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve
    cg=t.groupby('symbol').net_bps.mean(); week=pd.to_datetime(t.exit_ts,unit='ms',utc=True).dt.to_period('W').astype(str); wg=t.assign(week=week).groupby('week').net_bps.mean()
    met={'trades':int(len(t)),'avg_net_bps':float(net.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((net>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0.),'positive_coin_fraction':float((cg>0).mean()),'positive_week_fraction':float((wg>0).mean()),'active_actions':int(t.action.nunique()),'selected_symbols':int(t.symbol.nunique()),'portfolio_return_bps':float(pnl.sum()),'portfolio_exit_epochs':int(len(pnl))}
    return met,trades

def score_validation(m):
    if m['trades']<120: return -1e30
    breadth=max(0.2,m['positive_coin_fraction'])*max(0.2,m['positive_week_fraction'])
    return m['avg_net_bps']*math.sqrt(m['trades'])*min(max(m['profit_factor'],0),2.2)*breadth/(1+m['max_drawdown_bps']/900.0)

def main():
    symbols=discover(); print('DISCOVERED',len(symbols),symbols[:20],flush=True)
    fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            inst=fut[f]
            try:
                iid,df,c=f.result(); fetched[iid]=df; calls+=c; print('FETCHED',iid,len(df),flush=True)
            except Exception as e: print('FAIL',inst,repr(e),flush=True)
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in symbols:
        df=fetched.get(inst,pd.DataFrame())
        # Need enough pre-validation and sealed-period history to make a meaningful autonomous policy test.
        if len(df)<1600 or (df.ts<VAL_START).sum()<1250 or (df.ts>=FINAL_START).sum()<150:
            print('SKIP',inst,'bars',len(df),flush=True); continue
        raw_hash.update(pd.util.hash_pandas_object(df,index=False).values.tobytes())
        symbol=inst.replace('-USDT-SWAP','')
        parts.append(individual_frame(df,symbol)); accepted.append({'instrument':inst,'bars':int(len(df)),'first_ts':int(df.ts.iloc[0]),'last_ts':int(df.ts.iloc[-1])})
    if len(parts)<24: raise RuntimeError(f'only {len(parts)} symbols with sufficient real OKX history')
    ds=add_market_state(pd.concat(parts,ignore_index=True)).replace([np.inf,-np.inf],np.nan)
    outcome_cols=[f'fret_{h}' for h in HORIZONS]+[f'long_mae_{h}' for h in HORIZONS]+[f'short_mae_{h}' for h in HORIZONS]
    ds=ds.dropna(subset=FEATURES+outcome_cols).sort_values(['ts','symbol']).reset_index(drop=True)
    # Purge MAX_H bars before both boundaries so no training/validation label crosses into the next partition.
    purge=MAX_H*BAR_MS
    train=ds[ds.ts<VAL_START-purge].copy(); val=ds[(ds.ts>=VAL_START)&(ds.ts<FINAL_START-purge)].copy(); final=ds[ds.ts>=FINAL_START].copy()
    print('ROWS',len(train),len(val),len(final),'SYMBOLS',ds.symbol.nunique(),flush=True)
    best=None; candidates=[]
    for rp in RISK_PENALTIES:
        bundle=fit_ensemble(train,rp); mq,sq=predict_policy(bundle,val)
        for up in UNCERTAINTY_PENALTIES:
            for cap in CAPACITIES:
                met,_=simulate(val,mq,sq,up,cap); sc=score_validation(met); rec={'risk_penalty':rp,'uncertainty_penalty':up,'capacity':cap,'score':sc,'metrics':met}; candidates.append(rec)
                print('VAL',json.dumps(rec,sort_keys=True),flush=True)
                if best is None or sc>best[0]: best=(sc,bundle,up,cap,met)
    if best is None or best[0]<=-1e20: raise RuntimeError('no dynamic policy had sufficient validation support')
    _,bundle,up,cap,vmet=best
    print('VALIDATION_SELECTED',json.dumps({'risk_penalty':bundle['risk_penalty'],'uncertainty_penalty':up,'capacity':cap,'metrics':vmet},sort_keys=True),flush=True)
    # SEALED FINAL: touched exactly once after architecture/calibration selection.
    fmq,fsq=predict_policy(bundle,final); fmet,trades=simulate(final,fmq,fsq,up,cap)
    reasons=[]
    gates=[(fmet['trades']>=120,'insufficient_final_trades'),(fmet['avg_net_bps']>=3.0,'weak_final_edge'),(fmet['profit_factor']>=1.15,'profit_factor_below_gate'),(fmet['positive_coin_fraction']>=0.50,'coin_breadth_below_gate'),(fmet['positive_week_fraction']>=0.50,'week_breadth_below_gate'),(fmet['max_drawdown_bps']<=1200,'drawdown_above_gate'),(fmet['active_actions']>=3,'policy_collapsed_to_too_few_actions')]
    for ok,r in gates:
        if not ok: reasons.append(r)
    qualified=not reasons
    artifact={'x_scaler':bundle['x_scaler'],'y_scaler':bundle['y_scaler'],'models':bundle['models'],'features':FEATURES,'actions':ACTIONS,'risk_penalty':bundle['risk_penalty'],'uncertainty_penalty':up,'capacity':cap,'timeframe':'4h','roundtrip_cost_bps':COST_BPS}
    joblib.dump(artifact,OUT/'dynamic_policy.joblib')
    pd.DataFrame(trades).to_csv(OUT/'sealed_final_trades.csv',index=False)
    manifest={'version':'R1F-DYNAMIC-POLICY-OKX-4H','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'decision_authority':'LEARNED_ACTION_VALUE_POLICY','fixed_strategy_expert_authority':False,'timeframe':'4h','symbol_count':len(accepted),'rows':int(len(ds)),'train_rows':int(len(train)),'validation_rows':int(len(val)),'sealed_final_rows':int(len(final)),'roundtrip_cost_bps':COST_BPS,'actions':ACTIONS,'features':FEATURES,'selected_risk_penalty':bundle['risk_penalty'],'selected_uncertainty_penalty':up,'selected_capacity':cap,'validation':vmet,'qualification':{'qualified':qualified,**fmet,'reasons':reasons},'provenance':{'provider':'OKX public REST market history','endpoint':'/api/v5/market/history-candles','verified_public_archive':True,'retrieved_at':datetime.now(timezone.utc).isoformat(),'interval':BAR,'start':'2023-01-01','validation_start':'2026-07-01','sealed_final_start':'2026-08-01','accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls,'selection':'long-lived currently-live USDT crypto perpetuals; RWA groups excluded; live app universe remains dynamic'},'notes':['No named strategy expert decides trades.','Policy directly scores learned LONG/SHORT exposure and holding-horizon actions.','WAIT is the zero-reward baseline; policy trades only when a learned action value exceeds WAIT.','Named regimes/strategies may be shown later only for diagnostics and cannot gate the learned policy.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2)); (OUT/'validation_candidates.json').write_text(json.dumps(candidates,indent=2))
    print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
