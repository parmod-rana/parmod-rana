from __future__ import annotations
import json, math, os, hashlib, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from crypto_survival_training import train_okx_dynamic_brain as core

# GENERATION 3B: same autonomous learned action-value policy, enhanced for temporal robustness.
# IMPORTANT: The already-inspected 2026-08-01+ period is NOT used for calibration or training.
# No named trading strategy decides any action.

BAR=core.BAR; BAR_MS=core.BAR_MS; START_MS=core.START_MS
CUTOFF_MS=int(pd.Timestamp('2026-08-01',tz='UTC').timestamp()*1000)
PURGE=core.MAX_H*BAR_MS
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','72'))
COST_BPS=float(os.getenv('ROUNDTRIP_COST_BPS','30'))
MAX_TRAIN_ROWS=int(os.getenv('MAX_TRAIN_ROWS','150000'))
OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen3b')); OUT.mkdir(parents=True,exist_ok=True)
HORIZONS=core.HORIZONS; SIZES=core.SIZES; ACTIONS=core.ACTIONS
RISK_PENALTIES=(0.20,0.32)
UNCERTAINTY_PENALTIES=(0.0,0.5,1.0)
CAPACITIES=(2.0,3.0,5.0)
Q_FLOORS=(0.0,10.0,25.0,40.0,60.0,90.0)
SEEDS=(17,43)
FOLDS=[
    ('F1',pd.Timestamp('2025-07-01',tz='UTC'),pd.Timestamp('2025-09-01',tz='UTC')),
    ('F2',pd.Timestamp('2025-11-01',tz='UTC'),pd.Timestamp('2026-01-01',tz='UTC')),
    ('F3',pd.Timestamp('2026-03-01',tz='UTC'),pd.Timestamp('2026-05-01',tz='UTC')),
    ('F4',pd.Timestamp('2026-06-01',tz='UTC'),pd.Timestamp('2026-08-01',tz='UTC')),
]

EXTRA_FEATURES=['eff6','eff24','sign_persist12','down_frac12','rv_ratio6_24','drawdown24_bps','from_low24_bps','range_rel24','mkt_rv24','mkt_breadth24','mkt_dispersion24','mkt_range_bps']
FEATURES=core.FEATURES+EXTRA_FEATURES

def fetch_history(inst):
    rows=[]; after=None; calls=0
    for _ in range(42):
        p={'instId':inst,'bar':BAR,'limit':'300'}
        if after is not None: p['after']=str(after)
        data=core.get('/api/v5/market/history-candles',p); calls+=1
        if not data: break
        for z in data:
            if len(z)>=6 and (len(z)<9 or str(z[8])=='1'):
                rows.append([int(z[0]),float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5])])
        oldest=min(int(z[0]) for z in data)
        if oldest<=START_MS or (after is not None and oldest>=after): break
        after=oldest
    if not rows: return inst,pd.DataFrame(),calls
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts')
    return inst,d[d.ts>=START_MS].reset_index(drop=True),calls

def individual_frame(raw,symbol):
    d=core.individual_frame(raw,symbol)
    r=d.ret1.astype(float); c=d.close.astype(float)
    path6=r.abs().rolling(6,min_periods=6).sum(); path24=r.abs().rolling(24,min_periods=24).sum()
    d['eff6']=d.ret6.abs()/(path6+1e-9); d['eff24']=d.ret24.abs()/(path24+1e-9)
    s=np.sign(r); d['sign_persist12']=(s*s.shift(1)).rolling(12,min_periods=8).mean()
    d['down_frac12']=(r<0).astype(float).rolling(12,min_periods=8).mean()
    d['rv_ratio6_24']=d.rv6/(d.rv24+1e-9)
    hi24=c.rolling(24,min_periods=24).max(); lo24=c.rolling(24,min_periods=24).min()
    d['drawdown24_bps']=(c/hi24-1.0)*10000.0; d['from_low24_bps']=(c/lo24-1.0)*10000.0
    med_range=d.range_bps.rolling(24,min_periods=12).median(); d['range_rel24']=d.range_bps/(med_range+1e-9)
    return d

def add_market_state(ds):
    d=core.add_market_state(ds); g=d.groupby('ts',sort=False)
    d['mkt_rv24']=g.rv24.transform('median'); d['mkt_breadth24']=g.ret24.transform(lambda x:(x>0).mean())
    d['mkt_dispersion24']=g.ret24.transform('std'); d['mkt_range_bps']=g.range_bps.transform('median')
    return d

def clean_training_rows(df):
    outcome=[f'fret_{h}' for h in HORIZONS]+[f'long_mae_{h}' for h in HORIZONS]+[f'short_mae_{h}' for h in HORIZONS]
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+outcome).copy()

def recency_balanced_sample(df, cutoff_ms, limit):
    d=df.sort_values(['ts','symbol']).copy()
    if len(d)<=limit: return d
    day=86_400_000; age=(cutoff_ms-d.ts.to_numpy(np.int64))/day
    masks=[age<=365,(age>365)&(age<=730),age>730]; weights=(0.50,0.30,0.20); pieces=[]
    for mask,w in zip(masks,weights):
        part=d.loc[mask]
        n=min(len(part),max(1,int(limit*w)))
        if n and len(part): pieces.append(part.iloc[np.linspace(0,len(part)-1,n,dtype=int)])
    out=pd.concat(pieces).drop_duplicates().sort_values(['ts','symbol'])
    if len(out)<limit:
        rem=d.drop(index=out.index,errors='ignore'); n=min(limit-len(out),len(rem))
        if n: out=pd.concat([out,rem.iloc[np.linspace(0,len(rem)-1,n,dtype=int)]])
    return out.sort_values(['ts','symbol']).iloc[-limit:]

def reward_matrix(df,risk_penalty):
    ys=[]
    for a in ACTIONS:
        f=df[f'fret_{a["h"]}'].to_numpy(float); mae=df[('long_mae_' if a['dir']>0 else 'short_mae_')+str(a['h'])].to_numpy(float)
        raw=a['size']*a['dir']*f-COST_BPS*a['size']; risk=risk_penalty*(a['size']**1.55)*mae
        ys.append(raw-risk)
    return np.column_stack(ys)

def fit_bundle(train,risk_penalty,cutoff_ms):
    use=recency_balanced_sample(clean_training_rows(train),cutoff_ms,MAX_TRAIN_ROWS)
    X=use[FEATURES].to_numpy(float); Y=reward_matrix(use,risk_penalty)
    xs=StandardScaler().fit(X); ys=StandardScaler().fit(Y); Xs=xs.transform(X); Ys=ys.transform(Y)
    models=[]
    for seed in SEEDS:
        print('FIT',risk_penalty,'seed',seed,'rows',len(use),flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m=MLPRegressor(hidden_layer_sizes=(112,72,36),activation='relu',solver='adam',alpha=.0025,batch_size=1024,learning_rate_init=.00065,max_iter=100,early_stopping=True,validation_fraction=.08,n_iter_no_change=9,random_state=seed)
            m.fit(Xs,Ys)
        models.append(m)
    return {'x_scaler':xs,'y_scaler':ys,'models':models,'risk_penalty':risk_penalty,'rows':len(use)}

def predict(bundle,df):
    X=bundle['x_scaler'].transform(df[FEATURES].to_numpy(float)); pp=[]
    for m in bundle['models']: pp.append(bundle['y_scaler'].inverse_transform(m.predict(X)))
    a=np.stack(pp); return a.mean(0),a.std(0)

def actual_net(row,ai):
    a=ACTIONS[int(ai)]; return a['size']*a['dir']*float(row[f'fret_{a["h"]}'])-COST_BPS*a['size']

def simulate(df,mq,sq,up,capacity,q_floor):
    q=mq-up*sq; ai=np.argmax(q,axis=1); bq=q[np.arange(len(q)),ai]
    cand=df[['ts','symbol']].copy(); cand['i']=np.arange(len(cand)); cand['a']=ai; cand['q']=bq
    cand=cand[cand.q>q_floor].sort_values(['ts','q'],ascending=[True,False])
    active=[]; trades=[]; pnl_exit={}
    for ts,g in cand.groupby('ts',sort=True):
        active=[p for p in active if p['exit']>ts]; used=sum(p['size'] for p in active); avail=max(0.,capacity-used); active_sy={p['symbol'] for p in active}
        if avail<.49: continue
        for r in g.itertuples(index=False):
            if r.symbol in active_sy: continue
            a=ACTIONS[int(r.a)]
            if a['size']>avail+1e-9: continue
            src=df.iloc[int(r.i)]; net=actual_net(src,r.a); ex=int(r.ts+a['h']*BAR_MS)
            active.append({'symbol':r.symbol,'size':a['size'],'exit':ex}); active_sy.add(r.symbol); avail-=a['size']
            trades.append({'entry_ts':int(r.ts),'exit_ts':ex,'symbol':r.symbol,'action':a['name'],'size':a['size'],'pred_q':float(r.q),'net_bps':float(net)})
            pnl_exit[ex]=pnl_exit.get(ex,0.)+float(net)/capacity
            if avail<.49: break
    if not trades: return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_week_fraction':0.,'worst_week_bps':0.,'active_actions':0,'portfolio_return_bps':0.},[]
    t=pd.DataFrame(trades); n=t.net_bps.to_numpy(float); pos=n[n>0].sum(); neg=-n[n<0].sum(); pnl=pd.Series(pnl_exit).sort_index(); curve=pnl.cumsum().to_numpy(); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve
    cg=t.groupby('symbol').net_bps.mean(); wk=pd.to_datetime(t.exit_ts,unit='ms',utc=True).dt.strftime('%G-W%V'); wg=t.assign(week=wk).groupby('week').net_bps.mean()
    return {'trades':int(len(t)),'avg_net_bps':float(n.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((n>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0.),'positive_coin_fraction':float((cg>0).mean()),'positive_week_fraction':float((wg>0).mean()),'worst_week_bps':float(wg.min()),'active_actions':int(t.action.nunique()),'selected_symbols':int(t.symbol.nunique()),'portfolio_return_bps':float(pnl.sum()),'portfolio_exit_epochs':int(len(pnl))},trades

def robust_score(metrics):
    if any(m['trades']<45 for m in metrics): return -1e30
    total=sum(m['trades'] for m in metrics)
    if total<260: return -1e30
    av=np.array([m['avg_net_bps'] for m in metrics]); pf=np.array([m['profit_factor'] for m in metrics]); wk=np.array([m['positive_week_fraction'] for m in metrics]); coin=np.array([m['positive_coin_fraction'] for m in metrics]); dd=max(m['max_drawdown_bps'] for m in metrics)
    positive_fold=(av>0).mean()
    if positive_fold<.75: return -1e30
    base=np.median(av)*math.sqrt(total)*min(np.median(pf),2.0)*max(.2,np.median(wk))*max(.2,np.median(coin))/(1+dd/900.)
    stability=max(.15,1.0-np.std(av)/(abs(np.mean(av))+25.0)); downside=1.0 if av.min()>=0 else .35
    return float(base*stability*downside)

def main():
    symbols=core.discover()[:MAX_SYMBOLS]; print('DISCOVERED',len(symbols),symbols[:20],flush=True)
    fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            s=fut[f]
            try:
                iid,d,c=f.result(); fetched[iid]=d; calls+=c; print('FETCHED',iid,len(d),flush=True)
            except Exception as e: print('FAIL',s,repr(e),flush=True)
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in symbols:
        d=fetched.get(inst,pd.DataFrame())
        if len(d)<1600 or (d.ts<CUTOFF_MS).sum()<1500: continue
        raw_hash.update(pd.util.hash_pandas_object(d,index=False).values.tobytes()); sym=inst.replace('-USDT-SWAP',''); parts.append(individual_frame(d,sym)); accepted.append({'instrument':inst,'bars':int(len(d)),'first_ts':int(d.ts.iloc[0]),'last_ts':int(d.ts.iloc[-1])})
    if len(parts)<24: raise RuntimeError(f'only {len(parts)} usable symbols')
    ds=add_market_state(pd.concat(parts,ignore_index=True)); ds=clean_training_rows(ds).sort_values(['ts','symbol']).reset_index(drop=True)
    # Absolutely exclude the already-inspected Aug-Sep period from enhancement.
    research=ds[ds.ts<CUTOFF_MS-PURGE].copy(); print('RESEARCH_ROWS',len(research),'SYMBOLS',research.symbol.nunique(),flush=True)
    fold_cache={}; candidates=[]
    for rp in RISK_PENALTIES:
        preds=[]
        for name,vstart,vend in FOLDS:
            vs=int(vstart.timestamp()*1000); ve=int(vend.timestamp()*1000)
            tr=research[research.ts<vs-PURGE].copy(); va=research[(research.ts>=vs)&(research.ts<ve-PURGE)].copy()
            b=fit_bundle(tr,rp,vs); mq,sq=predict(b,va); preds.append((name,va,mq,sq))
        for up in UNCERTAINTY_PENALTIES:
            for cap in CAPACITIES:
                for floor in Q_FLOORS:
                    mets=[]
                    for name,va,mq,sq in preds:
                        met,_=simulate(va,mq,sq,up,cap,floor); mets.append(met)
                    sc=robust_score(mets); rec={'risk_penalty':rp,'uncertainty_penalty':up,'capacity':cap,'q_floor_bps':floor,'score':sc,'folds':mets}; candidates.append(rec)
        fold_cache[rp]=preds
    candidates.sort(key=lambda x:x['score'],reverse=True); best=candidates[0]
    if best['score']<=-1e20: raise RuntimeError('no Gen3B calibration survived rolling walk-forward support gates')
    print('SELECTED',json.dumps(best,sort_keys=True),flush=True)
    final_bundle=fit_bundle(research,best['risk_penalty'],CUTOFF_MS)
    artifact={'x_scaler':final_bundle['x_scaler'],'y_scaler':final_bundle['y_scaler'],'models':final_bundle['models'],'features':FEATURES,'actions':ACTIONS,'risk_penalty':best['risk_penalty'],'uncertainty_penalty':best['uncertainty_penalty'],'capacity':best['capacity'],'q_floor_bps':best['q_floor_bps'],'timeframe':'4h','roundtrip_cost_bps':COST_BPS,'policy_version':'GEN3B_WALK_FORWARD'}
    joblib.dump(artifact,OUT/'dynamic_policy.joblib')
    manifest={'version':'R1F-GEN3B-DYNAMIC-WALKFORWARD','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'decision_authority':'LEARNED_ACTION_VALUE_POLICY','fixed_strategy_expert_authority':False,'timeframe':'4h','symbol_count':len(accepted),'research_rows':int(len(research)),'features':FEATURES,'actions':ACTIONS,'selected_calibration':{k:best[k] for k in ('risk_penalty','uncertainty_penalty','capacity','q_floor_bps','score')},'walk_forward_folds':best['folds'],'qualification':{'qualified':False,'reasons':['no_fresh_untouched_period_after_gen3_retraining'],'authority':'SHADOW_PAPER_ONLY'},'provenance':{'provider':'OKX public REST market history','interval':BAR,'start':'2023-01-01','enhancement_cutoff':'2026-08-01','already_inspected_period_excluded':True,'accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls},'notes':['No named strategy expert decides trades.','Gen3B calibration uses rolling pre-August walk-forward folds only.','The already inspected 2026-08-01+ period is excluded from all training and calibration.','A fresh forward period is required before qualification can be restored.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'walkforward_candidates.json').write_text(json.dumps(candidates[:40],indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2))
    print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
