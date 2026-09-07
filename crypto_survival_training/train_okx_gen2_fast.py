from __future__ import annotations
import json, math, os, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from crypto_survival_training import train_real as core
BASE='https://www.okx.com'; BAR='4H'; BAR_MS=14_400_000
START_MS=int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()*1000); VAL_START=int(pd.Timestamp('2026-01-01',tz='UTC').timestamp()*1000); FINAL_START=int(pd.Timestamp('2026-08-01',tz='UTC').timestamp()*1000)
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','60')); HORIZON=int(os.getenv('HORIZON_BARS','2')); COST_BPS=float(os.getenv('ROUNDTRIP_COST_BPS','30')); OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen2_fast')); OUT.mkdir(parents=True,exist_ok=True)
_tls=threading.local(); _rate_lock=threading.Lock(); _next_req=0.0
def session():
    if not hasattr(_tls,'s'): _tls.s=requests.Session(); _tls.s.headers['User-Agent']='GLOBAL-CRYPTO-HUNTER-R1E-OKX/1.1'
    return _tls.s
def throttle():
    global _next_req
    with _rate_lock: now=time.monotonic(); target=max(now,_next_req); _next_req=target+0.115
    if target>now: time.sleep(target-now)
def get(path,params=None,retries=8):
    for i in range(retries):
        throttle(); r=session().get(BASE+path,params=params,timeout=30)
        if r.status_code==200:
            j=r.json()
            if str(j.get('code','0'))=='0': return j.get('data') or []
        time.sleep(min(6,0.4*(2**i)))
    raise RuntimeError(f'OKX failed {path} {params} status={r.status_code} body={r.text[:120]}')
def discover():
    inst=get('/api/v5/public/instruments',{'instType':'SWAP'}); ticks=get('/api/v5/market/tickers',{'instType':'SWAP'}); tmap={x.get('instId'):x for x in ticks}; ranked=[]
    for x in inst:
        if x.get('state')!='live' or x.get('settleCcy')!='USDT' or str(x.get('groupId') or '') in {'6','7'}: continue
        iid=x.get('instId','')
        if not iid.endswith('-USDT-SWAP'): continue
        t=tmap.get(iid,{})
        try: turnover=float(t.get('last') or 0)*float(t.get('volCcy24h') or 0)
        except: turnover=0
        ranked.append((turnover,iid))
    ranked.sort(reverse=True); return [x for _,x in ranked[:MAX_SYMBOLS]]
def fetch_history(inst):
    rows=[]; after=None; calls=0
    for _ in range(100):
        p={'instId':inst,'bar':BAR,'limit':'100'}
        if after is not None: p['after']=str(after)
        data=get('/api/v5/market/history-candles',p); calls+=1
        if not data: break
        for z in data:
            if len(z)>=6 and (len(z)<9 or str(z[8])=='1'): rows.append([int(z[0]),float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5])])
        oldest=min(int(z[0]) for z in data)
        if oldest<=START_MS or (after is not None and oldest>=after): break
        after=oldest
    if not rows: return inst,pd.DataFrame(),calls
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts'); df=df[df.ts>=START_MS].reset_index(drop=True); return inst,df,calls
def expert_arrays(df,edges,min_support=80):
    n=len(df); best_edge=np.zeros(n); best_dir=np.zeros(n,dtype=np.int8); best_name=np.full(n,'NONE',object); regs=df.regime.astype(str).to_numpy(); arch=df.archetype.astype(str).to_numpy()
    for name,ser in core.strategy_directions(df).items():
        dirs=ser.fillna(0).astype(int).to_numpy(); lookup={}
        for r,a in set(zip(regs,arch)):
            e=edges.get(f'{r}|{a}|{name}',{}); lookup[(r,a)]=(float(e.get('shrunk_edge_bps',0) or 0),int(e.get('n',0) or 0))
        vals=np.fromiter((lookup[(r,a)][0] for r,a in zip(regs,arch)),float,n); sup=np.fromiter((lookup[(r,a)][1] for r,a in zip(regs,arch)),int,n)
        mask=(dirs!=0)&(sup>=min_support)&(vals>best_edge)&(vals>0); best_edge[mask]=vals[mask]; best_dir[mask]=dirs[mask]; best_name[mask]=name
    return best_dir,best_edge,best_name
def simulate(df,pred,expert,min_pred_net,min_expert_edge,max_positions):
    edir,eedge,_=expert; pred=np.asarray(pred,float); y=df.future_ret_bps.to_numpy(float); pred_net=edir*pred-COST_BPS; epoch=((df.ts.to_numpy(np.int64)//BAR_MS)%HORIZON)==0; idx=np.flatnonzero((edir!=0)&epoch&(eedge>=min_expert_edge)&(pred_net>=min_pred_net))
    if not len(idx): return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_regime_fraction':0.,'portfolio_epochs':0,'selected_symbols':0}
    cand=pd.DataFrame({'i':idx,'ts':df.ts.to_numpy()[idx],'pred_net':pred_net[idx],'expert_edge':eedge[idx]}); cand['rank']=cand.pred_net+.35*cand.expert_edge; chosen=np.asarray([i for _,g in cand.groupby('ts',sort=True) for i in g.nlargest(max_positions,'rank').i],int)
    net=edir[chosen]*y[chosen]-COST_BPS; sy=df.symbol.astype(str).to_numpy()[chosen]; rg=df.regime.astype(str).to_numpy()[chosen]; ts=df.ts.to_numpy()[chosen]; pos=net[net>0].sum(); neg=-net[net<0].sum(); t=pd.DataFrame({'ts':ts,'net':net,'symbol':sy,'regime':rg}); er=t.groupby('ts').net.mean().sort_index().to_numpy(); curve=np.cumsum(er); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve; cg=t.groupby('symbol').net.mean(); rr=t.groupby('regime').net.mean()
    return {'trades':int(len(net)),'avg_net_bps':float(net.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((net>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0),'positive_coin_fraction':float((cg>0).mean()),'positive_regime_fraction':float((rr>0).mean()),'portfolio_epochs':int(len(er)),'selected_symbols':int(len(cg))}
def thresholds(pred,expert):
    d,_,_=expert; x=(d*np.asarray(pred)-COST_BPS)[d!=0]; vals=[0,5,10,15,20,30,40,60,80]
    if len(x): vals += [float(np.quantile(x,q)) for q in (.45,.55,.65,.75,.82,.88,.92)]
    return sorted(set(round(max(0,float(v)),6) for v in vals))
def main():
    symbols=discover(); print('DISCOVERED',len(symbols),symbols[:15]); fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            inst=fut[f]
            try: iid,df,c=f.result(); fetched[iid]=df; calls+=c; print('FETCHED',iid,len(df))
            except Exception as e: print('FAIL',inst,repr(e))
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in symbols:
        df=fetched.get(inst,pd.DataFrame())
        if len(df)<1400 or (df.ts<VAL_START).sum()<1000 or (df.ts>=FINAL_START).sum()<100: print('SKIP',inst,'bars',len(df)); continue
        raw_hash.update(pd.util.hash_pandas_object(df,index=False).values.tobytes()); f=core.feature_frame(df,HORIZON).dropna(subset=['future_ret_bps']).replace([np.inf,-np.inf],np.nan); f['symbol']=inst.replace('-SWAP','').replace('-',''); parts.append(f); accepted.append({'instrument':inst,'bars':int(len(df)),'first_ts':int(df.ts.iloc[0]),'last_ts':int(df.ts.iloc[-1])})
    if len(parts)<20: raise RuntimeError(f'only {len(parts)} symbols with sufficient OKX history')
    ds=pd.concat(parts,ignore_index=True); train=ds[ds.ts<VAL_START].copy(); val=ds[(ds.ts>=VAL_START)&(ds.ts<FINAL_START)].copy(); final=ds[ds.ts>=FINAL_START].copy(); print('ROWS',len(train),len(val),len(final),'SYMBOLS',ds.symbol.nunique()); edges=core.edge_table(train); ve=expert_arrays(val,edges,80); fe=expert_arrays(final,edges,80)
    specs=[('abs15',dict(loss='absolute_error',max_iter=240,learning_rate=.045,max_leaf_nodes=15,l2_regularization=2.5)),('abs31',dict(loss='absolute_error',max_iter=280,learning_rate=.035,max_leaf_nodes=31,l2_regularization=3.5)),('sq15',dict(loss='squared_error',max_iter=220,learning_rate=.035,max_leaf_nodes=15,l2_regularization=4.5))]; best=None
    for name,kw in specs:
        print('FIT',name); model=HistGradientBoostingRegressor(random_state=23,**kw).fit(core.matrix(train),train.future_ret_bps.to_numpy()); pv=model.predict(core.matrix(val))
        for th in thresholds(pv,ve):
            for ee in (0.,5.,10.,20.,30.):
                for mp in (3,5):
                    met=simulate(val,pv,ve,th,ee,mp)
                    if met['trades']<600 or met['portfolio_epochs']<150: continue
                    breadth=min(met['positive_coin_fraction'],met['positive_regime_fraction']); score=met['avg_net_bps']*math.sqrt(met['trades'])*min(max(met['profit_factor'],0),2)*max(.2,breadth)/(1+met['max_drawdown_bps']/1500)
                    if best is None or score>best[0]: best=(score,name,kw,model,th,ee,mp,met)
    if best is None: raise RuntimeError('no validation candidate met support requirements')
    _,name,kw,model,th,ee,mp,vmet=best; print('VALIDATION_BEST',name,th,ee,mp,json.dumps(vmet,sort_keys=True)); pf=model.predict(core.matrix(final)); fm=simulate(final,pf,fe,th,ee,mp); reasons=[]
    for ok,r in [(fm['trades']>=300,'insufficient_final_trades'),(fm['avg_net_bps']>=1,'weak_final_edge'),(fm['profit_factor']>=1.08,'profit_factor_below_gate'),(fm['positive_coin_fraction']>=.45,'coin_breadth_below_gate'),(fm['positive_regime_fraction']>=.45,'regime_breadth_below_gate'),(fm['max_drawdown_bps']<=2500,'drawdown_above_gate')]:
        if not ok: reasons.append(r)
    joblib.dump(model,OUT/'model.joblib'); (OUT/'strategy_edges.json').write_text(json.dumps(edges,indent=2)); manifest={'version':'R1E-OKX-4H-REAL-FAST','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'timeframe':'4h','symbol_count':len(accepted),'rows':int(len(ds)),'horizon_bars':HORIZON,'historical_roundtrip_cost_bps':COST_BPS,'prediction_threshold_bps':float(th),'min_expert_edge_bps':float(ee),'max_positions':int(mp),'model_file':'model.joblib','edge_table_file':'strategy_edges.json','model_variant':name,'validation':vmet,'provenance':{'provider':'OKX public REST market history','endpoint':'/api/v5/market/history-candles','verified_public_archive':True,'retrieved_at':datetime.now(timezone.utc).isoformat(),'interval':BAR,'start':'2023-01-01','validation_start':'2026-01-01','sealed_final_start':'2026-08-01','accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls,'selection':'dynamic high-turnover live USDT crypto perpetuals; OKX RWA groups excluded'},'qualification':{'qualified':not reasons,**fm,'reasons':reasons}}; (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2)); print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
