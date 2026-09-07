from __future__ import annotations
import json, math, os, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import requests, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from crypto_survival_training import train_real as core

BASE='https://www.okx.com'
BAR='4H'
BAR_MS=4*60*60*1000
START_MS=int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()*1000)
VAL_START=int(pd.Timestamp('2026-01-01',tz='UTC').timestamp()*1000)
FINAL_START=int(pd.Timestamp('2026-08-01',tz='UTC').timestamp()*1000)
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','60'))
HORIZON=int(os.getenv('HORIZON_BARS','2'))
COST_BPS=float(os.getenv('ROUNDTRIP_COST_BPS','30'))
OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen2')); OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='GLOBAL-CRYPTO-HUNTER-R1E-OKX/1.0'

def get(path,params=None,retries=8):
    for i in range(retries):
        r=S.get(BASE+path,params=params,timeout=30)
        if r.status_code==200:
            j=r.json()
            if str(j.get('code','0'))=='0': return j.get('data') or []
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(8,0.5*(2**i))); continue
        time.sleep(min(4,0.4*(i+1)))
    raise RuntimeError(f'OKX request failed {path} {params} status={r.status_code} body={r.text[:160]}')

def discover():
    inst=get('/api/v5/public/instruments',{'instType':'SWAP'})
    ticks=get('/api/v5/market/tickers',{'instType':'SWAP'})
    tmap={x.get('instId'):x for x in ticks}
    ranked=[]
    for x in inst:
        if x.get('state')!='live' or x.get('settleCcy')!='USDT': continue
        if str(x.get('groupId') or '') in {'6','7'}: continue
        iid=x.get('instId','')
        if not iid.endswith('-USDT-SWAP'): continue
        t=tmap.get(iid,{})
        try:
            last=float(t.get('last') or 0); vol=float(t.get('volCcy24h') or 0); turnover=last*vol
        except Exception: turnover=0
        ranked.append((turnover,iid))
    ranked.sort(reverse=True)
    return [x for _,x in ranked[:MAX_SYMBOLS]]

def fetch_history(inst):
    rows=[]; after=None; calls=0
    for _ in range(100):
        p={'instId':inst,'bar':BAR,'limit':'100'}
        if after is not None: p['after']=str(after)
        data=get('/api/v5/market/history-candles',p); calls+=1
        if not data: break
        good=[]
        for z in data:
            if len(z)<6: continue
            if len(z)>=9 and str(z[8])!='1': continue
            good.append([int(z[0]),float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5])])
        rows.extend(good)
        oldest=min(int(z[0]) for z in data)
        if oldest<=START_MS: break
        if after is not None and oldest>=after: break
        after=oldest
        time.sleep(0.12)
    if not rows: return pd.DataFrame(),calls
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts')
    df=df[df.ts>=START_MS].reset_index(drop=True)
    return df,calls

def build_expert(df, edges, min_support=80):
    n=len(df); best_edge=np.zeros(n,float); best_dir=np.zeros(n,int); best_name=np.array(['NONE']*n,dtype=object)
    regimes=df.regime.astype(str).to_numpy(); arches=df.archetype.astype(str).to_numpy()
    for name,ser in core.strategy_directions(df).items():
        dirs=ser.fillna(0).astype(int).to_numpy(); vals=np.zeros(n,float); supports=np.zeros(n,int)
        for i,(r,a) in enumerate(zip(regimes,arches)):
            e=edges.get(f'{r}|{a}|{name}',{}); vals[i]=float(e.get('shrunk_edge_bps',0) or 0); supports[i]=int(e.get('n',0) or 0)
        mask=(dirs!=0)&(supports>=min_support)&(vals>best_edge)&(vals>0)
        best_edge[mask]=vals[mask]; best_dir[mask]=dirs[mask]; best_name[mask]=name
    return best_dir,best_edge,best_name

def simulate(df,pred,edges,min_pred_net,min_expert_edge,max_positions):
    if df.empty: return None
    edir,eedge,_=build_expert(df,edges,80); pred=np.asarray(pred,float); y=df.future_ret_bps.to_numpy(float)
    pred_net=edir*pred-COST_BPS; epoch=((df.ts.to_numpy(np.int64)//BAR_MS)%HORIZON)==0
    mask=(edir!=0)&epoch&(eedge>=min_expert_edge)&(pred_net>=min_pred_net); idx=np.flatnonzero(mask)
    if not len(idx): return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_regime_fraction':0.,'portfolio_epochs':0}
    cand=pd.DataFrame({'i':idx,'ts':df.ts.to_numpy()[idx],'symbol':df.symbol.astype(str).to_numpy()[idx],'regime':df.regime.astype(str).to_numpy()[idx],'dir':edir[idx],'pred_net':pred_net[idx],'expert_edge':eedge[idx]})
    cand['rank']=cand.pred_net+0.35*cand.expert_edge; chosen=[]
    for _,g in cand.groupby('ts',sort=True): chosen.extend(g.nlargest(max_positions,'rank').i.tolist())
    chosen=np.asarray(chosen,int)
    if not len(chosen): return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_regime_fraction':0.,'portfolio_epochs':0}
    net=edir[chosen]*y[chosen]-COST_BPS; sy=df.symbol.astype(str).to_numpy()[chosen]; rg=df.regime.astype(str).to_numpy()[chosen]; ts=df.ts.to_numpy()[chosen]
    pos=net[net>0].sum(); neg=-net[net<0].sum(); tdf=pd.DataFrame({'ts':ts,'net':net,'symbol':sy,'regime':rg})
    epoch_ret=tdf.groupby('ts').net.mean().sort_index().to_numpy(); curve=np.cumsum(epoch_ret); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve
    cg=tdf.groupby('symbol').net.mean(); rgg=tdf.groupby('regime').net.mean()
    return {'trades':int(len(net)),'avg_net_bps':float(net.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((net>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0),'positive_coin_fraction':float((cg>0).mean()),'positive_regime_fraction':float((rgg>0).mean()),'portfolio_epochs':int(len(epoch_ret)),'selected_symbols':int(len(cg))}

def candidate_thresholds(pred,df,edges):
    edir,_,_=build_expert(df,edges,80); raw=edir*np.asarray(pred)-COST_BPS; x=raw[(edir!=0)&np.isfinite(raw)]
    vals=[0,5,10,15,20,30,40,60,80]
    if len(x): vals += [float(np.quantile(x,q)) for q in (.45,.55,.65,.75,.82,.88,.92)]
    return sorted(set(round(max(0,float(v)),6) for v in vals))

def main():
    symbols=discover(); print('DISCOVERED',len(symbols),symbols[:15]); parts=[]; accepted=[]; calls=0; raw_hash=hashlib.sha256()
    for i,inst in enumerate(symbols,1):
        try: df,c=fetch_history(inst); calls+=c
        except Exception as e: print('FAIL',inst,repr(e)); continue
        if len(df)<1800 or (df.ts<VAL_START).sum()<1500 or (df.ts>=FINAL_START).sum()<100:
            print('SKIP',inst,'bars',len(df),'train',int((df.ts<VAL_START).sum()),'final',int((df.ts>=FINAL_START).sum())); continue
        raw_hash.update(pd.util.hash_pandas_object(df,index=False).values.tobytes())
        f=core.feature_frame(df,HORIZON).dropna(subset=['future_ret_bps']).replace([np.inf,-np.inf],np.nan); f['symbol']=inst.replace('-SWAP','').replace('-',''); parts.append(f)
        accepted.append({'instrument':inst,'bars':int(len(df)),'first_ts':int(df.ts.iloc[0]),'last_ts':int(df.ts.iloc[-1])})
        print(f'[{i}/{len(symbols)}] {inst} bars={len(df)} train={(df.ts<VAL_START).sum()} val={((df.ts>=VAL_START)&(df.ts<FINAL_START)).sum()} final={(df.ts>=FINAL_START).sum()}')
    if len(parts)<20: raise RuntimeError(f'only {len(parts)} symbols with sufficient OKX history')
    ds=pd.concat(parts,ignore_index=True); train=ds[ds.ts<VAL_START].copy(); val=ds[(ds.ts>=VAL_START)&(ds.ts<FINAL_START)].copy(); final=ds[ds.ts>=FINAL_START].copy()
    print('ROWS',len(train),len(val),len(final),'SYMBOLS',ds.symbol.nunique()); edges=core.edge_table(train)
    models=[('abs15',dict(loss='absolute_error',max_iter=260,learning_rate=.045,max_leaf_nodes=15,l2_regularization=2.0)),('abs31',dict(loss='absolute_error',max_iter=300,learning_rate=.035,max_leaf_nodes=31,l2_regularization=3.0)),('sq15',dict(loss='squared_error',max_iter=240,learning_rate=.035,max_leaf_nodes=15,l2_regularization=4.0))]
    best=None
    for name,kw in models:
        print('FIT',name); m=HistGradientBoostingRegressor(random_state=23,**kw).fit(core.matrix(train),train.future_ret_bps.to_numpy()); pv=m.predict(core.matrix(val))
        for th in candidate_thresholds(pv,val,edges):
          for ee in (0.,5.,10.,20.,30.):
            for mp in (3,5):
              met=simulate(val,pv,edges,th,ee,mp)
              if met['trades']<600 or met['portfolio_epochs']<150: continue
              breadth=min(met['positive_coin_fraction'],met['positive_regime_fraction']); score=met['avg_net_bps']*math.sqrt(met['trades'])*min(max(met['profit_factor'],0),2.0)*max(.2,breadth)/(1+met['max_drawdown_bps']/1500)
              row=(score,name,kw,m,th,ee,mp,met)
              if best is None or score>best[0]: best=row
    if best is None: raise RuntimeError('no validation candidate met support requirements')
    _,name,kw,model,th,ee,mp,vmet=best; print('VALIDATION_BEST',name,'threshold',th,'expert_edge',ee,'max_positions',mp,json.dumps(vmet,sort_keys=True))
    pf=model.predict(core.matrix(final)); fmet=simulate(final,pf,edges,th,ee,mp); reasons=[]
    checks=[(fmet['trades']>=300,'insufficient_final_trades'),(fmet['avg_net_bps']>=1.0,'weak_final_edge'),(fmet['profit_factor']>=1.08,'profit_factor_below_gate'),(fmet['positive_coin_fraction']>=.45,'coin_breadth_below_gate'),(fmet['positive_regime_fraction']>=.45,'regime_breadth_below_gate'),(fmet['max_drawdown_bps']<=2500,'drawdown_above_gate')]
    for ok,r in checks:
      if not ok: reasons.append(r)
    qualified=not reasons; joblib.dump(model,OUT/'model.joblib'); (OUT/'strategy_edges.json').write_text(json.dumps(edges,indent=2))
    manifest={'version':'R1E-OKX-4H-REAL','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'timeframe':'4h','symbol_count':len(accepted),'rows':int(len(ds)),'horizon_bars':HORIZON,'historical_roundtrip_cost_bps':COST_BPS,'prediction_threshold_bps':float(th),'min_expert_edge_bps':float(ee),'max_positions':int(mp),'model_file':'model.joblib','edge_table_file':'strategy_edges.json','model_variant':name,'validation':vmet,'provenance':{'provider':'OKX public REST market history','endpoint':'/api/v5/market/history-candles','verified_public_archive':True,'retrieved_at':datetime.now(timezone.utc).isoformat(),'interval':BAR,'start':'2023-01-01','validation_start':'2026-01-01','sealed_final_start':'2026-08-01','accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls,'selection':'dynamic high-turnover live USDT crypto perpetuals; OKX RWA groups excluded'},'qualification':{'qualified':qualified,**fmet,'reasons':reasons}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2)); print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
