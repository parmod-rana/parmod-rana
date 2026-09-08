from __future__ import annotations
import io, json, math, os, time, zipfile, hashlib, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import joblib, requests
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from crypto_survival_training import train_okx_dynamic_brain as core
from crypto_survival_training import train_okx_dynamic_brain_gen3b as g
from crypto_survival_training import train_okx_dynamic_brain_gen3c as c

# GENERATION 3F — LEVERAGE-STATE DYNAMIC POLICY
# Same autonomous action-value concept as Gen3C, but the brain now receives
# genuine causal funding/crowding and historical open-interest state.
# No named strategy chooses an action. 2026-08-01+ stays excluded from all
# enhancement training/evaluation selection because it was already inspected.

CUTOFF_MS=c.CUTOFF_MS; PURGE=c.PURGE; BAR_MS=c.BAR_MS
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','72')); COST_BPS=c.COST_BPS
MAX_ROWS=int(os.getenv('MAX_TRAIN_ROWS','110000')); OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen3f')); OUT.mkdir(parents=True,exist_ok=True)
ACTIONS=c.ACTIONS; HORIZONS=c.HORIZONS; WINDOWS=c.WINDOWS; FOLDS=g.FOLDS
CAPACITY=2.0; Q_FLOOR=60.0; RISK_PENALTY=0.28
START_DATE=pd.Timestamp('2023-01-01',tz='UTC'); FUNDING_END=pd.Timestamp('2026-07-31',tz='UTC')
FUNDING_FEATURES=['funding_last','funding_mean3','funding_sum24h','funding_sum7d','funding_z21','funding_abs','funding_available']
OI_FEATURES=['oi_logusd','oi_ret1d','oi_ret3d','oi_ret7d','oi_z30','oi_available']
LEVERAGE_MARKET_FEATURES=['mkt_funding_median','mkt_funding_dispersion','funding_rel','rank_funding','mkt_oi_ret1d','mkt_oi_dispersion','rank_oi_ret1d']
FEATURES=c.FEATURES+FUNDING_FEATURES+OI_FEATURES+LEVERAGE_MARKET_FEATURES

GEN3C_FOLDS=[
 {'avg':34.55688496860242,'pf':1.1658361774788726,'dd':3215.4513180143645,'trades':135},
 {'avg':-84.1294575172906,'pf':0.7215288019729019,'dd':6595.604675176961,'trades':131},
 {'avg':15.434978385664031,'pf':1.0786186037822436,'dd':2420.303322194652,'trades':131},
 {'avg':65.53242550951644,'pf':1.336867564714191,'dd':2399.7351700318854,'trades':126},
]
GEN3C_MEDIAN_AVG=float(np.median([x['avg'] for x in GEN3C_FOLDS]))


def funding_url(day):
    return f"https://static.okx.com/cdn/okex/traderecords/swaprates/daily/{day.strftime('%Y%m%d')}/allswap-fundingrates-{day.strftime('%Y-%m-%d')}.zip?v=999"

def fetch_funding_day(day):
    url=funding_url(day)
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=30)
            if r.status_code==404: return day,None,url,None
            if r.status_code in (429,500,502,503,504): time.sleep(1.2*(attempt+1)); continue
            r.raise_for_status(); raw=r.content
            z=zipfile.ZipFile(io.BytesIO(raw)); names=z.namelist()
            if not names: return day,None,url,None
            df=pd.read_csv(io.BytesIO(z.read(names[0])))
            req={'instrument_name','funding_rate','funding_time'}
            if not req.issubset(df.columns): raise RuntimeError(f'funding schema changed: {list(df.columns)}')
            df=df[['instrument_name','funding_rate','funding_time']].copy(); df['funding_time']=pd.to_numeric(df.funding_time,errors='coerce'); df['funding_rate']=pd.to_numeric(df.funding_rate,errors='coerce'); df=df.dropna()
            return day,df,url,hashlib.sha256(raw).hexdigest()
        except Exception:
            if attempt==2: raise
            time.sleep(.8*(attempt+1))

def load_funding(symbols):
    days=list(pd.date_range(START_DATE,FUNDING_END,freq='D'))
    frames=[]; provenance=[]; misses=0
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut={ex.submit(fetch_funding_day,d):d for d in days}
        for i,f in enumerate(as_completed(fut),1):
            d=fut[f]
            try:
                day,df,url,sha=f.result()
                if df is None: misses+=1; continue
                df=df[df.instrument_name.isin(symbols)]
                if len(df): frames.append(df)
                provenance.append({'date':day.strftime('%Y-%m-%d'),'url':url,'sha256':sha,'rows_for_universe':int(len(df))})
            except Exception as e:
                print('FUNDING_FAIL',d.date(),repr(e),flush=True); misses+=1
            if i%150==0: print('FUNDING_PROGRESS',i,'of',len(days),'accepted_days',len(provenance),flush=True)
    if len(provenance)<900: raise RuntimeError(f'insufficient genuine funding archive coverage: {len(provenance)} days')
    allf=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['instrument_name','funding_rate','funding_time'])
    allf=allf.drop_duplicates(['instrument_name','funding_time']).sort_values(['instrument_name','funding_time'])
    return allf,provenance,misses

def fetch_oi_symbol(inst):
    url='https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history'
    start=int(START_DATE.timestamp()*1000); end=int(pd.Timestamp('2026-07-31 23:59:59',tz='UTC').timestamp()*1000); rows={}; calls=0
    for page in range(20):
        q={'instId':inst,'period':'1D','begin':str(start),'end':str(end),'limit':'100'}
        for attempt in range(5):
            r=requests.get(url,params=q,timeout=30); calls+=1
            if r.status_code==429: time.sleep(1.0+attempt*.8); continue
            try: body=r.json()
            except Exception: body={}
            if body.get('code')=='0': break
            if attempt==4: return inst,pd.DataFrame(),calls
            time.sleep(.6)
        data=body.get('data',[])
        if not data: break
        ts=[]
        for x in data:
            try:
                t=int(x[0]); oi=float(x[1]); oi_ccy=float(x[2]); oi_usd=float(x[3]); rows[t]=(t,oi,oi_ccy,oi_usd); ts.append(t)
            except Exception: pass
        if not ts: break
        oldest=min(ts); next_end=oldest-1
        if oldest<=start or next_end>=end or len(data)<100: break
        end=next_end; time.sleep(.12)
    d=pd.DataFrame(list(rows.values()),columns=['ts','oi','oi_ccy','oi_usd']).sort_values('ts') if rows else pd.DataFrame(columns=['ts','oi','oi_ccy','oi_usd'])
    return inst,d,calls

def load_oi(symbols):
    out={}; calls=0
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut={ex.submit(fetch_oi_symbol,s):s for s in symbols}
        for f in as_completed(fut):
            s=fut[f]
            try:
                inst,d,c=f.result(); out[inst]=d; calls+=c; print('OI',inst,len(d),flush=True)
            except Exception as e: print('OI_FAIL',s,repr(e),flush=True); out[s]=pd.DataFrame()
    return out,calls

def funding_features(events):
    if events is None or len(events)==0: return pd.DataFrame(columns=['ts']+FUNDING_FEATURES)
    e=events.sort_values('funding_time').copy(); r=e.funding_rate.astype(float)
    e['ts']=e.funding_time.astype('int64'); e['funding_last']=r; e['funding_mean3']=r.rolling(3,min_periods=1).mean(); e['funding_sum24h']=r.rolling(3,min_periods=1).sum(); e['funding_sum7d']=r.rolling(21,min_periods=3).sum()
    m=r.rolling(63,min_periods=12).mean(); sd=r.rolling(63,min_periods=12).std(); e['funding_z21']=(r-m)/(sd+1e-9); e['funding_abs']=r.abs(); e['funding_available']=1.0
    return e[['ts']+FUNDING_FEATURES]

def oi_features(raw):
    if raw is None or len(raw)==0: return pd.DataFrame(columns=['ts']+OI_FEATURES)
    e=raw.sort_values('ts').copy(); lv=np.log1p(np.maximum(e.oi_usd.astype(float),0.0)); e['oi_logusd']=lv
    e['oi_ret1d']=lv.diff(1); e['oi_ret3d']=lv.diff(3); e['oi_ret7d']=lv.diff(7); m=lv.rolling(30,min_periods=10).mean(); sd=lv.rolling(30,min_periods=10).std(); e['oi_z30']=(lv-m)/(sd+1e-9); e['oi_available']=1.0
    return e[['ts']+OI_FEATURES]

def merge_leverage(base, inst, funding_all, oi_map):
    d=base.sort_values('ts').copy()
    f=funding_features(funding_all[funding_all.instrument_name==inst])
    if len(f): d=pd.merge_asof(d,f.sort_values('ts'),on='ts',direction='backward',allow_exact_matches=True)
    else:
        for x in FUNDING_FEATURES: d[x]=np.nan
    o=oi_features(oi_map.get(inst))
    if len(o): d=pd.merge_asof(d,o.sort_values('ts'),on='ts',direction='backward',allow_exact_matches=True,tolerance=3*86_400_000)
    else:
        for x in OI_FEATURES: d[x]=np.nan
    for x in FUNDING_FEATURES+OI_FEATURES: d[x]=pd.to_numeric(d[x],errors='coerce').fillna(0.0)
    return d

def add_leverage_market(ds):
    d=ds.copy(); grp=d.groupby('ts',sort=False)
    d['mkt_funding_median']=grp.funding_last.transform('median'); d['mkt_funding_dispersion']=grp.funding_last.transform('std').fillna(0.0); d['funding_rel']=d.funding_last-d.mkt_funding_median; d['rank_funding']=grp.funding_last.rank(pct=True).fillna(.5)
    d['mkt_oi_ret1d']=grp.oi_ret1d.transform('median'); d['mkt_oi_dispersion']=grp.oi_ret1d.transform('std').fillna(0.0); d['rank_oi_ret1d']=grp.oi_ret1d.rank(pct=True).fillna(.5)
    return d

def clean(df):
    outcome=[f'fret_{h}' for h in HORIZONS]+[f'long_mae_{h}' for h in HORIZONS]+[f'short_mae_{h}' for h in HORIZONS]
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=c.FEATURES+outcome).copy().fillna({x:0.0 for x in FUNDING_FEATURES+OI_FEATURES+LEVERAGE_MARKET_FEATURES})

def window_sample(df,cutoff_ms,window_days):
    d=clean(df)
    if window_days: d=d[d.ts>=cutoff_ms-window_days*86_400_000]
    d=d.sort_values(['ts','symbol'])
    if len(d)>MAX_ROWS: d=d.iloc[np.linspace(0,len(d)-1,MAX_ROWS,dtype=int)]
    return d

def fit_one(train,cutoff_ms,window_days,seed=17):
    use=window_sample(train,cutoff_ms,window_days)
    if len(use)<25000: raise RuntimeError(f'not enough enriched rows {window_days}: {len(use)}')
    X=use[FEATURES].to_numpy(float); Y=g.reward_matrix(use,RISK_PENALTY)
    xs=StandardScaler().fit(X); ys=StandardScaler().fit(Y); Xs=xs.transform(X); Ys=ys.transform(Y)
    print('FIT_GEN3F','full' if window_days==0 else window_days,'rows',len(use),flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m=MLPRegressor(hidden_layer_sizes=(128,80,40),activation='relu',solver='adam',alpha=.003,batch_size=1024,learning_rate_init=.0006,max_iter=105,early_stopping=True,validation_fraction=.08,n_iter_no_change=10,random_state=seed)
        m.fit(Xs,Ys)
    return {'x_scaler':xs,'y_scaler':ys,'model':m,'window_days':window_days,'rows':len(use)}

def predict_one(bundle,df):
    X=bundle['x_scaler'].transform(df[FEATURES].to_numpy(float)); return bundle['y_scaler'].inverse_transform(bundle['model'].predict(X))

def evaluate_fold(research,name,start,end,j):
    vs=int(start.timestamp()*1000); ve=int(end.timestamp()*1000); tr=research[research.ts<vs-PURGE].copy(); va=research[(research.ts>=vs)&(research.ts<ve-PURGE)].copy()
    bs=[fit_one(tr,vs,w,17+j*31+i*7) for i,w in enumerate(WINDOWS)]; q=np.stack([predict_one(b,va) for b in bs]).mean(0); zeros=np.zeros_like(q); met,trades=g.simulate(va,q,zeros,0.0,CAPACITY,Q_FLOOR)
    return met,trades

def promotion(metrics):
    total=sum(m['trades'] for m in metrics); robust=g.robust_score(metrics)>-1e20; worst_avg=min(m['avg_net_bps'] for m in metrics); worst_pf=min(m['profit_factor'] for m in metrics); worst_dd=max(m['max_drawdown_bps'] for m in metrics); med=float(np.median([m['avg_net_bps'] for m in metrics])); coverage=all(m['trades']>=80 for m in metrics) and total>=400
    gates={'robust_pass':robust,'coverage_pass':coverage,'total_trades':total,'worst_avg_net_bps':worst_avg,'worst_profit_factor':worst_pf,'worst_max_drawdown_bps':worst_dd,'median_avg_net_bps':med,'gen3c_median_avg_net_bps':GEN3C_MEDIAN_AVG,'improves_worst_avg':worst_avg>-84.1294575172906,'improves_worst_pf':worst_pf>0.7215288019729019,'improves_worst_dd':worst_dd<6595.604675176961,'improves_median_avg':med>GEN3C_MEDIAN_AVG}
    promote=all([gates['robust_pass'],gates['coverage_pass'],gates['improves_worst_avg'],gates['improves_worst_pf'],gates['improves_worst_dd'],gates['improves_median_avg']]); return promote,gates

def main():
    symbols=core.discover()[:MAX_SYMBOLS]; print('DISCOVERED',len(symbols),symbols[:20],flush=True)
    fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(g.fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            s=fut[f]
            try: iid,d,cc=f.result(); fetched[iid]=d; calls+=cc; print('OHLCV',iid,len(d),flush=True)
            except Exception as e: print('OHLCV_FAIL',s,repr(e),flush=True)
    usable=[s for s in symbols if len(fetched.get(s,pd.DataFrame()))>=1600 and (fetched[s].ts<CUTOFF_MS).sum()>=1500]
    if len(usable)<24: raise RuntimeError(f'only {len(usable)} usable symbols')
    funding, funding_prov, funding_misses=load_funding(set(usable)); print('FUNDING_ROWS',len(funding),'DAYS',len(funding_prov),'MISSES',funding_misses,flush=True)
    oi_map,oi_calls=load_oi(usable)
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in usable:
        raw=fetched[inst]; raw_hash.update(pd.util.hash_pandas_object(raw,index=False).values.tobytes()); sym=inst.replace('-USDT-SWAP',''); base=g.individual_frame(raw,sym); parts.append(merge_leverage(base,inst,funding,oi_map)); accepted.append({'instrument':inst,'bars':int(len(raw)),'funding_events':int((funding.instrument_name==inst).sum()),'oi_days':int(len(oi_map.get(inst,pd.DataFrame())))})
    ds=g.add_market_state(pd.concat(parts,ignore_index=True)); ds=add_leverage_market(ds); ds=clean(ds).sort_values(['ts','symbol']).reset_index(drop=True); research=ds[ds.ts<CUTOFF_MS-PURGE].copy(); print('RESEARCH_ROWS',len(research),'SYMBOLS',research.symbol.nunique(),flush=True)
    reports=[]; alltr=[]
    for j,(name,start,end) in enumerate(FOLDS):
        met,tr=evaluate_fold(research,name,start,end,j); reports.append({'name':name,'metrics':met}); alltr.extend([dict(x,fold=name) for x in tr]); print('FOLD',name,json.dumps(met,sort_keys=True),flush=True)
    metrics=[x['metrics'] for x in reports]; promote,gates=promotion(metrics); print('PROMOTION',promote,json.dumps(gates,sort_keys=True),flush=True)
    final=[fit_one(research,CUTOFF_MS,w,59+i*13) for i,w in enumerate(WINDOWS)]
    artifact={'sub_policies':final,'features':FEATURES,'actions':ACTIONS,'risk_penalty':RISK_PENALTY,'consensus_penalty':0.0,'capacity':CAPACITY,'q_floor_bps':Q_FLOOR,'timeframe':'4h','roundtrip_cost_bps':COST_BPS,'policy_version':'GEN3F_LEVERAGE_STATE'}; joblib.dump(artifact,OUT/'dynamic_policy.joblib')
    manifest={'version':'R1F-GEN3F-LEVERAGE-STATE','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'decision_authority':'LEARNED_TEMPORAL_ACTION_VALUE_POLICY_WITH_LEVERAGE_STATE','fixed_strategy_expert_authority':False,'timeframe':'4h','symbol_count':len(accepted),'research_rows':int(len(research)),'features':FEATURES,'actions':ACTIONS,'walk_forward_folds':reports,'promotion_vs_gen3c':{'promote_to_active_shadow':promote,**gates},'qualification':{'qualified':False,'reasons':['no_fresh_untouched_period_after_gen3_retraining'],'authority':'SHADOW_PAPER_ONLY'},'provenance':{'ohlcv_provider':'OKX public REST market history','funding_provider':'OKX static historical funding archives','funding_archive_schema':['instrument_name','funding_rate','funding_time'],'funding_days':len(funding_prov),'funding_missing_days':funding_misses,'funding_manifest_sample':funding_prov[:5],'oi_provider':'OKX /api/v5/rubik/stat/contracts/open-interest-history','oi_api_calls':oi_calls,'accepted_symbols':accepted,'ohlcv_raw_sha256':raw_hash.hexdigest()},'notes':['Same autonomous action-value concept as Gen3C; no named strategy chooses trades.','Funding and OI are merged causally backward: only values timestamped at or before each 4h decision are visible.','OI has an explicit availability flag; unavailable history is never fabricated.','Funding archive files are schema-validated before use.','2026-08-01+ remains excluded from enhancement selection.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'walkforward_trades.json').write_text(json.dumps(alltr,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2)); print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
