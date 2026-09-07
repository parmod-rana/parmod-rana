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
from crypto_survival_training import train_okx_dynamic_brain_gen3b as g

# GENERATION 3C — TEMPORAL ENSEMBLE DYNAMIC POLICY
# Three learned action-value policies see different historical horizons:
# full history, 730 days and 365 days. Their learned action values are combined
# continuously; no named strategy/rule chooses direction, size or holding time.
# 2026-08-01+ stays excluded because it has already been inspected.

CUTOFF_MS=g.CUTOFF_MS; PURGE=g.PURGE; BAR_MS=g.BAR_MS
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','72')); COST_BPS=g.COST_BPS
MAX_ROWS=int(os.getenv('MAX_TRAIN_ROWS','110000')); OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen3c')); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=g.FEATURES; ACTIONS=g.ACTIONS; HORIZONS=g.HORIZONS
RISK_PENALTY=0.28
WINDOWS=(0,730,365) # 0 = all available history
CONSENSUS_PENALTIES=(0.0,0.35,0.7,1.1,1.5)
CAPACITIES=(2.0,3.0,5.0); Q_FLOORS=(0.0,10.0,25.0,40.0,60.0,90.0)
FOLDS=g.FOLDS

def window_sample(df,cutoff_ms,window_days):
    d=g.clean_training_rows(df)
    if window_days:
        d=d[d.ts>=cutoff_ms-window_days*86_400_000]
    d=d.sort_values(['ts','symbol'])
    if len(d)>MAX_ROWS:
        d=d.iloc[np.linspace(0,len(d)-1,MAX_ROWS,dtype=int)]
    return d

def fit_one(train,cutoff_ms,window_days,seed=17):
    use=window_sample(train,cutoff_ms,window_days)
    if len(use)<25000: raise RuntimeError(f'not enough rows for window {window_days}: {len(use)}')
    X=use[FEATURES].to_numpy(float); Y=g.reward_matrix(use,RISK_PENALTY)
    xs=StandardScaler().fit(X); ys=StandardScaler().fit(Y); Xs=xs.transform(X); Ys=ys.transform(Y)
    print('FIT_TEMPORAL','full' if window_days==0 else window_days,'rows',len(use),flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m=MLPRegressor(hidden_layer_sizes=(112,72,36),activation='relu',solver='adam',alpha=.0025,batch_size=1024,learning_rate_init=.00065,max_iter=100,early_stopping=True,validation_fraction=.08,n_iter_no_change=9,random_state=seed)
        m.fit(Xs,Ys)
    return {'x_scaler':xs,'y_scaler':ys,'model':m,'window_days':window_days,'rows':len(use)}

def predict_one(bundle,df):
    X=bundle['x_scaler'].transform(df[FEATURES].to_numpy(float)); return bundle['y_scaler'].inverse_transform(bundle['model'].predict(X))

def combine(preds,penalty):
    p=np.stack(preds,axis=0); return p.mean(0)-penalty*p.std(0)

def simulate_consensus(df,q,capacity,q_floor):
    # q already includes temporal-disagreement penalty, so std_q/up are unnecessary here.
    zeros=np.zeros_like(q)
    return g.simulate(df,q,zeros,0.0,capacity,q_floor)

def exploratory_score(mets):
    total=sum(m['trades'] for m in mets)
    if total<80: return -1e30
    av=np.array([m['avg_net_bps'] for m in mets]); pf=np.array([m['profit_factor'] for m in mets]); wk=np.array([m['positive_week_fraction'] for m in mets]); coin=np.array([m['positive_coin_fraction'] for m in mets]); dd=max(m['max_drawdown_bps'] for m in mets)
    return float(np.median(av)*math.sqrt(total)*max(.2,np.median(pf))*max(.15,np.median(wk))*max(.15,np.median(coin))/(1+dd/900.0))

def main():
    symbols=core.discover()[:MAX_SYMBOLS]; print('DISCOVERED',len(symbols),symbols[:20],flush=True)
    fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(g.fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            s=fut[f]
            try:
                iid,d,c=f.result(); fetched[iid]=d; calls+=c; print('FETCHED',iid,len(d),flush=True)
            except Exception as e: print('FAIL',s,repr(e),flush=True)
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in symbols:
        d=fetched.get(inst,pd.DataFrame())
        if len(d)<1600 or (d.ts<CUTOFF_MS).sum()<1500: continue
        raw_hash.update(pd.util.hash_pandas_object(d,index=False).values.tobytes()); sym=inst.replace('-USDT-SWAP',''); parts.append(g.individual_frame(d,sym)); accepted.append({'instrument':inst,'bars':int(len(d)),'first_ts':int(d.ts.iloc[0]),'last_ts':int(d.ts.iloc[-1])})
    if len(parts)<24: raise RuntimeError(f'only {len(parts)} usable symbols')
    ds=g.add_market_state(pd.concat(parts,ignore_index=True)); ds=g.clean_training_rows(ds).sort_values(['ts','symbol']).reset_index(drop=True)
    research=ds[ds.ts<CUTOFF_MS-PURGE].copy(); print('RESEARCH_ROWS',len(research),'SYMBOLS',research.symbol.nunique(),flush=True)
    fold_predictions=[]
    for name,vstart,vend in FOLDS:
        vs=int(vstart.timestamp()*1000); ve=int(vend.timestamp()*1000); tr=research[research.ts<vs-PURGE].copy(); va=research[(research.ts>=vs)&(research.ts<ve-PURGE)].copy()
        bundles=[fit_one(tr,vs,w,17) for w in WINDOWS]; preds=[predict_one(b,va) for b in bundles]
        fold_predictions.append((name,va,preds))
    candidates=[]
    for cp in CONSENSUS_PENALTIES:
        qs=[(name,va,combine(preds,cp)) for name,va,preds in fold_predictions]
        for cap in CAPACITIES:
            for floor in Q_FLOORS:
                mets=[]
                for name,va,q in qs:
                    met,_=simulate_consensus(va,q,cap,floor); mets.append(met)
                strict=g.robust_score(mets); explore=exploratory_score(mets)
                candidates.append({'consensus_penalty':cp,'capacity':cap,'q_floor_bps':floor,'strict_score':strict,'exploratory_score':explore,'folds':mets})
    candidates.sort(key=lambda x:(x['strict_score']>-1e20,x['strict_score'] if x['strict_score']>-1e20 else x['exploratory_score']),reverse=True)
    best=candidates[0]; strict_pass=best['strict_score']>-1e20
    print('SELECTED',json.dumps(best,sort_keys=True),flush=True)
    # Final enhanced brain is trained ONLY through pre-August research data.
    final_bundles=[fit_one(research,CUTOFF_MS,w,29) for w in WINDOWS]
    artifact={'sub_policies':final_bundles,'features':FEATURES,'actions':ACTIONS,'risk_penalty':RISK_PENALTY,'consensus_penalty':best['consensus_penalty'],'capacity':best['capacity'],'q_floor_bps':best['q_floor_bps'],'timeframe':'4h','roundtrip_cost_bps':COST_BPS,'policy_version':'GEN3C_TEMPORAL_ENSEMBLE'}
    joblib.dump(artifact,OUT/'dynamic_policy.joblib')
    manifest={'version':'R1F-GEN3C-TEMPORAL-ENSEMBLE','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'decision_authority':'LEARNED_TEMPORAL_ENSEMBLE_ACTION_VALUE_POLICY','fixed_strategy_expert_authority':False,'timeframe':'4h','symbol_count':len(accepted),'research_rows':int(len(research)),'features':FEATURES,'actions':ACTIONS,'temporal_windows_days':list(WINDOWS),'selected_calibration':{k:best[k] for k in ('consensus_penalty','capacity','q_floor_bps','strict_score','exploratory_score')},'walk_forward_folds':best['folds'],'walk_forward_robust_pass':strict_pass,'qualification':{'qualified':False,'reasons':['no_fresh_untouched_period_after_gen3_retraining'],'authority':'SHADOW_PAPER_ONLY'},'provenance':{'provider':'OKX public REST market history','interval':core.BAR,'start':'2023-01-01','enhancement_cutoff':'2026-08-01','already_inspected_period_excluded':True,'accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls},'notes':['Three neural policy memories are combined by learned action value, not named strategies.','Temporal disagreement reduces action value continuously rather than invoking a hand-written strategy gate.','All training and calibration exclude 2026-08-01+ because that period was already inspected.','Fresh forward shadow evidence is required for qualification.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'walkforward_candidates.json').write_text(json.dumps(candidates[:60],indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2))
    print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
