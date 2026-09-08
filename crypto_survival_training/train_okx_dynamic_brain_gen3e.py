from __future__ import annotations
import json, math, os, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from crypto_survival_training import train_okx_dynamic_brain as core
from crypto_survival_training import train_okx_dynamic_brain_gen3b as g
from crypto_survival_training import train_okx_dynamic_brain_gen3c as c

# GENERATION 3E — CONSERVATIVE EXPERIENCE / TRADE-QUALITY BRAIN
# The temporal neural policy still chooses coin-side/size/horizon action values.
# A second learned layer is trained ONLY on causal out-of-fold policy proposals
# and their real subsequent net outcomes. It estimates the conservative lower
# quantile of the proposed trade's future net return. If that estimate <= 0,
# the autonomous action is WAIT. No named strategy or hand-written regime
# chooses a trade. 2026-08-01+ remains excluded from all enhancement learning.

CUTOFF_MS=c.CUTOFF_MS; PURGE=c.PURGE; BAR_MS=c.BAR_MS
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','72'))
COST_BPS=c.COST_BPS
BASE_ROWS=int(os.getenv('BASE_TRAIN_ROWS','85000'))
OUT=Path(os.getenv('OUT_DIR','trained_artifact_gen3e')); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=c.FEATURES; ACTIONS=c.ACTIONS; WINDOWS=c.WINDOWS
CAPACITY=2.0

# These windows generate truly out-of-fold experience for the quality learner.
# Hyperparameters below are frozen; no search is performed on Gen3C/Gen3D test folds.
EXPERIENCE_WINDOWS=[
    ('E1',pd.Timestamp('2024-07-01',tz='UTC'),pd.Timestamp('2024-10-01',tz='UTC')),
    ('E2',pd.Timestamp('2024-10-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC')),
    ('E3',pd.Timestamp('2025-01-01',tz='UTC'),pd.Timestamp('2025-04-01',tz='UTC')),
    ('E4',pd.Timestamp('2025-04-01',tz='UTC'),pd.Timestamp('2025-07-01',tz='UTC')),
]
TEST_FOLDS=g.FOLDS
PROPOSAL_FEATURES=[
    'policy_best_q','policy_second_q','policy_margin_q','memory_std_chosen_q',
    'memory_min_chosen_q','memory_max_chosen_q','memory_action_agreement',
    'chosen_dir','chosen_size','chosen_horizon_bars'
]
QUALITY_FEATURES=FEATURES+PROPOSAL_FEATURES

GEN3C_REFERENCE={
    'worst_avg_net_bps':-84.1294575172906,
    'worst_profit_factor':0.7215288019729019,
    'worst_max_drawdown_bps':6595.604675176961,
}

def sample_window(df, cutoff_ms, window_days):
    d=g.clean_training_rows(df)
    if window_days:
        d=d[d.ts>=cutoff_ms-window_days*86_400_000]
    d=d.sort_values(['ts','symbol'])
    if len(d)>BASE_ROWS:
        d=d.iloc[np.linspace(0,len(d)-1,BASE_ROWS,dtype=int)]
    return d

def fit_memory(train, cutoff_ms, window_days, seed):
    # Temporarily reuse Gen3C architecture with a smaller deterministic sample budget.
    old=c.MAX_ROWS
    try:
        c.MAX_ROWS=BASE_ROWS
        return c.fit_one(train,cutoff_ms,window_days,seed)
    finally:
        c.MAX_ROWS=old

def temporal_proposals(train, eval_df, cutoff_ms, seed_base):
    bundles=[fit_memory(train,cutoff_ms,w,seed_base+i*11) for i,w in enumerate(WINDOWS)]
    preds=[c.predict_one(b,eval_df) for b in bundles]
    stack=np.stack(preds,axis=0)
    q=stack.mean(0)  # Gen3C's selected consensus penalty was 0.0.
    ai=np.argmax(q,axis=1)
    order=np.partition(q,-2,axis=1)
    best=q[np.arange(len(q)),ai]
    second=order[:,-2]
    chosen_mem=np.take_along_axis(stack,ai[None,:,None],axis=2).squeeze(2).T
    mem_arg=np.argmax(stack,axis=2).T
    agree=(mem_arg==ai[:,None]).mean(axis=1)
    meta=pd.DataFrame({
        'policy_best_q':best,
        'policy_second_q':second,
        'policy_margin_q':best-second,
        'memory_std_chosen_q':chosen_mem.std(axis=1),
        'memory_min_chosen_q':chosen_mem.min(axis=1),
        'memory_max_chosen_q':chosen_mem.max(axis=1),
        'memory_action_agreement':agree,
        'chosen_dir':[ACTIONS[int(i)]['dir'] for i in ai],
        'chosen_size':[ACTIONS[int(i)]['size'] for i in ai],
        'chosen_horizon_bars':[ACTIONS[int(i)]['h'] for i in ai],
        'action_index':ai,
    },index=eval_df.index)
    out=eval_df.copy()
    for k in PROPOSAL_FEATURES: out[k]=meta[k]
    out['action_index']=ai
    out['actual_net_bps']=[g.actual_net(out.loc[idx],int(a)) for idx,a in zip(out.index,ai)]
    return bundles,out

def fit_quality(experience):
    d=experience.replace([np.inf,-np.inf],np.nan).dropna(subset=QUALITY_FEATURES+['actual_net_bps']).copy()
    if len(d)<30000: raise RuntimeError(f'not enough causal experience rows: {len(d)}')
    X=d[QUALITY_FEATURES].to_numpy(float); y=d.actual_net_bps.to_numpy(float)
    # Deliberately low-complexity, strongly regularized learners to avoid Gen3D-style meta overfit.
    median=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.045,max_iter=180,max_leaf_nodes=11,min_samples_leaf=160,l2_regularization=8.0,random_state=301)
    lower=HistGradientBoostingRegressor(loss='quantile',quantile=.25,learning_rate=.04,max_iter=200,max_leaf_nodes=9,min_samples_leaf=180,l2_regularization=10.0,random_state=302)
    median.fit(X,y); lower.fit(X,y)
    return {'median':median,'lower_q25':lower,'rows':len(d)}

def quality_predict(qm, df):
    X=df[QUALITY_FEATURES].to_numpy(float)
    med=qm['median'].predict(X); lo=qm['lower_q25'].predict(X)
    conservative=np.minimum(med,lo)
    return med,lo,conservative

def simulate_quality(df, conservative):
    ai=df.action_index.to_numpy(int)
    cand=df[['ts','symbol']].copy(); cand['i']=np.arange(len(df)); cand['a']=ai; cand['quality']=conservative
    # Zero is an economic threshold: predicted lower-quartile net after costs must be positive.
    cand=cand[cand.quality>0.0].sort_values(['ts','quality'],ascending=[True,False])
    active=[]; trades=[]; pnl_exit={}
    for ts,grp in cand.groupby('ts',sort=True):
        active=[p for p in active if p['exit']>ts]
        used=sum(p['size'] for p in active); avail=max(0.,CAPACITY-used); active_sy={p['symbol'] for p in active}
        if avail<.49: continue
        for r in grp.itertuples(index=False):
            if r.symbol in active_sy: continue
            a=ACTIONS[int(r.a)]
            if a['size']>avail+1e-9: continue
            src=df.iloc[int(r.i)]; net=g.actual_net(src,int(r.a)); ex=int(r.ts+a['h']*BAR_MS)
            active.append({'symbol':r.symbol,'size':a['size'],'exit':ex}); active_sy.add(r.symbol); avail-=a['size']
            trades.append({'entry_ts':int(r.ts),'exit_ts':ex,'symbol':r.symbol,'action':a['name'],'size':a['size'],'quality_q25_bps':float(r.quality),'net_bps':float(net)})
            pnl_exit[ex]=pnl_exit.get(ex,0.)+float(net)/CAPACITY
            if avail<.49: break
    if not trades:
        return {'trades':0,'avg_net_bps':0.,'profit_factor':0.,'win_rate':0.,'max_drawdown_bps':0.,'positive_coin_fraction':0.,'positive_week_fraction':0.,'worst_week_bps':0.,'active_actions':0,'selected_symbols':0,'portfolio_return_bps':0.},[]
    t=pd.DataFrame(trades); n=t.net_bps.to_numpy(float); pos=n[n>0].sum(); neg=-n[n<0].sum()
    pnl=pd.Series(pnl_exit).sort_index(); curve=pnl.cumsum().to_numpy(); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve
    cg=t.groupby('symbol').net_bps.mean(); wk=pd.to_datetime(t.exit_ts,unit='ms',utc=True).dt.strftime('%G-W%V'); wg=t.assign(week=wk).groupby('week').net_bps.mean()
    return {'trades':int(len(t)),'avg_net_bps':float(n.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((n>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0.),'positive_coin_fraction':float((cg>0).mean()),'positive_week_fraction':float((wg>0).mean()),'worst_week_bps':float(wg.min()),'active_actions':int(t.action.nunique()),'selected_symbols':int(t.symbol.nunique()),'portfolio_return_bps':float(pnl.sum())},trades

def promotion_check(metrics):
    robust=g.robust_score(metrics)>-1e20
    worst_avg=min(m['avg_net_bps'] for m in metrics)
    worst_pf=min(m['profit_factor'] for m in metrics)
    worst_dd=max(m['max_drawdown_bps'] for m in metrics)
    improves=(worst_avg>GEN3C_REFERENCE['worst_avg_net_bps'] and worst_pf>GEN3C_REFERENCE['worst_profit_factor'] and worst_dd<GEN3C_REFERENCE['worst_max_drawdown_bps'])
    return robust and improves, {'robust_pass':robust,'worst_avg_net_bps':worst_avg,'worst_profit_factor':worst_pf,'worst_max_drawdown_bps':worst_dd,'improves_all_gen3c_bad_regime_metrics':improves}

def main():
    symbols=core.discover()[:MAX_SYMBOLS]; print('DISCOVERED',len(symbols),symbols[:20],flush=True)
    fetched={}; calls=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(g.fetch_history,s):s for s in symbols}
        for f in as_completed(fut):
            s=fut[f]
            try:
                iid,d,cc=f.result(); fetched[iid]=d; calls+=cc; print('FETCHED',iid,len(d),flush=True)
            except Exception as e: print('FAIL',s,repr(e),flush=True)
    parts=[]; accepted=[]; raw_hash=hashlib.sha256()
    for inst in symbols:
        d=fetched.get(inst,pd.DataFrame())
        if len(d)<1600 or (d.ts<CUTOFF_MS).sum()<1500: continue
        raw_hash.update(pd.util.hash_pandas_object(d,index=False).values.tobytes()); sym=inst.replace('-USDT-SWAP','')
        parts.append(g.individual_frame(d,sym)); accepted.append({'instrument':inst,'bars':int(len(d)),'first_ts':int(d.ts.iloc[0]),'last_ts':int(d.ts.iloc[-1])})
    if len(parts)<24: raise RuntimeError(f'only {len(parts)} usable symbols')
    ds=g.add_market_state(pd.concat(parts,ignore_index=True)); ds=g.clean_training_rows(ds).sort_values(['ts','symbol']).reset_index(drop=True)
    research=ds[ds.ts<CUTOFF_MS-PURGE].copy(); print('RESEARCH_ROWS',len(research),'SYMBOLS',research.symbol.nunique(),flush=True)

    experience=[]; experience_reports=[]
    for j,(name,start,end) in enumerate(EXPERIENCE_WINDOWS):
        vs=int(start.timestamp()*1000); ve=int(end.timestamp()*1000)
        tr=research[research.ts<vs-PURGE].copy(); ev=research[(research.ts>=vs)&(research.ts<ve-PURGE)].copy()
        _,prop=temporal_proposals(tr,ev,vs,401+j*37)
        experience.append(prop); experience_reports.append({'name':name,'rows':int(len(prop)),'start':start.isoformat(),'end':end.isoformat()})
        print('EXPERIENCE_WINDOW',name,len(prop),flush=True)
    quality_history=pd.concat(experience,ignore_index=True)

    fold_reports=[]; fold_trades=[]
    # Strict rolling evaluation: quality model for each fold uses only proposal/outcome experience available before that fold.
    for j,(name,start,end) in enumerate(TEST_FOLDS):
        vs=int(start.timestamp()*1000); ve=int(end.timestamp()*1000)
        hist=quality_history[quality_history.ts<vs-PURGE].copy(); qm=fit_quality(hist)
        tr=research[research.ts<vs-PURGE].copy(); ev=research[(research.ts>=vs)&(research.ts<ve-PURGE)].copy()
        _,prop=temporal_proposals(tr,ev,vs,701+j*41)
        med,lo,cons=quality_predict(qm,prop); met,trades=simulate_quality(prop,cons)
        rep={'name':name,'quality_train_rows':qm['rows'],'metrics':met,'mean_pred_median_bps':float(np.mean(med)),'mean_pred_q25_bps':float(np.mean(lo)),'wait_fraction':float((cons<=0).mean())}
        fold_reports.append(rep); fold_trades.extend([dict(x,fold=name) for x in trades]); print('TEST_FOLD',name,json.dumps(rep,sort_keys=True),flush=True)
        # Once this fold has occurred in time, its outcomes become legitimate experience for the next fold.
        quality_history=pd.concat([quality_history,prop],ignore_index=True)

    metrics=[x['metrics'] for x in fold_reports]
    promote,promotion=promotion_check(metrics); print('PROMOTION',promote,json.dumps(promotion,sort_keys=True),flush=True)

    # Final shadow brain: temporal memories trained through pre-August research;
    # quality learner trained on all causally generated pre-August proposal experience.
    final_memories=[fit_memory(research,CUTOFF_MS,w,1001+i*19) for i,w in enumerate(WINDOWS)]
    final_qm=fit_quality(quality_history[quality_history.ts<CUTOFF_MS-PURGE])
    artifact={'sub_policies':final_memories,'quality_models':{'median':final_qm['median'],'lower_q25':final_qm['lower_q25']},'features':FEATURES,'quality_features':QUALITY_FEATURES,'proposal_features':PROPOSAL_FEATURES,'actions':ACTIONS,'capacity':CAPACITY,'timeframe':'4h','roundtrip_cost_bps':COST_BPS,'policy_version':'GEN3E_CONSERVATIVE_EXPERIENCE','decision_rule':'learned_lower_quartile_net_gt_zero_else_WAIT'}
    joblib.dump(artifact,OUT/'dynamic_policy.joblib')
    manifest={'version':'R1F-GEN3E-CONSERVATIVE-EXPERIENCE','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'decision_authority':'LEARNED_TEMPORAL_POLICY_PLUS_CAUSAL_TRADE_QUALITY','fixed_strategy_expert_authority':False,'timeframe':'4h','symbol_count':len(accepted),'research_rows':int(len(research)),'features':FEATURES,'proposal_features':PROPOSAL_FEATURES,'quality_features':QUALITY_FEATURES,'actions':ACTIONS,'experience_windows':experience_reports,'walk_forward_test_folds':fold_reports,'promotion_vs_gen3c':{'promote_to_active_shadow':promote,**promotion},'qualification':{'qualified':False,'reasons':['no_fresh_untouched_period_after_gen3_retraining'],'authority':'SHADOW_PAPER_ONLY'},'provenance':{'provider':'OKX public REST market history','interval':core.BAR,'start':'2023-01-01','enhancement_cutoff':'2026-08-01','already_inspected_period_excluded':True,'accepted_symbols':accepted,'raw_history_sha256':raw_hash.hexdigest(),'api_calls':calls},'notes':['No named trading strategy chooses an action.','Temporal neural memories propose actions; a conservative learned quality layer estimates the lower-quartile realized net of the proposal from causal out-of-fold experience.','WAIT is selected when learned conservative expected net is not positive after costs.','Quality hyperparameters and zero economic threshold are fixed before the Gen3C/Gen3D walk-forward folds; there is no threshold search on those folds.','All enhancement learning excludes 2026-08-01+; fresh forward shadow evidence remains mandatory for qualification.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'walkforward_trades.json').write_text(json.dumps(fold_trades,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2))
    print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
