from __future__ import annotations
import json, math, os, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import requests, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE='https://fapi.binance.com'
START_MS=int(pd.Timestamp(os.getenv('TRAIN_START','2023-01-01'),tz='UTC').timestamp()*1000)
END_MS=int(pd.Timestamp(os.getenv('TRAIN_END','2026-08-31 23:59:59'),tz='UTC').timestamp()*1000)
MAX_SYMBOLS=int(os.getenv('MAX_SYMBOLS','80'))
INTERVAL=os.getenv('INTERVAL','1h')
HORIZON=int(os.getenv('HORIZON_BARS','4'))
COST_BPS=float(os.getenv('ROUNDTRIP_COST_BPS','30'))
OUT=Path(os.getenv('OUT_DIR','trained_artifact')); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=['ret1','ret2','ret4','ret8','ret16','ret32','ema8_21_atr','ema21_55_atr','ema8_slope_atr','ema21_slope_atr','rsi14','atr_pct','rv8','rv32','vol_ratio','volume_z32','range20_pos','range55_pos','breakout20_atr','breakdown20_atr','efficiency20','autocorr20','body_frac','upper_wick_frac','lower_wick_frac','down_share20','dist_ema21_atr','hour_sin','hour_cos']
REGIME_CODE={'RANGE_QUIET':0,'RANGE_VOLATILE':1,'TREND_UP':2,'TREND_DOWN':3,'BREAKOUT_UP':4,'BREAKOUT_DOWN':5,'CHAOTIC':6}
ARCHETYPE_CODE={'BALANCED':0,'HIGH_BETA':1,'MEAN_REVERTING':2,'TREND_PERSISTENT':3,'LOW_VOL':4}
S=requests.Session(); S.headers['User-Agent']='GLOBAL-CRYPTO-HUNTER-R1D-REAL-TRAIN/1.0'

def get_json(path,params=None,retries=6):
    for i in range(retries):
        r=S.get(BASE+path,params=params,timeout=30)
        if r.status_code==200: return r.json()
        if r.status_code in (418,429):
            time.sleep(max(2,2**i)); continue
        if r.status_code==451: raise RuntimeError('Binance API geo-blocked on runner')
        time.sleep(min(10,1.5**i))
    raise RuntimeError(f'HTTP failure {path}: {r.status_code} {r.text[:200]}')

def discover_symbols():
    info=get_json('/fapi/v1/exchangeInfo'); ticks=get_json('/fapi/v1/ticker/24hr')
    qv={x['symbol']:float(x.get('quoteVolume') or 0) for x in ticks if 'symbol' in x}
    syms=[]
    for x in info['symbols']:
        if x.get('status')!='TRADING' or x.get('contractType')!='PERPETUAL' or x.get('quoteAsset')!='USDT': continue
        s=x['symbol']; syms.append((qv.get(s,0.0),s))
    syms.sort(reverse=True)
    return [s for _,s in syms[:MAX_SYMBOLS]]

def fetch_klines(symbol):
    rows=[]; start=START_MS; calls=0
    while start<END_MS:
        data=get_json('/fapi/v1/klines',{'symbol':symbol,'interval':INTERVAL,'startTime':start,'endTime':END_MS,'limit':1000})
        calls+=1
        if not data: break
        rows.extend(data)
        nxt=int(data[-1][0])+1
        if nxt<=start: break
        start=nxt
        if len(data)<1000: break
        time.sleep(0.13)
    if not rows: return pd.DataFrame(),calls
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume','close_ts','quote_volume','trades','taker_base','taker_quote','ignore'])
    for c in ['ts','open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df[['ts','open','high','low','close','volume']].dropna().drop_duplicates('ts').sort_values('ts'),calls

def safe_div(a,b): return a/b.replace(0,np.nan)
def feature_frame(df,h=4):
    if len(df)<256: return pd.DataFrame()
    x=df.copy().reset_index(drop=True); c,o,hi,lo,v=x.close,x.open,x.high,x.low,x.volume.clip(lower=0); r1=c.pct_change()
    for n in (1,2,4,8,16,32): x[f'ret{n}']=c.pct_change(n)
    e8,e21,e55=(c.ewm(span=n,adjust=False).mean() for n in (8,21,55)); prev=c.shift(1)
    tr=pd.concat([(hi-lo).abs(),(hi-prev).abs(),(lo-prev).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    x['ema8_21_atr']=(e8-e21)/atr; x['ema21_55_atr']=(e21-e55)/atr; x['ema8_slope_atr']=(e8-e8.shift(4))/atr; x['ema21_slope_atr']=(e21-e21.shift(4))/atr
    d=c.diff(); gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=safe_div(gain,loss); x['rsi14']=100-(100/(1+rs))
    x['atr_pct']=atr/c; x['rv8']=r1.rolling(8).std(); x['rv32']=r1.rolling(32).std(); x['vol_ratio']=safe_div(x.rv8,x.rv32)
    lv=np.log1p(v); x['volume_z32']=safe_div(lv-lv.rolling(32).mean(),lv.rolling(32).std())
    hi20=hi.rolling(20).max(); lo20=lo.rolling(20).min(); hi55=hi.rolling(55).max(); lo55=lo.rolling(55).min(); x['range20_pos']=safe_div(c-lo20,hi20-lo20); x['range55_pos']=safe_div(c-lo55,hi55-lo55)
    x['breakout20_atr']=(c-hi20.shift(1))/atr; x['breakdown20_atr']=(lo20.shift(1)-c)/atr; travel=c.diff().abs().rolling(20).sum(); x['efficiency20']=safe_div((c-c.shift(20)).abs(),travel); x['autocorr20']=r1.rolling(20).corr(r1.shift(1))
    rng=(hi-lo).replace(0,np.nan); x['body_frac']=(c-o)/rng; x['upper_wick_frac']=(hi-pd.concat([o,c],axis=1).max(axis=1))/rng; x['lower_wick_frac']=(pd.concat([o,c],axis=1).min(axis=1)-lo)/rng; x['down_share20']=(r1<0).astype(float).rolling(20).mean(); x['dist_ema21_atr']=(c-e21)/atr
    dt=pd.to_datetime(x.ts,unit='ms',utc=True); hour=dt.dt.hour+dt.dt.minute/60; x['hour_sin']=np.sin(2*np.pi*hour/24); x['hour_cos']=np.cos(2*np.pi*hour/24)
    volmed=x.rv32.rolling(256,min_periods=64).median(); arche=np.full(len(x),'BALANCED',dtype=object); arche=np.where(x.rv32<.65*volmed,'LOW_VOL',arche); arche=np.where(x.rv32>1.65*volmed,'HIGH_BETA',arche); arche=np.where((x.autocorr20<-.12)&(x.efficiency20<.35),'MEAN_REVERTING',arche); arche=np.where((x.autocorr20>.08)&(x.efficiency20>.38),'TREND_PERSISTENT',arche); x['archetype']=arche
    regime=np.full(len(x),'RANGE_VOLATILE',dtype=object); regime=np.where(x.vol_ratio>1.75,'CHAOTIC',regime); regime=np.where((x.efficiency20<.25)&(x.vol_ratio<.9),'RANGE_QUIET',regime); regime=np.where((x.ema8_21_atr>.45)&(x.ema21_slope_atr>.15),'TREND_UP',regime); regime=np.where((x.ema8_21_atr<-.45)&(x.ema21_slope_atr<-.15),'TREND_DOWN',regime); regime=np.where((x.breakout20_atr>.30)&(x.volume_z32>.35),'BREAKOUT_UP',regime); regime=np.where((x.breakdown20_atr>.30)&(x.volume_z32>.35),'BREAKOUT_DOWN',regime); x['regime']=regime
    x['regime_code']=[REGIME_CODE.get(z,0) for z in x.regime]; x['archetype_code']=[ARCHETYPE_CODE.get(z,0) for z in x.archetype]
    x['future_ret_bps']=(c.shift(-h)/c-1)*10000; x['future_ts']=x.ts.shift(-h); return x

def strategy_directions(x):
    z={}; z['trend_following']=np.sign(x.ema8_21_atr).where((x.efficiency20>.30)&(x.ema21_slope_atr.abs()>.08),0); z['breakout']=pd.Series(np.where(x.breakout20_atr>.25,1,np.where(x.breakdown20_atr>.25,-1,0)),index=x.index); z['pullback_continuation']=pd.Series(np.where((x.ema8_21_atr>.35)&(x.ret1<0)&(x.dist_ema21_atr>0),1,np.where((x.ema8_21_atr<-.35)&(x.ret1>0)&(x.dist_ema21_atr<0),-1,0)),index=x.index); z['mean_reversion']=pd.Series(np.where(x.dist_ema21_atr>1.2,-1,np.where(x.dist_ema21_atr<-1.2,1,0)),index=x.index); z['range_fade']=pd.Series(np.where((x.range20_pos>.86)&(x.efficiency20<.35),-1,np.where((x.range20_pos<.14)&(x.efficiency20<.35),1,0)),index=x.index); z['volatility_expansion']=np.sign(x.ret4).where((x.vol_ratio>1.25)&(x.volume_z32>.2),0); z['momentum_acceleration']=np.sign(x.ret4).where((x.ret4.abs()>x.rv32.fillna(0)*2.5)&(x.efficiency20>.28),0); z['exhaustion_reversal']=pd.Series(np.where((x.rsi14>76)&(x.upper_wick_frac>.25),-1,np.where((x.rsi14<24)&(x.lower_wick_frac>.25),1,0)),index=x.index); return z

def matrix(x): return x[FEATURES+['regime_code','archetype_code']].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
def sample_spread(x,cap=5000):
    if len(x)<=cap:return x
    idx=np.linspace(0,len(x)-1,cap,dtype=int); return x.iloc[np.unique(idx)].copy()
def split_purged(g,h):
    n=len(g); a=int(.70*n); b=int(.85*n); return g.iloc[:max(0,a-h)].copy(),g.iloc[a:max(a,b-h)].copy(),g.iloc[b:].copy()
def metrics(pred,y,thr,symbols,regimes):
    pred=np.asarray(pred); y=np.asarray(y); act=np.abs(pred)>=thr; net=np.sign(pred[act])*y[act]-COST_BPS
    if not len(net): return {'trades':0,'avg_net_bps':0,'profit_factor':0,'win_rate':0,'max_drawdown_bps':0,'positive_coin_fraction':0,'positive_regime_fraction':0}
    pos=net[net>0].sum(); neg=-net[net<0].sum(); curve=np.cumsum(net); peak=np.maximum.accumulate(np.r_[0.,curve]); dd=peak[1:]-curve; d=pd.DataFrame({'net':net,'symbol':np.asarray(symbols)[act],'regime':np.asarray(regimes)[act]}); cg=d.groupby('symbol').net.mean(); rg=d.groupby('regime').net.mean()
    return {'trades':int(len(net)),'avg_net_bps':float(net.mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'win_rate':float((net>0).mean()),'max_drawdown_bps':float(dd.max() if len(dd) else 0),'positive_coin_fraction':float((cg>0).mean()),'positive_regime_fraction':float((rg>0).mean())}
def choose_threshold(pred,val):
    cand=sorted(set([5.,8.,10.,12.,15.,20.,25.,30.,40.,50.,75.,100.]+[float(np.quantile(np.abs(pred),q)) for q in (.5,.6,.7,.8,.85,.9,.93,.95)])); best=None
    for t in cand:
        m=metrics(pred,val.future_ret_bps.to_numpy(),t,val.symbol.to_numpy(),val.regime.to_numpy())
        if m['trades']<250: continue
        score=m['avg_net_bps']*math.sqrt(m['trades'])*min(1.5,max(.1,m['profit_factor']))
        if best is None or score>best[0]: best=(score,t,m)
    return float(best[1] if best else np.quantile(np.abs(pred),.9))
def edge_table(train,shrink_n=100):
    out={}
    for name,ser in strategy_directions(train).items():
        t=train[['regime','archetype','future_ret_bps']].copy(); t['dir']=ser.astype(int); t=t[t.dir!=0]; t['net']=t.dir*t.future_ret_bps-COST_BPS
        for (r,a),g in t.groupby(['regime','archetype']):
            n=len(g); avg=float(g.net.mean()); pos=float(g.loc[g.net>0,'net'].sum()); neg=float(-g.loc[g.net<0,'net'].sum()); out[f'{r}|{a}|{name}']={'n':n,'avg_net_bps':avg,'win_rate':float((g.net>0).mean()),'profit_factor':float(pos/neg) if neg>0 else 99.,'shrunk_edge_bps':avg*n/(n+shrink_n)}
    return out

def main():
    symbols=discover_symbols(); print('DISCOVERED',len(symbols),symbols[:10]); parts=[]; source_hash=hashlib.sha256(); calls=0; accepted=[]
    for i,s in enumerate(symbols,1):
        try: df,c=fetch_klines(s); calls+=c
        except Exception as e: print('FAIL',s,e); continue
        if len(df)<2500: print('SKIP',s,len(df)); continue
        source_hash.update(pd.util.hash_pandas_object(df,index=False).values.tobytes())
        f=feature_frame(df,HORIZON).dropna(subset=['future_ret_bps']); f=sample_spread(f,5000); f['symbol']=s; parts.append(f); accepted.append({'symbol':s,'bars':int(len(df)),'first_ts':int(df.ts.iloc[0]),'last_ts':int(df.ts.iloc[-1])}); print(f'[{i}/{len(symbols)}] {s} bars={len(df)} sampled={len(f)}')
    if len(parts)<20: raise RuntimeError(f'only {len(parts)} symbols had sufficient real history')
    ds=pd.concat(parts,ignore_index=True); trs=[]; vas=[]; hos=[]
    for s,g in ds.groupby('symbol',sort=False):
        tr,va,ho=split_purged(g.sort_values('ts').reset_index(drop=True),HORIZON); trs.append(tr); vas.append(va); hos.append(ho)
    train=pd.concat(trs,ignore_index=True).replace([np.inf,-np.inf],np.nan); val=pd.concat(vas,ignore_index=True).replace([np.inf,-np.inf],np.nan); hold=pd.concat(hos,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=300,learning_rate=.045,max_leaf_nodes=31,l2_regularization=2.0,random_state=17).fit(matrix(train),train.future_ret_bps.to_numpy())
    pv=model.predict(matrix(val)); thr=choose_threshold(pv,val); ph=model.predict(matrix(hold)); hm=metrics(ph,hold.future_ret_bps.to_numpy(),thr,hold.symbol.to_numpy(),hold.regime.to_numpy())
    reasons=[]
    checks=[(hm['trades']>=300,'insufficient_holdout_trades'),(hm['avg_net_bps']>=1.0,'weak_holdout_edge'),(hm['profit_factor']>=1.08,'profit_factor_below_gate'),(hm['positive_coin_fraction']>=.45,'coin_breadth_below_gate'),(hm['positive_regime_fraction']>=.45,'regime_breadth_below_gate'),(hm['max_drawdown_bps']<=3500,'drawdown_above_gate')]
    for ok,r in checks:
        if not ok: reasons.append(r)
    qualified=not reasons
    joblib.dump(model,OUT/'model.joblib'); (OUT/'strategy_edges.json').write_text(json.dumps(edge_table(train),indent=2))
    prov={'provider':'Binance USD-M Futures REST public market data','endpoint':'https://fapi.binance.com/fapi/v1/klines','verified_public_archive':True,'retrieved_at':datetime.now(timezone.utc).isoformat(),'interval':INTERVAL,'requested_start':START_MS,'requested_end':END_MS,'accepted_symbols':accepted,'raw_history_sha256':source_hash.hexdigest(),'api_calls':calls,'selection':'dynamic top active USDT perpetuals by 24h quote volume','requested_symbol_count':MAX_SYMBOLS}
    manifest={'version':'R1D-REAL','trained_at':datetime.now(timezone.utc).isoformat(),'trained_on_real_history':True,'timeframe':INTERVAL,'symbol_count':len(accepted),'rows':len(ds),'horizon_bars':HORIZON,'historical_roundtrip_cost_bps':COST_BPS,'prediction_threshold_bps':thr,'model_file':'model.joblib','edge_table_file':'strategy_edges.json','provenance':prov,'qualification':{'qualified':qualified,**hm,'reasons':reasons}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); (OUT/'TRAINING_REPORT.txt').write_text(json.dumps(manifest,indent=2)); print('FINAL_MANIFEST'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
