from __future__ import annotations
import hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from crypto_survival_training.microstructure_features import extract_csv_gz

API='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('real_microstructure_smoke'); OUT.mkdir(exist_ok=True)
BEGIN='1735689600000'  # 2025-01-01 00:00 UTC
END='1735776000000'    # 2025-01-02 00:00 UTC
INST='BAT-USDT'


def archive_url():
    p={'module':'6','instType':'SPOT','instIdList':INST,'dateAggrType':'daily','begin':BEGIN,'end':END}
    r=requests.get(API,params=p,timeout=30); r.raise_for_status(); body=r.json()
    if body.get('code')!='0': raise RuntimeError(body)
    files=[]
    for d in body.get('data',[]):
        for detail in d.get('details',[]) or []:
            for g in detail.get('groupDetails',[]) or []:
                if str(g.get('filename','')).endswith('.csv.gz'):
                    files.append(g)
    if not files: raise RuntimeError('no module-6 BAT-USDT archive returned')
    target=next((x for x in files if '2025-01-01' in str(x.get('filename'))),files[-1])
    return target


def main():
    meta=archive_url(); url=meta['url']; fn=OUT/str(meta['filename'])
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        h=hashlib.sha256(); total=0
        with fn.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if not chunk: continue
                f.write(chunk); h.update(chunk); total+=len(chunk)
    print('REAL_TBT_DOWNLOAD',{'instrument':INST,'file':fn.name,'bytes':total,'sha256':h.hexdigest()},flush=True)
    if total<1_000_000: raise RuntimeError(f'archive unexpectedly small: {total}')

    states=extract_csv_gz(fn,chunksize=25_000)
    print('REAL_TBT_STATES',states.to_dict(orient='records'),flush=True)
    if len(states)<5 or len(states)>7: raise RuntimeError(f'unexpected 4h state count: {len(states)}')
    if int(states.book_updates.sum())<1000: raise RuntimeError('too few genuine book updates parsed')
    if not states.state_time_ms.is_monotonic_increasing: raise RuntimeError('state time is not monotonic')
    if states.state_time_ms.duplicated().any(): raise RuntimeError('duplicate completed state timestamps')
    if int(states.state_time_ms.min()) <= int(BEGIN): raise RuntimeError('bucket-end chronology invariant failed')
    if int(states.state_time_ms.max()) > int(END)+4*60*60*1000: raise RuntimeError('unexpected state timestamp beyond archive day')

    required=['spread_bps_mean','microprice_dev_bps_mean','qty_imb_50_mean','ord_imb_50_mean',
              'notional_imb_50_mean','depth_notional_50_mean','top_ofi_norm_mean','capture_latency_ms_mean']
    missing=[c for c in required if c not in states.columns]
    if missing: raise RuntimeError(f'missing extracted features: {missing}')
    for c in ['qty_imb_50_mean','ord_imb_50_mean','notional_imb_50_mean']:
        v=pd.to_numeric(states[c],errors='coerce').dropna()
        if len(v)==0 or not ((v>=-1.000001)&(v<=1.000001)).all(): raise RuntimeError(f'invalid imbalance {c}')
    spread=pd.to_numeric(states['spread_bps_mean'],errors='coerce')
    if spread.isna().all() or (spread<0).any(): raise RuntimeError('invalid spread state')
    latency=pd.to_numeric(states['capture_latency_ms_mean'],errors='coerce')
    if latency.isna().all(): raise RuntimeError('missing capture latency')

    evidence={
        'instrument':INST,'archive_filename':fn.name,'archive_bytes':total,'archive_sha256':h.hexdigest(),
        'completed_4h_states':int(len(states)),'parsed_book_updates':int(states.book_updates.sum()),
        'first_state_time_ms':int(states.state_time_ms.min()),'last_state_time_ms':int(states.state_time_ms.max()),
        'feature_columns':list(states.columns),'result':'PASS'
    }
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
    states.to_csv(OUT/'states.csv',index=False)
    print('REAL_OKX_MICROSTRUCTURE_SMOKE PASS',json.dumps(evidence,sort_keys=True),flush=True)

if __name__=='__main__': main()
