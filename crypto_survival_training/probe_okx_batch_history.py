from __future__ import annotations
import json, requests
from pathlib import Path

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('batch_history_probe'); OUT.mkdir(exist_ok=True)
BEGIN='1735689600000'  # 2025-01-01 UTC
END='1736035200000'    # 2025-01-05 UTC
MODULES=['1','2','3','4','5','6','11','fundingRate','funding','openInterest','volume']
AGGR=['1D','1M']

def main():
    results=[]
    for module in MODULES:
        for aggr in AGGR:
            p={'module':module,'instType':'SWAP','dateAggrType':aggr,'begin':BEGIN,'end':END,'instIdList':'BTC-USDT-SWAP'}
            try:
                r=requests.get(URL,params=p,timeout=25)
                try: body=r.json()
                except Exception: body={'raw':r.text[:1000]}
                data=body.get('data') if isinstance(body,dict) else None
                sample=data[:1] if isinstance(data,list) else data
                rec={'module':module,'aggr':aggr,'http':r.status_code,'code':body.get('code') if isinstance(body,dict) else None,'msg':body.get('msg') if isinstance(body,dict) else None,'sample':sample}
            except Exception as e:
                rec={'module':module,'aggr':aggr,'error':repr(e)}
            results.append(rec); print(json.dumps(rec,default=str),flush=True)
    (OUT/'probe.json').write_text(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
