from __future__ import annotations
import json, time, requests
from pathlib import Path

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('batch_history_probe'); OUT.mkdir(exist_ok=True)
CASES=[]
# Changelog documents numeric modules 1..6; module 3 is the primary funding candidate.
# Test aggregation enum and date encoding conservatively with rate-limit spacing.
for module in ['2','3','4']:
    for aggr in ['1','2','day','month','daily','monthly']:
        CASES.append((module,aggr,'1735689600000','1736035200000','ms'))
# Also test date strings in case begin/end aggregation uses calendar encoding.
for aggr in ['1','2','day','month']:
    CASES.append(('3',aggr,'20250101','20250105','ymd'))

def main():
    results=[]
    for module,aggr,begin,end,encoding in CASES:
        p={'module':module,'instType':'SWAP','dateAggrType':aggr,'begin':begin,'end':end,'instIdList':'BTC-USDT-SWAP'}
        try:
            r=requests.get(URL,params=p,timeout=25)
            try: body=r.json()
            except Exception: body={'raw':r.text[:1000]}
            data=body.get('data') if isinstance(body,dict) else None
            sample=data[:1] if isinstance(data,list) else data
            rec={'module':module,'aggr':aggr,'date_encoding':encoding,'http':r.status_code,'code':body.get('code') if isinstance(body,dict) else None,'msg':body.get('msg') if isinstance(body,dict) else None,'sample':sample}
        except Exception as e:
            rec={'module':module,'aggr':aggr,'date_encoding':encoding,'error':repr(e)}
        results.append(rec); print(json.dumps(rec,default=str),flush=True); time.sleep(0.65)
    (OUT/'probe_v2.json').write_text(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
