from __future__ import annotations
import json, requests, time
from pathlib import Path

BASE='https://www.okx.com'
OUT=Path('oi_history_probe'); OUT.mkdir(exist_ok=True)
URL=BASE+'/api/v5/rubik/stat/contracts/open-interest-history'
CASES=[
    {'instId':'BTC-USDT-SWAP','period':'1D'},
    {'instId':'ETH-USDT-SWAP','period':'1D'},
    {'instId':'SOL-USDT-SWAP','period':'1D'},
    {'instId':'XRP-USDT-SWAP','period':'1D'},
]

def main():
    out=[]
    for p in CASES:
        q=dict(p); q['begin']='1672531200000'; q['end']='1788825600000'; q['limit']='100'
        r=requests.get(URL,params=q,timeout=30)
        try: body=r.json()
        except Exception: body={'raw':r.text[:1200]}
        data=body.get('data') if isinstance(body,dict) else None
        rec={'params':q,'http':r.status_code,'code':body.get('code') if isinstance(body,dict) else None,'msg':body.get('msg') if isinstance(body,dict) else None,'records':len(data) if isinstance(data,list) else None,'first':data[-1] if isinstance(data,list) and data else None,'last':data[0] if isinstance(data,list) and data else None}
        out.append(rec); print(json.dumps(rec,default=str),flush=True); time.sleep(.8)
    (OUT/'probe_v2.json').write_text(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
