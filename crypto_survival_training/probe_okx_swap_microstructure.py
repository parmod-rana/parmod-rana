from __future__ import annotations
import json, time
from pathlib import Path
import requests

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('swap_microstructure_probe'); OUT.mkdir(exist_ok=True)
BEGIN='1735689600000'; END='1735776000000'
CASES=[
    ('4','BAT-USDT-SWAP'),('6','BAT-USDT-SWAP'),
    ('4','BTC-USDT-SWAP'),('6','BTC-USDT-SWAP')
]


def files(body):
    out=[]
    for d in body.get('data',[]) if isinstance(body,dict) else []:
        for detail in d.get('details',[]) or []:
            for g in detail.get('groupDetails',[]) or []:
                out.append({'filename':g.get('filename'),'sizeMB':g.get('sizeMB'),'url':g.get('url'),
                            'instType':detail.get('instType'),'instId':detail.get('instId')})
    return out


def main():
    results=[]
    for module,inst in CASES:
        p={'module':module,'instType':'SWAP','instIdList':inst,'dateAggrType':'daily','begin':BEGIN,'end':END}
        r=requests.get(URL,params=p,timeout=30)
        try: body=r.json()
        except Exception: body={'raw':r.text[:1000]}
        rec={'module':module,'instrument':inst,'http':r.status_code,
             'code':body.get('code') if isinstance(body,dict) else None,
             'msg':body.get('msg') if isinstance(body,dict) else None,
             'files':files(body)}
        print(json.dumps(rec,default=str),flush=True); results.append(rec); time.sleep(.8)
    (OUT/'probe.json').write_text(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
