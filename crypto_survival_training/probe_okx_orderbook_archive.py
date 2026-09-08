from __future__ import annotations
import io, json, time, zipfile
from pathlib import Path
import requests

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('orderbook_archive_probe'); OUT.mkdir(exist_ok=True)
# Probe the six documented original modules on a date where archives are known to exist.
# We do not assume module semantics; filenames returned by OKX identify them.
MODULES=['1','2','3','4','5','6']
BEGIN='1735689600000'  # 2025-01-01 UTC
END='1735776000000'    # 2025-01-02 UTC


def main():
    results=[]
    for module in MODULES:
        p={'module':module,'instType':'SWAP','dateAggrType':'daily','begin':BEGIN,'end':END,'instIdList':'BTC-USDT-SWAP'}
        r=requests.get(URL,params=p,timeout=30)
        try: body=r.json()
        except Exception: body={'raw':r.text[:1200]}
        rec={'module':module,'http':r.status_code,'code':body.get('code') if isinstance(body,dict) else None,'msg':body.get('msg') if isinstance(body,dict) else None,'files':[]}
        data=body.get('data',[]) if isinstance(body,dict) else []
        for d in data if isinstance(data,list) else []:
            for detail in d.get('details',[]) or []:
                for g in detail.get('groupDetails',[]) or []:
                    rec['files'].append({'filename':g.get('filename'),'url':g.get('url'),'sizeMB':g.get('sizeMB'),'instType':detail.get('instType'),'instId':detail.get('instId')})
        print(json.dumps(rec,default=str),flush=True)
        results.append(rec)
        time.sleep(.7)

    # Inspect one returned archive for each module that has a small enough file (<15 MB).
    for rec in results:
        if not rec['files']:
            continue
        f=rec['files'][0]
        try:
            size=float(f.get('sizeMB') or 0)
        except Exception:
            size=0
        if size<=0 or size>15:
            continue
        try:
            rr=requests.get(f['url'],timeout=45); rr.raise_for_status()
            z=zipfile.ZipFile(io.BytesIO(rr.content)); names=z.namelist()
            samples=[]
            for name in names[:2]:
                raw=z.read(name)
                text=raw[:6000].decode('utf-8','replace')
                samples.append({'name':name,'bytes':len(raw),'first_lines':text.splitlines()[:6]})
            rec['archive_sample']=samples
        except Exception as e:
            rec['archive_error']=repr(e)
    (OUT/'probe.json').write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps(results,indent=2,default=str),flush=True)

if __name__=='__main__': main()
