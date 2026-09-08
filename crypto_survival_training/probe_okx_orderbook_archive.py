from __future__ import annotations
import io, json, time, zipfile
from pathlib import Path
import requests

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('orderbook_archive_probe'); OUT.mkdir(exist_ok=True)
MODULES=['4','5','6']
DAY_BEGIN='1735689600000'   # 2025-01-01 UTC
DAY_END='1735776000000'     # 2025-01-02 UTC
MONTH_BEGIN='1735689600000' # 2025-01-01 UTC
MONTH_END='1738281600000'   # 2025-01-31 UTC

# The first probe showed modules 4/5/6 return no files under a broad all-SWAP request.
# This V2 probe does not assume the instrument selector name or aggregation mode.
# It tests exact instrument selectors for both SWAP and SPOT plus daily/monthly aggregation.
VARIANTS=[
    ('swap_list_daily', {'instType':'SWAP','instIdList':'BTC-USDT-SWAP','dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}),
    ('swap_inst_daily', {'instType':'SWAP','instId':'BTC-USDT-SWAP','dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}),
    ('swap_onlyinst_daily', {'instId':'BTC-USDT-SWAP','dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}),
    ('spot_list_daily', {'instType':'SPOT','instIdList':'BTC-USDT','dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}),
    ('spot_inst_daily', {'instType':'SPOT','instId':'BTC-USDT','dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}),
    ('swap_list_monthly', {'instType':'SWAP','instIdList':'BTC-USDT-SWAP','dateAggrType':'monthly','begin':MONTH_BEGIN,'end':MONTH_END}),
    ('swap_inst_monthly', {'instType':'SWAP','instId':'BTC-USDT-SWAP','dateAggrType':'monthly','begin':MONTH_BEGIN,'end':MONTH_END}),
]


def extract_files(body):
    files=[]
    data=body.get('data',[]) if isinstance(body,dict) else []
    for d in data if isinstance(data,list) else []:
        for detail in d.get('details',[]) or []:
            for g in detail.get('groupDetails',[]) or []:
                files.append({
                    'filename':g.get('filename'),'url':g.get('url'),'sizeMB':g.get('sizeMB'),
                    'instType':detail.get('instType'),'instId':detail.get('instId'),
                    'instFamily':detail.get('instFamily'),'ccy':detail.get('ccy')})
    return files


def main():
    results=[]
    for module in MODULES:
        for name,params in VARIANTS:
            p={'module':module,**params}
            r=requests.get(URL,params=p,timeout=30)
            try: body=r.json()
            except Exception: body={'raw':r.text[:1200]}
            rec={'module':module,'variant':name,'params':p,'http':r.status_code,
                 'code':body.get('code') if isinstance(body,dict) else None,
                 'msg':body.get('msg') if isinstance(body,dict) else None,
                 'files':extract_files(body)}
            print(json.dumps(rec,default=str),flush=True)
            results.append(rec)
            time.sleep(.7)

    # Inspect only a small returned archive, if any. Large L2 files are deliberately not downloaded here.
    for rec in results:
        if not rec['files']:
            continue
        f=rec['files'][0]
        try: size=float(f.get('sizeMB') or 0)
        except Exception: size=0
        if size<=0 or size>12:
            continue
        try:
            rr=requests.get(f['url'],timeout=45); rr.raise_for_status()
            z=zipfile.ZipFile(io.BytesIO(rr.content)); names=z.namelist(); samples=[]
            for member in names[:2]:
                raw=z.read(member); text=raw[:6000].decode('utf-8','replace')
                samples.append({'name':member,'bytes':len(raw),'first_lines':text.splitlines()[:6]})
            rec['archive_sample']=samples
        except Exception as e:
            rec['archive_error']=repr(e)

    (OUT/'probe_v2.json').write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps(results,indent=2,default=str),flush=True)

if __name__=='__main__': main()
