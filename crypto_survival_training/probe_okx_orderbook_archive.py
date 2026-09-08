from __future__ import annotations
import gzip, io, json, tarfile, time
from pathlib import Path
import requests

URL='https://www.okx.com/api/v5/public/market-data-history'
OUT=Path('orderbook_archive_probe'); OUT.mkdir(exist_ok=True)
DAY_BEGIN='1735689600000'  # 2025-01-01 UTC
DAY_END='1735776000000'    # 2025-01-02 UTC
MODULES=['4','5','6']
# Smaller established spot markets are used to discover a manageable real L2/TBT sample.
SYMBOLS=['1INCH-USDT','ALGO-USDT','BAT-USDT','XLM-USDT','XTZ-USDT','ZRX-USDT','IOST-USDT','ONT-USDT']
MAX_SAMPLE_MB=30.0


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


def sample_archive(f):
    size=float(f.get('sizeMB') or 0)
    if size<=0 or size>MAX_SAMPLE_MB:
        return {'skipped':f'size {size} MB outside sample budget'}
    r=requests.get(f['url'],timeout=90); r.raise_for_status(); raw=r.content
    name=str(f.get('filename') or '')
    result={'download_bytes':len(raw),'filename':name}
    if name.endswith('.tar.gz'):
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
            members=[m for m in tf.getmembers() if m.isfile()]
            result['members']=[m.name for m in members[:4]]
            if members:
                fh=tf.extractfile(members[0]); sample=fh.read(12000).decode('utf-8','replace') if fh else ''
                result['first_lines']=sample.splitlines()[:8]
    elif name.endswith('.csv.gz') or name.endswith('.gz'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw),mode='rb') as gz:
            sample=gz.read(12000).decode('utf-8','replace')
        result['first_lines']=sample.splitlines()[:8]
    else:
        result['magic']=raw[:32].hex()
    return result


def main():
    results=[]
    for module in MODULES:
        for sym in SYMBOLS:
            p={'module':module,'instType':'SPOT','instIdList':sym,'dateAggrType':'daily','begin':DAY_BEGIN,'end':DAY_END}
            r=requests.get(URL,params=p,timeout=30)
            try: body=r.json()
            except Exception: body={'raw':r.text[:1000]}
            rec={'module':module,'symbol':sym,'http':r.status_code,
                 'code':body.get('code') if isinstance(body,dict) else None,
                 'msg':body.get('msg') if isinstance(body,dict) else None,
                 'files':extract_files(body)}
            if rec['files']:
                rec['min_size_mb']=min(float(x.get('sizeMB') or 0) for x in rec['files'])
            print(json.dumps(rec,default=str),flush=True); results.append(rec); time.sleep(.65)

    # One smallest genuine sample per module, under a strict download budget.
    for module in MODULES:
        candidates=[]
        for rec in results:
            if rec['module']!=module: continue
            for f in rec['files']:
                try: size=float(f.get('sizeMB') or 0)
                except Exception: continue
                if 0<size<=MAX_SAMPLE_MB: candidates.append((size,rec,f))
        if not candidates: continue
        size,rec,f=min(candidates,key=lambda x:x[0])
        try:
            rec['archive_sample']=sample_archive(f)
            print('ARCHIVE_SAMPLE',module,rec['symbol'],size,json.dumps(rec['archive_sample'],default=str),flush=True)
        except Exception as e:
            rec['archive_sample_error']=repr(e); print('ARCHIVE_SAMPLE_FAIL',module,rec['symbol'],repr(e),flush=True)

    (OUT/'probe_v3.json').write_text(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
