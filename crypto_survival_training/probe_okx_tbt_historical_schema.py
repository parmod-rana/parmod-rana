from __future__ import annotations
import gzip, io, json
from datetime import datetime, timezone, timedelta
import requests

URL='https://www.okx.com/api/v5/public/market-data-history'
CASES=[('BAT-USDT','2024-07-01'),('ZRX-USDT','2024-11-01'),('ATOM-USDT','2025-03-01'),('BAT-USDT','2026-03-15')]

def day_ms(day):
    d=datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc)
    return str(int(d.timestamp()*1000)),str(int((d+timedelta(days=1)).timestamp()*1000))

def files(body):
    out=[]
    for d in body.get('data',[]) or []:
        for detail in d.get('details',[]) or []:
            for g in detail.get('groupDetails',[]) or []:
                if g.get('url'): out.append({'filename':g.get('filename'),'url':g.get('url'),'sizeMB':g.get('sizeMB')})
    return sorted(out,key=lambda x:str(x.get('filename') or ''))

def header_from_url(url):
    # Stream only enough compressed bytes to decode the CSV header and first row.
    r=requests.get(url,stream=True,timeout=60); r.raise_for_status()
    raw=io.BytesIO()
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            raw.write(chunk)
        if raw.tell()>=2*1024*1024: break
    raw.seek(0)
    try:
        with gzip.GzipFile(fileobj=raw,mode='rb') as gz:
            text=gz.read(256000).decode('utf-8','replace')
    except EOFError:
        # Gzip streaming may reach the artificial truncation boundary after
        # already producing useful decoded bytes; retry with a larger prefix.
        r=requests.get(url,stream=True,timeout=60); r.raise_for_status(); raw=io.BytesIO()
        for chunk in r.iter_content(chunk_size=262144):
            if chunk: raw.write(chunk)
            if raw.tell()>=8*1024*1024: break
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw,mode='rb') as gz:
            try: text=gz.read(256000).decode('utf-8','replace')
            except EOFError: text=''
    lines=text.splitlines()
    return {'header':lines[0] if lines else None,'first_row':lines[1] if len(lines)>1 else None}

def main():
    s=requests.Session()
    for sym,day in CASES:
        begin,end=day_ms(day)
        p={'module':'6','instType':'SPOT','instIdList':sym,'dateAggrType':'daily','begin':begin,'end':end}
        r=s.get(URL,params=p,timeout=30); body=r.json(); fs=files(body)
        print('CASE',sym,day,'http',r.status_code,'code',body.get('code'),'files',[(f['filename'],f['sizeMB']) for f in fs],flush=True)
        for f in fs[:1]:
            try: print('SCHEMA',sym,day,json.dumps(header_from_url(f['url'])),flush=True)
            except Exception as e: print('SCHEMA_FAIL',sym,day,repr(e),flush=True)

if __name__=='__main__': main()
