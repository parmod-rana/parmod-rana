from __future__ import annotations
import json, requests, time
from pathlib import Path
from datetime import datetime, timezone

BASE='https://www.okx.com'
OUT=Path('oi_history_probe'); OUT.mkdir(exist_ok=True)
URL=BASE+'/api/v5/rubik/stat/contracts/open-interest-history'

def iso(ms): return datetime.fromtimestamp(ms/1000,timezone.utc).isoformat()

def fetch_pages(inst):
    rows={}; end=1788825600000; pages=[]
    for i in range(8):
        q={'instId':inst,'period':'1D','begin':'1672531200000','end':str(end),'limit':'100'}
        r=requests.get(URL,params=q,timeout=30)
        body=r.json(); data=body.get('data',[]) if isinstance(body,dict) else []
        if body.get('code')!='0' or not data:
            pages.append({'page':i+1,'http':r.status_code,'code':body.get('code'),'msg':body.get('msg'),'records':len(data)}); break
        ts=[int(x[0]) for x in data]
        for x in data: rows[int(x[0])]=x
        oldest=min(ts); newest=max(ts)
        pages.append({'page':i+1,'records':len(data),'oldest':oldest,'oldest_iso':iso(oldest),'newest':newest,'newest_iso':iso(newest)})
        next_end=oldest-1
        if next_end>=end: break
        end=next_end; time.sleep(.8)
    keys=sorted(rows)
    return {'instrument':inst,'unique_records':len(keys),'first_ts':keys[0] if keys else None,'first_iso':iso(keys[0]) if keys else None,'last_ts':keys[-1] if keys else None,'last_iso':iso(keys[-1]) if keys else None,'pages':pages,'first_row':rows[keys[0]] if keys else None,'last_row':rows[keys[-1]] if keys else None}

def main():
    report=[fetch_pages(x) for x in ['BTC-USDT-SWAP','ETH-USDT-SWAP','SOL-USDT-SWAP','XRP-USDT-SWAP']]
    print(json.dumps(report,indent=2)); (OUT/'probe_v3_pagination.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
