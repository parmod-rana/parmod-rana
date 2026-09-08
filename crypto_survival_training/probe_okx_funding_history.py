from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from crypto_survival_training import train_okx_dynamic_brain as core

OUT=Path('funding_probe'); OUT.mkdir(exist_ok=True)
START_MS=int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()*1000)

def fetch(inst):
    rows=[]; after=None; calls=0
    for _ in range(16):
        p={'instId':inst,'limit':'400'}
        if after is not None: p['after']=str(after)
        data=core.get('/api/v5/public/funding-rate-history',p); calls+=1
        if not data: break
        for z in data:
            try: rows.append((int(z['fundingTime']),float(z.get('realizedRate') or z.get('fundingRate') or 0.0)))
            except Exception: pass
        oldest=min(int(z['fundingTime']) for z in data)
        if oldest<=START_MS or (after is not None and oldest>=after): break
        after=oldest
    rows=sorted(set(rows))
    return {'instrument':inst,'records':len(rows),'calls':calls,'first_ts':rows[0][0] if rows else None,'last_ts':rows[-1][0] if rows else None,'first_iso':datetime.fromtimestamp(rows[0][0]/1000,timezone.utc).isoformat() if rows else None,'last_iso':datetime.fromtimestamp(rows[-1][0]/1000,timezone.utc).isoformat() if rows else None}

def main():
    syms=core.discover()[:12]
    report=[fetch(x) for x in syms]
    ok=sum(1 for r in report if r['records']>=500 and r['first_ts'] and r['first_ts']<=int(pd.Timestamp('2024-01-01',tz='UTC').timestamp()*1000))
    out={'symbols_probed':len(report),'deep_history_symbols':ok,'report':report}
    print(json.dumps(out,indent=2)); (OUT/'funding_probe.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__': main()
