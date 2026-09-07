from __future__ import annotations
import pandas as pd
from crypto_survival_training import train_okx_dynamic_brain as core

# Transport-only optimization. OKX documents max limit=300 for
# /api/v5/market/history-candles. All model/state/action/split/gate logic remains
# exactly in train_okx_dynamic_brain.py.
def fetch_history_300(inst):
    rows=[]; after=None; calls=0
    for _ in range(42):
        p={'instId':inst,'bar':core.BAR,'limit':'300'}
        if after is not None: p['after']=str(after)
        data=core.get('/api/v5/market/history-candles',p); calls+=1
        if not data: break
        for z in data:
            if len(z)>=6 and (len(z)<9 or str(z[8])=='1'):
                rows.append([int(z[0]),float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5])])
        oldest=min(int(z[0]) for z in data)
        if oldest<=core.START_MS or (after is not None and oldest>=after): break
        after=oldest
    if not rows: return inst,pd.DataFrame(),calls
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts')
    df=df[df.ts>=core.START_MS].reset_index(drop=True)
    return inst,df,calls

core.fetch_history=fetch_history_300
if __name__=='__main__':
    core.main()
