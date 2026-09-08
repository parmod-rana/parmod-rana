from __future__ import annotations

import numpy as np
import pandas as pd

import r3e_runner as core


def make_market(scale: float, market: str) -> pd.DataFrame:
    rows=[]
    base=pd.Timestamp('2023-01-02 09:15', tz='Asia/Kolkata')
    for d in range(8):
        day=(base + pd.Timedelta(days=d)).date()
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        px=18000.0*scale + d*7.0
        for m in range(75):
            ts=pd.Timestamp(f'{day} 09:15', tz='Asia/Kolkata') + pd.Timedelta(minutes=m)
            drift=0.05*np.sin(m/7.0)+0.02*np.cos(m/5.0)
            o=px
            c=px+drift
            h=max(o,c)+0.15
            l=min(o,c)-0.15
            rows.append((ts,o,h,l,c,market))
            px=c
    return pd.DataFrame(rows, columns=['timestamp','open','high','low','close','market'])


frames={
    'NIFTY': make_market(1.0,'NIFTY'),
    'SENSEX': make_market(3.5,'SENSEX'),
    'VIX': make_market(0.001,'VIX'),
}
state=core.add_targets(core.build_state(frames))
assert len(core.FEATURES)==42
assert all(c in state.columns for c in core.FEATURES)
assert set(state['year'].unique())=={2023}
assert not any(c.startswith('y_') for c in core.FEATURES)

# Runtime check of selector/gate path without changing frozen policy.
rows=[]
for i,ts in enumerate(state['timestamp'].iloc[80:120:4]):
    for side,sign in [('CE',1.0),('PE',-1.0)]:
        rows.append({
            'timestamp':ts,
            'date':ts.date(),
            'month':str(ts.to_period('M')),
            'gross_underlying':sign*(6.0 if i%2==0 else -4.0),
            'exit_timestamp':ts+pd.Timedelta(minutes=15),
            'side':side,
            'horizon':15,
            'raw_score':float(i)+(0.1 if side=='CE' else 0.0),
            'percentile':0.999,
            'T_h':10.0,
        })
cand=pd.DataFrame(rows)
tr=core.simulate(cand,0.99)
sm=core.summarize(tr)
assert 'net3' in sm and 'net5' in sm and 'pass' in sm
print('R3E_RUNTIME_SMOKE_PASS', len(state), len(tr))
