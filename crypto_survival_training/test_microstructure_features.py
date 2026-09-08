from __future__ import annotations
import gzip, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from crypto_survival_training.microstructure_features import snapshot_features, aggregate_to_completed_bars, extract_csv_gz


def make_book():
    rows=[]
    start=1735689600000
    for j in range(137):
        # Cross a 4h bucket boundary deliberately and vary prices/queues.
        t=start+j*180000
        shift=(j%17)*0.00001
        r={'timeMs':t+2+(j%3),'exchTimeMs':t,'symbol':'SYNTH.OK'}
        for i in range(1,51):
            r[f'bid_{i}_px']=0.25+shift-i*0.0001
            r[f'ask_{i}_px']=0.25+shift+i*0.0001
            r[f'bid_{i}_qty']=10+i+(j%5)
            r[f'ask_{i}_qty']=9+i+((j+2)%7)
            r[f'bid_{i}_ordCnt']=1+(i+j)%4
            r[f'ask_{i}_ordCnt']=1+(i+2*j)%5
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    raw=make_book()
    direct=aggregate_to_completed_bars(snapshot_features(raw)).sort_values('state_time_ms').reset_index(drop=True)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'book.csv.gz'
        raw.to_csv(p,index=False,compression='gzip')
        streamed=extract_csv_gz(p,chunksize=13).sort_values('state_time_ms').reset_index(drop=True)
    assert direct.state_time_ms.tolist()==streamed.state_time_ms.tolist()
    assert direct.book_updates.tolist()==streamed.book_updates.tolist()
    assert int(streamed.book_updates.sum())==len(raw)
    common=[c for c in direct.columns if c in streamed.columns and c not in ('state_time_ms','book_updates')]
    for c in common:
        a=pd.to_numeric(direct[c],errors='coerce').to_numpy(float)
        b=pd.to_numeric(streamed[c],errors='coerce').to_numpy(float)
        if not np.allclose(a,b,rtol=1e-9,atol=1e-9,equal_nan=True):
            raise AssertionError(f'mismatch {c}: {a} vs {b}')
    for c in [x for x in streamed.columns if 'qty_imb_' in x or 'ord_imb_' in x or 'notional_imb_' in x]:
        v=pd.to_numeric(streamed[c],errors='coerce').dropna()
        assert ((v>=-1.0000001)&(v<=1.0000001)).all(), c
    print('MICROSTRUCTURE_STREAMING_EQUIVALENCE PASS',len(raw),'snapshots',len(streamed),'completed buckets',len(common),'verified features')

if __name__=='__main__': main()
