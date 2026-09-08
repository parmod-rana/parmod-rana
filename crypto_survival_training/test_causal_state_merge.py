from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_survival_training.causal_state_merge import merge_completed_state


def main():
    decisions=pd.DataFrame({
        'decision_time_ms':[50,100,150,250,450],
        'id':['pre','exact','between','late','stale']
    })
    states=pd.DataFrame({
        'state_time_ms':[100,200,300],
        'signal':[1.0,2.0,3.0],
    })
    out=merge_completed_state(decisions,states,prefix='micro_',max_age_ms=120)
    # 50: no history. 100: exact 100. 150: back to 100. 250: back to 200. 450: 300 is 150 old -> stale.
    expected_time=[np.nan,100,100,200,np.nan]
    expected_avail=[0.0,1.0,1.0,1.0,0.0]
    expected_signal=[np.nan,1.0,1.0,2.0,np.nan]
    assert out.id.tolist()==decisions.id.tolist()
    for got,exp in zip(out.micro_state_time_ms.tolist(),expected_time):
        assert (pd.isna(got) and pd.isna(exp)) or int(got)==int(exp), (got,exp)
    assert out.micro_available.tolist()==expected_avail
    for got,exp in zip(out.micro_signal.tolist(),expected_signal):
        assert (pd.isna(got) and pd.isna(exp)) or float(got)==float(exp), (got,exp)
    matched=out.micro_available==1
    assert (out.loc[matched,'micro_state_time_ms']<=out.loc[matched,'decision_time_ms']).all()
    assert (out.loc[matched,'micro_age_ms']>=0).all()
    print('CAUSAL_STATE_MERGE PASS',out.to_dict(orient='records'))

if __name__=='__main__': main()
