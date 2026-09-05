"""R3 sequence dataset consuming only frozen teacher inputs + exact target artifacts."""
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import Dataset
from .r3_train_contract import HORIZONS,apply_contract,regime_labels

class R3SequenceDataset(Dataset):
    def __init__(self,input_npz,target_npz,contract_npz,seq_len=384,step=1):
        self.i=np.load(input_npz,allow_pickle=False);self.t=np.load(target_npz,allow_pickle=False);self.c=np.load(contract_npz,allow_pickle=False)
        if not np.array_equal(self.i['TS_UTC_NS'],self.t['TS_UTC_NS']):raise RuntimeError('INPUT_TARGET_TIMESTAMP_MISMATCH')
        self.lower,self.ctx=apply_contract(self.i['LOWER'],self.i['CTX15'],self.c)
        self.regime=regime_labels(self.i,self.c);self.seq_len=int(seq_len);self.thr=float(self.c['DIRECTION_ABS_BPS_THRESHOLD'])
        valid=self.t['VALID'].astype(bool).all(axis=1)
        # Sequence windows may span overnight/weekend gaps by design (gap state is in inputs),
        # but never cross an index artifact boundary because each Dataset owns exactly one index.
        self.idx=np.flatnonzero(valid & (np.arange(len(valid))>=self.seq_len-1))[::max(1,int(step))]
    def __len__(self):return len(self.idx)
    def __getitem__(self,j):
        k=int(self.idx[j]);s=k-self.seq_len+1;ret=self.t['RET_BPS'][k].astype(np.float32);mfe=self.t['MFE_BPS'][k].astype(np.float32);mae=self.t['MAE_BPS'][k].astype(np.float32)
        r5=float(ret[HORIZONS.index(5)]);action=0 if r5 < -self.thr else (2 if r5 > self.thr else 1)
        barrier=self.t['UP_BEFORE_DOWN_1ATR'][k].astype(np.float32);bvalid=self.t['BARRIER_VALID'][k].astype(np.float32)
        return {'lower':torch.from_numpy(self.lower[s:k+1]),'ctx15':torch.from_numpy(self.ctx[s:k+1]),'action':torch.tensor(action,dtype=torch.long),'regime':torch.tensor(int(self.regime[k]),dtype=torch.long),'ret':torch.from_numpy(ret),'mfe':torch.from_numpy(mfe),'mae':torch.from_numpy(mae),'barrier':torch.from_numpy(barrier),'barrier_valid':torch.from_numpy(bvalid)}

class R3DualChronologicalDataset(Dataset):
    """Merge NIFTY and SENSEX sequence endpoints in nondecreasing market time.

    Each sample's causal history remains inside its own index artifact. Equal timestamps are
    deterministically ordered NIFTY then SENSEX. The merged stream never rewinds from 2024
    back to 2022 merely because the target index changes.
    """
    def __init__(self,nifty_input,nifty_targets,sensex_input,sensex_targets,contract_npz,seq_len=384,step=1):
        self.ds=[R3SequenceDataset(nifty_input,nifty_targets,contract_npz,seq_len,step),R3SequenceDataset(sensex_input,sensex_targets,contract_npz,seq_len,step)]
        t0=self.ds[0].i['TS_UTC_NS'][self.ds[0].idx].astype(np.int64);t1=self.ds[1].i['TS_UTC_NS'][self.ds[1].idx].astype(np.int64)
        ts=np.concatenate([t0,t1]);src=np.concatenate([np.zeros(len(t0),np.uint8),np.ones(len(t1),np.uint8)]);loc=np.concatenate([np.arange(len(t0),dtype=np.int64),np.arange(len(t1),dtype=np.int64)])
        order=np.lexsort((src,ts));self.ts=ts[order];self.src=src[order];self.loc=loc[order]
        if len(self.ts)>1 and np.any(np.diff(self.ts)<0):raise RuntimeError('DUAL_DATASET_CHRONOLOGY_REWIND')
    def __len__(self):return len(self.ts)
    def __getitem__(self,j):
        s=int(self.src[j]);b=self.ds[s][int(self.loc[j])];b['target_index_id']=torch.tensor(s,dtype=torch.int64);b['endpoint_ts_utc_ns']=torch.tensor(int(self.ts[j]),dtype=torch.int64);return b
