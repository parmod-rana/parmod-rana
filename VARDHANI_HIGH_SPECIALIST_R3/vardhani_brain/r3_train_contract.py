"""Development-only normalization, action-label and regime contract for R3.

This module MUST only be called with 2022-2024 teacher-input and target artifacts.
It never opens 2025 validation files. The resulting contract is frozen before any
2025 model evaluation and is not an economic promotion threshold.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

HORIZONS=(1,3,5,10,15)
DIRECTION_QUANTILE=0.34  # inherited V0.2 representation-label rule; fixed pre-validation


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def _robust(a):
    a=np.asarray(a,np.float32)
    med=np.median(a,axis=0).astype(np.float32)
    q1=np.percentile(a,25,axis=0).astype(np.float32);q3=np.percentile(a,75,axis=0).astype(np.float32)
    scale=np.maximum(q3-q1,1e-6).astype(np.float32)
    return med,scale


def _year_guard(inp):
    y=np.asarray([int(str(x)[:4]) for x in inp['DAY'].astype(str)],dtype=np.int16)
    if not np.all((y>=2022)&(y<=2024)):raise RuntimeError('TRAIN_CONTRACT_NONDEVELOPMENT_ROW')


def fit_contract(nifty_input:Path,sensex_input:Path,nifty_targets:Path,sensex_targets:Path,out_npz:Path,out_report:Path):
    ni=np.load(nifty_input,allow_pickle=False);si=np.load(sensex_input,allow_pickle=False)
    nt=np.load(nifty_targets,allow_pickle=False);st=np.load(sensex_targets,allow_pickle=False)
    _year_guard(ni);_year_guard(si)
    if not np.array_equal(ni['TS_UTC_NS'],nt['TS_UTC_NS']):raise RuntimeError('NIFTY_INPUT_TARGET_TIMESTAMP_MISMATCH')
    if not np.array_equal(si['TS_UTC_NS'],st['TS_UTC_NS']):raise RuntimeError('SENSEX_INPUT_TARGET_TIMESTAMP_MISMATCH')
    nn=[str(x) for x in ni['LOWER_NAMES'].tolist()];sn=[str(x) for x in si['LOWER_NAMES'].tolist()]
    nc=[str(x) for x in ni['CTX15_NAMES'].tolist()];sc=[str(x) for x in si['CTX15_NAMES'].tolist()]
    if nn!=sn or nc!=sc:raise RuntimeError('TARGET_INDEX_FEATURE_CONTRACT_MISMATCH')
    if any('tf15_' in x for x in nn):raise RuntimeError('15M_IN_LOWER_TRAIN_CONTRACT')

    lower=np.concatenate([ni['LOWER'].astype(np.float32),si['LOWER'].astype(np.float32)],axis=0)
    ctx=np.concatenate([ni['CTX15'].astype(np.float32),si['CTX15'].astype(np.float32)],axis=0)
    lm,ls=_robust(lower);cm,cs=_robust(ctx)

    h5=HORIZONS.index(5);rets=[]
    for t in (nt,st):
        v=t['VALID'][:,h5].astype(bool);r=t['RET_BPS'][:,h5].astype(np.float32);rets.append(r[v])
    r5=np.concatenate(rets);direction_thr=float(np.quantile(np.abs(r5),DIRECTION_QUANTILE))

    rv_col=nn.index('target_rv30')
    rv=np.concatenate([ni['LOWER'][:,rv_col],si['LOWER'][:,rv_col]]).astype(np.float32)
    vol_q=np.quantile(rv,[.25,.5,.75]).astype(np.float32)

    np.savez(out_npz,LOWER_MEDIAN=lm,LOWER_SCALE=ls,CTX15_MEDIAN=cm,CTX15_SCALE=cs,
             DIRECTION_ABS_BPS_THRESHOLD=np.asarray(direction_thr,np.float32),REGIME_RV30_QUARTILES=vol_q,
             LOWER_NAMES=np.asarray(nn,dtype='U96'),CTX15_NAMES=np.asarray(nc,dtype='U96'),
             DIRECTION_QUANTILE=np.asarray(DIRECTION_QUANTILE,np.float32))
    rep={
        'format':'VARDHANI_R3_TRAIN_ONLY_REPRESENTATION_CONTRACT_V1','status':'PASS','fit_years':[2022,2023,2024],
        'validation_2025_read_during_fit':False,'sealed_2026_read_during_fit':False,
        'direction_quantile_fixed':DIRECTION_QUANTILE,'direction_abs_bps_threshold':direction_thr,
        'regime_rv30_quartiles':vol_q.tolist(),'lower_dim':128,'context15_dim':32,'15m_in_lower':False,
        'classification':'TRAIN_ONLY_REPRESENTATION_AND_AUXILIARY_LABEL_CONTRACT_NOT_ECONOMIC_PROMOTION_RULE',
        'sources':{
            'nifty_train_input_sha256':sha256_file(nifty_input),'sensex_train_input_sha256':sha256_file(sensex_input),
            'nifty_train_targets_sha256':sha256_file(nifty_targets),'sensex_train_targets_sha256':sha256_file(sensex_targets)
        },
        'contract_npz':out_npz.name,'contract_sha256':sha256_file(out_npz),'optimizer_steps':0
    }
    json.dump(rep,open(out_report,'w'),indent=2);return rep


def apply_contract(lower,ctx,contract):
    lo=(np.asarray(lower,np.float32)-contract['LOWER_MEDIAN'])/contract['LOWER_SCALE']
    cx=(np.asarray(ctx,np.float32)-contract['CTX15_MEDIAN'])/contract['CTX15_SCALE']
    return np.clip(lo,-12,12).astype(np.float32),np.clip(cx,-12,12).astype(np.float32)


def regime_labels(inp,contract):
    names=[str(x) for x in inp['LOWER_NAMES'].tolist()];x=inp['LOWER'].astype(np.float32)
    rv=x[:,names.index('target_rv30')];trend=x[:,names.index('target_ema_sep_atr')]
    tf5=x[:,names.index('target_tf5_closed_bar_ret')]
    vbin=np.digitize(rv,contract['REGIME_RV30_QUARTILES']).astype(np.int64)
    tbit=(trend>0).astype(np.int64);agree=(np.sign(trend)==np.sign(tf5)).astype(np.int64)
    return vbin*4+tbit*2+agree
