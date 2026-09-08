from __future__ import annotations

"""Causal feature extraction for verified OKX 50-level TBT order-book snapshots.

This module does NOT select trades. It converts raw book states into a compact,
portable market-state representation for future learned-policy experiments.
All aggregates are keyed to the END of their observation bucket so a caller can
merge them only after that bucket has fully completed.
"""

from pathlib import Path
from typing import Iterable
import math
import numpy as np
import pandas as pd

LEVELS = (1, 5, 10, 20, 50)
BAR_MS = 4 * 60 * 60 * 1000
LAST_COLS = [
    'spread_bps','microprice_dev_bps','qty_imb_1','qty_imb_5','qty_imb_10','qty_imb_20','qty_imb_50',
    'ord_imb_5','ord_imb_20','ord_imb_50','notional_imb_10','notional_imb_50','depth_notional_10',
    'depth_notional_50','top1_concentration','top_ofi_norm','capture_latency_ms'
]
MEAN_COLS = [
    'spread_bps','microprice_dev_bps','qty_imb_5','qty_imb_20','qty_imb_50','ord_imb_20','ord_imb_50',
    'notional_imb_50','depth_notional_50','top1_concentration','top_ofi_norm','capture_latency_ms','update_interval_ms'
]
STD_COLS = ['spread_bps','microprice_dev_bps','qty_imb_20','qty_imb_50','top_ofi_norm','depth_notional_50']


def _series_sum(parts: Iterable[pd.Series]) -> pd.Series:
    parts = list(parts)
    out = parts[0].astype(float).copy()
    for p in parts[1:]:
        out = out + p.astype(float)
    return out


def snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized causal features from full 50-level snapshots."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    bp = pd.to_numeric(df['bid_1_px'], errors='coerce')
    ap = pd.to_numeric(df['ask_1_px'], errors='coerce')
    bq = pd.to_numeric(df['bid_1_qty'], errors='coerce')
    aq = pd.to_numeric(df['ask_1_qty'], errors='coerce')
    mid = (bp + ap) / 2.0
    out['mid'] = mid
    out['spread_bps'] = (ap - bp) / mid.replace(0, np.nan) * 1e4
    micro = (ap * bq + bp * aq) / (bq + aq).replace(0, np.nan)
    out['microprice_dev_bps'] = (micro - mid) / mid.replace(0, np.nan) * 1e4

    for n in LEVELS:
        bid_qty = _series_sum(pd.to_numeric(df[f'bid_{i}_qty'], errors='coerce') for i in range(1, n + 1))
        ask_qty = _series_sum(pd.to_numeric(df[f'ask_{i}_qty'], errors='coerce') for i in range(1, n + 1))
        bid_orders = _series_sum(pd.to_numeric(df[f'bid_{i}_ordCnt'], errors='coerce') for i in range(1, n + 1))
        ask_orders = _series_sum(pd.to_numeric(df[f'ask_{i}_ordCnt'], errors='coerce') for i in range(1, n + 1))
        bid_notional = _series_sum(
            pd.to_numeric(df[f'bid_{i}_px'], errors='coerce') * pd.to_numeric(df[f'bid_{i}_qty'], errors='coerce')
            for i in range(1, n + 1)
        )
        ask_notional = _series_sum(
            pd.to_numeric(df[f'ask_{i}_px'], errors='coerce') * pd.to_numeric(df[f'ask_{i}_qty'], errors='coerce')
            for i in range(1, n + 1)
        )
        out[f'qty_imb_{n}'] = (bid_qty - ask_qty) / (bid_qty + ask_qty).replace(0, np.nan)
        out[f'ord_imb_{n}'] = (bid_orders - ask_orders) / (bid_orders + ask_orders).replace(0, np.nan)
        out[f'notional_imb_{n}'] = (bid_notional - ask_notional) / (bid_notional + ask_notional).replace(0, np.nan)
        out[f'depth_notional_{n}'] = bid_notional + ask_notional

    depth10 = _series_sum(
        pd.to_numeric(df[f'bid_{i}_qty'], errors='coerce') + pd.to_numeric(df[f'ask_{i}_qty'], errors='coerce')
        for i in range(1, 11)
    )
    out['top1_concentration'] = (bq + aq) / depth10.replace(0, np.nan)

    # Cont-style top-of-book order-flow pressure from consecutive full snapshots.
    prev_bp, prev_bq = bp.shift(1), bq.shift(1)
    prev_ap, prev_aq = ap.shift(1), aq.shift(1)
    bid_flow = np.where(bp > prev_bp, bq, np.where(bp == prev_bp, bq - prev_bq, -prev_bq))
    ask_flow = np.where(ap < prev_ap, -aq, np.where(ap == prev_ap, -(aq - prev_aq), prev_aq))
    norm_depth = ((bq + aq + prev_bq + prev_aq) / 2.0).replace(0, np.nan)
    out['top_ofi_norm'] = (bid_flow + ask_flow) / norm_depth

    out['capture_latency_ms'] = pd.to_numeric(df['timeMs'], errors='coerce') - pd.to_numeric(df['exchTimeMs'], errors='coerce')
    out['update_interval_ms'] = pd.to_numeric(df['exchTimeMs'], errors='coerce').diff()
    out['exchTimeMs'] = pd.to_numeric(df['exchTimeMs'], errors='coerce')
    return out.replace([np.inf, -np.inf], np.nan)


def aggregate_to_completed_bars(features: pd.DataFrame, bar_ms: int = BAR_MS) -> pd.DataFrame:
    """Exact in-memory aggregation, timestamped at each observation bucket END."""
    if features.empty:
        return features.copy()
    d = features.dropna(subset=['exchTimeMs']).copy()
    d['state_time_ms'] = (d['exchTimeMs'].astype('int64') // bar_ms + 1) * bar_ms
    rows=[]
    for state_time, g in d.groupby('state_time_ms', sort=True):
        rec={'state_time_ms':int(state_time),'book_updates':int(len(g))}
        for c in LAST_COLS:
            if c in g:
                valid=g[c].dropna(); rec[c+'_last']=float(valid.iloc[-1]) if len(valid) else np.nan
        for c in MEAN_COLS:
            if c in g: rec[c+'_mean']=float(g[c].mean())
        for c in STD_COLS:
            if c in g: rec[c+'_std']=float(g[c].std(ddof=0))
        if 'spread_bps' in g: rec['spread_bps_max']=float(g['spread_bps'].max())
        if 'top_ofi_norm' in g:
            rec['top_ofi_abs_max']=float(g['top_ofi_norm'].abs().max())
            rec['top_ofi_sum']=float(g['top_ofi_norm'].sum())
        if 'depth_notional_50' in g: rec['depth_notional_50_min']=float(g['depth_notional_50'].min())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('state_time_ms').reset_index(drop=True)


def _accumulate(acc: dict, features: pd.DataFrame, bar_ms: int) -> None:
    """Associative online aggregation so large TBT days can be streamed safely."""
    if features.empty:
        return
    d=features.dropna(subset=['exchTimeMs']).copy()
    d['state_time_ms']=(d['exchTimeMs'].astype('int64')//bar_ms+1)*bar_ms
    for state_time,g in d.groupby('state_time_ms',sort=False):
        k=int(state_time)
        a=acc.setdefault(k,{'book_updates':0,'last':{},'sum':{},'sumsq':{},'n':{},'spread_max':-np.inf,'ofi_abs_max':-np.inf,'ofi_sum':0.0,'depth_min':np.inf})
        a['book_updates']+=int(len(g))
        for c in LAST_COLS:
            if c in g:
                valid=g[c].dropna()
                if len(valid): a['last'][c]=float(valid.iloc[-1])
        for c in set(MEAN_COLS+STD_COLS):
            if c not in g: continue
            vals=pd.to_numeric(g[c],errors='coerce').dropna().to_numpy(float)
            if not len(vals): continue
            a['sum'][c]=a['sum'].get(c,0.0)+float(vals.sum())
            a['sumsq'][c]=a['sumsq'].get(c,0.0)+float(np.square(vals).sum())
            a['n'][c]=a['n'].get(c,0)+int(len(vals))
        if 'spread_bps' in g and g['spread_bps'].notna().any(): a['spread_max']=max(a['spread_max'],float(g['spread_bps'].max()))
        if 'top_ofi_norm' in g and g['top_ofi_norm'].notna().any():
            vals=g['top_ofi_norm'].dropna().to_numpy(float); a['ofi_abs_max']=max(a['ofi_abs_max'],float(np.abs(vals).max())); a['ofi_sum']+=float(vals.sum())
        if 'depth_notional_50' in g and g['depth_notional_50'].notna().any(): a['depth_min']=min(a['depth_min'],float(g['depth_notional_50'].min()))


def _finalize_accumulator(acc: dict) -> pd.DataFrame:
    rows=[]
    for state_time in sorted(acc):
        a=acc[state_time]; rec={'state_time_ms':state_time,'book_updates':a['book_updates']}
        for c,v in a['last'].items(): rec[c+'_last']=v
        for c in MEAN_COLS:
            n=a['n'].get(c,0)
            if n: rec[c+'_mean']=a['sum'][c]/n
        for c in STD_COLS:
            n=a['n'].get(c,0)
            if n:
                mean=a['sum'][c]/n; var=max(0.0,a['sumsq'][c]/n-mean*mean); rec[c+'_std']=math.sqrt(var)
        rec['spread_bps_max']=a['spread_max'] if np.isfinite(a['spread_max']) else np.nan
        rec['top_ofi_abs_max']=a['ofi_abs_max'] if np.isfinite(a['ofi_abs_max']) else np.nan
        rec['top_ofi_sum']=a['ofi_sum']
        rec['depth_notional_50_min']=a['depth_min'] if np.isfinite(a['depth_min']) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def extract_csv_gz(path: str | Path, chunksize: int = 50_000, bar_ms: int = BAR_MS) -> pd.DataFrame:
    """Stream a module-6 .csv.gz while preserving cross-chunk OFI chronology exactly."""
    acc={}; previous_raw=None
    for raw in pd.read_csv(path,compression='gzip',chunksize=chunksize):
        if raw.empty: continue
        if previous_raw is not None:
            joined=pd.concat([previous_raw,raw],ignore_index=True)
            f=snapshot_features(joined).iloc[1:].copy()
        else:
            f=snapshot_features(raw)
        _accumulate(acc,f,bar_ms)
        previous_raw=raw.iloc[[-1]].copy()
    return _finalize_accumulator(acc)
