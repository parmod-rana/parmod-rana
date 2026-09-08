from __future__ import annotations

"""Causal feature extraction for verified OKX 50-level TBT order-book snapshots.

This module does NOT select trades. It converts raw book states into a compact,
portable market-state representation for future learned-policy experiments.
All aggregates are keyed to the END of their observation bucket so a caller can
merge them only when that bucket has fully completed.
"""

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

LEVELS = (1, 5, 10, 20, 50)
BAR_MS = 4 * 60 * 60 * 1000


def _series_sum(parts: Iterable[pd.Series]) -> pd.Series:
    parts = list(parts)
    out = parts[0].astype(float).copy()
    for p in parts[1:]:
        out = out + p.astype(float)
    return out


def snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized features from full 50-level snapshots.

    Expected verified module-6 columns include timeMs, exchTimeMs and, for each
    level 1..50, bid/ask price, quantity and order-count fields.
    """
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
    """Aggregate tick states to completed buckets and timestamp each row at bucket END."""
    if features.empty:
        return features.copy()
    d = features.dropna(subset=['exchTimeMs']).copy()
    d['state_time_ms'] = (d['exchTimeMs'].astype('int64') // bar_ms + 1) * bar_ms
    core = [c for c in d.columns if c not in ('exchTimeMs', 'state_time_ms', 'mid')]
    last_cols = ['spread_bps','microprice_dev_bps','qty_imb_1','qty_imb_5','qty_imb_10','qty_imb_20','qty_imb_50',
                 'ord_imb_5','ord_imb_20','ord_imb_50','notional_imb_10','notional_imb_50','depth_notional_10',
                 'depth_notional_50','top1_concentration','top_ofi_norm','capture_latency_ms']
    mean_cols = ['spread_bps','microprice_dev_bps','qty_imb_5','qty_imb_20','qty_imb_50','ord_imb_20','ord_imb_50',
                 'notional_imb_50','depth_notional_50','top1_concentration','top_ofi_norm','capture_latency_ms','update_interval_ms']
    std_cols = ['spread_bps','microprice_dev_bps','qty_imb_20','qty_imb_50','top_ofi_norm','depth_notional_50']

    rows=[]
    for state_time, g in d.groupby('state_time_ms', sort=True):
        rec={'state_time_ms':int(state_time),'book_updates':int(len(g))}
        for c in last_cols:
            if c in g: rec[c+'_last']=float(g[c].iloc[-1]) if pd.notna(g[c].iloc[-1]) else np.nan
        for c in mean_cols:
            if c in g: rec[c+'_mean']=float(g[c].mean())
        for c in std_cols:
            if c in g: rec[c+'_std']=float(g[c].std(ddof=0))
        if 'spread_bps' in g: rec['spread_bps_max']=float(g['spread_bps'].max())
        if 'top_ofi_norm' in g:
            rec['top_ofi_abs_p90']=float(g['top_ofi_norm'].abs().quantile(.90))
            rec['top_ofi_sum']=float(g['top_ofi_norm'].sum())
        if 'depth_notional_50' in g: rec['depth_notional_50_min']=float(g['depth_notional_50'].min())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('state_time_ms').reset_index(drop=True)


def extract_csv_gz(path: str | Path, chunksize: int = 50_000, bar_ms: int = BAR_MS) -> pd.DataFrame:
    """Stream a verified module-6 .csv.gz file without materializing the full raw book in memory."""
    chunks=[]
    carry=None
    for raw in pd.read_csv(path, compression='gzip', chunksize=chunksize):
        if carry is not None:
            raw=pd.concat([carry,raw],ignore_index=True)
        f=snapshot_features(raw)
        agg=aggregate_to_completed_bars(f,bar_ms=bar_ms)
        if len(agg)>1:
            chunks.append(agg.iloc[:-1].copy())
        carry=raw.iloc[[-1]].copy() if len(raw) else None
    if carry is not None:
        f=snapshot_features(carry)
        chunks.append(aggregate_to_completed_bars(f,bar_ms=bar_ms))
    if not chunks:
        return pd.DataFrame()
    # Re-aggregate duplicate bucket ends created at chunk boundaries conservatively.
    d=pd.concat(chunks,ignore_index=True)
    return d.sort_values('state_time_ms').drop_duplicates('state_time_ms',keep='last').reset_index(drop=True)
