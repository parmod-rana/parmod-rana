from __future__ import annotations
import hashlib, io, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests, joblib
from crypto_survival_training import train_real as core

MANIFEST_URL = 'https://raw.githubusercontent.com/olaxbt/ai-market-maker/main/data/MANIFEST.json'
RAW_BASE = 'https://raw.githubusercontent.com/olaxbt/ai-market-maker/main/'


def load_real_frames():
    s = requests.Session()
    s.headers['User-Agent'] = 'GLOBAL-CRYPTO-HUNTER-R1D-VERIFIED/1.0'
    r = s.get(MANIFEST_URL, timeout=30)
    r.raise_for_status()
    source_manifest = r.json()
    frames, accepted = {}, []
    combined = hashlib.sha256()
    for item in source_manifest.get('ohlcv_vision', []):
        if item.get('timeframe') != '1d':
            continue
        path = item['path']
        rr = s.get(RAW_BASE + path, timeout=45)
        rr.raise_for_status()
        raw = rr.content
        actual = hashlib.sha256(raw).hexdigest().lower()
        expected = str(item.get('sha256') or '').lower()
        if not expected or actual != expected:
            raise RuntimeError(f'checksum mismatch: {path}')
        df = pd.read_csv(io.BytesIO(raw)).rename(columns={'timestamp_ms': 'ts'})
        for c in ['ts', 'open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume']].dropna().drop_duplicates('ts').sort_values('ts')
        symbol = str(item.get('vision_symbol') or item.get('symbol', '').replace('/', '')).upper()
        frames[symbol] = df
        combined.update(actual.encode())
        accepted.append({'symbol': symbol, 'bars': int(len(df)), 'first_ts': int(df.ts.iloc[0]), 'last_ts': int(df.ts.iloc[-1]), 'sha256': actual})
        print(f'REAL {symbol}: bars={len(df)} sha256=PASS')
    if len(frames) < 10:
        raise RuntimeError(f'only {len(frames)} verified histories found')
    provenance = {
        'provider': 'OlaXBT public dataset built from Binance Vision monthly klines',
        'source_repository': 'olaxbt/ai-market-maker',
        'source_manifest': MANIFEST_URL,
        'source_declared_as': 'binance_vision_monthly_klines',
        'verified_public_archive': True,
        'verified_source_manifest': True,
        'interval': '1d',
        'accepted_symbols': accepted,
        'raw_history_sha256': combined.hexdigest(),
        'selection': 'every OHLCV symbol listed in the checksum-verified public source manifest'
    }
    return frames, provenance


def main():
    frames, provenance = load_real_frames()
    parts = []
    for symbol, df in frames.items():
        f = core.feature_frame(df, core.HORIZON).dropna(subset=['future_ret_bps'])
        f = core.sample_spread(f, 5000)
        f['symbol'] = symbol
        parts.append(f)
    dataset = pd.concat(parts, ignore_index=True)

    train_parts, val_parts, hold_parts = [], [], []
    for _, g in dataset.groupby('symbol', sort=False):
        tr, va, ho = core.split_purged(g.sort_values('ts').reset_index(drop=True), core.HORIZON)
        train_parts.append(tr); val_parts.append(va); hold_parts.append(ho)
    train = pd.concat(train_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    val = pd.concat(val_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    hold = pd.concat(hold_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)

    model = core.HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=.045, max_leaf_nodes=31, l2_regularization=2.0, random_state=17)
    model.fit(core.matrix(train), train.future_ret_bps.to_numpy())
    pred_val = model.predict(core.matrix(val))
    threshold = core.choose_threshold(pred_val, val)
    pred_hold = model.predict(core.matrix(hold))
    hm = core.metrics(pred_hold, hold.future_ret_bps.to_numpy(), threshold, hold.symbol.to_numpy(), hold.regime.to_numpy())

    reasons = []
    checks = [
        (hm['trades'] >= 300, 'insufficient_holdout_trades'),
        (hm['avg_net_bps'] >= 1.0, 'weak_holdout_edge'),
        (hm['profit_factor'] >= 1.08, 'profit_factor_below_gate'),
        (hm['positive_coin_fraction'] >= .45, 'coin_breadth_below_gate'),
        (hm['positive_regime_fraction'] >= .45, 'regime_breadth_below_gate'),
        (hm['max_drawdown_bps'] <= 3500, 'drawdown_above_gate'),
    ]
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    qualified = not reasons

    core.OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, core.OUT / 'model.joblib')
    (core.OUT / 'strategy_edges.json').write_text(json.dumps(core.edge_table(train), indent=2))
    manifest = {
        'version': 'R1D-REAL-MACRO',
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'trained_on_real_history': True,
        'timeframe': '1d',
        'symbol_count': int(dataset.symbol.nunique()),
        'rows': int(len(dataset)),
        'horizon_bars': core.HORIZON,
        'historical_roundtrip_cost_bps': core.COST_BPS,
        'prediction_threshold_bps': float(threshold),
        'model_file': 'model.joblib',
        'edge_table_file': 'strategy_edges.json',
        'provenance': provenance,
        'qualification': {'qualified': qualified, **hm, 'reasons': reasons},
    }
    (core.OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (core.OUT / 'TRAINING_REPORT.txt').write_text(json.dumps(manifest, indent=2))
    print('FINAL_MANIFEST')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
