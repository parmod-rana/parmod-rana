from __future__ import annotations
import json, math, os, hashlib, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from crypto_survival_training import train_okx_dynamic_brain as core
from crypto_survival_training import train_okx_dynamic_brain_gen3b as g
from crypto_survival_training import train_okx_dynamic_brain_gen3c as c

# GENERATION 3D — LEARNED TEMPORAL MEMORY GATE
# The policy remains fully learned. Three temporal neural memories propose action
# values. A separate meta learner, trained only on causal out-of-fold proposals,
# learns which memory is likely to be reliable in the current market state.
# No named strategy or regime rule selects direction, size, horizon, or memory.
# 2026-08-01+ remains excluded from ALL training/calibration because that period
# was already inspected by Generation 3.

CUTOFF_MS = g.CUTOFF_MS
PURGE = g.PURGE
BAR_MS = g.BAR_MS
MAX_SYMBOLS = int(os.getenv('MAX_SYMBOLS', '72'))
COST_BPS = g.COST_BPS
OUT = Path(os.getenv('OUT_DIR', 'trained_artifact_gen3d'))
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = g.FEATURES
ACTIONS = g.ACTIONS
WINDOWS = c.WINDOWS
RISK_PENALTY = 0.28
BASE_ROWS = int(os.getenv('BASE_TRAIN_ROWS', '90000'))

META_TRAIN_WINDOWS = [
    ('M1', pd.Timestamp('2024-07-01', tz='UTC'), pd.Timestamp('2024-09-01', tz='UTC')),
    ('M2', pd.Timestamp('2024-11-01', tz='UTC'), pd.Timestamp('2025-01-01', tz='UTC')),
]
META_CAL_WINDOW = ('MC', pd.Timestamp('2025-03-01', tz='UTC'), pd.Timestamp('2025-05-01', tz='UTC'))
TEST_FOLDS = g.FOLDS

TEMPERATURES = (20.0, 45.0, 80.0, 140.0)
DISAGREE_PENALTIES = (0.0, 0.25, 0.5, 0.8)
META_QUALITY_FLOORS = (-40.0, -10.0, 10.0, 30.0)
Q_FLOORS = (0.0, 25.0, 50.0, 75.0)
CAPACITIES = (2.0, 3.0, 5.0)


def window_sample(df, cutoff_ms, window_days):
    d = g.clean_training_rows(df)
    if window_days:
        d = d[d.ts >= cutoff_ms - window_days * 86_400_000]
    d = d.sort_values(['ts', 'symbol'])
    if len(d) > BASE_ROWS:
        d = d.iloc[np.linspace(0, len(d) - 1, BASE_ROWS, dtype=int)]
    return d


def fit_memory(train, cutoff_ms, window_days, seed):
    use = window_sample(train, cutoff_ms, window_days)
    if len(use) < 22000:
        raise RuntimeError(f'not enough rows for memory window {window_days}: {len(use)}')
    X = use[FEATURES].to_numpy(float)
    Y = g.reward_matrix(use, RISK_PENALTY)
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    xs = StandardScaler().fit(X)
    ys = StandardScaler().fit(Y)
    Xs = xs.transform(X)
    Ys = ys.transform(Y)
    print('FIT_MEMORY', 'full' if window_days == 0 else window_days, 'rows', len(use), 'seed', seed, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = MLPRegressor(hidden_layer_sizes=(112, 72, 36), activation='relu', solver='adam', alpha=.0025, batch_size=1024, learning_rate_init=.00065, max_iter=95, early_stopping=True, validation_fraction=.08, n_iter_no_change=9, random_state=seed)
        m.fit(Xs, Ys)
    return {'x_scaler': xs, 'y_scaler': ys, 'model': m, 'window_days': window_days, 'rows': len(use)}


def predict_memory(bundle, df):
    X = bundle['x_scaler'].transform(df[FEATURES].to_numpy(float))
    return bundle['y_scaler'].inverse_transform(bundle['model'].predict(X))


def fit_memories(train, cutoff_ms, seed):
    return [fit_memory(train, cutoff_ms, w, seed + i * 11) for i, w in enumerate(WINDOWS)]


def predict_memories(bundles, df):
    return [predict_memory(b, df) for b in bundles]


def action_arrays(pred):
    ai = np.argmax(pred, axis=1)
    top = pred[np.arange(len(pred)), ai]
    part = np.partition(pred, -2, axis=1)
    second = part[:, -2]
    acts = [ACTIONS[int(i)] for i in ai]
    direction = np.array([a['dir'] for a in acts], dtype=float)
    size = np.array([a['size'] for a in acts], dtype=float)
    horizon = np.array([a['h'] for a in acts], dtype=float)
    return ai, top, top - second, direction, size, horizon


def proposal_frame(df, preds):
    base = df[FEATURES].reset_index(drop=True).copy()
    action_ids = []
    top_qs = []
    dirs_all = []
    for j, p in enumerate(preds):
        ai, top, margin, direction, size, horizon = action_arrays(p)
        action_ids.append(ai)
        top_qs.append(top)
        dirs_all.append(direction)
        base[f'm{j}_top_q'] = top
        base[f'm{j}_margin'] = margin
        base[f'm{j}_dir'] = direction
        base[f'm{j}_size'] = size
        base[f'm{j}_h'] = horizon
    tq = np.column_stack(top_qs)
    dirs = np.column_stack(dirs_all)
    base['memory_q_std'] = tq.std(axis=1)
    base['memory_q_range'] = tq.max(axis=1) - tq.min(axis=1)
    base['direction_consensus'] = np.abs(dirs.mean(axis=1))
    base['topq_mean'] = tq.mean(axis=1)
    base['topq_max'] = tq.max(axis=1)
    return base, action_ids


def realized_targets(df, action_ids):
    ys = []
    for ai in action_ids:
        vals = np.empty(len(df), dtype=float)
        for k, idx in enumerate(ai):
            vals[k] = g.actual_net(df.iloc[k], int(idx))
        ys.append(np.clip(vals, -1800.0, 1800.0))
    return np.column_stack(ys)


def make_meta_records(df, preds):
    X, action_ids = proposal_frame(df, preds)
    Y = realized_targets(df.reset_index(drop=True), action_ids)
    return X, Y


def fit_gate(X, Y):
    models = []
    for j in range(Y.shape[1]):
        m = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.055, max_iter=170, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=2.0, random_state=101 + j)
        m.fit(X.to_numpy(float), Y[:, j])
        models.append(m)
    return {'models': models, 'meta_features': list(X.columns)}


def gate_quality(gate, X):
    arr = X[gate['meta_features']].to_numpy(float)
    return np.column_stack([m.predict(arr) for m in gate['models']])


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def combine_gated(preds, quality, temperature, disagree_penalty, meta_floor):
    w = softmax(quality / float(temperature))
    stack = np.stack(preds, axis=1)
    q = np.sum(stack * w[:, :, None], axis=1)
    if disagree_penalty > 0:
        q = q - disagree_penalty * np.std(stack, axis=1)
    weak = np.max(quality, axis=1) <= meta_floor
    if weak.any():
        q[weak, :] = -1e9
    return q, w


def sim(df, q, capacity, q_floor):
    zeros = np.zeros_like(q)
    return g.simulate(df, q, zeros, 0.0, capacity, q_floor)


def calibration_score(m):
    if m['trades'] < 70:
        return -1e30
    if m['avg_net_bps'] <= 0 or m['profit_factor'] <= 1.0:
        return -1e30
    return float(m['avg_net_bps'] * math.sqrt(m['trades']) * min(m['profit_factor'], 2.0) * max(.2, m['positive_week_fraction']) * max(.2, m['positive_coin_fraction']) / (1.0 + m['max_drawdown_bps'] / 900.0))


def train_predict_window(research, start, end, seed):
    vs = int(start.timestamp() * 1000)
    ve = int(end.timestamp() * 1000)
    tr = research[research.ts < vs - PURGE].copy()
    va = research[(research.ts >= vs) & (research.ts < ve - PURGE)].copy().reset_index(drop=True)
    bundles = fit_memories(tr, vs, seed)
    preds = predict_memories(bundles, va)
    return va, preds


def main():
    symbols = core.discover()[:MAX_SYMBOLS]
    print('DISCOVERED', len(symbols), symbols[:20], flush=True)
    fetched = {}; calls = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(g.fetch_history, s): s for s in symbols}
        for f in as_completed(fut):
            s = fut[f]
            try:
                iid, d, cc = f.result(); fetched[iid] = d; calls += cc
                print('FETCHED', iid, len(d), flush=True)
            except Exception as e:
                print('FAIL', s, repr(e), flush=True)

    parts = []; accepted = []; raw_hash = hashlib.sha256()
    for inst in symbols:
        d = fetched.get(inst, pd.DataFrame())
        if len(d) < 1600 or (d.ts < CUTOFF_MS).sum() < 1500:
            continue
        raw_hash.update(pd.util.hash_pandas_object(d, index=False).values.tobytes())
        sym = inst.replace('-USDT-SWAP', '')
        parts.append(g.individual_frame(d, sym))
        accepted.append({'instrument': inst, 'bars': int(len(d)), 'first_ts': int(d.ts.iloc[0]), 'last_ts': int(d.ts.iloc[-1])})
    if len(parts) < 24:
        raise RuntimeError(f'only {len(parts)} usable symbols')

    ds = g.add_market_state(pd.concat(parts, ignore_index=True))
    ds = g.clean_training_rows(ds).sort_values(['ts', 'symbol']).reset_index(drop=True)
    research = ds[ds.ts < CUTOFF_MS - PURGE].copy()
    print('RESEARCH_ROWS', len(research), 'SYMBOLS', research.symbol.nunique(), flush=True)

    meta_X = []; meta_Y = []; seed_records = []
    for idx, (name, start, end) in enumerate(META_TRAIN_WINDOWS):
        va, preds = train_predict_window(research, start, end, 17 + idx * 31)
        X, Y = make_meta_records(va, preds)
        meta_X.append(X); meta_Y.append(Y)
        seed_records.append({'name': name, 'rows': int(len(va))})
        print('META_TRAIN_WINDOW', name, len(va), flush=True)
    MX = pd.concat(meta_X, ignore_index=True)
    MY = np.vstack(meta_Y)
    gate0 = fit_gate(MX, MY)

    mc_name, mc_start, mc_end = META_CAL_WINDOW
    mc, mc_preds = train_predict_window(research, mc_start, mc_end, 83)
    mcX, mcY = make_meta_records(mc, mc_preds)
    qual = gate_quality(gate0, mcX)
    cal_candidates = []; best = None
    for temp in TEMPERATURES:
        for dp in DISAGREE_PENALTIES:
            for mf in META_QUALITY_FLOORS:
                q, _ = combine_gated(mc_preds, qual, temp, dp, mf)
                for cap in CAPACITIES:
                    for floor in Q_FLOORS:
                        met, _ = sim(mc, q, cap, floor)
                        sc = calibration_score(met)
                        rec = {'temperature': temp, 'disagree_penalty': dp, 'meta_quality_floor': mf, 'capacity': cap, 'q_floor_bps': floor, 'score': sc, 'metrics': met}
                        cal_candidates.append(rec)
                        if best is None or sc > best['score']:
                            best = rec
    if best is None or best['score'] <= -1e20:
        raise RuntimeError('no Gen3D calibration survived independent meta-calibration window')
    print('CALIBRATION_SELECTED', json.dumps(best, sort_keys=True), flush=True)

    gate = fit_gate(pd.concat([MX, mcX], ignore_index=True), np.vstack([MY, mcY]))

    test_metrics = []; test_records = []; oof_X = [MX, mcX]; oof_Y = [MY, mcY]
    for idx, (name, start, end) in enumerate(TEST_FOLDS):
        va, preds = train_predict_window(research, start, end, 131 + idx * 29)
        X, Y = make_meta_records(va, preds)
        quality = gate_quality(gate, X)
        q, weights = combine_gated(preds, quality, best['temperature'], best['disagree_penalty'], best['meta_quality_floor'])
        met, _ = sim(va, q, best['capacity'], best['q_floor_bps'])
        test_metrics.append(met)
        test_records.append({'name': name, 'metrics': met, 'mean_memory_weights': weights.mean(axis=0).tolist()})
        oof_X.append(X); oof_Y.append(Y)
        print('TEST_FOLD', name, json.dumps(test_records[-1], sort_keys=True), flush=True)

    robust = g.robust_score(test_metrics)
    robust_pass = robust > -1e20
    print('ROBUST_SCORE', robust, 'PASS', robust_pass, flush=True)

    final_bundles = fit_memories(research, CUTOFF_MS, 251)
    final_gate = fit_gate(pd.concat(oof_X, ignore_index=True), np.vstack(oof_Y))
    artifact = {'sub_policies': final_bundles, 'gate_models': final_gate['models'], 'meta_features': final_gate['meta_features'], 'features': FEATURES, 'actions': ACTIONS, 'risk_penalty': RISK_PENALTY, 'temperature': best['temperature'], 'disagree_penalty': best['disagree_penalty'], 'meta_quality_floor': best['meta_quality_floor'], 'capacity': best['capacity'], 'q_floor_bps': best['q_floor_bps'], 'timeframe': '4h', 'roundtrip_cost_bps': COST_BPS, 'policy_version': 'GEN3D_LEARNED_TEMPORAL_GATE'}
    joblib.dump(artifact, OUT / 'dynamic_policy.joblib')

    manifest = {'version': 'R1F-GEN3D-LEARNED-TEMPORAL-GATE', 'trained_at': datetime.now(timezone.utc).isoformat(), 'trained_on_real_history': True, 'decision_authority': 'LEARNED_META_GATED_ACTION_VALUE_POLICY', 'fixed_strategy_expert_authority': False, 'timeframe': '4h', 'symbol_count': len(accepted), 'research_rows': int(len(research)), 'features': FEATURES, 'actions': ACTIONS, 'temporal_windows_days': list(WINDOWS), 'meta_training_windows': seed_records, 'meta_calibration': {'name': mc_name, 'rows': int(len(mc)), 'selected': best}, 'walk_forward_test_folds': test_records, 'walk_forward_robust_score': robust, 'walk_forward_robust_pass': robust_pass, 'qualification': {'qualified': False, 'reasons': ['no_fresh_untouched_period_after_gen3_retraining'], 'authority': 'SHADOW_PAPER_ONLY'}, 'provenance': {'provider': 'OKX public REST market history', 'interval': core.BAR, 'start': '2023-01-01', 'enhancement_cutoff': '2026-08-01', 'already_inspected_period_excluded': True, 'accepted_symbols': accepted, 'raw_history_sha256': raw_hash.hexdigest(), 'api_calls': calls}, 'notes': ['No named strategy or regime rule chooses a trade.', 'Three neural temporal memories propose action values.', 'A meta learner trained only on causal out-of-fold proposals predicts which memory is likely to be reliable in the current state.', 'Memory weighting is state-dependent and continuous; disagreement can reduce action value.', 'All enhancement training/calibration excludes 2026-08-01+ because that period was already inspected.', 'Fresh forward shadow evidence remains mandatory for qualification.']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'TRAINING_REPORT.txt').write_text(json.dumps(manifest, indent=2))
    (OUT / 'calibration_candidates.json').write_text(json.dumps(sorted(cal_candidates, key=lambda x: x['score'], reverse=True)[:80], indent=2))
    print('FINAL_MANIFEST')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
