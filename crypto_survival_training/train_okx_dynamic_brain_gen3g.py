from __future__ import annotations

"""GENERATION 3G — CAUSAL RELIABILITY MEMORY.

Builds directly on frozen Gen3C. The three temporal action-value memories are
kept as the decision source; no named strategy, post-hoc trade-quality gate, or
hand-written regime chooses actions.

New intelligence: each temporal memory earns or loses influence from the real
net outcome of its OWN recent high-conviction proposals. Outcomes are revealed
to the reliability state only after their action horizon has completed. The
blend remains deliberately anchored to equal weights so the learner cannot
collapse into the Gen3D/Gen3E failure modes.

2026-08-01+ is excluded from all enhancement selection because it has already
been inspected. Hyperparameters and promotion gates below are frozen before
this generation's walk-forward result is observed.
"""

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crypto_survival_training import train_okx_dynamic_brain as core
from crypto_survival_training import train_okx_dynamic_brain_gen3b as g
from crypto_survival_training import train_okx_dynamic_brain_gen3c as c

CUTOFF_MS = g.CUTOFF_MS
PURGE = g.PURGE
BAR_MS = g.BAR_MS
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "72"))
COST_BPS = g.COST_BPS
MAX_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "110000"))
OUT = Path(os.getenv("OUT_DIR", "trained_artifact_gen3g"))
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = g.FEATURES
ACTIONS = g.ACTIONS
WINDOWS = c.WINDOWS
FOLDS = g.FOLDS
RISK_PENALTY = 0.28
CAPACITY = 2.0
Q_FLOOR_BPS = 60.0

# Frozen causal experience-memory design. No search on F1-F4.
EXPERIENCE_DAYS = 180
RELIABILITY_ALPHA = 0.12
RELIABILITY_NET_SCALE_BPS = 150.0
RELIABILITY_TEMPERATURE = 0.35
ADAPTIVE_WEIGHT_STRENGTH = 0.45
PROPOSALS_PER_MEMORY_PER_EPOCH = 2

GEN3C_REFERENCE = {
    "median_fold_avg_net_bps": 24.995931677133225,
    "worst_fold_avg_net_bps": -84.1294575172906,
    "worst_fold_profit_factor": 0.7215288019729019,
    "worst_fold_max_drawdown_bps": 6595.604675176961,
    "minimum_total_trades": 400,
    "minimum_trades_each_fold": 80,
}


def fit_memory(train: pd.DataFrame, cutoff_ms: int, window_days: int, seed: int):
    old = c.MAX_ROWS
    try:
        c.MAX_ROWS = MAX_ROWS
        return c.fit_one(train, cutoff_ms, window_days, seed)
    finally:
        c.MAX_ROWS = old


def fit_memories(train: pd.DataFrame, cutoff_ms: int, seed_base: int):
    return [fit_memory(train, cutoff_ms, w, seed_base + i * 17) for i, w in enumerate(WINDOWS)]


def predict_memories(bundles, df: pd.DataFrame):
    return [c.predict_one(b, df) for b in bundles]


def reliability_weights(utility: np.ndarray) -> np.ndarray:
    z = utility.astype(float) / RELIABILITY_TEMPERATURE
    z = z - np.max(z)
    e = np.exp(np.clip(z, -20.0, 20.0))
    soft = e / max(float(e.sum()), 1e-12)
    uniform = np.full(len(utility), 1.0 / len(utility), dtype=float)
    return (1.0 - ADAPTIVE_WEIGHT_STRENGTH) * uniform + ADAPTIVE_WEIGHT_STRENGTH * soft


def _queue_memory_proposals(df: pd.DataFrame, idx: np.ndarray, q: np.ndarray, pending: list[dict], memory_i: int) -> None:
    if len(idx) == 0:
        return
    ai = np.argmax(q, axis=1)
    best = q[np.arange(len(q)), ai]
    order = np.argsort(best)[::-1]
    picked = 0
    for local in order:
        if best[local] <= Q_FLOOR_BPS:
            break
        row_i = int(idx[local])
        action_i = int(ai[local])
        entry_ts = int(df.iloc[row_i].ts)
        due = entry_ts + int(ACTIONS[action_i]["h"]) * BAR_MS
        pending.append({"due": due, "row_i": row_i, "action_i": action_i, "memory_i": memory_i})
        picked += 1
        if picked >= PROPOSALS_PER_MEMORY_PER_EPOCH:
            break


def _reveal_due(df: pd.DataFrame, pending: list[dict], now_ts: int, utility: np.ndarray) -> list[dict]:
    keep = []
    for p in pending:
        if p["due"] <= now_ts:
            net = g.actual_net(df.iloc[p["row_i"]], p["action_i"])
            score = math.tanh(float(net) / RELIABILITY_NET_SCALE_BPS)
            m = int(p["memory_i"])
            utility[m] = (1.0 - RELIABILITY_ALPHA) * utility[m] + RELIABILITY_ALPHA * score
        else:
            keep.append(p)
    return keep


def run_reliability_stream(df: pd.DataFrame, preds: list[np.ndarray], initial_utility=None):
    if initial_utility is None:
        utility = np.zeros(len(preds), dtype=float)
    else:
        utility = np.array(initial_utility, dtype=float).copy()
    dynamic_q = np.zeros_like(preds[0], dtype=float)
    pending: list[dict] = []
    weight_rows = []

    ts_values = df.ts.to_numpy(np.int64)
    for ts in np.unique(ts_values):
        pending = _reveal_due(df, pending, int(ts), utility)
        w = reliability_weights(utility)
        idx = np.flatnonzero(ts_values == ts)
        q_stack = np.stack([p[idx] for p in preds], axis=0)
        dynamic_q[idx] = np.tensordot(w, q_stack, axes=(0, 0))
        weight_rows.append([int(ts), *map(float, w), *map(float, utility)])
        for m, p in enumerate(preds):
            _queue_memory_proposals(df, idx, p[idx], pending, m)

    # Do NOT reveal outcomes after the final observed timestamp; they would not
    # have been available to any decision inside this evaluation period.
    weights = np.array([r[1:1 + len(preds)] for r in weight_rows], dtype=float)
    diagnostics = {
        "initial_utility": list(map(float, np.zeros(len(preds)) if initial_utility is None else initial_utility)),
        "final_utility": list(map(float, utility)),
        "mean_weights": list(map(float, weights.mean(axis=0))) if len(weights) else [],
        "min_weights": list(map(float, weights.min(axis=0))) if len(weights) else [],
        "max_weights": list(map(float, weights.max(axis=0))) if len(weights) else [],
        "pending_unrevealed_at_end": len(pending),
        "epochs": len(weight_rows),
    }
    return dynamic_q, utility, diagnostics


def causal_initial_reliability(research: pd.DataFrame, fold_start_ms: int, seed_base: int):
    exp_end = fold_start_ms - PURGE
    exp_start = exp_end - EXPERIENCE_DAYS * 86_400_000
    train = research[research.ts < exp_start - PURGE].copy()
    experience = research[(research.ts >= exp_start) & (research.ts < exp_end)].copy()
    if len(train) < 30000 or len(experience) < 5000:
        raise RuntimeError(f"insufficient causal reliability experience: train={len(train)} exp={len(experience)}")
    bundles = fit_memories(train, exp_start, seed_base)
    preds = predict_memories(bundles, experience)
    _, utility, diag = run_reliability_stream(experience, preds, initial_utility=np.zeros(len(WINDOWS)))
    diag["experience_start_ms"] = int(exp_start)
    diag["experience_end_ms"] = int(exp_end)
    diag["experience_rows"] = int(len(experience))
    return utility, diag


def promotion_gate(metrics: list[dict]) -> dict:
    av = np.array([m["avg_net_bps"] for m in metrics], dtype=float)
    pf = np.array([m["profit_factor"] for m in metrics], dtype=float)
    dd = np.array([m["max_drawdown_bps"] for m in metrics], dtype=float)
    trades = np.array([m["trades"] for m in metrics], dtype=int)
    robust_pass = g.robust_score(metrics) > -1e20
    result = {
        "robust_pass": bool(robust_pass),
        "positive_fold_count": int((av > 0).sum()),
        "total_trades": int(trades.sum()),
        "minimum_trades_each_fold_observed": int(trades.min()),
        "median_avg_net_bps": float(np.median(av)),
        "worst_avg_net_bps": float(av.min()),
        "worst_profit_factor": float(pf.min()),
        "worst_max_drawdown_bps": float(dd.max()),
    }
    result["coverage_pass"] = bool(
        result["total_trades"] >= GEN3C_REFERENCE["minimum_total_trades"]
        and result["minimum_trades_each_fold_observed"] >= GEN3C_REFERENCE["minimum_trades_each_fold"]
    )
    result["improves_median_avg"] = result["median_avg_net_bps"] > GEN3C_REFERENCE["median_fold_avg_net_bps"]
    result["improves_worst_avg"] = result["worst_avg_net_bps"] > GEN3C_REFERENCE["worst_fold_avg_net_bps"]
    result["improves_worst_pf"] = result["worst_profit_factor"] > GEN3C_REFERENCE["worst_fold_profit_factor"]
    result["improves_worst_dd"] = result["worst_max_drawdown_bps"] < GEN3C_REFERENCE["worst_fold_max_drawdown_bps"]
    result["promote_to_active_shadow"] = bool(
        result["robust_pass"]
        and result["coverage_pass"]
        and result["positive_fold_count"] >= 3
        and result["improves_median_avg"]
        and result["improves_worst_avg"]
        and result["improves_worst_pf"]
        and result["improves_worst_dd"]
    )
    return result


def main() -> None:
    symbols = core.discover()[:MAX_SYMBOLS]
    print("DISCOVERED", len(symbols), symbols[:20], flush=True)
    fetched = {}
    calls = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(g.fetch_history, s): s for s in symbols}
        for f in as_completed(fut):
            s = fut[f]
            try:
                iid, d, c_calls = f.result()
                fetched[iid] = d
                calls += c_calls
                print("FETCHED", iid, len(d), flush=True)
            except Exception as exc:
                print("FAIL", s, repr(exc), flush=True)

    parts = []
    accepted = []
    raw_hash = hashlib.sha256()
    for inst in symbols:
        d = fetched.get(inst, pd.DataFrame())
        if len(d) < 1600 or (d.ts < CUTOFF_MS).sum() < 1500:
            continue
        raw_hash.update(pd.util.hash_pandas_object(d, index=False).values.tobytes())
        sym = inst.replace("-USDT-SWAP", "")
        parts.append(g.individual_frame(d, sym))
        accepted.append({
            "instrument": inst,
            "bars": int(len(d)),
            "first_ts": int(d.ts.iloc[0]),
            "last_ts": int(d.ts.iloc[-1]),
        })
    if len(parts) < 24:
        raise RuntimeError(f"only {len(parts)} usable symbols")

    ds = g.add_market_state(pd.concat(parts, ignore_index=True))
    ds = g.clean_training_rows(ds).sort_values(["ts", "symbol"]).reset_index(drop=True)
    research = ds[ds.ts < CUTOFF_MS - PURGE].copy()
    print("RESEARCH_ROWS", len(research), "SYMBOLS", research.symbol.nunique(), flush=True)

    fold_metrics = []
    fold_diagnostics = []
    for fold_i, (name, vstart, vend) in enumerate(FOLDS):
        vs = int(vstart.timestamp() * 1000)
        ve = int(vend.timestamp() * 1000)
        tr = research[research.ts < vs - PURGE].copy()
        va = research[(research.ts >= vs) & (research.ts < ve - PURGE)].copy()

        initial_utility, init_diag = causal_initial_reliability(research, vs, 1100 + fold_i * 100)
        bundles = fit_memories(tr, vs, 2100 + fold_i * 100)
        preds = predict_memories(bundles, va)
        q, _, live_diag = run_reliability_stream(va, preds, initial_utility=initial_utility)
        met, _ = g.simulate(va, q, np.zeros_like(q), 0.0, CAPACITY, Q_FLOOR_BPS)
        fold_metrics.append(met)
        fold_diagnostics.append({"name": name, "initialization": init_diag, "validation": live_diag})
        print("FOLD", name, json.dumps(met, sort_keys=True), flush=True)
        print("RELIABILITY", name, json.dumps(live_diag, sort_keys=True), flush=True)

    promotion = promotion_gate(fold_metrics)
    print("PROMOTION", promotion["promote_to_active_shadow"], json.dumps(promotion, sort_keys=True), flush=True)

    # Final deployable shadow artifact: final memories train through pre-August
    # research; reliability state comes only from a causal pre-cutoff experience
    # window whose memories were trained before that window.
    final_initial_utility, final_init_diag = causal_initial_reliability(research, CUTOFF_MS, 5100)
    final_bundles = fit_memories(research, CUTOFF_MS, 6100)
    artifact = {
        "sub_policies": final_bundles,
        "features": FEATURES,
        "actions": ACTIONS,
        "risk_penalty": RISK_PENALTY,
        "capacity": CAPACITY,
        "q_floor_bps": Q_FLOOR_BPS,
        "timeframe": "4h",
        "roundtrip_cost_bps": COST_BPS,
        "policy_version": "GEN3G_CAUSAL_RELIABILITY_MEMORY",
        "reliability": {
            "initial_utility": list(map(float, final_initial_utility)),
            "alpha": RELIABILITY_ALPHA,
            "net_scale_bps": RELIABILITY_NET_SCALE_BPS,
            "temperature": RELIABILITY_TEMPERATURE,
            "adaptive_weight_strength": ADAPTIVE_WEIGHT_STRENGTH,
            "proposals_per_memory_per_epoch": PROPOSALS_PER_MEMORY_PER_EPOCH,
            "experience_days": EXPERIENCE_DAYS,
            "causal_update_lag": "chosen_action_horizon",
        },
    }
    joblib.dump(artifact, OUT / "dynamic_policy.joblib")

    manifest = {
        "version": "R1F-GEN3G-CAUSAL-RELIABILITY-MEMORY",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_real_history": True,
        "decision_authority": "LEARNED_TEMPORAL_ACTION_VALUE_POLICY_WITH_CAUSAL_RELIABILITY_MEMORY",
        "fixed_strategy_expert_authority": False,
        "timeframe": "4h",
        "symbol_count": len(accepted),
        "research_rows": int(len(research)),
        "features": FEATURES,
        "actions": ACTIONS,
        "temporal_windows_days": list(WINDOWS),
        "frozen_reliability_configuration": artifact["reliability"],
        "frozen_economic_configuration": {
            "capacity": CAPACITY,
            "q_floor_bps": Q_FLOOR_BPS,
            "risk_penalty": RISK_PENALTY,
            "roundtrip_cost_bps": COST_BPS,
        },
        "walk_forward_folds": [
            {"name": name, "metrics": met, "reliability": diag}
            for (name, _, _), met, diag in zip(FOLDS, fold_metrics, fold_diagnostics)
        ],
        "promotion_vs_gen3c": promotion,
        "qualification": {
            "qualified": False,
            "reasons": ["no_fresh_untouched_period_after_gen3_retraining"],
            "authority": "SHADOW_PAPER_ONLY",
        },
        "provenance": {
            "provider": "OKX public REST market history",
            "interval": core.BAR,
            "start": "2023-01-01",
            "enhancement_cutoff": "2026-08-01",
            "already_inspected_period_excluded": True,
            "accepted_symbols": accepted,
            "raw_history_sha256": raw_hash.hexdigest(),
            "api_calls": calls,
        },
        "final_reliability_initialization": final_init_diag,
        "notes": [
            "Gen3C temporal action-value memories remain the only source of coin/direction/size/horizon values.",
            "Memory influence changes only from outcomes whose own action horizon has already completed.",
            "Reliability adaptation is continuously anchored to equal weighting; no memory can become a hard gate.",
            "No Gen3D-style meta learner and no Gen3E-style trade rejection layer is used.",
            "No funding, OI, order-book, liquidation or microstructure input enters this frozen Gen3G evaluation.",
            "Fresh forward shadow evidence remains required for qualification even if historical promotion gates pass.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "TRAINING_REPORT.txt").write_text(json.dumps(manifest, indent=2))
    print("FINAL_MANIFEST")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
