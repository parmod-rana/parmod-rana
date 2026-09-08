from __future__ import annotations

"""Frozen fresh-forward evaluator for historically breadth-confirmed microstructure knowledge.

This module evaluates only the single relationship that survived the preregistered
20-symbol historical breadth confirmation. It performs no feature/sign/symbol/date
reselection, no threshold retuning, no model fitting, and has no trading authority.
"""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT",
    "LINK-USDT", "AVAX-USDT", "DOT-USDT", "LTC-USDT", "TRX-USDT", "UNI-USDT",
    "AAVE-USDT", "NEAR-USDT", "ETC-USDT", "FIL-USDT", "CRV-USDT", "SUSHI-USDT",
    "ALGO-USDT", "XLM-USDT",
]
BLOCKS = {
    "P1": ("2026-09-15", "2026-10-01"),
    "P2": ("2026-10-15", "2026-11-01"),
    "P3": ("2026-11-15", "2026-12-01"),
    "P4": ("2026-12-15", "2027-01-01"),
}
EXPECTED_DATES = [d for pair in BLOCKS.values() for d in pair]
EXPECTED_RELATIONSHIP = {
    "horizon_minutes": 60,
    "feature": "top_ofi_norm_mean",
    "teacher_sign": "NEGATIVE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def load_teacher_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_micro_teacher_forward", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen teacher module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_lock(lock: dict) -> dict:
    if lock.get("knowledge_set") != "MICROSTRUCTURE_FRESH_FORWARD_ELIGIBLE_KNOWLEDGE_V1":
        raise RuntimeError("wrong forward-eligible knowledge set")
    if lock.get("authority") != "FUTURE_KNOWLEDGE_QUALIFICATION_ONLY_NO_TRADE_AUTHORITY":
        raise RuntimeError("wrong forward knowledge authority")
    rels = lock.get("eligible_relationships") or []
    if len(rels) != 1:
        raise RuntimeError("forward eligible relationship count changed")
    rel = rels[0]
    for k, v in EXPECTED_RELATIONSHIP.items():
        if rel.get(k) != v:
            raise RuntimeError(f"forward relationship changed: {k}")
    if rel.get("historical_breadth_confirmed") is not True:
        raise RuntimeError("relationship lacks historical breadth confirmation")
    if lock.get("eligible_trust_candidates") != []:
        raise RuntimeError("trust candidates resurrected after historical failure")
    if lock.get("eligible_model_families") != []:
        raise RuntimeError("model family resurrected after teacher failure")
    if lock.get("ofi_incremental_eligible") is not False:
        raise RuntimeError("OFI incremental block resurrected")
    if lock.get("gen3c_change") is not False or lock.get("real_money_authority") is not False:
        raise RuntimeError("authority escalation detected")
    return rel


def validate_holdout(holdout: dict) -> None:
    if holdout.get("holdout") != "MICROSTRUCTURE_FRESH_FORWARD_HOLDOUT_V1":
        raise RuntimeError("wrong fresh-forward holdout")
    if holdout.get("status") != "FROZEN_BEFORE_20_SYMBOL_HISTORICAL_TRANSFER_RESULT_AND_BEFORE_ALL_HOLDOUT_DATES":
        raise RuntimeError("holdout freeze status changed")
    if holdout.get("symbols") != EXPECTED_SYMBOLS:
        raise RuntimeError("future symbols changed")
    blocks = holdout.get("future_blocks") or {}
    if {k: tuple(v) for k, v in blocks.items()} != BLOCKS:
        raise RuntimeError("future dates/blocks changed")
    gate = holdout.get("forward_relationship_gate") or {}
    if gate.get("same_sign_symbols_min") != 12:
        raise RuntimeError("same-sign symbol gate changed")
    if float(gate.get("symbol_median_abs_spearman_min", -1)) != 0.02:
        raise RuntimeError("median Spearman gate changed")
    if gate.get("same_sign_future_blocks_min") != 3 or gate.get("all_conditions_required") is not True:
        raise RuntimeError("future block gate changed")
    if gate.get("teacher_sign_is_immutable") is not True:
        raise RuntimeError("teacher sign mutability detected")
    gov = holdout.get("governance") or {}
    for key in ["future_dates_reselection", "future_symbols_reselection", "feature_reselection", "sign_reselection", "threshold_retuning", "post_result_hyperparameter_search", "gen3c_change", "real_money_authority"]:
        if gov.get(key) is not False:
            raise RuntimeError(f"forward governance changed: {key}")


def validate_forward_bank(bank_dir: Path) -> tuple[dict, Path]:
    man_path = bank_dir / "manifest.json"
    states_path = bank_dir / "states_15m.csv.gz"
    man = json.loads(man_path.read_text())
    if man.get("version") != "MICROSTRUCTURE_FRESH_FORWARD_BANK_V1":
        raise RuntimeError("wrong fresh-forward bank version")
    if man.get("authority") != "FUTURE_EVIDENCE_DATA_ONLY_NO_TRADE_AUTHORITY":
        raise RuntimeError("wrong fresh-forward bank authority")
    if man.get("symbols") != EXPECTED_SYMBOLS:
        raise RuntimeError("forward bank symbols changed")
    if man.get("dates") != EXPECTED_DATES:
        raise RuntimeError("forward bank dates changed")
    if man.get("pair_count") != 160 or man.get("rows") != 15360:
        raise RuntimeError("forward bank frozen size mismatch")
    checks = man.get("checks") or {}
    required_true = [
        "all_160_frozen_pairs_present", "all_pairs_exact_96_state_grid", "unique_symbol_state_time",
        "no_forward_returns_used_in_capture", "no_model_fit", "no_trade_authority",
        "all_observation_dates_after_protocol_freeze", "exact_okx_module6_spot_50_level_tbt",
    ]
    for key in required_true:
        if checks.get(key) is not True:
            raise RuntimeError(f"forward bank check failed: {key}")
    if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
        raise RuntimeError("forward bank finite-mid check failed")
    actual = sha256(states_path)
    if actual != man.get("states_sha256"):
        raise RuntimeError("forward bank state hash mismatch")
    return man, states_path


def evaluate_relationship(teacher_mod, raw: pd.DataFrame, rel: dict) -> dict:
    h = int(rel["horizon_minutes"])
    feature = str(rel["feature"])
    expected_sign = -1
    labeled = teacher_mod.label_horizon(raw, h)
    if feature not in labeled.columns:
        raise RuntimeError("locked forward feature missing")

    symbol_corr = {}
    same_symbols = 0
    abs_symbols = []
    for symbol in EXPECTED_SYMBOLS:
        sub = labeled[labeled.symbol.astype(str).eq(symbol)]
        r = teacher_mod.spearman(sub[feature], sub["target_return_bps"])
        symbol_corr[symbol] = r
        if r is not None:
            abs_symbols.append(abs(float(r)))
            same_symbols += int(np.sign(float(r)) == expected_sign)

    block_corr = {}
    same_blocks = 0
    for block, dates in BLOCKS.items():
        sub = labeled[labeled.date.astype(str).isin(dates)]
        r = teacher_mod.spearman(sub[feature], sub["target_return_bps"])
        block_corr[block] = r
        if r is not None:
            same_blocks += int(np.sign(float(r)) == expected_sign)

    med_abs = safe_float(np.median(abs_symbols)) if abs_symbols else None
    qualified = bool(same_symbols >= 12 and med_abs is not None and med_abs >= 0.02 and same_blocks >= 3)
    return {
        "horizon_minutes": h,
        "feature": feature,
        "teacher_sign": rel["teacher_sign"],
        "forward_symbol_spearman": symbol_corr,
        "forward_symbols_same_sign_as_teacher": same_symbols,
        "forward_symbol_median_abs_spearman": med_abs,
        "forward_block_spearman": block_corr,
        "forward_blocks_same_sign_as_teacher": same_blocks,
        "frozen_gate": {
            "same_sign_symbols_min": 12,
            "symbol_median_abs_spearman_min": 0.02,
            "same_sign_future_blocks_min": 3,
            "all_conditions_required": True,
        },
        "fresh_forward_relationship_qualified": qualified,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-bank-dir", required=True)
    ap.add_argument("--teacher-module", required=True)
    ap.add_argument("--eligible-knowledge", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", default="microstructure_fresh_forward_result")
    args = ap.parse_args()

    bank_dir = Path(args.forward_bank_dir)
    teacher_module_path = Path(args.teacher_module)
    eligible_path = Path(args.eligible_knowledge)
    holdout_path = Path(args.holdout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = json.loads(eligible_path.read_text())
    holdout = json.loads(holdout_path.read_text())
    rel = validate_lock(lock)
    validate_holdout(holdout)
    man, states_path = validate_forward_bank(bank_dir)
    teacher_mod = load_teacher_module(teacher_module_path)
    raw = pd.read_csv(states_path)
    if len(raw) != 15360:
        raise RuntimeError("forward bank actual row count mismatch")
    if raw.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate forward symbol/state_time")
    if set(raw["symbol"].astype(str).unique()) != set(EXPECTED_SYMBOLS):
        raise RuntimeError("forward state symbols mismatch")
    if set(raw["date"].astype(str).unique()) != set(EXPECTED_DATES):
        raise RuntimeError("forward state dates mismatch")
    group_sizes = raw.groupby(["symbol", "date"], sort=False).size()
    if len(group_sizes) != 160 or not bool((group_sizes == 96).all()):
        raise RuntimeError("forward bank actual 96-state grid count mismatch")
    for (_, _), g in raw.groupby(["symbol", "date"], sort=False):
        ts = np.sort(pd.to_numeric(g["state_time_ms"], errors="coerce").to_numpy(float))
        if not np.isfinite(ts).all() or len(ts) != 96:
            raise RuntimeError("non-finite or incomplete forward state grid")
        if not bool(np.all(np.diff(ts) == 900000.0)):
            raise RuntimeError("forward state timestamps are not exact 15m increments")
    result_rel = evaluate_relationship(teacher_mod, raw, rel)

    result = {
        "version": "MICROSTRUCTURE_FRESH_FORWARD_QUALIFICATION_V1_RESULT",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "FUTURE_KNOWLEDGE_QUALIFICATION_ONLY_NO_TRADE_AUTHORITY",
        "input_evidence": {
            "forward_bank_states_sha256": man["states_sha256"],
            "forward_bank_rows": man["rows"],
            "forward_pair_count": man["pair_count"],
            "eligible_knowledge_sha256": sha256(eligible_path),
            "holdout_sha256": sha256(holdout_path),
            "frozen_teacher_module_sha256": sha256(teacher_module_path),
        },
        "relationship_evaluation": result_rel,
        "summary": {
            "eligible_relationships": 1,
            "fresh_forward_relationships_qualified": int(result_rel["fresh_forward_relationship_qualified"]),
            "economic_edge_claim": False,
            "trading_strategy_claim": False,
            "trade_authority": False,
        },
        "governance": {
            "feature_reselection": False,
            "symbol_reselection": False,
            "date_reselection": False,
            "sign_reselection": False,
            "threshold_retuning": False,
            "model_fit": False,
            "trading_rule_selection": False,
            "gen3c_change": False,
            "future_trading_integration_requires_separately_pre_registered_generation": True,
            "real_money_authority": False,
        },
    }
    (out / "fresh_forward_result.json").write_text(json.dumps(result, indent=2))
    print("MICROSTRUCTURE_FRESH_FORWARD_FINAL", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(
        "REL", result_rel["horizon_minutes"], result_rel["feature"],
        result_rel["fresh_forward_relationship_qualified"],
        result_rel["forward_symbols_same_sign_as_teacher"],
        result_rel["forward_symbol_median_abs_spearman"],
        result_rel["forward_blocks_same_sign_as_teacher"], flush=True,
    )


if __name__ == "__main__":
    main()
