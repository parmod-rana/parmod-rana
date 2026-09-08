from __future__ import annotations

"""Frozen breadth confirmation for V1.1-accepted microstructure knowledge.

This evaluator does not search features, signs, thresholds, models or symbols.
It mechanically tests only candidates already locked in
MICROSTRUCTURE_ACCEPTED_KNOWLEDGE_V1 against the pre-registered 20-symbol panel.
It has no trading authority and cannot modify Gen3C.
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
TEACHER_SYMBOLS = {"BAT-USDT", "ZRX-USDT", "ATOM-USDT", "BCH-USDT"}
FOLDS = {
    "F1": ("2025-07-15", "2025-08-15"),
    "F2": ("2025-11-15", "2025-12-15"),
    "F3": ("2026-03-15", "2026-04-15"),
    "F4": ("2026-06-15", "2026-07-15"),
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
    spec = importlib.util.spec_from_file_location("frozen_micro_teacher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen teacher module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def relationship_confirmation(teacher_mod, labeled: dict[int, pd.DataFrame], knowledge: dict) -> list[dict]:
    out = []
    for cand in knowledge["eligible_stable_feature_relationships"]:
        h = int(cand["horizon_minutes"])
        feature = str(cand["feature"])
        expected_sign = 1 if cand["teacher_sign"] == "POSITIVE" else -1
        df = labeled[h]
        if feature not in df.columns:
            raise RuntimeError(f"locked feature missing from confirmation bank: {feature}")

        symbol_corr = {}
        same_symbols = 0
        abs_symbols = []
        for symbol in EXPECTED_SYMBOLS:
            sub = df[df.symbol.astype(str).eq(symbol)]
            r = teacher_mod.spearman(sub[feature], sub["target_return_bps"])
            symbol_corr[symbol] = r
            if r is not None:
                abs_symbols.append(abs(float(r)))
                same_symbols += int(np.sign(float(r)) == expected_sign)

        fold_corr = {}
        same_folds = 0
        for fold, dates in FOLDS.items():
            sub = df[df.date.astype(str).isin(dates)]
            r = teacher_mod.spearman(sub[feature], sub["target_return_bps"])
            fold_corr[fold] = r
            if r is not None:
                same_folds += int(np.sign(float(r)) == expected_sign)

        med_abs = safe_float(np.median(abs_symbols)) if abs_symbols else None
        passed = bool(same_symbols >= 12 and med_abs is not None and med_abs >= 0.02 and same_folds >= 3)
        out.append({
            "horizon_minutes": h,
            "feature": feature,
            "teacher_sign": cand["teacher_sign"],
            "teacher_spearman": cand["teacher_spearman"],
            "confirmation_symbol_spearman": symbol_corr,
            "confirmation_symbols_same_sign_as_teacher": same_symbols,
            "confirmation_symbol_median_abs_spearman": med_abs,
            "confirmation_fold_spearman": fold_corr,
            "confirmation_folds_same_sign_as_teacher": same_folds,
            "frozen_gate": {
                "confirmation_symbols_same_sign_as_teacher_min": 12,
                "confirmation_symbol_median_abs_spearman_min": 0.02,
                "confirmation_folds_same_sign_as_teacher_min": 3,
            },
            "breadth_confirmed": passed,
        })
    return out


def _error_ratio(err: np.ndarray, values: np.ndarray, direction: str, q20: float, q80: float) -> dict:
    finite = np.isfinite(values) & np.isfinite(err)
    if direction == "HIGH_IS_BAD":
        bad = finite & (values >= q80)
        good = finite & (values <= q20)
    elif direction == "LOW_IS_BAD":
        bad = finite & (values <= q20)
        good = finite & (values >= q80)
    else:
        raise RuntimeError(f"unknown bad-state direction {direction}")
    bad_mae = float(np.mean(err[bad])) if bad.any() else np.nan
    good_mae = float(np.mean(err[good])) if good.any() else np.nan
    ratio = bad_mae / good_mae if np.isfinite(bad_mae) and np.isfinite(good_mae) and good_mae > 0 else np.nan
    return {
        "bad_state_samples": int(bad.sum()),
        "good_state_samples": int(good.sum()),
        "bad_state_mae": safe_float(bad_mae),
        "good_state_mae": safe_float(good_mae),
        "bad_over_good_error_ratio": safe_float(ratio),
    }


def trust_confirmation(teacher_mod, teacher_raw: pd.DataFrame, teacher_eval_raw: pd.DataFrame, labeled_confirmation: dict[int, pd.DataFrame], knowledge: dict) -> list[dict]:
    out = []
    candidates_by_horizon: dict[int, list[dict]] = {}
    for c in knowledge["eligible_trust_information"]:
        candidates_by_horizon.setdefault(int(c["horizon_minutes"]), []).append(c)

    for h, candidates in sorted(candidates_by_horizon.items()):
        t = teacher_mod.label_horizon(teacher_raw, h)
        e = teacher_mod.label_horizon(teacher_eval_raw, h)
        cdf = labeled_confirmation[h].copy()
        cols = teacher_mod.feature_columns(t, e)
        xt = teacher_mod.transform_features(t, cols)
        xc = teacher_mod.transform_features(cdf, cols)
        model = teacher_mod.nonlinear_regressor()
        model.fit(xt, t["target_return_bps"].to_numpy(float))
        pred = model.predict(xc)
        err = np.abs(cdf["target_return_bps"].to_numpy(float) - pred)

        for cand in candidates:
            feature = str(cand["feature"])
            if feature not in cdf.columns:
                raise RuntimeError(f"locked trust feature missing: {feature}")
            direction = str(cand["bad_state_definition"])
            q20 = float(cand["teacher_q20"])
            q80 = float(cand["teacher_q80"])
            vals = pd.to_numeric(cdf[feature], errors="coerce").to_numpy(float)

            symbols = {}
            symbol_pass = 0
            for symbol in EXPECTED_SYMBOLS:
                mask = cdf.symbol.astype(str).eq(symbol).to_numpy()
                rec = _error_ratio(err[mask], vals[mask], direction, q20, q80)
                symbols[symbol] = rec
                ratio = rec["bad_over_good_error_ratio"]
                symbol_pass += int(ratio is not None and ratio >= 1.10)

            folds = {}
            fold_pass = 0
            for fold, dates in FOLDS.items():
                mask = cdf.date.astype(str).isin(dates).to_numpy()
                rec = _error_ratio(err[mask], vals[mask], direction, q20, q80)
                folds[fold] = rec
                ratio = rec["bad_over_good_error_ratio"]
                fold_pass += int(ratio is not None and ratio >= 1.10)

            passed = bool(symbol_pass >= 12 and fold_pass >= 3)
            out.append({
                "horizon_minutes": h,
                "feature": feature,
                "bad_state_definition": direction,
                "teacher_q20": q20,
                "teacher_q80": q80,
                "teacher_nonlinear_feature_count": len(cols),
                "confirmation_symbol_diagnostic": symbols,
                "confirmation_symbols_bad_over_good_error_ratio_at_least_1_10": symbol_pass,
                "confirmation_fold_diagnostic": folds,
                "confirmation_folds_bad_over_good_error_ratio_at_least_1_10": fold_pass,
                "frozen_gate": {
                    "confirmation_symbols_bad_over_good_error_ratio_at_least_1_10_min": 12,
                    "confirmation_folds_bad_over_good_error_ratio_at_least_1_10_min": 3,
                },
                "breadth_confirmed": passed,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-bank-dir", required=True)
    ap.add_argument("--confirmation-bank-dir", required=True)
    ap.add_argument("--teacher-module", required=True)
    ap.add_argument("--knowledge", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", default="microstructure_transfer_confirmation_result")
    args = ap.parse_args()

    tdir = Path(args.teacher_bank_dir)
    cdir = Path(args.confirmation_bank_dir)
    teacher_module_path = Path(args.teacher_module)
    knowledge_path = Path(args.knowledge)
    protocol_path = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    knowledge = json.loads(knowledge_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if knowledge.get("knowledge_set") != "MICROSTRUCTURE_ACCEPTED_KNOWLEDGE_V1":
        raise RuntimeError("wrong accepted-knowledge set")
    if protocol.get("study") != "MICROSTRUCTURE_TRANSFER_BREADTH_CONFIRMATION_V1":
        raise RuntimeError("wrong confirmation protocol")
    if len(knowledge.get("eligible_stable_feature_relationships", [])) != 7:
        raise RuntimeError("accepted stable relationship count changed")
    if len(knowledge.get("eligible_trust_information", [])) != 5:
        raise RuntimeError("accepted trust candidate count changed")
    if knowledge.get("ineligible_classes", {}).get("magnitude_models") != "NO_V1_1_MODEL_FAMILY_PASSED_STABILITY_GATE":
        raise RuntimeError("model eligibility changed")

    tman = json.loads((tdir / "manifest.json").read_text())
    cman = json.loads((cdir / "manifest.json").read_text())
    if tman.get("version") != "MICROSTRUCTURE_EXPERIENCE_BANK_FULL_V1":
        raise RuntimeError("wrong frozen teacher bank")
    if cman.get("version") != "MICROSTRUCTURE_TRANSFER_BANK_V1":
        raise RuntimeError("wrong transfer bank")
    if cman.get("pair_count") != 160 or set(cman.get("symbols", [])) != set(EXPECTED_SYMBOLS):
        raise RuntimeError("transfer bank breadth mismatch")
    if set(cman.get("symbols", [])) & TEACHER_SYMBOLS:
        raise RuntimeError("teacher symbol leaked into confirmation")
    checks = cman.get("checks") or {}
    for k in ["all_160_frozen_pairs_present", "teacher_symbols_excluded", "unique_symbol_state_time", "no_forward_returns_used", "no_model_fit", "no_trade_authority"]:
        if checks.get(k) is not True:
            raise RuntimeError(f"transfer bank check failed {k}")
    if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
        raise RuntimeError("transfer bank finite-mid check failed")

    teacher_path = tdir / "teacher_states_15m.csv.gz"
    eval_path = tdir / "historical_evaluation_states_15m.csv.gz"
    confirmation_path = cdir / "states_15m.csv.gz"
    expected_teacher_sha = knowledge["teacher_evidence"]["input_bank_data_sha256"]["teacher"]
    expected_eval_sha = knowledge["teacher_evidence"]["input_bank_data_sha256"]["historical_evaluation"]
    if sha256(teacher_path) != expected_teacher_sha or tman["sha256"]["teacher"] != expected_teacher_sha:
        raise RuntimeError("teacher state hash mismatch")
    if sha256(eval_path) != expected_eval_sha or tman["sha256"]["historical_evaluation"] != expected_eval_sha:
        raise RuntimeError("teacher historical-evaluation hash mismatch")
    if sha256(confirmation_path) != cman.get("states_sha256"):
        raise RuntimeError("confirmation state hash mismatch")

    teacher_mod = load_teacher_module(teacher_module_path)
    teacher_raw = pd.read_csv(teacher_path)
    teacher_eval_raw = pd.read_csv(eval_path)
    confirmation_raw = pd.read_csv(confirmation_path)

    needed_horizons = sorted(set([int(x["horizon_minutes"]) for x in knowledge["eligible_stable_feature_relationships"]] + [int(x["horizon_minutes"]) for x in knowledge["eligible_trust_information"]]))
    labeled_confirmation = {h: teacher_mod.label_horizon(confirmation_raw, h) for h in needed_horizons}

    relationships = relationship_confirmation(teacher_mod, labeled_confirmation, knowledge)
    trust = trust_confirmation(teacher_mod, teacher_raw, teacher_eval_raw, labeled_confirmation, knowledge)
    confirmed_relationships = [x for x in relationships if x["breadth_confirmed"]]
    confirmed_trust = [x for x in trust if x["breadth_confirmed"]]

    result = {
        "version": "MICROSTRUCTURE_TRANSFER_BREADTH_CONFIRMATION_V1_RESULT",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "CONFIRMATORY_RESEARCH_ONLY_NO_TRADE_AUTHORITY",
        "input_evidence": {
            "teacher_bank_teacher_sha256": expected_teacher_sha,
            "teacher_bank_historical_evaluation_sha256": expected_eval_sha,
            "confirmation_bank_states_sha256": cman["states_sha256"],
            "confirmation_bank_rows": cman["rows"],
            "confirmation_pair_count": cman["pair_count"],
            "accepted_knowledge_sha256": sha256(knowledge_path),
            "protocol_sha256": sha256(protocol_path),
            "frozen_teacher_module_sha256": sha256(teacher_module_path),
        },
        "stable_feature_relationship_confirmation": relationships,
        "trust_information_confirmation": trust,
        "summary": {
            "pre_registered_relationship_candidates": len(relationships),
            "breadth_confirmed_relationships": len(confirmed_relationships),
            "pre_registered_trust_candidates": len(trust),
            "breadth_confirmed_trust_candidates": len(confirmed_trust),
            "any_transferable_knowledge_confirmed": bool(confirmed_relationships or confirmed_trust),
            "eligible_model_families_confirmed": 0,
            "ofi_incremental_confirmed": false,
            "economic_edge_claim": false,
            "trade_authority": false
        },
        "governance": {
            "feature_reselection": false,
            "symbol_reselection": false,
            "sign_reselection": false,
            "threshold_retuning": false,
            "hyperparameter_search": false,
            "confirmation_model_refit_on_confirmation_data": false,
            "trading_rule_selection": false,
            "gen3c_change": false,
            "historical_confirmation_is_not_fresh_forward_qualification": true,
            "real_money_authority": false
        }
    }
    (out / "transfer_confirmation_result.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(relationships).to_json(out / "relationship_summary.jsonl", orient="records", lines=True)
    pd.DataFrame(trust).to_json(out / "trust_summary.jsonl", orient="records", lines=True)
    print("TRANSFER_CONFIRMATION_FINAL", json.dumps(result["summary"], sort_keys=True), flush=True)
    for r in relationships:
        print("REL", r["horizon_minutes"], r["feature"], r["breadth_confirmed"], r["confirmation_symbols_same_sign_as_teacher"], r["confirmation_symbol_median_abs_spearman"], r["confirmation_folds_same_sign_as_teacher"], flush=True)
    for r in trust:
        print("TRUST", r["horizon_minutes"], r["feature"], r["breadth_confirmed"], r["confirmation_symbols_bad_over_good_error_ratio_at_least_1_10"], r["confirmation_folds_bad_over_good_error_ratio_at_least_1_10"], flush=True)


if __name__ == "__main__":
    main()
