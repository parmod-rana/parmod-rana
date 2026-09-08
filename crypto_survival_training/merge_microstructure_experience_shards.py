from __future__ import annotations

"""Fail-closed merger for the pre-registered 16-shard TBT experience bank.

This preserves the teacher/evaluation boundary and performs no economic label,
return calculation, model fitting or trading decision.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_survival_training.build_microstructure_experience_shard import BLOCKS, SYMBOLS

EXPECTED = {(s, b) for s in SYMBOLS for b in BLOCKS}
TRAIN_DATES = set(BLOCKS["B0"] + BLOCKS["B1"] + BLOCKS["B2"])
EVAL_DATES = set(BLOCKS["EVAL"])


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="downloaded_shards")
    ap.add_argument("--out", default="microstructure_experience_bank_full")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifests = []
    frames = []
    seen = set()
    for mp in sorted(root.rglob("manifest.json")):
        m = json.loads(mp.read_text())
        if m.get("version") != "MICROSTRUCTURE_EXPERIENCE_SHARD_V1":
            continue
        key = (str(m.get("symbol")), str(m.get("block")))
        if key in seen:
            raise RuntimeError(f"duplicate shard manifest {key}")
        if key not in EXPECTED:
            raise RuntimeError(f"unexpected shard {key}")
        state_path = mp.parent / "states_15m.csv.gz"
        if not state_path.exists():
            raise RuntimeError(f"missing states for {key}")
        actual_sha = file_sha(state_path)
        if actual_sha != m.get("states_sha256"):
            raise RuntimeError(f"states hash mismatch {key}: {actual_sha} != {m.get('states_sha256')}")
        checks = m.get("checks") or {}
        required_true = [
            "all_requested_dates_present",
            "unique_symbol_state_time",
            "no_forward_returns_used",
            "no_model_fit",
            "no_trade_authority",
            "exact_utc_day_filter_in_underlying_builder",
        ]
        if not all(checks.get(k) is True for k in required_true):
            raise RuntimeError(f"failed shard checks {key}: {checks}")
        if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
            raise RuntimeError(f"non-finite mid coverage {key}: {checks.get('finite_mid_fraction')}")
        df = pd.read_csv(state_path)
        if df.empty:
            raise RuntimeError(f"empty shard {key}")
        frames.append(df)
        manifests.append({
            "symbol": key[0],
            "block": key[1],
            "states_sha256": actual_sha,
            "rows": int(len(df)),
            "manifest_path": str(mp),
        })
        seen.add(key)

    missing = sorted(EXPECTED - seen)
    if missing:
        raise RuntimeError(f"missing frozen shards: {missing}")
    if len(seen) != 16:
        raise RuntimeError(f"expected 16 unique shards, found {len(seen)}")

    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "state_time_ms"]).reset_index(drop=True)
    if full.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time after merge")
    if set(full.symbol.astype(str).unique()) != set(SYMBOLS):
        raise RuntimeError("merged symbol set differs from frozen protocol")

    observed_dates = set(full.date.astype(str).unique())
    expected_dates = TRAIN_DATES | EVAL_DATES
    if observed_dates != expected_dates:
        raise RuntimeError(f"date set mismatch missing={sorted(expected_dates-observed_dates)} extra={sorted(observed_dates-expected_dates)}")

    train = full[full.date.astype(str).isin(TRAIN_DATES)].copy()
    evaluation = full[full.date.astype(str).isin(EVAL_DATES)].copy()
    if len(train) + len(evaluation) != len(full):
        raise RuntimeError("teacher/evaluation split is not exhaustive")
    if set(train.date.astype(str).unique()) & set(evaluation.date.astype(str).unique()):
        raise RuntimeError("teacher/evaluation date leakage")

    paths = {
        "all": out / "all_states_15m.csv.gz",
        "teacher": out / "teacher_states_15m.csv.gz",
        "historical_evaluation": out / "historical_evaluation_states_15m.csv.gz",
    }
    full.to_csv(paths["all"], index=False, compression="gzip")
    train.to_csv(paths["teacher"], index=False, compression="gzip")
    evaluation.to_csv(paths["historical_evaluation"], index=False, compression="gzip")

    manifest = {
        "version": "MICROSTRUCTURE_EXPERIENCE_BANK_FULL_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "economic_role": "NONE_DATA_ENGINEERING_ONLY",
        "symbols": SYMBOLS,
        "teacher_dates": sorted(TRAIN_DATES),
        "historical_evaluation_dates": sorted(EVAL_DATES),
        "shards": manifests,
        "shard_count": len(manifests),
        "symbol_date_pairs": len(SYMBOLS) * len(expected_dates),
        "rows": {
            "all": int(len(full)),
            "teacher": int(len(train)),
            "historical_evaluation": int(len(evaluation)),
        },
        "sha256": {k: file_sha(v) for k, v in paths.items()},
        "checks": {
            "all_16_frozen_shards_present": True,
            "all_128_frozen_symbol_date_pairs_present": True,
            "unique_symbol_state_time": True,
            "teacher_evaluation_dates_disjoint": True,
            "finite_mid_fraction_all": float(np.isfinite(pd.to_numeric(full.mid_last, errors="coerce")).mean()),
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "missing_pair_policy": "FAIL_CLOSED_DO_NOT_IMPUTE",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FULL_BANK_MANIFEST", json.dumps({
        "shard_count": manifest["shard_count"],
        "symbol_date_pairs": manifest["symbol_date_pairs"],
        "rows": manifest["rows"],
        "sha256": manifest["sha256"],
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
