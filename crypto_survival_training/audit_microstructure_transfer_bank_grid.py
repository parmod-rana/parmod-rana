from __future__ import annotations

"""Fail-closed exact 15-minute grid audit for the frozen 20x8 transfer bank.

This is data-integrity only. It computes no future return, feature relationship,
model output, economic metric or trading decision.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT",
    "LINK-USDT", "AVAX-USDT", "DOT-USDT", "LTC-USDT", "TRX-USDT", "UNI-USDT",
    "AAVE-USDT", "NEAR-USDT", "ETC-USDT", "FIL-USDT", "CRV-USDT", "SUSHI-USDT",
    "ALGO-USDT", "XLM-USDT",
]
DATES = [
    "2025-07-15", "2025-08-15", "2025-11-15", "2025-12-15",
    "2026-03-15", "2026-04-15", "2026-06-15", "2026-07-15",
]
BAR_MS = 15 * 60 * 1000
EXPECTED_STATES_PER_PAIR = 96
EXPECTED_PAIRS = 160
EXPECTED_ROWS = EXPECTED_PAIRS * EXPECTED_STATES_PER_PAIR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(bank_dir: Path) -> dict:
    manifest_path = bank_dir / "manifest.json"
    states_path = bank_dir / "states_15m.csv.gz"
    if not manifest_path.exists() or not states_path.exists():
        raise RuntimeError("transfer bank manifest/states missing")
    m = json.loads(manifest_path.read_text())
    if m.get("version") != "MICROSTRUCTURE_TRANSFER_BANK_V1":
        raise RuntimeError("wrong transfer bank version")
    if int(m.get("pair_count", -1)) != EXPECTED_PAIRS:
        raise RuntimeError("wrong transfer pair count")
    if int(m.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError(f"wrong transfer bank row count {m.get('rows')} != {EXPECTED_ROWS}")
    if set(m.get("symbols", [])) != set(SYMBOLS) or set(m.get("dates", [])) != set(DATES):
        raise RuntimeError("frozen symbol/date set mismatch")
    actual_sha = sha256(states_path)
    if actual_sha != m.get("states_sha256"):
        raise RuntimeError("transfer bank states hash mismatch")
    checks = m.get("checks") or {}
    for k in ["all_160_frozen_pairs_present", "teacher_symbols_excluded", "unique_symbol_state_time", "no_forward_returns_used", "no_model_fit", "no_trade_authority"]:
        if checks.get(k) is not True:
            raise RuntimeError(f"manifest integrity check failed {k}")
    if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
        raise RuntimeError("manifest finite-mid coverage failed")

    df = pd.read_csv(states_path)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"states rows {len(df)} != {EXPECTED_ROWS}")
    if df.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time")
    if float(np.isfinite(pd.to_numeric(df.mid_last, errors="coerce")).mean()) < 0.999999:
        raise RuntimeError("states finite-mid coverage failed")

    expected_keys = {(s, d) for s in SYMBOLS for d in DATES}
    observed_keys = set(zip(df.symbol.astype(str), df.date.astype(str)))
    if observed_keys != expected_keys:
        raise RuntimeError(f"pair-key mismatch missing={sorted(expected_keys-observed_keys)[:5]} extra={sorted(observed_keys-expected_keys)[:5]}")

    pair_summary = []
    for (symbol, day), g0 in df.groupby(["symbol", "date"], sort=True):
        g = g0.sort_values("state_time_ms")
        ts = pd.to_numeric(g.state_time_ms, errors="coerce").to_numpy(dtype=np.int64)
        if len(ts) != EXPECTED_STATES_PER_PAIR:
            raise RuntimeError(f"{symbol} {day}: {len(ts)} states != 96")
        start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
        expected = start + np.arange(1, EXPECTED_STATES_PER_PAIR + 1, dtype=np.int64) * BAR_MS
        if not np.array_equal(ts, expected):
            mismatch = np.flatnonzero(ts != expected)
            idx = int(mismatch[0]) if len(mismatch) else -1
            raise RuntimeError(f"{symbol} {day}: exact 15m grid mismatch at index={idx}")
        pair_summary.append({
            "symbol": symbol,
            "date": day,
            "states": int(len(ts)),
            "first_state_time_ms": int(ts[0]),
            "last_state_time_ms": int(ts[-1]),
        })

    if len(pair_summary) != EXPECTED_PAIRS:
        raise RuntimeError("grouped pair count mismatch")
    return {
        "version": "MICROSTRUCTURE_TRANSFER_BANK_EXACT_GRID_AUDIT_V1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "authority": "DATA_INTEGRITY_ONLY_NO_TRADE_AUTHORITY",
        "states_sha256": actual_sha,
        "pairs": EXPECTED_PAIRS,
        "states_per_pair": EXPECTED_STATES_PER_PAIR,
        "rows": EXPECTED_ROWS,
        "exact_15m_grid_all_pairs": True,
        "finite_mid_fraction": 1.0,
        "no_forward_returns_used": True,
        "no_model_fit": True,
        "no_trade_authority": True,
        "pair_summary": pair_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--out", default="microstructure_transfer_bank_grid_audit.json")
    args = ap.parse_args()
    result = audit(Path(args.bank_dir))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("TRANSFER_BANK_EXACT_GRID_AUDIT_PASS", json.dumps({k: result[k] for k in ["pairs", "states_per_pair", "rows", "states_sha256", "exact_15m_grid_all_pairs"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
