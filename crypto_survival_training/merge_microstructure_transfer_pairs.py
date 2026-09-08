from __future__ import annotations

"""Fail-closed merger for the frozen 20-symbol x 8-date breadth bank.

No future labels, models, economic metrics or trading decisions are computed.
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
EXPECTED = {(s, d) for s in SYMBOLS for d in DATES}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="downloaded_transfer_pairs")
    ap.add_argument("--out", default="microstructure_transfer_bank_v1")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, frames, pair_records = set(), [], []

    for mp in sorted(root.rglob("manifest.json")):
        m = json.loads(mp.read_text())
        if m.get("version") != "MICROSTRUCTURE_TRANSFER_PAIR_V1":
            continue
        key = (str(m.get("symbol")), str(m.get("date")))
        if key in seen:
            raise RuntimeError(f"duplicate transfer pair {key}")
        if key not in EXPECTED:
            raise RuntimeError(f"unexpected transfer pair {key}")
        checks = m.get("checks") or {}
        for k in [
            "frozen_symbol", "frozen_date", "teacher_symbol_excluded",
            "unique_symbol_state_time", "no_forward_returns_used", "no_model_fit",
            "no_trade_authority", "exact_utc_day_filter_in_underlying_builder",
        ]:
            if checks.get(k) is not True:
                raise RuntimeError(f"failed check {k} for {key}: {checks}")
        if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
            raise RuntimeError(f"non-finite mid {key}")
        sp = mp.parent / "states_15m.csv.gz"
        if not sp.exists():
            raise RuntimeError(f"missing states {key}")
        actual = sha256(sp)
        if actual != m.get("states_sha256"):
            raise RuntimeError(f"state hash mismatch {key}")
        df = pd.read_csv(sp)
        if df.empty:
            raise RuntimeError(f"empty pair {key}")
        if set(df.symbol.astype(str).unique()) != {key[0]} or set(df.date.astype(str).unique()) != {key[1]}:
            raise RuntimeError(f"content key mismatch {key}")
        frames.append(df)
        pair_records.append({"symbol": key[0], "date": key[1], "rows": int(len(df)), "states_sha256": actual})
        seen.add(key)

    missing = sorted(EXPECTED - seen)
    if missing:
        raise RuntimeError(f"missing frozen transfer pairs count={len(missing)} first={missing[:10]}")
    if len(seen) != 160:
        raise RuntimeError(f"expected 160 pairs, got {len(seen)}")

    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "state_time_ms"]).reset_index(drop=True)
    if full.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time after transfer merge")
    if set(full.symbol.astype(str).unique()) != set(SYMBOLS):
        raise RuntimeError("transfer symbol set mismatch")
    if set(full.date.astype(str).unique()) != set(DATES):
        raise RuntimeError("transfer date set mismatch")
    finite_mid = float(np.isfinite(pd.to_numeric(full.mid_last, errors="coerce")).mean())
    if finite_mid < 0.999999:
        raise RuntimeError(f"merged finite-mid failure {finite_mid}")

    states_path = out / "states_15m.csv.gz"
    full.to_csv(states_path, index=False, compression="gzip")
    manifest = {
        "version": "MICROSTRUCTURE_TRANSFER_BANK_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "CONFIRMATORY_DATA_ONLY_NO_TRADE_AUTHORITY",
        "protocol": "MICROSTRUCTURE_TRANSFER_BREADTH_CONFIRMATION_V1",
        "symbols": SYMBOLS,
        "dates": DATES,
        "pair_count": 160,
        "rows": int(len(full)),
        "states_sha256": sha256(states_path),
        "pairs": pair_records,
        "checks": {
            "all_160_frozen_pairs_present": True,
            "teacher_symbols_excluded": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": finite_mid,
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "missing_pair_policy": "FAIL_CLOSED_DO_NOT_IMPUTE",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("TRANSFER_BANK_MANIFEST", json.dumps({
        "pair_count": manifest["pair_count"],
        "rows": manifest["rows"],
        "states_sha256": manifest["states_sha256"],
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
