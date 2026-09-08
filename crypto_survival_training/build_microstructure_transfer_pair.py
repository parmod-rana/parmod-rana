from __future__ import annotations

"""Build one frozen non-teacher crypto microstructure confirmation pair.

Data engineering only: genuine OKX module-6 50-level TBT book snapshots are
streamed, hashed and causally aggregated to completed 15-minute states. No
forward label, model fit, economic result or trade decision is produced here.
"""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from crypto_survival_training import build_microstructure_experience_bank as bank

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
TEACHER_SYMBOLS = {"BAT-USDT", "ZRX-USDT", "ATOM-USDT", "BCH-USDT"}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=SYMBOLS)
    ap.add_argument("--date", required=True, choices=DATES)
    ap.add_argument("--out-root", default="microstructure_transfer_pairs")
    args = ap.parse_args()

    if args.symbol in TEACHER_SYMBOLS:
        raise RuntimeError("teacher symbol escaped frozen exclusion")

    out = Path(args.out_root) / f"{args.symbol.replace('-', '_')}__{args.date}"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    with tempfile.TemporaryDirectory(prefix="microtransfer_") as t:
        states, archives, target_rows = bank.process_symbol_day(
            session, args.symbol, args.date, Path(t)
        )

    if states.empty:
        raise RuntimeError("empty causal state pair")
    if set(states["symbol"].astype(str).unique()) != {args.symbol}:
        raise RuntimeError("symbol mismatch in pair")
    if set(states["date"].astype(str).unique()) != {args.date}:
        raise RuntimeError("date mismatch in pair")
    if states.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time in pair")
    finite_mid = float(np.isfinite(pd.to_numeric(states.mid_last, errors="coerce")).mean())
    if finite_mid < 0.999999:
        raise RuntimeError(f"non-finite mid coverage {finite_mid}")

    states_path = out / "states_15m.csv.gz"
    states.to_csv(states_path, index=False, compression="gzip")
    manifest = {
        "version": "MICROSTRUCTURE_TRANSFER_PAIR_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "DATA_ONLY_NO_TRADE_AUTHORITY",
        "protocol": "MICROSTRUCTURE_TRANSFER_BREADTH_CONFIRMATION_V1",
        "symbol": args.symbol,
        "date": args.date,
        "rows": int(len(states)),
        "states_sha256": file_sha(states_path),
        "genuine_rows_inside_target_utc_day": int(target_rows),
        "archives": archives,
        "checks": {
            "frozen_symbol": True,
            "frozen_date": True,
            "teacher_symbol_excluded": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": finite_mid,
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "exact_utc_day_filter_in_underlying_builder": True,
            "raw_archives_retained": False,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("TRANSFER_PAIR", json.dumps({
        "symbol": args.symbol,
        "date": args.date,
        "rows": manifest["rows"],
        "states_sha256": manifest["states_sha256"],
        "target_rows": target_rows,
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
