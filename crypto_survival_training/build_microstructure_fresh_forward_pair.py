from __future__ import annotations

"""Build one frozen fresh-forward microstructure pair from genuine OKX module-6 data.

Data engineering only. The target UTC day must already be complete. No future label,
model fit, economic metric, trading rule, Gen3C change, or trade authority is produced.
"""

import argparse
import hashlib
import json
import tempfile
from datetime import date, datetime, timezone
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
    "2026-09-15", "2026-10-01", "2026-10-15", "2026-11-01",
    "2026-11-15", "2026-12-01", "2026-12-15", "2027-01-01",
]
PROTOCOL_FROZEN_AT = "2026-09-08T04:55:00Z"
STATE_MS = 15 * 60 * 1000


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=SYMBOLS)
    ap.add_argument("--date", required=True, choices=DATES)
    ap.add_argument("--out-root", default="microstructure_fresh_forward_pairs")
    args = ap.parse_args()

    target = date.fromisoformat(args.date)
    if datetime.now(timezone.utc).date() <= target:
        raise RuntimeError("target UTC day is not yet complete; future evidence cannot be captured early")

    out = Path(args.out_root) / f"{args.symbol.replace('-', '_')}__{args.date}"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    with tempfile.TemporaryDirectory(prefix="microforward_") as t:
        states, archives, target_rows = bank.process_symbol_day(session, args.symbol, args.date, Path(t))

    if len(states) != 96:
        raise RuntimeError(f"expected exact 96 completed 15m states, got {len(states)}")
    if set(states["symbol"].astype(str).unique()) != {args.symbol}:
        raise RuntimeError("symbol mismatch in fresh-forward pair")
    if set(states["date"].astype(str).unique()) != {args.date}:
        raise RuntimeError("date mismatch in fresh-forward pair")
    if states.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time in fresh-forward pair")
    ts = np.sort(pd.to_numeric(states.state_time_ms, errors="coerce").to_numpy(float))
    if not np.isfinite(ts).all() or not bool(np.all(np.diff(ts) == STATE_MS)):
        raise RuntimeError("fresh-forward pair is not an exact 15m grid")
    finite_mid = float(np.isfinite(pd.to_numeric(states.mid_last, errors="coerce")).mean())
    if finite_mid < 0.999999:
        raise RuntimeError(f"non-finite mid coverage {finite_mid}")

    states_path = out / "states_15m.csv.gz"
    states.to_csv(states_path, index=False, compression="gzip")
    manifest = {
        "version": "MICROSTRUCTURE_FRESH_FORWARD_PAIR_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "FUTURE_EVIDENCE_DATA_ONLY_NO_TRADE_AUTHORITY",
        "protocol": "MICROSTRUCTURE_FRESH_FORWARD_PROTOCOL_V1",
        "holdout": "MICROSTRUCTURE_FRESH_FORWARD_HOLDOUT_V1",
        "protocol_frozen_at_utc": PROTOCOL_FROZEN_AT,
        "symbol": args.symbol,
        "date": args.date,
        "rows": 96,
        "states_sha256": file_sha(states_path),
        "genuine_rows_inside_target_utc_day": int(target_rows),
        "archives": archives,
        "checks": {
            "frozen_symbol": True,
            "frozen_future_date": True,
            "target_utc_day_completed_before_capture": True,
            "observation_date_after_protocol_freeze": True,
            "exact_96_state_grid": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": finite_mid,
            "exact_okx_module6_spot_50_level_tbt": True,
            "no_forward_returns_used_in_capture": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "raw_archives_retained": False,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FRESH_FORWARD_PAIR", json.dumps({
        "symbol": args.symbol, "date": args.date, "rows": 96,
        "states_sha256": manifest["states_sha256"], "target_rows": target_rows,
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
