from __future__ import annotations

"""Build one pre-registered microstructure experience-bank shard.

No economic label, return, model or trade decision is produced here. Symbols
and date blocks are fixed by MICROSTRUCTURE_BANK_SHARD_PROTOCOL.json. The
underlying parser enforces exact UTC-day exchange timestamps and lossless
historical schema normalization.
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

SYMBOLS = ["BAT-USDT", "ZRX-USDT", "ATOM-USDT", "BCH-USDT"]
BLOCKS = {
    "B0": ["2024-07-01", "2024-07-15", "2024-08-01", "2024-08-15", "2024-09-01", "2024-09-15", "2024-10-01", "2024-10-15"],
    "B1": ["2024-11-01", "2024-11-15", "2024-12-01", "2024-12-15", "2025-01-01", "2025-01-15", "2025-02-01", "2025-02-15"],
    "B2": ["2025-03-01", "2025-03-15", "2025-04-01", "2025-04-15", "2025-05-01", "2025-05-15", "2025-06-01", "2025-06-15"],
    "EVAL": ["2025-07-15", "2025-08-15", "2025-11-15", "2025-12-15", "2026-03-15", "2026-04-15", "2026-06-15", "2026-07-15"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=SYMBOLS)
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--out-root", default="microstructure_experience_shards")
    args = ap.parse_args()

    symbol = args.symbol
    block = args.block
    dates = BLOCKS[block]
    out = Path(args.out_root) / f"{symbol.replace('-', '_')}__{block}"
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    frames = []
    provenance = []
    with tempfile.TemporaryDirectory(prefix="microshard_") as t:
        tmpdir = Path(t)
        for day in dates:
            states, archives, target_rows = bank.process_symbol_day(session, symbol, day, tmpdir)
            frames.append(states)
            provenance.append({
                "symbol": symbol,
                "date": day,
                "genuine_rows_inside_target_utc_day": int(target_rows),
                "completed_15m_states": int(len(states)),
                "first_state_time_ms": int(states.state_time_ms.iloc[0]),
                "last_state_time_ms": int(states.state_time_ms.iloc[-1]),
                "archives": archives,
            })
            print("SHARD_PAIR", symbol, block, day, "rows", target_rows, "states", len(states), flush=True)

    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "state_time_ms"]).reset_index(drop=True)
    if df.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("duplicate symbol/state_time in shard")
    if set(df["date"].astype(str).unique()) != set(dates):
        raise RuntimeError("shard did not contain every frozen requested date")
    if set(df["symbol"].astype(str).unique()) != {symbol}:
        raise RuntimeError("unexpected symbol escaped shard")

    states_path = out / "states_15m.csv.gz"
    df.to_csv(states_path, index=False, compression="gzip")
    states_sha = hashlib.sha256(states_path.read_bytes()).hexdigest()
    manifest = {
        "version": "MICROSTRUCTURE_EXPERIENCE_SHARD_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "economic_role": "NONE_DATA_ENGINEERING_ONLY",
        "symbol": symbol,
        "block": block,
        "dates": dates,
        "expected_pairs": len(dates),
        "rows": int(len(df)),
        "states_sha256": states_sha,
        "feature_columns": [c for c in df.columns if c not in {"symbol", "date", "state_time_ms", "mid_last"}],
        "provenance": provenance,
        "checks": {
            "all_requested_dates_present": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": float(np.isfinite(pd.to_numeric(df.mid_last, errors="coerce")).mean()),
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "raw_archives_retained": False,
            "exact_utc_day_filter_in_underlying_builder": True,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SHARD_MANIFEST", json.dumps({
        "symbol": symbol,
        "block": block,
        "rows": manifest["rows"],
        "states_sha256": states_sha,
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
