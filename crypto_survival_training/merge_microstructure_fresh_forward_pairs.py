from __future__ import annotations

"""Fail-closed merger for the frozen 20-symbol x 8-date fresh-forward bank."""

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
    "2026-09-15", "2026-10-01", "2026-10-15", "2026-11-01",
    "2026-11-15", "2026-12-01", "2026-12-15", "2027-01-01",
]
EXPECTED = {(s, d) for s in SYMBOLS for d in DATES}
STATE_MS = 15 * 60 * 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="downloaded_fresh_forward_pairs")
    ap.add_argument("--out", default="microstructure_fresh_forward_bank_v1")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, frames, pair_records = set(), [], []

    for mp in sorted(root.rglob("manifest.json")):
        m = json.loads(mp.read_text())
        if m.get("version") != "MICROSTRUCTURE_FRESH_FORWARD_PAIR_V1":
            continue
        key = (str(m.get("symbol")), str(m.get("date")))
        if key in seen:
            raise RuntimeError(f"duplicate fresh-forward pair {key}")
        if key not in EXPECTED:
            raise RuntimeError(f"unexpected fresh-forward pair {key}")
        if m.get("authority") != "FUTURE_EVIDENCE_DATA_ONLY_NO_TRADE_AUTHORITY" or m.get("rows") != 96:
            raise RuntimeError(f"pair authority/size mismatch {key}")
        checks = m.get("checks") or {}
        for k in [
            "frozen_symbol", "frozen_future_date", "target_utc_day_completed_before_capture",
            "observation_date_after_protocol_freeze", "exact_96_state_grid", "unique_symbol_state_time",
            "exact_okx_module6_spot_50_level_tbt", "no_forward_returns_used_in_capture",
            "no_model_fit", "no_trade_authority",
        ]:
            if checks.get(k) is not True:
                raise RuntimeError(f"failed check {k} for {key}")
        if float(checks.get("finite_mid_fraction", 0.0)) < 0.999999:
            raise RuntimeError(f"non-finite mid {key}")
        sp = mp.parent / "states_15m.csv.gz"
        if not sp.exists() or sha256(sp) != m.get("states_sha256"):
            raise RuntimeError(f"state hash mismatch {key}")
        df = pd.read_csv(sp)
        if len(df) != 96:
            raise RuntimeError(f"pair row mismatch {key}")
        if set(df.symbol.astype(str).unique()) != {key[0]} or set(df.date.astype(str).unique()) != {key[1]}:
            raise RuntimeError(f"content key mismatch {key}")
        ts = np.sort(pd.to_numeric(df.state_time_ms, errors="coerce").to_numpy(float))
        if not np.isfinite(ts).all() or not bool(np.all(np.diff(ts) == STATE_MS)):
            raise RuntimeError(f"non-exact 15m grid {key}")
        frames.append(df)
        pair_records.append({"symbol": key[0], "date": key[1], "rows": 96, "states_sha256": sha256(sp)})
        seen.add(key)

    missing = sorted(EXPECTED - seen)
    if missing or len(seen) != 160:
        raise RuntimeError(f"fresh-forward bank incomplete: seen={len(seen)} missing={len(missing)} first={missing[:10]}")

    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date", "state_time_ms"]).reset_index(drop=True)
    if len(full) != 15360 or full.duplicated(["symbol", "state_time_ms"]).any():
        raise RuntimeError("merged fresh-forward bank size/uniqueness failure")
    finite_mid = float(np.isfinite(pd.to_numeric(full.mid_last, errors="coerce")).mean())
    if finite_mid < 0.999999:
        raise RuntimeError("merged fresh-forward finite-mid failure")

    states_path = out / "states_15m.csv.gz"
    full.to_csv(states_path, index=False, compression="gzip")
    manifest = {
        "version": "MICROSTRUCTURE_FRESH_FORWARD_BANK_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "FUTURE_EVIDENCE_DATA_ONLY_NO_TRADE_AUTHORITY",
        "protocol": "MICROSTRUCTURE_FRESH_FORWARD_PROTOCOL_V1",
        "holdout": "MICROSTRUCTURE_FRESH_FORWARD_HOLDOUT_V1",
        "symbols": SYMBOLS,
        "dates": DATES,
        "pair_count": 160,
        "rows": 15360,
        "states_sha256": sha256(states_path),
        "pairs": pair_records,
        "checks": {
            "all_160_frozen_pairs_present": True,
            "all_pairs_exact_96_state_grid": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": finite_mid,
            "all_observation_dates_after_protocol_freeze": True,
            "exact_okx_module6_spot_50_level_tbt": True,
            "no_forward_returns_used_in_capture": True,
            "no_model_fit": True,
            "no_trade_authority": True,
            "missing_pair_policy": "FAIL_CLOSED_DO_NOT_IMPUTE_OR_SUBSTITUTE",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FRESH_FORWARD_BANK_MANIFEST", json.dumps({
        "pair_count": 160, "rows": 15360, "states_sha256": manifest["states_sha256"],
        "checks": manifest["checks"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
