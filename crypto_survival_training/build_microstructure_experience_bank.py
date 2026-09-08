from __future__ import annotations

"""Build a compact causal experience bank from genuine OKX module-6 TBT books.

This program performs data engineering only. It does not inspect forward
returns, fit a model, select a strategy, or alter any active/challenger brain.
Archives are streamed one at a time, hashed, aggregated to completed 15-minute
states, then deleted so raw TBT size cannot dominate the research workflow.
Cross-file OFI continuity is preserved within each symbol/day.
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from crypto_survival_training import microstructure_features as mf

URL = "https://www.okx.com/api/v5/public/market-data-history"
MODULE = "6"
BAR_MS_15M = 15 * 60 * 1000
CHUNKSIZE = int(os.getenv("MICRO_CHUNKSIZE", "40000"))

FULL_SYMBOLS = ["BAT-USDT", "ZRX-USDT", "ATOM-USDT", "BCH-USDT"]
FULL_TRAIN_DATES = [
    "2024-07-01", "2024-07-15", "2024-08-01", "2024-08-15",
    "2024-09-01", "2024-09-15", "2024-10-01", "2024-10-15",
    "2024-11-01", "2024-11-15", "2024-12-01", "2024-12-15",
    "2025-01-01", "2025-01-15", "2025-02-01", "2025-02-15",
    "2025-03-01", "2025-03-15", "2025-04-01", "2025-04-15",
    "2025-05-01", "2025-05-15", "2025-06-01", "2025-06-15",
]
FULL_EVAL_DATES = [
    "2025-07-15", "2025-08-15", "2025-11-15", "2025-12-15",
    "2026-03-15", "2026-04-15", "2026-06-15", "2026-07-15",
]

# Infrastructure pilot only: no forward-return labels or economic metrics.
PILOT_SYMBOLS = ["BAT-USDT", "ZRX-USDT"]
PILOT_DATES = ["2024-07-01", "2024-11-01", "2025-03-01", "2026-03-15"]


def day_ms(day: str) -> tuple[str, str]:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return str(int(d.timestamp() * 1000)), str(int((d + timedelta(days=1)).timestamp() * 1000))


def archive_files(session: requests.Session, symbol: str, day: str) -> list[dict]:
    begin, end = day_ms(day)
    params = {
        "module": MODULE,
        "instType": "SPOT",
        "instIdList": symbol,
        "dateAggrType": "daily",
        "begin": begin,
        "end": end,
    }
    r = session.get(URL, params=params, timeout=45)
    r.raise_for_status()
    body = r.json()
    if str(body.get("code")) != "0":
        raise RuntimeError(f"archive API code={body.get('code')} msg={body.get('msg')}")
    files = []
    for d in body.get("data", []) or []:
        for detail in d.get("details", []) or []:
            for group in detail.get("groupDetails", []) or []:
                if group.get("url"):
                    files.append({
                        "filename": str(group.get("filename") or ""),
                        "url": group.get("url"),
                        "sizeMB": group.get("sizeMB"),
                    })
    return sorted(files, key=lambda x: x["filename"])


def download_stream(session: requests.Session, url: str, destination: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with session.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                h.update(chunk)
                fh.write(chunk)
                n += len(chunk)
    return h.hexdigest(), n


def accumulate_chunk(acc: dict, mid_last: dict[int, float], features: pd.DataFrame) -> None:
    if features.empty:
        return
    mf._accumulate(acc, features, BAR_MS_15M)
    d = features.dropna(subset=["exchTimeMs"]).copy()
    if d.empty:
        return
    d["state_time_ms"] = (d["exchTimeMs"].astype("int64") // BAR_MS_15M + 1) * BAR_MS_15M
    for state_time, g in d.groupby("state_time_ms", sort=False):
        valid = pd.to_numeric(g["mid"], errors="coerce").dropna()
        if len(valid):
            mid_last[int(state_time)] = float(valid.iloc[-1])


def process_one_gz(path: Path, acc: dict, mid_last: dict[int, float], previous_raw: pd.DataFrame | None):
    for raw in pd.read_csv(path, compression="gzip", chunksize=CHUNKSIZE):
        if raw.empty:
            continue
        if previous_raw is not None:
            joined = pd.concat([previous_raw, raw], ignore_index=True)
            features = mf.snapshot_features(joined).iloc[1:].copy()
        else:
            features = mf.snapshot_features(raw)
        accumulate_chunk(acc, mid_last, features)
        previous_raw = raw.iloc[[-1]].copy()
    return previous_raw


def process_symbol_day(session: requests.Session, symbol: str, day: str, tmpdir: Path):
    files = archive_files(session, symbol, day)
    if not files:
        raise RuntimeError(f"no module-6 files for {symbol} {day}")
    acc: dict = {}
    mid_last: dict[int, float] = {}
    previous_raw = None
    provenance = []
    for i, f in enumerate(files):
        local = tmpdir / f"{symbol.replace('-', '_')}_{day}_{i}.csv.gz"
        sha, nbytes = download_stream(session, f["url"], local)
        previous_raw = process_one_gz(local, acc, mid_last, previous_raw)
        provenance.append({
            "filename": f["filename"],
            "reported_size_mb": f.get("sizeMB"),
            "download_bytes": nbytes,
            "sha256": sha,
        })
        local.unlink(missing_ok=True)
    states = mf._finalize_accumulator(acc)
    if states.empty:
        raise RuntimeError(f"no completed states for {symbol} {day}")
    states["mid_last"] = states.state_time_ms.map(mid_last)
    states.insert(0, "date", day)
    states.insert(0, "symbol", symbol)
    states = states.sort_values("state_time_ms").reset_index(drop=True)
    if states.state_time_ms.duplicated().any():
        raise RuntimeError(f"duplicate completed states {symbol} {day}")
    if not states.state_time_ms.is_monotonic_increasing:
        raise RuntimeError(f"nonmonotonic completed states {symbol} {day}")
    spread = pd.to_numeric(states["spread_bps_last"], errors="coerce").dropna()
    if (spread < 0).any():
        raise RuntimeError(f"negative spread {symbol} {day}")
    return states, provenance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--out", default="microstructure_experience_bank")
    args = ap.parse_args()

    if args.mode == "pilot":
        symbols = PILOT_SYMBOLS
        dates = PILOT_DATES
        purpose = "INFRASTRUCTURE_ONLY_NO_ECONOMIC_LABELS"
    else:
        symbols = FULL_SYMBOLS
        dates = FULL_TRAIN_DATES + FULL_EVAL_DATES
        purpose = "PRE_REGISTERED_EXPERIENCE_BANK_DATA_ONLY"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    frames = []
    all_provenance = []

    with tempfile.TemporaryDirectory(prefix="microbank_") as t:
        tmpdir = Path(t)
        for day in dates:
            for symbol in symbols:
                states, provenance = process_symbol_day(session, symbol, day, tmpdir)
                frames.append(states)
                all_provenance.append({
                    "symbol": symbol,
                    "date": day,
                    "completed_15m_states": int(len(states)),
                    "first_state_time_ms": int(states.state_time_ms.iloc[0]),
                    "last_state_time_ms": int(states.state_time_ms.iloc[-1]),
                    "archives": provenance,
                })
                print("BUILT", symbol, day, "states", len(states), "archives", len(provenance), flush=True)

    bank = pd.concat(frames, ignore_index=True).sort_values(["symbol", "state_time_ms"]).reset_index(drop=True)
    # Compact durable representation. Raw archives have already been deleted.
    bank.to_csv(out / "states_15m.csv.gz", index=False, compression="gzip")
    bank_hash = hashlib.sha256((out / "states_15m.csv.gz").read_bytes()).hexdigest()
    manifest = {
        "version": "MICROSTRUCTURE_EXPERIENCE_BANK_V1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "purpose": purpose,
        "provider": "OKX historical market-data archive",
        "module": MODULE,
        "market_type": "SPOT",
        "state_resolution": "15m_completed_bucket",
        "symbols": symbols,
        "dates": dates,
        "pairs": len(symbols) * len(dates),
        "rows": int(len(bank)),
        "feature_columns": [c for c in bank.columns if c not in {"symbol", "date", "state_time_ms", "mid_last"}],
        "states_sha256": bank_hash,
        "raw_archives_retained": False,
        "cross_file_ofi_continuity": True,
        "provenance": all_provenance,
        "checks": {
            "unique_symbol_state_time": bool(not bank.duplicated(["symbol", "state_time_ms"]).any()),
            "finite_mid_fraction": float(np.isfinite(pd.to_numeric(bank.mid_last, errors="coerce")).mean()),
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True
        }
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("MANIFEST", json.dumps({k: manifest[k] for k in ["version", "mode", "pairs", "rows", "states_sha256", "checks"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
