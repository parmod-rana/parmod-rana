from __future__ import annotations

"""Breadth audit for genuine OKX historical 50-level TBT order-book archives.

This is a DATA AVAILABILITY probe only. It never sees forward returns and cannot
select a trading rule. Dates are fixed in advance to cover all four frozen
Gen3C walk-forward regimes. The goal is to decide whether historical
microstructure is broad enough to justify a future Gen3G learning experiment.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

URL = "https://www.okx.com/api/v5/public/market-data-history"
MODULE = "6"  # verified 50-level TBT full snapshots (.csv.gz)
OUT = Path("microstructure_breadth_probe")
OUT.mkdir(exist_ok=True)

# Frozen before the probe: broad cross-section spanning majors and smaller names.
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT",
    "LINK-USDT", "AVAX-USDT", "DOT-USDT", "LTC-USDT", "BCH-USDT", "TRX-USDT",
    "UNI-USDT", "AAVE-USDT", "NEAR-USDT", "ETC-USDT", "FIL-USDT", "ATOM-USDT",
    "CRV-USDT", "SUSHI-USDT", "ALGO-USDT", "XLM-USDT", "BAT-USDT", "ZRX-USDT",
]

# Two fixed dates inside each frozen Gen3C validation regime.
DATES = [
    "2025-07-15", "2025-08-15",  # F1
    "2025-11-15", "2025-12-15",  # F2
    "2026-03-15", "2026-04-15",  # F3
    "2026-06-15", "2026-07-15",  # F4
]

# Non-economic breadth gate frozen before seeing probe results.
MIN_DATES_PER_SYMBOL = 6
MIN_SYMBOLS_PASSING = 12
MIN_TOTAL_AVAILABLE_PAIRS = 60


def day_ms(day: str) -> tuple[str, str]:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return str(int(d.timestamp() * 1000)), str(int((d + timedelta(days=1)).timestamp() * 1000))


def extract_files(body: dict) -> list[dict]:
    files: list[dict] = []
    for d in body.get("data", []) if isinstance(body, dict) else []:
        for detail in d.get("details", []) or []:
            for group in detail.get("groupDetails", []) or []:
                files.append({
                    "filename": group.get("filename"),
                    "url": group.get("url"),
                    "sizeMB": group.get("sizeMB"),
                    "instType": detail.get("instType"),
                    "instId": detail.get("instId"),
                })
    return files


def main() -> None:
    rows: list[dict] = []
    session = requests.Session()
    for day in DATES:
        begin, end = day_ms(day)
        for symbol in SYMBOLS:
            params = {
                "module": MODULE,
                "instType": "SPOT",
                "instIdList": symbol,
                "dateAggrType": "daily",
                "begin": begin,
                "end": end,
            }
            try:
                r = session.get(URL, params=params, timeout=30)
                body = r.json()
                files = extract_files(body)
                sizes = []
                for f in files:
                    try:
                        sizes.append(float(f.get("sizeMB") or 0.0))
                    except Exception:
                        pass
                rec = {
                    "date": day,
                    "symbol": symbol,
                    "http": r.status_code,
                    "code": body.get("code") if isinstance(body, dict) else None,
                    "available": bool(files),
                    "file_count": len(files),
                    "total_size_mb": float(sum(sizes)),
                    "min_size_mb": float(min(sizes)) if sizes else None,
                    "max_size_mb": float(max(sizes)) if sizes else None,
                }
            except Exception as exc:
                rec = {
                    "date": day,
                    "symbol": symbol,
                    "http": None,
                    "code": None,
                    "available": False,
                    "file_count": 0,
                    "total_size_mb": 0.0,
                    "min_size_mb": None,
                    "max_size_mb": None,
                    "error": repr(exc),
                }
            rows.append(rec)
            print("PAIR", json.dumps(rec, sort_keys=True), flush=True)
            time.sleep(0.12)

    by_symbol = {}
    by_date = {}
    for rec in rows:
        by_symbol.setdefault(rec["symbol"], 0)
        by_date.setdefault(rec["date"], 0)
        if rec["available"]:
            by_symbol[rec["symbol"]] += 1
            by_date[rec["date"]] += 1

    passing_symbols = sorted([s for s, n in by_symbol.items() if n >= MIN_DATES_PER_SYMBOL])
    available_pairs = sum(1 for r in rows if r["available"])
    available_sizes = [r["total_size_mb"] for r in rows if r["available"] and r["total_size_mb"] > 0]
    available_sizes.sort()
    median_size = available_sizes[len(available_sizes) // 2] if available_sizes else None

    summary = {
        "probe": "GEN3G_MICROSTRUCTURE_BREADTH_V1",
        "module": MODULE,
        "market_type": "SPOT",
        "symbols_tested": len(SYMBOLS),
        "dates_tested": len(DATES),
        "total_pairs": len(rows),
        "available_pairs": available_pairs,
        "availability_fraction": available_pairs / len(rows) if rows else 0.0,
        "available_dates_by_symbol": by_symbol,
        "available_symbols_by_date": by_date,
        "passing_symbols": passing_symbols,
        "passing_symbol_count": len(passing_symbols),
        "median_available_daily_file_size_mb": median_size,
        "frozen_breadth_gate": {
            "min_dates_per_symbol": MIN_DATES_PER_SYMBOL,
            "min_symbols_passing": MIN_SYMBOLS_PASSING,
            "min_total_available_pairs": MIN_TOTAL_AVAILABLE_PAIRS,
        },
        "breadth_pass": (
            len(passing_symbols) >= MIN_SYMBOLS_PASSING
            and available_pairs >= MIN_TOTAL_AVAILABLE_PAIRS
        ),
        "notes": [
            "No forward returns or trading outcomes are used by this probe.",
            "Probe dates were fixed before availability results and span all four frozen Gen3C walk-forward regimes.",
            "A breadth pass only authorizes a future frozen learning experiment; it is not evidence of economic edge.",
        ],
    }
    (OUT / "pairs.json").write_text(json.dumps(rows, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
