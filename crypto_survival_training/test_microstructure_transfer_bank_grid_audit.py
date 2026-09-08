from __future__ import annotations

# Synthetic integrity tests for the frozen 20x8 exact-grid transfer bank.

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crypto_survival_training import audit_microstructure_transfer_bank_grid as audit


def make_bank(root: Path, mutate=None) -> Path:
    rows = []
    for symbol in audit.SYMBOLS:
        for day in audit.DATES:
            start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
            for i in range(1, 97):
                rows.append({
                    "symbol": symbol,
                    "date": day,
                    "state_time_ms": start + i * audit.BAR_MS,
                    "mid_last": 100.0 + i,
                })
    df = pd.DataFrame(rows)
    if mutate is not None:
        df = mutate(df.copy())
    bank = root / "bank"
    bank.mkdir()
    states = bank / "states_15m.csv.gz"
    df.to_csv(states, index=False, compression="gzip")
    sha = hashlib.sha256(states.read_bytes()).hexdigest()
    manifest = {
        "version": "MICROSTRUCTURE_TRANSFER_BANK_V1",
        "symbols": audit.SYMBOLS,
        "dates": audit.DATES,
        "pair_count": 160,
        "rows": int(len(df)),
        "states_sha256": sha,
        "checks": {
            "all_160_frozen_pairs_present": True,
            "teacher_symbols_excluded": True,
            "unique_symbol_state_time": True,
            "finite_mid_fraction": 1.0,
            "no_forward_returns_used": True,
            "no_model_fit": True,
            "no_trade_authority": True,
        },
    }
    (bank / "manifest.json").write_text(json.dumps(manifest))
    return bank


class TransferBankGridAuditTest(unittest.TestCase):
    def test_complete_exact_grid_passes(self):
        with tempfile.TemporaryDirectory() as t:
            result = audit.audit(make_bank(Path(t)))
            self.assertTrue(result["exact_15m_grid_all_pairs"])
            self.assertEqual(result["pairs"], 160)
            self.assertEqual(result["states_per_pair"], 96)
            self.assertEqual(result["rows"], 15360)

    def test_single_missing_state_fails_closed(self):
        def mutate(df):
            return df.drop(index=df.index[500]).reset_index(drop=True)
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(RuntimeError):
                audit.audit(make_bank(Path(t), mutate))

    def test_shifted_timestamp_fails_exact_grid(self):
        def mutate(df):
            df.loc[500, "state_time_ms"] = int(df.loc[500, "state_time_ms"]) + 60_000
            return df
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(RuntimeError):
                audit.audit(make_bank(Path(t), mutate))


if __name__ == "__main__":
    unittest.main()
