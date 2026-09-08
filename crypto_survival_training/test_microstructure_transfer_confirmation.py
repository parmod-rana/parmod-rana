from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crypto_survival_training import run_microstructure_transfer_confirmation as confirm


class DummyTeacher:
    @staticmethod
    def spearman(x, y):
        a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
        b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if int(ok.sum()) < 12 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
            return None
        return float(spearmanr(a[ok], b[ok]).statistic)


def make_confirmation_frame(negative_symbols: set[str] | None = None) -> pd.DataFrame:
    negative_symbols = negative_symbols or set()
    rows = []
    for si, symbol in enumerate(confirm.EXPECTED_SYMBOLS):
        for fold_i, (_, dates) in enumerate(confirm.FOLDS.items()):
            for date_i, date in enumerate(dates):
                for k in range(16):
                    feature = float(k + date_i * 20 + fold_i * 50)
                    # Positive candidate by default; reverse target only for selected symbols.
                    target = feature if symbol not in negative_symbols else -feature
                    rows.append({
                        "symbol": symbol,
                        "date": date,
                        "locked_feature": feature,
                        "target_return_bps": target + si * 1e-6,
                    })
    return pd.DataFrame(rows)


class TransferConfirmationGateTest(unittest.TestCase):
    def test_relationship_passes_only_when_symbol_and_fold_gates_pass(self):
        frame = make_confirmation_frame()
        knowledge = {
            "eligible_stable_feature_relationships": [{
                "horizon_minutes": 60,
                "feature": "locked_feature",
                "teacher_sign": "POSITIVE",
                "teacher_spearman": 0.10,
            }]
        }
        result = confirm.relationship_confirmation(DummyTeacher, {60: frame}, knowledge)[0]
        self.assertTrue(result["breadth_confirmed"])
        self.assertEqual(result["confirmation_symbols_same_sign_as_teacher"], 20)
        self.assertEqual(result["confirmation_folds_same_sign_as_teacher"], 4)
        self.assertGreaterEqual(result["confirmation_symbol_median_abs_spearman"], 0.02)

    def test_relationship_fails_when_only_eleven_symbols_keep_teacher_sign(self):
        reverse = set(confirm.EXPECTED_SYMBOLS[11:])
        frame = make_confirmation_frame(reverse)
        knowledge = {
            "eligible_stable_feature_relationships": [{
                "horizon_minutes": 60,
                "feature": "locked_feature",
                "teacher_sign": "POSITIVE",
                "teacher_spearman": 0.10,
            }]
        }
        result = confirm.relationship_confirmation(DummyTeacher, {60: frame}, knowledge)[0]
        self.assertEqual(result["confirmation_symbols_same_sign_as_teacher"], 11)
        self.assertFalse(result["breadth_confirmed"])

    def test_high_is_bad_error_ratio(self):
        values = np.arange(10, dtype=float)
        err = np.array([1, 1, 1, 1, 1, 1, 2, 2, 3, 4], dtype=float)
        rec = confirm._error_ratio(err, values, "HIGH_IS_BAD", q20=2.0, q80=7.0)
        self.assertEqual(rec["good_state_samples"], 3)
        self.assertEqual(rec["bad_state_samples"], 3)
        self.assertGreater(rec["bad_over_good_error_ratio"], 1.10)

    def test_low_is_bad_error_ratio(self):
        values = np.arange(10, dtype=float)
        err = np.array([4, 3, 2, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        rec = confirm._error_ratio(err, values, "LOW_IS_BAD", q20=2.0, q80=7.0)
        self.assertEqual(rec["bad_state_samples"], 3)
        self.assertEqual(rec["good_state_samples"], 3)
        self.assertGreater(rec["bad_over_good_error_ratio"], 1.10)

    def test_unknown_trust_direction_fails_closed(self):
        with self.assertRaises(RuntimeError):
            confirm._error_ratio(np.ones(4), np.arange(4), "UNKNOWN", 1.0, 2.0)


if __name__ == "__main__":
    unittest.main()
