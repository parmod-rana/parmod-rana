from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from crypto_survival_training.run_microstructure_experience_teacher import (
    FOLDS,
    STATE_MS,
    feature_relationships,
    label_horizon,
    transform_features,
)


class MicrostructureTeacherCausalityTest(unittest.TestCase):
    def make_day(self, symbol: str, date: str, n: int = 96) -> pd.DataFrame:
        start = pd.Timestamp(date, tz="UTC").value // 1_000_000
        state_time = start + (np.arange(n) + 1) * STATE_MS
        return pd.DataFrame({
            "symbol": symbol,
            "date": date,
            "state_time_ms": state_time.astype(np.int64),
            "mid_last": 100.0 + np.arange(n) * 0.1,
            "spread_bps_mean": np.linspace(1, 3, n),
            "depth_notional_50_mean": np.linspace(1000, 2000, n),
            "capture_latency_ms_mean": np.linspace(-2, 8, n),
            "update_interval_ms_mean": np.linspace(1, 5, n),
            "book_updates": np.arange(n) + 100,
            "top_ofi_sum": np.sin(np.arange(n) / 5),
        })

    def test_exact_horizon_counts(self):
        df = pd.concat([
            self.make_day("BAT-USDT", "2025-01-01"),
            self.make_day("ZRX-USDT", "2025-01-01"),
        ], ignore_index=True)
        self.assertEqual(len(label_horizon(df, 15)), 2 * 95)
        self.assertEqual(len(label_horizon(df, 60)), 2 * 92)
        self.assertEqual(len(label_horizon(df, 240)), 2 * 80)

    def test_missing_state_is_not_interpolated(self):
        df = self.make_day("BAT-USDT", "2025-01-01", n=20)
        # Remove the fifth completed state. A 60m label needing that exact state
        # must become unavailable rather than jumping to the next row.
        missing_ts = int(df.loc[4, "state_time_ms"])
        df = df[df.state_time_ms != missing_ts].reset_index(drop=True)
        out = label_horizon(df, 60)
        first_ts = int(df.loc[0, "state_time_ms"])
        self.assertFalse((out.state_time_ms == first_ts).any())

    def test_labels_do_not_cross_sampled_days(self):
        d1 = self.make_day("BAT-USDT", "2025-01-01", n=4)
        d2 = self.make_day("BAT-USDT", "2025-01-02", n=4)
        out = label_horizon(pd.concat([d1, d2], ignore_index=True), 15)
        # Three valid next-state labels per day, never day1 last -> day2 first.
        self.assertEqual(len(out), 6)

    def test_frozen_transformations_are_finite_where_input_is_finite(self):
        df = self.make_day("BAT-USDT", "2025-01-01", n=10)
        cols = [
            "spread_bps_mean",
            "depth_notional_50_mean",
            "capture_latency_ms_mean",
            "update_interval_ms_mean",
            "book_updates",
            "top_ofi_sum",
        ]
        x = transform_features(df, cols)
        self.assertEqual(list(x.columns), cols)
        self.assertTrue(np.isfinite(x.to_numpy(float)).all())
        self.assertTrue((x["depth_notional_50_mean"] > 0).all())

    def test_stable_feature_requires_cross_symbol_evidence(self):
        symbols = ["BAT-USDT", "ZRX-USDT", "ATOM-USDT", "BCH-USDT"]
        teacher_parts = []
        for symbol in symbols:
            signal = np.linspace(-2.0, 2.0, 40)
            teacher_parts.append(pd.DataFrame({
                "symbol": symbol,
                "date": "2025-01-15",
                "signal": signal,
                "target_return_bps": signal * 20.0,
            }))
        teacher = pd.concat(teacher_parts, ignore_index=True)

        eval_parts = []
        for fold, dates in FOLDS.items():
            for symbol in symbols:
                signal = np.linspace(-1.5, 1.5, 24)
                eval_parts.append(pd.DataFrame({
                    "symbol": symbol,
                    "date": dates[0],
                    "signal": signal,
                    "target_return_bps": signal * 10.0,
                }))
        evaluation = pd.concat(eval_parts, ignore_index=True)
        row = feature_relationships(teacher, evaluation, ["signal"])[0]
        self.assertTrue(row["stable_relationship"])
        self.assertEqual(row["same_sign_teacher_symbols"], 4)
        self.assertEqual(row["same_sign_eval_folds"], 4)
        self.assertEqual(set(row["teacher_symbol_spearman"]), set(symbols))


if __name__ == "__main__":
    unittest.main()
