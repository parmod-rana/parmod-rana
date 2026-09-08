import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ff", HERE / "run_microstructure_fresh_forward_evaluator.py")
ff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ff)


class FakeTeacher:
    @staticmethod
    def label_horizon(frame, horizon_min):
        assert horizon_min == 60
        return frame.copy()

    @staticmethod
    def spearman(x, y):
        a = pd.to_numeric(pd.Series(x), errors="coerce")
        b = pd.to_numeric(pd.Series(y), errors="coerce")
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 12 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
            return None
        return float(a[ok].corr(b[ok], method="spearman"))


def make_frame():
    rows = []
    neg = set(ff.EXPECTED_SYMBOLS[:13])
    for si, symbol in enumerate(ff.EXPECTED_SYMBOLS):
        for date in ff.EXPECTED_DATES:
            for i in range(24):
                x = float(i) + si * 0.001
                y = -x if symbol in neg else x
                rows.append({
                    "symbol": symbol,
                    "date": date,
                    "top_ofi_norm_mean": x,
                    "target_return_bps": y,
                })
    return pd.DataFrame(rows)


def valid_lock():
    return {
        "knowledge_set": "MICROSTRUCTURE_FRESH_FORWARD_ELIGIBLE_KNOWLEDGE_V1",
        "authority": "FUTURE_KNOWLEDGE_QUALIFICATION_ONLY_NO_TRADE_AUTHORITY",
        "eligible_relationships": [{
            "horizon_minutes": 60,
            "feature": "top_ofi_norm_mean",
            "teacher_sign": "NEGATIVE",
            "historical_breadth_confirmed": True,
        }],
        "eligible_trust_candidates": [],
        "eligible_model_families": [],
        "ofi_incremental_eligible": False,
        "gen3c_change": False,
        "real_money_authority": False,
    }


def test_only_historical_survivor_is_allowed():
    rel = ff.validate_lock(valid_lock())
    assert rel["feature"] == "top_ofi_norm_mean"
    broken = valid_lock()
    broken["eligible_relationships"][0]["feature"] = "depth_notional_10_last"
    try:
        ff.validate_lock(broken)
    except RuntimeError:
        pass
    else:
        raise AssertionError("feature reselection was not rejected")


def test_frozen_gate_mechanics():
    rel = ff.validate_lock(valid_lock())
    out = ff.evaluate_relationship(FakeTeacher(), make_frame(), rel)
    assert out["forward_symbols_same_sign_as_teacher"] == 13
    assert out["forward_symbol_median_abs_spearman"] >= 0.02
    assert out["forward_blocks_same_sign_as_teacher"] == 4
    assert out["fresh_forward_relationship_qualified"] is True


def test_gate_fails_closed_when_symbol_breadth_is_insufficient():
    frame = make_frame()
    for symbol in ff.EXPECTED_SYMBOLS[11:13]:
        m = frame.symbol.eq(symbol)
        frame.loc[m, "target_return_bps"] = frame.loc[m, "top_ofi_norm_mean"]
    out = ff.evaluate_relationship(FakeTeacher(), frame, ff.validate_lock(valid_lock()))
    assert out["forward_symbols_same_sign_as_teacher"] == 11
    assert out["fresh_forward_relationship_qualified"] is False
