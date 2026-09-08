from __future__ import annotations

"""Frozen research-only microstructure experience teacher V1.1.

Consumes the fail-closed MICROSTRUCTURE_EXPERIENCE_BANK_FULL_V1 and answers
pre-registered questions about genuine order-flow information. It cannot select
trades, modify Gen3C, or promote any future generation.

V1.1 was frozen before any full-bank evaluation was exposed. It adds explicit
cross-symbol robustness diagnostics/acceptance to the original cross-regime
checks. Model families, hyperparameters, labels and horizons are unchanged.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (15, 60, 240)
STATE_MS = 15 * 60 * 1000
FOLDS = {
    "F1": ("2025-07-15", "2025-08-15"),
    "F2": ("2025-11-15", "2025-12-15"),
    "F3": ("2026-03-15", "2026-04-15"),
    "F4": ("2026-06-15", "2026-07-15"),
}
EXCLUDE = {"symbol", "date", "state_time_ms", "mid_last"}
TRUST_VARIABLES = {
    "spread_bps_mean": "HIGH_IS_BAD",
    "depth_notional_50_mean": "LOW_IS_BAD",
    "capture_latency_ms_mean": "HIGH_IS_BAD",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def spearman(x, y) -> float | None:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 12 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return None
    r = spearmanr(a[ok], b[ok], nan_policy="omit").statistic
    return safe_float(r)


def label_horizon(frame: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Attach exact future labels; never interpolate or cross a sampled UTC day."""
    steps = horizon_min // 15
    target_delta = steps * STATE_MS
    parts = []
    for (_, _), g0 in frame.groupby(["symbol", "date"], sort=False):
        g = g0.sort_values("state_time_ms").copy()
        ts = pd.to_numeric(g["state_time_ms"], errors="coerce")
        entry = pd.to_numeric(g["mid_last"], errors="coerce")
        future_ts = ts.shift(-steps)
        future_mid = entry.shift(-steps)
        exact = future_ts.eq(ts + target_delta)
        ret = (future_mid / entry - 1.0) * 1e4
        ret = ret.where(exact & entry.gt(0) & future_mid.gt(0))
        g["target_return_bps"] = ret
        g["target_direction"] = np.where(ret.notna(), (ret > 0).astype(float), np.nan)

        path_returns = []
        exact_path = pd.Series(True, index=g.index)
        for k in range(1, steps + 1):
            kth_ts = ts.shift(-k)
            kth_mid = entry.shift(-k)
            kth_exact = kth_ts.eq(ts + k * STATE_MS)
            exact_path &= kth_exact
            path_returns.append(((kth_mid / entry) - 1.0) * 1e4)
        if path_returns:
            path = pd.concat(path_returns, axis=1)
            g["target_mfe_bps"] = path.max(axis=1).where(exact_path)
            g["target_mae_bps"] = path.min(axis=1).where(exact_path)
        else:
            g["target_mfe_bps"] = np.nan
            g["target_mae_bps"] = np.nan
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out[out["target_return_bps"].notna()].reset_index(drop=True)


def feature_columns(teacher: pd.DataFrame, evaluation: pd.DataFrame) -> list[str]:
    cols = []
    for c in teacher.columns:
        if c in EXCLUDE or c.startswith("target_") or c not in evaluation.columns:
            continue
        s = pd.to_numeric(teacher[c], errors="coerce")
        if s.notna().sum() >= max(100, int(len(teacher) * 0.10)):
            cols.append(c)
    if not cols:
        raise RuntimeError("no numeric feature columns survived frozen policy")
    return cols


def transform_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").astype(float)
        name = c.lower()
        if "depth_notional" in name or name == "book_updates":
            s = np.log1p(s.clip(lower=0.0))
        elif "capture_latency_ms" in name:
            s = np.sign(s) * np.log1p(np.abs(s))
        elif "update_interval_ms" in name:
            s = np.log1p(s.clip(lower=0.0))
        x[c] = s.replace([np.inf, -np.inf], np.nan)
    return x


def linear_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def linear_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def nonlinear_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=30,
            l2_regularization=5.0,
            random_state=42,
        )),
    ])


def nonlinear_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=30,
            l2_regularization=5.0,
            random_state=42,
        )),
    ])


def magnitude_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae": safe_float(mean_absolute_error(y, pred)),
        "rmse": safe_float(np.sqrt(mean_squared_error(y, pred))),
        "spearman": spearman(y, pred),
    }


def direction_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    auc = None
    if len(np.unique(y)) == 2:
        auc = safe_float(roc_auc_score(y, p))
    return {
        "roc_auc": auc,
        "balanced_accuracy": safe_float(balanced_accuracy_score(y, pred)),
        "log_loss": safe_float(log_loss(y, p, labels=[0, 1])),
        "brier": safe_float(brier_score_loss(y, p)),
    }


def _evaluation_slice(
    sub: pd.DataFrame,
    x: pd.DataFrame,
    fitted: dict,
    constant_return: float,
    constant_prob: float,
) -> dict:
    y = sub["target_return_bps"].to_numpy(float)
    yd = sub["target_direction"].to_numpy(int)
    rec = {
        "samples": int(len(sub)),
        "baseline": {
            "magnitude": magnitude_metrics(y, np.full(len(y), constant_return)),
            "direction": direction_metrics(yd, np.full(len(yd), constant_prob)),
        },
    }
    for name, (reg, clf) in fitted.items():
        rec[name] = {
            "magnitude": magnitude_metrics(y, reg.predict(x)),
            "direction": direction_metrics(yd, clf.predict_proba(x)[:, 1]),
        }
    return rec


def eval_models(
    teacher: pd.DataFrame,
    evaluation: pd.DataFrame,
    cols: list[str],
) -> tuple[dict, dict, dict]:
    xt = transform_features(teacher, cols)
    xe = transform_features(evaluation, cols)
    yt = teacher["target_return_bps"].to_numpy(float)
    ydt = teacher["target_direction"].to_numpy(int)

    constant_return = float(np.nanmedian(yt))
    constant_prob = float(np.clip(np.nanmean(ydt), 1e-6, 1 - 1e-6))

    models = {
        "linear": (linear_regressor(), linear_classifier()),
        "nonlinear": (nonlinear_regressor(), nonlinear_classifier()),
    }
    fitted = {}
    for name, (reg, clf) in models.items():
        reg.fit(xt, yt)
        clf.fit(xt, ydt)
        fitted[name] = (reg, clf)

    fold_results = {}
    for fold, dates in FOLDS.items():
        mask = evaluation["date"].astype(str).isin(dates).to_numpy()
        sub = evaluation.loc[mask]
        if sub.empty:
            raise RuntimeError(f"historical evaluation fold {fold} is empty")
        rec = _evaluation_slice(sub, xe.loc[mask], fitted, constant_return, constant_prob)
        rec["symbols"] = sorted(sub["symbol"].astype(str).unique().tolist())
        rec["dates"] = list(dates)
        fold_results[fold] = rec

    symbol_results = {}
    for symbol in sorted(evaluation["symbol"].astype(str).unique()):
        mask = evaluation["symbol"].astype(str).eq(symbol).to_numpy()
        sub = evaluation.loc[mask]
        if sub.empty:
            raise RuntimeError(f"historical evaluation symbol {symbol} is empty")
        rec = _evaluation_slice(sub, xe.loc[mask], fitted, constant_return, constant_prob)
        rec["dates"] = sorted(sub["date"].astype(str).unique().tolist())
        symbol_results[symbol] = rec

    stability = {}
    for name in models:
        mag_fold_beats = 0
        dir_fold_beats = 0
        fold_rs = []
        fold_aucs = []
        for fold in FOLDS:
            r = fold_results[fold]
            if r[name]["magnitude"]["mae"] < r["baseline"]["magnitude"]["mae"]:
                mag_fold_beats += 1
            if r[name]["direction"]["log_loss"] < r["baseline"]["direction"]["log_loss"]:
                dir_fold_beats += 1
            sr = r[name]["magnitude"]["spearman"]
            auc = r[name]["direction"]["roc_auc"]
            if sr is not None:
                fold_rs.append(sr)
            if auc is not None:
                fold_aucs.append(auc)

        mag_symbol_beats = 0
        dir_symbol_beats = 0
        positive_symbol_rs = 0
        good_symbol_auc = 0
        for symbol, r in symbol_results.items():
            if r[name]["magnitude"]["mae"] < r["baseline"]["magnitude"]["mae"]:
                mag_symbol_beats += 1
            if r[name]["direction"]["log_loss"] < r["baseline"]["direction"]["log_loss"]:
                dir_symbol_beats += 1
            sr = r[name]["magnitude"]["spearman"]
            auc = r[name]["direction"]["roc_auc"]
            positive_symbol_rs += int(sr is not None and sr > 0)
            good_symbol_auc += int(auc is not None and auc > 0.5)

        median_fold_spearman = safe_float(np.median(fold_rs)) if fold_rs else None
        median_fold_auc = safe_float(np.median(fold_aucs)) if fold_aucs else None
        magnitude_stable = bool(
            mag_fold_beats >= 3
            and median_fold_spearman is not None
            and median_fold_spearman > 0
            and mag_symbol_beats >= 3
            and positive_symbol_rs >= 3
        )
        direction_stable = bool(
            dir_fold_beats >= 3
            and median_fold_auc is not None
            and median_fold_auc > 0.5
            and dir_symbol_beats >= 3
            and good_symbol_auc >= 3
        )
        stability[name] = {
            "magnitude_folds_beating_baseline_MAE": mag_fold_beats,
            "median_eval_fold_spearman": median_fold_spearman,
            "magnitude_symbols_beating_baseline_MAE": mag_symbol_beats,
            "symbols_with_positive_eval_spearman": positive_symbol_rs,
            "magnitude_stable": magnitude_stable,
            "direction_folds_beating_baseline_log_loss": dir_fold_beats,
            "median_eval_fold_ROC_AUC": median_fold_auc,
            "direction_symbols_beating_baseline_log_loss": dir_symbol_beats,
            "symbols_with_eval_ROC_AUC_above_0_5": good_symbol_auc,
            "direction_stable": direction_stable,
        }
    return fold_results, symbol_results, stability


def feature_relationships(teacher: pd.DataFrame, evaluation: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    y_teacher = teacher["target_return_bps"]
    teacher_symbols = sorted(teacher["symbol"].astype(str).unique())
    for c in cols:
        tr = spearman(teacher[c], y_teacher)
        fold_corr = {}
        same_fold_sign = 0
        abs_eval = []
        for fold, dates in FOLDS.items():
            sub = evaluation[evaluation["date"].astype(str).isin(dates)]
            r = spearman(sub[c], sub["target_return_bps"])
            fold_corr[fold] = r
            if r is not None:
                abs_eval.append(abs(r))
                if tr is not None and tr != 0 and np.sign(r) == np.sign(tr):
                    same_fold_sign += 1

        symbol_corr = {}
        same_symbol_sign = 0
        abs_symbol = []
        for symbol in teacher_symbols:
            sub = teacher[teacher["symbol"].astype(str).eq(symbol)]
            r = spearman(sub[c], sub["target_return_bps"])
            symbol_corr[symbol] = r
            if r is not None:
                abs_symbol.append(abs(r))
                if tr is not None and tr != 0 and np.sign(r) == np.sign(tr):
                    same_symbol_sign += 1

        med_abs_eval = safe_float(np.median(abs_eval)) if abs_eval else None
        med_abs_symbol = safe_float(np.median(abs_symbol)) if abs_symbol else None
        stable = bool(
            tr is not None
            and abs(tr) >= 0.02
            and same_fold_sign >= 3
            and med_abs_eval is not None
            and med_abs_eval >= 0.02
            and same_symbol_sign >= 3
            and med_abs_symbol is not None
            and med_abs_symbol >= 0.02
        )
        rows.append({
            "feature": c,
            "teacher_spearman": tr,
            "eval_fold_spearman": fold_corr,
            "same_sign_eval_folds": same_fold_sign,
            "median_abs_eval_spearman": med_abs_eval,
            "teacher_symbol_spearman": symbol_corr,
            "same_sign_teacher_symbols": same_symbol_sign,
            "median_abs_teacher_symbol_spearman": med_abs_symbol,
            "stable_relationship": stable,
        })
    rows.sort(key=lambda r: abs(r["teacher_spearman"] or 0.0), reverse=True)
    return rows


def ofi_incremental(teacher: pd.DataFrame, evaluation: pd.DataFrame, cols: list[str]) -> dict:
    no_ofi = [c for c in cols if "top_ofi" not in c.lower()]
    if len(no_ofi) == len(cols):
        return {"available": False, "reason": "no top_ofi features in bank"}
    out = {"available": True, "removed_features": sorted(set(cols) - set(no_ofi)), "models": {}}
    for model_name, reg_factory, clf_factory in [
        ("linear", linear_regressor, linear_classifier),
        ("nonlinear", nonlinear_regressor, nonlinear_classifier),
    ]:
        x_full = transform_features(teacher, cols)
        x_no = transform_features(teacher, no_ofi)
        y = teacher["target_return_bps"].to_numpy(float)
        yd = teacher["target_direction"].to_numpy(int)
        rf, cf = reg_factory(), clf_factory()
        rn, cn = reg_factory(), clf_factory()
        rf.fit(x_full, y)
        cf.fit(x_full, yd)
        rn.fit(x_no, y)
        cn.fit(x_no, yd)
        folds = {}
        mae_wins = 0
        ll_wins = 0
        for fold, dates in FOLDS.items():
            sub = evaluation[evaluation["date"].astype(str).isin(dates)]
            yf = sub["target_return_bps"].to_numpy(float)
            ydf = sub["target_direction"].to_numpy(int)
            xf = transform_features(sub, cols)
            xn = transform_features(sub, no_ofi)
            full_mae = mean_absolute_error(yf, rf.predict(xf))
            no_mae = mean_absolute_error(yf, rn.predict(xn))
            full_ll = log_loss(ydf, np.clip(cf.predict_proba(xf)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
            no_ll = log_loss(ydf, np.clip(cn.predict_proba(xn)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
            mae_win = full_mae < no_mae
            ll_win = full_ll < no_ll
            mae_wins += int(mae_win)
            ll_wins += int(ll_win)
            folds[fold] = {
                "full_mae": safe_float(full_mae),
                "no_ofi_mae": safe_float(no_mae),
                "full_minus_no_ofi_mae": safe_float(full_mae - no_mae),
                "full_log_loss": safe_float(full_ll),
                "no_ofi_log_loss": safe_float(no_ll),
                "full_minus_no_ofi_log_loss": safe_float(full_ll - no_ll),
            }
        out["models"][model_name] = {
            "folds": folds,
            "folds_full_beats_no_ofi_MAE": mae_wins,
            "folds_full_beats_no_ofi_log_loss": ll_wins,
            "stable_incremental_information": bool(mae_wins >= 3 and ll_wins >= 3),
        }
    return out


def trust_state_diagnostic(
    teacher: pd.DataFrame,
    evaluation: pd.DataFrame,
    pred_by_fold: dict[str, np.ndarray],
) -> dict:
    out = {}
    for var, direction in TRUST_VARIABLES.items():
        if var not in teacher.columns or var not in evaluation.columns:
            out[var] = {"available": False}
            continue
        train_v = pd.to_numeric(teacher[var], errors="coerce")
        q20 = safe_float(train_v.quantile(0.20))
        q80 = safe_float(train_v.quantile(0.80))
        if q20 is None or q80 is None or q20 >= q80:
            out[var] = {"available": False, "reason": "invalid teacher quintiles"}
            continue
        folds = {}
        material = 0
        for fold, dates in FOLDS.items():
            sub = evaluation[evaluation["date"].astype(str).isin(dates)].copy()
            pred = pred_by_fold[fold]
            err = np.abs(sub["target_return_bps"].to_numpy(float) - pred)
            v = pd.to_numeric(sub[var], errors="coerce").to_numpy(float)
            if direction == "HIGH_IS_BAD":
                bad = np.isfinite(v) & (v >= q80)
                good = np.isfinite(v) & (v <= q20)
            else:
                bad = np.isfinite(v) & (v <= q20)
                good = np.isfinite(v) & (v >= q80)
            bad_mae = float(np.mean(err[bad])) if bad.any() else np.nan
            good_mae = float(np.mean(err[good])) if good.any() else np.nan
            ratio = bad_mae / good_mae if np.isfinite(bad_mae) and np.isfinite(good_mae) and good_mae > 0 else np.nan
            if np.isfinite(ratio) and ratio >= 1.10:
                material += 1
            folds[fold] = {
                "bad_state_samples": int(bad.sum()),
                "good_state_samples": int(good.sum()),
                "bad_state_mae": safe_float(bad_mae),
                "good_state_mae": safe_float(good_mae),
                "bad_over_good_error_ratio": safe_float(ratio),
            }
        out[var] = {
            "available": True,
            "teacher_q20": q20,
            "teacher_q80": q80,
            "bad_state_definition": direction,
            "folds": folds,
            "material_bad_state_error_folds": material,
            "stable_trust_information": bool(material >= 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default="microstructure_experience_bank_full")
    ap.add_argument("--protocol", default="crypto_survival_training/MICROSTRUCTURE_TEACHER_MODEL_PROTOCOL.json")
    ap.add_argument("--out", default="microstructure_teacher_result")
    args = ap.parse_args()

    bank = Path(args.bank_dir)
    protocol_path = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = bank / "manifest.json"
    teacher_path = bank / "teacher_states_15m.csv.gz"
    eval_path = bank / "historical_evaluation_states_15m.csv.gz"
    for p in (manifest_path, teacher_path, eval_path, protocol_path):
        if not p.exists():
            raise RuntimeError(f"required input missing: {p}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "MICROSTRUCTURE_EXPERIENCE_BANK_FULL_V1":
        raise RuntimeError(f"unexpected bank version {manifest.get('version')}")
    checks = manifest.get("checks") or {}
    required = [
        "all_16_frozen_shards_present",
        "all_128_frozen_symbol_date_pairs_present",
        "unique_symbol_state_time",
        "teacher_evaluation_dates_disjoint",
        "no_forward_returns_used",
        "no_model_fit",
        "no_trade_authority",
    ]
    if not all(checks.get(k) is True for k in required):
        raise RuntimeError(f"bank integrity gate failed: {checks}")
    if float(checks.get("finite_mid_fraction_all", 0.0)) < 0.999999:
        raise RuntimeError("bank finite mid coverage below frozen requirement")

    raw_teacher = pd.read_csv(teacher_path)
    raw_eval = pd.read_csv(eval_path)
    if set(raw_teacher["date"].astype(str)) & set(raw_eval["date"].astype(str)):
        raise RuntimeError("teacher/evaluation date leakage detected before study")
    if len(raw_teacher["symbol"].astype(str).unique()) != 4 or len(raw_eval["symbol"].astype(str).unique()) != 4:
        raise RuntimeError("cross-symbol teacher requires all four frozen representative symbols")

    result = {
        "version": "MICROSTRUCTURE_EXPERIENCE_TEACHER_V1_1_RESULT",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "authority": "RESEARCH_ONLY_NO_TRADE_AUTHORITY",
        "input_bank": {
            "manifest_sha256": sha256(manifest_path),
            "teacher_sha256": sha256(teacher_path),
            "historical_evaluation_sha256": sha256(eval_path),
            "teacher_rows_raw": int(len(raw_teacher)),
            "historical_evaluation_rows_raw": int(len(raw_eval)),
        },
        "frozen_model_protocol_sha256": sha256(protocol_path),
        "horizons": {},
        "governance": {
            "no_hyperparameter_search": True,
            "no_post_result_threshold_change": True,
            "no_trade_rule_selected": True,
            "no_gen3c_change": True,
            "direct_integration_forbidden": True,
            "historical_evaluation_not_fresh_forward_qualification": True,
            "broader_non_teacher_symbol_confirmation_required_before_future_integration": True,
            "real_money_authority": False,
        },
    }

    stable_rows_all = []
    for h in HORIZONS:
        teacher = label_horizon(raw_teacher, h)
        evaluation = label_horizon(raw_eval, h)
        cols = feature_columns(teacher, evaluation)
        fold_models, symbol_models, stability = eval_models(teacher, evaluation, cols)
        relationships = feature_relationships(teacher, evaluation, cols)
        stable_relationships = [r for r in relationships if r["stable_relationship"]]
        stable_rows_all.extend([{"horizon_minutes": h, **r} for r in stable_relationships])
        ofi = ofi_incremental(teacher, evaluation, cols)

        # Frozen trust-state diagnostic uses the fixed nonlinear magnitude model.
        reg = nonlinear_regressor()
        reg.fit(transform_features(teacher, cols), teacher["target_return_bps"].to_numpy(float))
        pred_by_fold = {}
        for fold, dates in FOLDS.items():
            sub = evaluation[evaluation["date"].astype(str).isin(dates)]
            pred_by_fold[fold] = reg.predict(transform_features(sub, cols))
        trust = trust_state_diagnostic(teacher, evaluation, pred_by_fold)

        result["horizons"][str(h)] = {
            "teacher_samples": int(len(teacher)),
            "historical_evaluation_samples": int(len(evaluation)),
            "feature_count": int(len(cols)),
            "features": cols,
            "model_evaluation_by_regime": fold_models,
            "model_evaluation_by_symbol": symbol_models,
            "model_stability": stability,
            "stable_feature_relationship_count": int(len(stable_relationships)),
            "stable_feature_relationships": stable_relationships,
            "ofi_incremental_information": ofi,
            "trust_state_diagnostic": trust,
        }
        print("TEACHER_HORIZON", h, json.dumps({
            "teacher_samples": len(teacher),
            "eval_samples": len(evaluation),
            "stable_features": len(stable_relationships),
            "model_stability": stability,
            "ofi": {k: v for k, v in ofi.items() if k != "models"},
        }, sort_keys=True), flush=True)

    stable_df = pd.DataFrame(stable_rows_all)
    if not stable_df.empty:
        stable_df.to_csv(out / "stable_feature_relationships.csv", index=False)
    else:
        (out / "stable_feature_relationships.csv").write_text("horizon_minutes,feature\n")

    result["summary"] = {
        "stable_relationships_total": int(len(stable_rows_all)),
        "horizons_with_any_stable_relationship": int(sum(
            1 for h in result["horizons"].values() if h["stable_feature_relationship_count"] > 0
        )),
        "cross_symbol_robustness_required": True,
        "broader_non_teacher_symbol_confirmation_required": True,
        "future_integration_status": "REQUIRES_SEPARATE_PRE_REGISTERED_GENERATION_AFTER_BREADTH_CONFIRMATION",
        "economic_edge_claim": False,
        "trade_authority": False,
    }
    (out / "teacher_result.json").write_text(json.dumps(result, indent=2))
    print("TEACHER_FINAL", json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
