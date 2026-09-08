from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

ARCHIVE_SHA256 = "bd7df469f6e7d95bee62a7c51d794a9119478cbc3c95b1e68debcafb4adc5b20"
ALLOWED_CONTENT_YEARS = {2022, 2023, 2024}
HORIZONS = (15, 30, 60)
Q_GRID = (0.99, 0.995, 0.9975)
TRAIN_DAYS = 252
MIN_TRAIN_DATES = 60
MODEL_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.04,
    num_leaves=7,
    max_depth=3,
    min_child_samples=200,
    reg_lambda=30.0,
    random_state=7584,
    n_jobs=1,
    verbosity=-1,
)

FEATURES = [
    *[f"n_ret_{h}" for h in (1, 2, 3, 5, 10, 15, 30)],
    *[f"s_ret_{h}" for h in (1, 2, 3, 5, 10, 15)],
    *[f"div_ret_{h}" for h in (1, 3, 5, 10, 15)],
    *[f"vix_ret_{h}" for h in (1, 5, 15)],
    "n_body_frac", "n_range_frac", "n_upper_wick_frac", "n_lower_wick_frac",
    "n_atr14_pct", "n_rv_5", "n_rv_15", "n_rv_30",
    "n_ema9_21_gap_pct", "n_ema21_50_gap_pct", "n_ema9_slope3_pct",
    "n_rsi14",
    "n_trend_eff_5", "n_trend_eff_15", "n_trend_eff_30",
    "n_session_pos", "n_dist_session_high_pct", "n_dist_session_low_pct",
    "n_opening_gap_pct",
    "minute_sin", "minute_cos",
]
assert len(FEATURES) == 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_candles(raw: bytes, market: str) -> pd.DataFrame:
    obj = json.loads(raw)
    candles = obj.get("data", {}).get("candles", []) if isinstance(obj, dict) else []
    if not candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    rows = []
    for r in candles:
        if not isinstance(r, list) or len(r) < 5:
            continue
        rows.append((r[0], r[1], r[2], r[3], r[4]))
    out = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out[(out.close > 0) & (out.open > 0) & (out.high > 0) & (out.low > 0)]
    out["market"] = market
    return out


def member_year(name: str) -> int | None:
    m = re.search(r"/(20\d{2})-\d{2}-\d{2}--", name)
    return int(m.group(1)) if m else None


def load_authorized_underlyings(archive: Path) -> dict[str, pd.DataFrame]:
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise RuntimeError("FAIL_CLOSED: genuine raw archive SHA256 mismatch")
    prefixes = {
        "NIFTY": "underlying/nifty-50/minutes-1/",
        "SENSEX": "underlying/sensex/minutes-1/",
        "VIX": "underlying/india-vix/minutes-1/",
    }
    buckets = {k: [] for k in prefixes}
    with zipfile.ZipFile(archive, "r") as z:
        for name in z.namelist():
            market = next((k for k, p in prefixes.items() if name.startswith(p)), None)
            if market is None:
                continue
            y = member_year(name)
            if y is None or y not in ALLOWED_CONTENT_YEARS:
                # Do not read the member body. 2025/2026 market content stays sealed.
                continue
            buckets[market].append(parse_candles(z.read(name), market))
    out = {}
    for market, parts in buckets.items():
        if not parts:
            raise RuntimeError(f"FAIL_CLOSED: no authorized {market} 1m members")
        x = pd.concat(parts, ignore_index=True)
        x = x.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        if not set(x.timestamp.dt.year.unique()).issubset(ALLOWED_CONTENT_YEARS):
            raise RuntimeError(f"FAIL_CLOSED: forbidden year entered {market} frame")
        out[market] = x
    return out


def wilder_atr14(g: pd.DataFrame) -> pd.Series:
    prev = g.close.shift(1)
    tr = pd.concat([(g.high-g.low).abs(), (g.high-prev).abs(), (g.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def session_return(close: pd.Series, dates: pd.Series, lag: int) -> pd.Series:
    prev = close.groupby(dates).shift(lag)
    return (close / prev - 1.0) * 10000.0


def realized_vol(close: pd.Series, dates: pd.Series, n: int) -> pd.Series:
    r = np.log(close / close.groupby(dates).shift(1))
    return r.groupby(dates).rolling(n, min_periods=n).std(ddof=0).reset_index(level=0, drop=True) * 10000.0


def trend_eff(close: pd.Series, dates: pd.Series, n: int) -> pd.Series:
    net = (close - close.groupby(dates).shift(n)).abs()
    step = close.groupby(dates).diff().abs()
    path = step.groupby(dates).rolling(n, min_periods=n).sum().reset_index(level=0, drop=True)
    return net / path.replace(0, np.nan)


def build_state(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    n = frames["NIFTY"].copy().rename(columns={c: f"n_{c}" for c in ["open","high","low","close"]})
    s = frames["SENSEX"][["timestamp","close"]].copy().rename(columns={"close":"s_close"})
    v = frames["VIX"][["timestamp","close"]].copy().rename(columns={"close":"vix_close"})
    x = n.merge(s, on="timestamp", how="left").merge(v, on="timestamp", how="left")
    x = x.sort_values("timestamp").reset_index(drop=True)
    x["date"] = x.timestamp.dt.date
    x["year"] = x.timestamp.dt.year
    x["month"] = x.timestamp.dt.to_period("M").astype(str)

    for h in (1,2,3,5,10,15,30):
        x[f"n_ret_{h}"] = session_return(x.n_close, x.date, h)
    for h in (1,2,3,5,10,15):
        x[f"s_ret_{h}"] = session_return(x.s_close, x.date, h)
    for h in (1,3,5,10,15):
        x[f"div_ret_{h}"] = x[f"n_ret_{h}"] - x[f"s_ret_{h}"]
    for h in (1,5,15):
        x[f"vix_ret_{h}"] = session_return(x.vix_close, x.date, h)

    denom = x.n_close.replace(0, np.nan)
    body_hi = pd.concat([x.n_open, x.n_close], axis=1).max(axis=1)
    body_lo = pd.concat([x.n_open, x.n_close], axis=1).min(axis=1)
    x["n_body_frac"] = (x.n_close - x.n_open) / denom
    x["n_range_frac"] = (x.n_high - x.n_low) / denom
    x["n_upper_wick_frac"] = (x.n_high - body_hi) / denom
    x["n_lower_wick_frac"] = (body_lo - x.n_low) / denom

    x["n_atr14"] = x.groupby("date", group_keys=False).apply(lambda g: wilder_atr14(g.rename(columns={"n_open":"open","n_high":"high","n_low":"low","n_close":"close"})), include_groups=False).reset_index(level=0, drop=True)
    x["n_atr14_pct"] = x.n_atr14 / denom * 10000.0
    for h in (5,15,30):
        x[f"n_rv_{h}"] = realized_vol(x.n_close, x.date, h)

    ema9 = x.groupby("date").n_close.transform(lambda q: q.ewm(span=9, adjust=False, min_periods=9).mean())
    ema21 = x.groupby("date").n_close.transform(lambda q: q.ewm(span=21, adjust=False, min_periods=21).mean())
    ema50 = x.groupby("date").n_close.transform(lambda q: q.ewm(span=50, adjust=False, min_periods=50).mean())
    x["n_ema9_21_gap_pct"] = (ema9-ema21)/denom*10000.0
    x["n_ema21_50_gap_pct"] = (ema21-ema50)/denom*10000.0
    x["n_ema9_slope3_pct"] = (ema9-ema9.groupby(x.date).shift(3))/denom*10000.0
    x["n_rsi14"] = x.groupby("date").n_close.transform(rsi14)
    for h in (5,15,30):
        x[f"n_trend_eff_{h}"] = trend_eff(x.n_close, x.date, h)

    sh = x.groupby("date").n_high.cummax()
    sl = x.groupby("date").n_low.cummin()
    width = (sh-sl).replace(0, np.nan)
    x["n_session_pos"] = (x.n_close-sl)/width
    x["n_dist_session_high_pct"] = (sh-x.n_close)/denom*10000.0
    x["n_dist_session_low_pct"] = (x.n_close-sl)/denom*10000.0
    first_open = x.groupby("date").n_open.transform("first")
    prev_close_by_date = x.groupby("date").n_close.last().shift(1)
    prev_map = pd.Series(prev_close_by_date.values, index=prev_close_by_date.index)
    x["n_prev_session_close"] = x["date"].map(prev_map)
    x["n_opening_gap_pct"] = (first_open/x.n_prev_session_close-1.0)*10000.0

    minute = (x.timestamp.dt.hour*60+x.timestamp.dt.minute) - (9*60+15)
    phase = 2*np.pi*minute/375.0
    x["minute_sin"] = np.sin(phase)
    x["minute_cos"] = np.cos(phase)

    if FEATURES != list(dict.fromkeys(FEATURES)) or len(FEATURES) != 42:
        raise RuntimeError("FAIL_CLOSED: feature schema is not exactly 42 unique columns")
    return x


def add_targets(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    for h in HORIZONS:
        entry = y.groupby("date").n_open.shift(-1)
        exitc = y.groupby("date").n_close.shift(-h)
        y[f"y_{h}"] = (exitc/entry-1.0)*10000.0
        y[f"exit_ts_{h}"] = y.groupby("date").timestamp.shift(-h)
    return y


def fit_binary(X: pd.DataFrame, target: np.ndarray):
    u = np.unique(target[np.isfinite(target)])
    if len(u) < 2:
        p = float(u[0]) if len(u) else 0.0
        return ("constant", p)
    m = LGBMClassifier(objective="binary", **MODEL_PARAMS)
    m.fit(X, target.astype(int))
    return ("model", m)


def pred_binary(obj, X: pd.DataFrame) -> np.ndarray:
    kind, payload = obj
    if kind == "constant":
        return np.full(len(X), float(payload))
    return payload.predict_proba(X)[:, 1]


def percentile_against_train(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    a = np.sort(train_scores[np.isfinite(train_scores)])
    if len(a) == 0:
        return np.full(len(scores), np.nan)
    return np.searchsorted(a, scores, side="right") / len(a)


@dataclass
class MonthPack:
    month: str
    action_train_scores: dict[tuple[str,int], np.ndarray]
    eval_rows: pd.DataFrame


def learn_month(x: pd.DataFrame, month_start: pd.Timestamp) -> MonthPack | None:
    train_start = month_start - pd.Timedelta(days=TRAIN_DAYS)
    train = x[(x.timestamp >= train_start) & (x.timestamp < month_start)].copy()
    eval_end = month_start + pd.offsets.MonthBegin(1)
    ev = x[(x.timestamp >= month_start) & (x.timestamp < eval_end)].copy()
    if train.date.nunique() < MIN_TRAIN_DATES or ev.empty:
        return None
    preds = []
    train_score_map = {}
    for h in HORIZONS:
        tr = train[np.isfinite(train[f"y_{h}"])].copy()
        ee = ev.copy()
        if tr.empty:
            continue
        Xtr = tr[FEATURES]
        Xev = ee[FEATURES]
        T = float(np.nanquantile(np.abs(tr[f"y_{h}"].to_numpy()), 0.75))
        reg = LGBMRegressor(objective="huber", **MODEL_PARAMS)
        reg.fit(Xtr, tr[f"y_{h}"].to_numpy())
        up = fit_binary(Xtr, (tr[f"y_{h}"].to_numpy() >= T).astype(int))
        dn = fit_binary(Xtr, (tr[f"y_{h}"].to_numpy() <= -T).astype(int))
        mu_tr = reg.predict(Xtr); pu_tr = pred_binary(up, Xtr); pd_tr = pred_binary(dn, Xtr)
        mu_ev = reg.predict(Xev); pu_ev = pred_binary(up, Xev); pd_ev = pred_binary(dn, Xev)
        for side in ("CE","PE"):
            sign = 1.0 if side == "CE" else -1.0
            raw_tr = sign*mu_tr + T*sign*(pu_tr-pd_tr)
            raw_ev = sign*mu_ev + T*sign*(pu_ev-pd_ev)
            pct = percentile_against_train(raw_tr, raw_ev)
            train_score_map[(side,h)] = raw_tr
            z = ee[["timestamp","date","month",f"y_{h}",f"exit_ts_{h}"]].copy()
            z.columns = ["timestamp","date","month","gross_underlying", "exit_timestamp"]
            z["gross_underlying"] = sign*z.gross_underlying
            z["side"] = side; z["horizon"] = h; z["raw_score"] = raw_ev; z["percentile"] = pct; z["T_h"] = T
            preds.append(z)
    if not preds:
        return None
    allp = pd.concat(preds, ignore_index=True)
    return MonthPack(str(month_start.to_period("M")), train_score_map, allp)


def build_year_candidates(x: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError("FAIL_CLOSED: R3E evaluation year must be 2023 or 2024")
    packs = []
    for m in range(1,13):
        ms = pd.Timestamp(year=year, month=m, day=1, tz="Asia/Kolkata")
        p = learn_month(x, ms)
        if p is not None:
            packs.append(p.eval_rows)
    if not packs:
        return pd.DataFrame()
    return pd.concat(packs, ignore_index=True)


def pf(v: pd.Series) -> float:
    a = v.to_numpy(float); gp = a[a>0].sum(); gl = -a[a<0].sum()
    return float(gp/gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)


def stats(v: pd.Series) -> dict:
    a = v.dropna().astype(float)
    return {"n": int(len(a)), "mean": float(a.mean()) if len(a) else None, "pf": pf(a) if len(a) else None,
            "sum": float(a.sum()) if len(a) else 0.0, "win": float((a>0).mean()) if len(a) else None}


def simulate(candidates: pd.DataFrame, q: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    c = candidates[np.isfinite(candidates.gross_underlying) & np.isfinite(candidates.percentile)].copy()
    c = c[c.percentile >= q]
    # Fixed tie order: percentile, raw score, shorter horizon, CE.
    c["side_tie"] = (c.side == "CE").astype(int)
    c = c.sort_values(["timestamp","percentile","raw_score","horizon","side_tie"], ascending=[True,False,False,True,False])
    c = c.drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    rows=[]; busy_until=None
    for r in c.itertuples(index=False):
        if busy_until is not None and r.timestamp < busy_until:
            continue
        rows.append(r._asdict())
        busy_until = r.exit_timestamp
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["net3"] = out.gross_underlying - 3.0
    out["net5"] = out.gross_underlying - 5.0
    out["half"] = np.where(pd.to_datetime(out.timestamp).dt.month <= 6, "H1", "H2")
    return out


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"net3":{"n":0},"net5":{"n":0},"pass":False}
    out={}
    for col in ("net3","net5"):
        s=stats(trades[col]); s["H1"]=stats(trades.loc[trades.half=="H1",col]); s["H2"]=stats(trades.loc[trades.half=="H2",col])
        month_sum=trades.groupby("month")[col].sum()
        s["active_months"]=int(len(month_sum)); s["positive_month_frac"]=float((month_sum>0).mean()) if len(month_sum) else 0.0
        k=max(1, int(math.ceil(len(trades)*0.01)))
        rm=trades.drop(index=trades.nlargest(k,col).index)
        s["rm_top1"]=stats(rm[col])
        out[col]=s
    n3=out["net3"]; n5=out["net5"]
    passed=(n3["n"]>=60 and n3["H1"]["n"]>=15 and n3["H2"]["n"]>=15 and
            n3["mean"]>0 and n3["pf"]>1 and n5["mean"]>0 and n5["pf"]>1 and
            n3["H1"]["mean"]>0 and n3["H1"]["pf"]>1 and n3["H2"]["mean"]>0 and n3["H2"]["pf"]>1 and
            n3["rm_top1"]["mean"]>0 and n3["rm_top1"]["pf"]>1 and
            n3["active_months"]>=8 and n3["positive_month_frac"]>=0.60)
    out["pass"]=bool(passed)
    return out


def worst_half_mean(s: dict) -> float:
    return min(s["net3"]["H1"]["mean"], s["net3"]["H2"]["mean"])


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("r3e_output"))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    frames=load_authorized_underlyings(args.archive)
    state=add_targets(build_state(frames))
    if len(FEATURES)!=42 or any(c not in state.columns for c in FEATURES):
        raise RuntimeError("FAIL_CLOSED: exact 42-feature state unavailable")

    cand23=build_year_candidates(state, 2023)
    grid={}; passing=[]; trade_by_q={}
    for q in Q_GRID:
        tr=simulate(cand23,q); sm=summarize(tr); grid[str(q)]=sm; trade_by_q[q]=tr
        if sm["pass"]: passing.append(q)
    selected=None
    if passing:
        selected=sorted(passing, key=lambda q:(worst_half_mean(grid[str(q)]), grid[str(q)]["net3"]["mean"], grid[str(q)]["net5"]["mean"], q), reverse=True)[0]

    result={
        "format":"VARDHANI_R3_TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1",
        "archive_sha256":sha256_file(args.archive),
        "feature_count":42,
        "feature_names":FEATURES,
        "boundary":{"student_optimizer_steps":0,"2025_used":False,"2026_used":False,"real_orders":False},
        "calibration_year":2023,"calibration_grid":grid,"selected_q":selected,
        "audit_2024_opened":False,"economic_edge_claimed":False
    }
    if selected is None:
        result["status"]="RETIRED_NO_2023_CALIBRATION_PASS__2024_NOT_OPENED_FOR_R3E"
    else:
        cand24=build_year_candidates(state,2024)
        tr24=simulate(cand24,selected); sm24=summarize(tr24)
        result["audit_2024_opened"]=True; result["audit_2024"]=sm24
        result["status"]="UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED" if sm24["pass"] else "RETIRED_2024_AUDIT_FAIL"
        trade_by_q[selected].to_csv(args.out_dir/"R3E_2023_SELECTED_TRADES.csv",index=False)
        tr24.to_csv(args.out_dir/"R3E_2024_AUDIT_TRADES.csv",index=False)
    (args.out_dir/"TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1.json").write_text(json.dumps(result,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
