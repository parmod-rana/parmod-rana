from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import r3e_runner as core


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def read_month(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    for c in ("timestamp", "exit_timestamp"):
        x[c] = pd.to_datetime(x[c], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    return x


def month_candidates(state: pd.DataFrame, year: int, month: int, out_dir: Path, progress: dict) -> pd.DataFrame:
    month_key = f"{year}-{month:02d}"
    path = out_dir / f"R3E_CANDIDATES_{month_key}.csv"
    if path.exists():
        x = read_month(path)
        progress.setdefault("resumed_months", []).append(month_key)
        return x

    ms = pd.Timestamp(year=year, month=month, day=1, tz="Asia/Kolkata")
    pack = core.learn_month(state, ms)
    x = pd.DataFrame() if pack is None else pack.eval_rows
    x.to_csv(path, index=False)
    progress.setdefault("completed_months", []).append(month_key)
    progress["last_completed_month"] = month_key
    write_json(out_dir / "R3E_CPU_PROGRESS_V1.json", progress)
    return x


def build_year_resumable(state: pd.DataFrame, year: int, out_dir: Path, progress: dict) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise RuntimeError("FAIL_CLOSED: R3E evaluation year must be 2023 or 2024")
    parts = []
    for month in range(1, 13):
        x = month_candidates(state, year, month, out_dir, progress)
        if not x.empty:
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_q(candidates: pd.DataFrame):
    grid = {}
    passing = []
    trades = {}
    for q in core.Q_GRID:
        tr = core.simulate(candidates, q)
        sm = core.summarize(tr)
        grid[str(q)] = sm
        trades[q] = tr
        if sm["pass"]:
            passing.append(q)
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda q: (
                core.worst_half_mean(grid[str(q)]),
                grid[str(q)]["net3"]["mean"],
                grid[str(q)]["net5"]["mean"],
                q,
            ),
            reverse=True,
        )[0]
    return grid, trades, selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("r3e_cpu_output"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    authority = json.loads((Path(__file__).with_name("R3E_CPU_ONLY_EXECUTION_AUTHORITY_V1.json")).read_text())
    if authority["compute"]["device"] != "CPU_ONLY" or authority["compute"]["gpu_allowed"] is not False:
        raise RuntimeError("FAIL_CLOSED: CPU-only authority invalid")
    if core.MODEL_PARAMS.get("n_jobs") != 1:
        raise RuntimeError("FAIL_CLOSED: frozen R3E LightGBM n_jobs changed")

    progress = {
        "format": "VARDHANI_R3_TEACHER_ECONOMIC_R3E_CPU_PROGRESS_V1",
        "status": "CPU_EXECUTION_IN_PROGRESS",
        "archive_sha256": core.sha256_file(args.archive),
        "device": "CPU_ONLY",
        "student_optimizer_steps": 0,
        "2025_used": False,
        "2026_used": False,
        "real_orders": False,
        "economic_edge_claimed": False,
    }
    if progress["archive_sha256"] != core.ARCHIVE_SHA256:
        raise RuntimeError("FAIL_CLOSED: genuine raw archive SHA256 mismatch")
    write_json(args.out_dir / "R3E_CPU_PROGRESS_V1.json", progress)

    frames = core.load_authorized_underlyings(args.archive)
    state = core.add_targets(core.build_state(frames))
    if len(core.FEATURES) != 42 or any(c not in state.columns for c in core.FEATURES):
        raise RuntimeError("FAIL_CLOSED: exact 42-feature state unavailable")

    cand23 = build_year_resumable(state, 2023, args.out_dir, progress)
    grid23, trades23, selected = select_q(cand23)

    result = {
        "format": "VARDHANI_R3_TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1",
        "execution_mode": "CPU_ONLY_MONTH_CHECKPOINT_RESUME",
        "archive_sha256": core.sha256_file(args.archive),
        "feature_count": 42,
        "feature_names": core.FEATURES,
        "boundary": {
            "student_optimizer_steps": 0,
            "2025_used": False,
            "2026_used": False,
            "real_orders": False,
        },
        "calibration_year": 2023,
        "calibration_grid": grid23,
        "selected_q": selected,
        "audit_2024_opened": False,
        "economic_edge_claimed": False,
    }

    if selected is None:
        result["status"] = "RETIRED_NO_2023_CALIBRATION_PASS__2024_NOT_OPENED_FOR_R3E"
    else:
        trades23[selected].to_csv(args.out_dir / "R3E_2023_SELECTED_TRADES.csv", index=False)
        cand24 = build_year_resumable(state, 2024, args.out_dir, progress)
        tr24 = core.simulate(cand24, selected)
        sm24 = core.summarize(tr24)
        tr24.to_csv(args.out_dir / "R3E_2024_AUDIT_TRADES.csv", index=False)
        result["audit_2024_opened"] = True
        result["audit_2024"] = sm24
        result["status"] = (
            "UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED"
            if sm24["pass"]
            else "RETIRED_2024_AUDIT_FAIL"
        )

    write_json(args.out_dir / "TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1.json", result)
    progress["status"] = "CPU_EXECUTION_COMPLETE"
    progress["result_status"] = result["status"]
    progress["selected_q"] = selected
    progress["audit_2024_opened"] = result["audit_2024_opened"]
    write_json(args.out_dir / "R3E_CPU_PROGRESS_V1.json", progress)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
