from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R3E_DIR = HERE.parent / 'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0, str(R3E_DIR))
import r3e_exact_member_cpu as source
import r3h_runner as core

PREREG = 'TEACHER_ECONOMIC_R3H_OPENING_DISLOCATION_ROUTER_PREREG_V1.json'


def evaluate(frames: dict[str, pd.DataFrame], outdir: Path) -> dict:
    state = core.build_state(frames)
    grid = {}
    passing = []
    trade_by_q = {}
    for q in core.Q_GRID:
        cand = core.build_year_candidates(state, 2023, q)
        tr = core.simulate(cand)
        sm = core.summarize(tr)
        grid[str(q)] = sm
        trade_by_q[q] = tr
        print('R3H_2023_Q', q, 'candidates', len(cand), json.dumps(sm, default=str), flush=True)
        if sm['pass']:
            passing.append(q)

    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda q: (
                core.worst_half_mean(grid[str(q)]),
                grid[str(q)]['net3']['mean'],
                grid[str(q)]['net5']['mean'],
                q,
            ),
            reverse=True,
        )[0]

    result = {
        'format': 'VARDHANI_R3_TEACHER_ECONOMIC_R3H_DEVELOPMENT_RESULT_V1',
        'preregistration': PREREG,
        'source_authority': 'R3E_EXACT_MEMBER_RECONSTRUCTION_AUTHORITY_V1.json',
        'authorized_member_count': 108,
        'authorized_years': [2022, 2023, 2024],
        'all_members_byte_identical_to_locked_manifest': True,
        'method': {
            'checkpoints_ist': ['09:45','10:00','10:15'],
            'adaptive_history_dates': core.LOOKBACK_DATES,
            'minimum_history_dates': core.MIN_HISTORY,
            'q_grid': list(core.Q_GRID),
            'horizons': list(core.HORIZONS),
            'direction': 'sign_nifty_ret30',
            'sensex_confirmation': True,
            'inverse_vix_confirmation': True,
            'session_location_confirmation': True,
        },
        'boundary': {
            'teacher_v2_changed': False,
            'student_optimizer_steps': 0,
            '2025_used': False,
            '2026_used': False,
            'real_orders': False,
            'gpu_used': False,
        },
        'calibration_year': 2023,
        'calibration_grid': grid,
        'selected_q': selected,
        'audit_2024_opened': False,
        'economic_edge_claimed': False,
    }
    if selected is None:
        result['status'] = 'RETIRED_NO_2023_CALIBRATION_PASS__2024_NOT_OPENED_FOR_R3H'
    else:
        cand24 = core.build_year_candidates(state, 2024, selected)
        tr24 = core.simulate(cand24)
        sm24 = core.summarize(tr24)
        result['audit_2024_opened'] = True
        result['audit_2024'] = sm24
        result['status'] = 'UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED' if sm24['pass'] else 'RETIRED_2024_AUDIT_FAIL'
        trade_by_q[selected].to_csv(outdir / 'R3H_2023_SELECTED_TRADES.csv', index=False)
        tr24.to_csv(outdir / 'R3H_2024_AUDIT_TRADES.csv', index=False)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path('r3h_exact_cpu_work'))
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((HERE / PREREG).read_text())
    if prereg.get('status') != 'FROZEN_BEFORE_ANY_R3H_OUTPUT':
        raise RuntimeError('FAIL_CLOSED: R3H preregistration not frozen')
    hard = prereg['hard_boundaries']
    if hard['student_optimizer_steps'] != 0 or hard['2025_used'] or hard['2026_used'] or hard['gpu_allowed']:
        raise RuntimeError('FAIL_CLOSED: R3H hard boundary mismatch')
    frames, audit = source.reconstruct_frames(a.work / 'exact_members')
    (a.work / 'R3H_EXACT_MEMBER_RECONSTRUCTION_AUDIT_V1.json').write_text(json.dumps({'status':'PASS_ALL_108_BYTE_IDENTICAL','members':audit},indent=2)+'\n')
    result = evaluate(frames, a.work)
    (a.work / 'TEACHER_ECONOMIC_R3H_DEVELOPMENT_RESULT_V1.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('R3H_FINAL_RESULT', json.dumps(result,default=str), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
