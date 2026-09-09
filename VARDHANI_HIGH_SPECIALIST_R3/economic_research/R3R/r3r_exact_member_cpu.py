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
import r3r_runner as core

PREREG = 'TEACHER_ECONOMIC_R3R_VIX_LED_RISK_SHOCK_PREREG_V1.json'


def evaluate(frames: dict[str, pd.DataFrame], outdir: Path) -> dict:
    state = core.build_state(frames)
    cand23 = core.build_year_candidates(state, 2023)
    tr23 = core.simulate(cand23)
    sm23 = core.summarize(tr23)
    print('R3R_2023_FIXED', 'candidates', len(cand23), json.dumps(sm23, default=str), flush=True)
    result = {
        'format': 'VARDHANI_R3_TEACHER_ECONOMIC_R3R_DEVELOPMENT_RESULT_V1',
        'preregistration': PREREG,
        'source_authority': 'R3E_EXACT_MEMBER_RECONSTRUCTION_AUTHORITY_V1.json',
        'authorized_member_count': 108,
        'authorized_years': [2022, 2023, 2024],
        'all_members_byte_identical_to_locked_manifest': True,
        'method': {
            'checkpoints_ist': list(core.CHECKPOINTS),
            'adaptive_history_dates_same_checkpoint': core.LOOKBACK_DATES,
            'minimum_history_dates': core.MIN_HISTORY,
            'vix_shock_quantile': core.VIX_SHOCK_Q,
            'vix_extreme_quantile': core.VIX_EXTREME_Q,
            'horizons': list(core.HORIZONS),
            'max_trades_per_date': core.MAX_TRADES_PER_DATE,
            'one_trade_at_a_time': True,
        },
        'boundary': {
            'teacher_v2_changed': False,
            'student_optimizer_steps': 0,
            '2025_used': False,
            '2026_used': False,
            'real_orders': False,
            'gpu_used': False,
        },
        'development_year': 2023,
        'candidate_count_2023': int(len(cand23)),
        'development_2023': sm23,
        'audit_2024_opened': False,
        'economic_edge_claimed': False,
    }
    if not sm23['pass']:
        result['status'] = 'RETIRED_NO_2023_GATE_PASS__2024_NOT_OPENED_FOR_R3R'
        return result

    tr23.to_csv(outdir / 'R3R_2023_FROZEN_TRADES.csv', index=False)
    cand24 = core.build_year_candidates(state, 2024)
    tr24 = core.simulate(cand24)
    sm24 = core.summarize(tr24)
    print('R3R_2024_FROZEN_AUDIT', 'candidates', len(cand24), json.dumps(sm24, default=str), flush=True)
    result['audit_2024_opened'] = True
    result['candidate_count_2024'] = int(len(cand24))
    result['audit_2024'] = sm24
    tr24.to_csv(outdir / 'R3R_2024_AUDIT_TRADES.csv', index=False)
    result['status'] = (
        'UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED'
        if sm24['pass'] else 'RETIRED_2024_AUDIT_FAIL'
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path('r3r_exact_cpu_work'))
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    prereg = json.loads((HERE / PREREG).read_text())
    if prereg.get('status') != 'FROZEN_BEFORE_ANY_R3R_OUTPUT':
        raise RuntimeError('FAIL_CLOSED: R3R preregistration not frozen')
    sig = prereg['signal']
    hard = prereg['hard_boundaries']
    if tuple(sig['checkpoints_ist']) != core.CHECKPOINTS:
        raise RuntimeError('FAIL_CLOSED: R3R checkpoints mismatch')
    if int(sig['adaptive_history'].split()[1]) != core.LOOKBACK_DATES:
        raise RuntimeError('FAIL_CLOSED: R3R history mismatch')
    if int(sig['minimum_history_dates']) != core.MIN_HISTORY:
        raise RuntimeError('FAIL_CLOSED: R3R minimum-history mismatch')
    if float(sig['vix_shock_quantile']) != core.VIX_SHOCK_Q or float(sig['vix_extreme_quantile']) != core.VIX_EXTREME_Q:
        raise RuntimeError('FAIL_CLOSED: R3R VIX threshold mismatch')
    if prereg['execution']['maximum_trades_per_date'] != core.MAX_TRADES_PER_DATE:
        raise RuntimeError('FAIL_CLOSED: R3R max-trades mismatch')
    if prereg['horizon_router']['if_abs_vix15_at_or_above_previous_history_q90'] != 60 or prereg['horizon_router']['otherwise'] != 30:
        raise RuntimeError('FAIL_CLOSED: R3R horizon router mismatch')
    if hard['teacher_v2_changed'] or hard['student_optimizer_steps'] != 0:
        raise RuntimeError('FAIL_CLOSED: R3R teacher/student boundary mismatch')
    if hard['gpu_allowed'] or hard['real_orders'] or hard['2025_used'] or hard['2026_used'] or hard['economic_edge_claimed']:
        raise RuntimeError('FAIL_CLOSED: R3R hard boundary mismatch')

    frames, audit = source.reconstruct_frames(args.work / 'exact_members')
    (args.work / 'R3R_EXACT_MEMBER_RECONSTRUCTION_AUDIT_V1.json').write_text(
        json.dumps({'status': 'PASS_ALL_108_BYTE_IDENTICAL', 'members': audit}, indent=2) + '\n'
    )
    result = evaluate(frames, args.work)
    (args.work / 'TEACHER_ECONOMIC_R3R_DEVELOPMENT_RESULT_V1.json').write_text(
        json.dumps(result, indent=2, default=str) + '\n'
    )
    print('R3R_FINAL_RESULT', json.dumps(result, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
