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
import r3k_runner as core

PREREG = 'TEACHER_ECONOMIC_R3K_OPENING_PERSISTENCE_CONFIRM_PREREG_V1.json'


def evaluate(frames: dict[str, pd.DataFrame], outdir: Path) -> dict:
    state = core.build_state(frames)
    cand23 = core.build_year_candidates(state, 2023)
    tr23 = core.simulate(cand23)
    sm23 = core.summarize(tr23)
    print('R3K_2023_FIXED_Q70', 'candidates', len(cand23), json.dumps(sm23, default=str), flush=True)

    result = {
        'format': 'VARDHANI_R3_TEACHER_ECONOMIC_R3K_DEVELOPMENT_RESULT_V1',
        'preregistration': PREREG,
        'source_authority': 'R3E_EXACT_MEMBER_RECONSTRUCTION_AUTHORITY_V1.json',
        'authorized_member_count': 108,
        'authorized_years': [2022, 2023, 2024],
        'all_members_byte_identical_to_locked_manifest': True,
        'method': {
            'initial_checkpoints_ist': list(core.INITIAL_CHECKPOINTS),
            'adaptive_history_dates_same_checkpoint': core.LOOKBACK_DATES,
            'minimum_history_dates': core.MIN_HISTORY,
            'fixed_q': core.FIXED_Q,
            'confirmation_delay_minutes': core.CONFIRM_DELAY_MIN,
            'confirmation': 'NIFTY ret5 same sign; SENSEX ret5 same sign; VIX ret5 opposite sign; session location preserved',
            'horizons': list(core.HORIZONS),
            'maximum_confirmed_trades_per_date': 1,
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
        'fixed_q': core.FIXED_Q,
        'candidate_count_2023': int(len(cand23)),
        'development_2023': sm23,
        'audit_2024_opened': False,
        'economic_edge_claimed': False,
    }

    if not sm23['pass']:
        result['status'] = 'RETIRED_NO_2023_GATE_PASS__2024_NOT_OPENED_FOR_R3K'
        return result

    # 2024 is touched only after the immutable 2023 gate has passed.
    tr23.to_csv(outdir / 'R3K_2023_FROZEN_TRADES.csv', index=False)
    cand24 = core.build_year_candidates(state, 2024)
    tr24 = core.simulate(cand24)
    sm24 = core.summarize(tr24)
    print('R3K_2024_FROZEN_AUDIT', 'candidates', len(cand24), json.dumps(sm24, default=str), flush=True)
    result['audit_2024_opened'] = True
    result['candidate_count_2024'] = int(len(cand24))
    result['audit_2024'] = sm24
    tr24.to_csv(outdir / 'R3K_2024_AUDIT_TRADES.csv', index=False)
    result['status'] = 'UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED' if sm24['pass'] else 'RETIRED_2024_AUDIT_FAIL'
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path('r3k_exact_cpu_work'))
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)

    p = json.loads((HERE / PREREG).read_text())
    if p.get('status') != 'FROZEN_BEFORE_ANY_R3K_OUTPUT':
        raise RuntimeError('FAIL_CLOSED: R3K preregistration not frozen')
    s = p['initial_signal']
    pc = p['persistence_confirmation']
    h = p['hard_boundaries']
    if tuple(s['checkpoints_ist']) != core.INITIAL_CHECKPOINTS:
        raise RuntimeError('FAIL_CLOSED: R3K initial checkpoints mismatch')
    if float(s['fixed_extremity_quantile']) != core.FIXED_Q:
        raise RuntimeError('FAIL_CLOSED: R3K fixed Q mismatch')
    if int(s['adaptive_history'].split()[1]) != core.LOOKBACK_DATES or int(s['minimum_history_dates']) != core.MIN_HISTORY:
        raise RuntimeError('FAIL_CLOSED: R3K history contract mismatch')
    if int(pc['delay_minutes']) != core.CONFIRM_DELAY_MIN or int(pc['maximum_confirmed_trades_per_date']) != 1:
        raise RuntimeError('FAIL_CLOSED: R3K persistence contract mismatch')
    if tuple(core.HORIZONS) != (60, 120):
        raise RuntimeError('FAIL_CLOSED: R3K horizon contract mismatch')
    if h['teacher_v2_changed'] or h['student_optimizer_steps'] != 0 or h['gpu_allowed'] or h['real_orders'] or h['2025_used'] or h['2026_used'] or h['economic_edge_claimed']:
        raise RuntimeError('FAIL_CLOSED: R3K hard boundary mismatch')

    frames, audit = source.reconstruct_frames(a.work / 'exact_members')
    (a.work / 'R3K_EXACT_MEMBER_RECONSTRUCTION_AUDIT_V1.json').write_text(
        json.dumps({'status':'PASS_ALL_108_BYTE_IDENTICAL','members':audit}, indent=2) + '\n'
    )
    result = evaluate(frames, a.work)
    (a.work / 'TEACHER_ECONOMIC_R3K_DEVELOPMENT_RESULT_V1.json').write_text(
        json.dumps(result, indent=2, default=str) + '\n'
    )
    print('R3K_FINAL_RESULT', json.dumps(result, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
