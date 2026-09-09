from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent
R3E_DIR=HERE.parent/'R3E'
if str(R3E_DIR) not in sys.path:
    sys.path.insert(0,str(R3E_DIR))
import r3e_exact_member_cpu as source
import r3l_runner as core

PREREG='TEACHER_ECONOMIC_R3L_DUAL_PATH_OPENING_ROUTER_PREREG_V1.json'


def evaluate(frames:dict[str,pd.DataFrame],outdir:Path)->dict:
    state=core.build_state(frames)
    cand23=core.build_year_candidates(state,2023)
    tr23=core.simulate(cand23)
    sm23=core.summarize(tr23)
    print('R3L_2023_FIXED', 'candidates',len(cand23),json.dumps(sm23,default=str),flush=True)
    result={
      'format':'VARDHANI_R3_TEACHER_ECONOMIC_R3L_DEVELOPMENT_RESULT_V1',
      'preregistration':PREREG,
      'source_authority':'R3E_EXACT_MEMBER_RECONSTRUCTION_AUTHORITY_V1.json',
      'authorized_member_count':108,'authorized_years':[2022,2023,2024],
      'all_members_byte_identical_to_locked_manifest':True,
      'method':{
        'checkpoints_ist':list(core.CHECKPOINTS),'adaptive_history_dates_same_checkpoint':core.LOOKBACK_DATES,
        'minimum_history_dates':core.MIN_HISTORY,'dislocation_quantile':core.DISLOCATION_Q,'wick_quantile':core.WICK_Q,
        'specialists':['CONT','REJECT'],'one_trade_at_a_time':True
      },
      'boundary':{'teacher_v2_changed':False,'student_optimizer_steps':0,'2025_used':False,'2026_used':False,'real_orders':False,'gpu_used':False},
      'development_year':2023,'candidate_count_2023':int(len(cand23)),'development_2023':sm23,
      'audit_2024_opened':False,'economic_edge_claimed':False
    }
    if not sm23['pass']:
        result['status']='RETIRED_NO_2023_GATE_PASS__2024_NOT_OPENED_FOR_R3L'
        return result
    tr23.to_csv(outdir/'R3L_2023_FROZEN_TRADES.csv',index=False)
    cand24=core.build_year_candidates(state,2024); tr24=core.simulate(cand24); sm24=core.summarize(tr24)
    print('R3L_2024_FROZEN_AUDIT','candidates',len(cand24),json.dumps(sm24,default=str),flush=True)
    result['audit_2024_opened']=True; result['candidate_count_2024']=int(len(cand24)); result['audit_2024']=sm24
    tr24.to_csv(outdir/'R3L_2024_AUDIT_TRADES.csv',index=False)
    result['status']='UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED' if sm24['pass'] else 'RETIRED_2024_AUDIT_FAIL'
    return result


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--work',type=Path,default=Path('r3l_exact_cpu_work'))
    a=ap.parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    p=json.loads((HERE/PREREG).read_text())
    if p.get('status')!='FROZEN_BEFORE_ANY_R3L_OUTPUT': raise RuntimeError('FAIL_CLOSED: R3L preregistration not frozen')
    s=p['common_state']; h=p['hard_boundaries']
    if tuple(s['checkpoints_ist'])!=core.CHECKPOINTS: raise RuntimeError('FAIL_CLOSED: R3L checkpoint mismatch')
    if int(s['adaptive_history'].split()[1])!=core.LOOKBACK_DATES or int(s['minimum_history_dates'])!=core.MIN_HISTORY: raise RuntimeError('FAIL_CLOSED: R3L history mismatch')
    if float(s['dislocation_quantile'])!=core.DISLOCATION_Q: raise RuntimeError('FAIL_CLOSED: R3L dislocation threshold mismatch')
    if h['teacher_v2_changed'] or h['student_optimizer_steps']!=0 or h['gpu_allowed'] or h['real_orders'] or h['2025_used'] or h['2026_used'] or h['economic_edge_claimed']:
        raise RuntimeError('FAIL_CLOSED: R3L hard boundary mismatch')
    frames,audit=source.reconstruct_frames(a.work/'exact_members')
    (a.work/'R3L_EXACT_MEMBER_RECONSTRUCTION_AUDIT_V1.json').write_text(json.dumps({'status':'PASS_ALL_108_BYTE_IDENTICAL','members':audit},indent=2)+'\n')
    result=evaluate(frames,a.work)
    (a.work/'TEACHER_ECONOMIC_R3L_DEVELOPMENT_RESULT_V1.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('R3L_FINAL_RESULT',json.dumps(result,default=str),flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
