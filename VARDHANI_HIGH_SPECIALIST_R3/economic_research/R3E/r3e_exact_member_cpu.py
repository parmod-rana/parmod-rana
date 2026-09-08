from __future__ import annotations

import argparse, calendar, hashlib, json, time, urllib.request
from pathlib import Path

import pandas as pd

import r3e_runner as core
from r3e_authorized_source_manifest import ROWS

KEYS={
 'NIFTY':'NSE_INDEX%7CNifty%2050',
 'SENSEX':'BSE_INDEX%7CSENSEX',
 'VIX':'NSE_INDEX%7CIndia%20VIX',
}
PREFIX={
 'NIFTY':'underlying/nifty-50/minutes-1',
 'SENSEX':'underlying/sensex/minutes-1',
 'VIX':'underlying/india-vix/minutes-1',
}


def exact_url(market:int|str, year:int, month:int)->str:
    last=calendar.monthrange(year,month)[1]
    key=KEYS[str(market)]
    return f'https://api.upstox.com/v3/historical-candle/{key}/minutes/1/{year}-{month:02d}-{last:02d}/{year}-{month:02d}-01'


def exact_name(market:str,year:int,month:int)->str:
    last=calendar.monthrange(year,month)[1]
    return f'{PREFIX[market]}/{year}-{month:02d}-01--{year}-{month:02d}-{last:02d}.json'


def digest(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def fetch_locked(url:str,want_bytes:int,want_sha:str,retries:int=4)->tuple[bytes,str]:
    last=None
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'VARDHANI-R3E-exact-member-reconstruction/1'})
            with urllib.request.urlopen(req,timeout=90) as r:
                raw=r.read()
            for label,b in [('raw',raw),('raw_plus_lf',raw+b'\n'),('raw_plus_crlf',raw+b'\r\n')]:
                if len(b)==want_bytes and digest(b)==want_sha:
                    return b,label
            raise RuntimeError(f'BYTE_IDENTITY_MISMATCH got_bytes={len(raw)} got_sha={digest(raw)} expected_bytes={want_bytes} expected_sha={want_sha}')
        except Exception as e:
            last=e
            if attempt+1<retries: time.sleep(2**attempt)
    raise RuntimeError(f'FAIL_CLOSED exact source retrieval failed: {url}: {last}')


def reconstruct_frames(cache:Path)->tuple[dict[str,pd.DataFrame],list[dict]]:
    cache.mkdir(parents=True,exist_ok=True)
    if len(ROWS)!=108 or {y for _,y,_,_,_ in ROWS}!={2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED authorized source manifest boundary invalid')
    buckets={k:[] for k in KEYS}; audit=[]
    for idx,(market,year,month,want_bytes,want_sha) in enumerate(ROWS,1):
        if year not in core.ALLOWED_CONTENT_YEARS:
            raise RuntimeError('FAIL_CLOSED forbidden year in authorized source manifest')
        name=exact_name(market,year,month); p=cache/name
        p.parent.mkdir(parents=True,exist_ok=True)
        mode='cache_exact'
        if p.exists() and p.stat().st_size==want_bytes and core.sha256_file(p)==want_sha:
            raw=p.read_bytes()
        else:
            raw,mode=fetch_locked(exact_url(market,year,month),want_bytes,want_sha)
            tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_bytes(raw); tmp.replace(p)
        if len(raw)!=want_bytes or digest(raw)!=want_sha:
            raise RuntimeError(f'FAIL_CLOSED postwrite identity mismatch {name}')
        frame=core.parse_candles(raw,market)
        if frame.empty:
            raise RuntimeError(f'FAIL_CLOSED empty authorized member {name}')
        if not set(frame.timestamp.dt.year.unique()).issubset({year}):
            raise RuntimeError(f'FAIL_CLOSED cross-year body {name}')
        buckets[market].append(frame)
        audit.append({'i':idx,'market':market,'year':year,'month':month,'path':name,'bytes':want_bytes,'sha256':want_sha,'transport':mode,'rows':int(len(frame))})
        print(f'EXACT_MEMBER_PASS {idx}/108 {market} {year}-{month:02d} rows={len(frame)} transport={mode}',flush=True)
    out={}
    for market,parts in buckets.items():
        x=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp',keep='last').sort_values('timestamp').reset_index(drop=True)
        if set(x.timestamp.dt.year.unique())!={2022,2023,2024}:
            raise RuntimeError(f'FAIL_CLOSED unexpected years in reconstructed {market}')
        out[market]=x
    return out,audit


def evaluate(frames:dict[str,pd.DataFrame],outdir:Path)->dict:
    state=core.add_targets(core.build_state(frames))
    if len(core.FEATURES)!=42 or any(c not in state.columns for c in core.FEATURES):
        raise RuntimeError('FAIL_CLOSED exact 42-feature state unavailable')
    if set(state.year.unique())!={2022,2023,2024}:
        raise RuntimeError('FAIL_CLOSED state year boundary invalid')
    cand23=core.build_year_candidates(state,2023)
    grid={}; passing=[]; trade_by_q={}
    for q in core.Q_GRID:
        tr=core.simulate(cand23,q); sm=core.summarize(tr); grid[str(q)]=sm; trade_by_q[q]=tr
        print('R3E_2023_Q',q,json.dumps(sm,default=str),flush=True)
        if sm['pass']: passing.append(q)
    selected=None
    if passing:
        selected=sorted(passing,key=lambda q:(core.worst_half_mean(grid[str(q)]),grid[str(q)]['net3']['mean'],grid[str(q)]['net5']['mean'],q),reverse=True)[0]
    result={
      'format':'VARDHANI_R3_TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1',
      'source_authority':'R3E_EXACT_MEMBER_RECONSTRUCTION_AUTHORITY_V1.json',
      'authorized_member_count':108,
      'authorized_years':[2022,2023,2024],
      'all_members_byte_identical_to_locked_manifest':True,
      'feature_count':42,'feature_names':core.FEATURES,
      'boundary':{'teacher_v2_changed':False,'student_optimizer_steps':0,'2025_used':False,'2026_used':False,'real_orders':False,'gpu_used':False},
      'calibration_year':2023,'calibration_grid':grid,'selected_q':selected,
      'audit_2024_opened':False,'economic_edge_claimed':False,
    }
    if selected is None:
        result['status']='RETIRED_NO_2023_CALIBRATION_PASS__2024_NOT_OPENED_FOR_R3E'
    else:
        cand24=core.build_year_candidates(state,2024)
        tr24=core.simulate(cand24,selected); sm24=core.summarize(tr24)
        result['audit_2024_opened']=True; result['audit_2024']=sm24
        result['status']='UNDERLYING_2023_2024_PASS__OPTION_GATE_REQUIRED' if sm24['pass'] else 'RETIRED_2024_AUDIT_FAIL'
        trade_by_q[selected].to_csv(outdir/'R3E_2023_SELECTED_TRADES.csv',index=False)
        tr24.to_csv(outdir/'R3E_2024_AUDIT_TRADES.csv',index=False)
    return result


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--work',type=Path,default=Path('r3e_exact_cpu_work'))
    a=ap.parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    frames,audit=reconstruct_frames(a.work/'exact_members')
    (a.work/'R3E_EXACT_MEMBER_RECONSTRUCTION_AUDIT_V1.json').write_text(json.dumps({'status':'PASS_ALL_108_BYTE_IDENTICAL','members':audit},indent=2)+'\n')
    result=evaluate(frames,a.work)
    (a.work/'TEACHER_ECONOMIC_R3E_DEVELOPMENT_RESULT_V1.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('R3E_FINAL_RESULT',json.dumps(result,default=str),flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
