"""Frozen full R3 ~568M teacher training entrypoint for an off-device CUDA runtime.

Hard rules:
- preflight must already be PASS_READY_FOR_HEAVY_OPTIMIZER;
- exactly one chronological 2022-2024 pass, NIFTY/SENSEX endpoints merged by timestamp;
- no shuffle, no 2025 optimizer steps, no early stopping or checkpoint selection on 2025;
- 2025 evaluation starts only after the final development checkpoint is frozen;
- 2026 is never opened;
- this trainer cannot enable broker orders or modify frozen R2.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from .config import BrainConfig
from .model import MarketBrain,parameter_report
from .r3_dataset import R3DualChronologicalDataset
from .r3_training_core import teacher_loss,seed_all

SEED=20260905
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=8
LR=1.0e-4
WEIGHT_DECAY=1.0e-2
MAX_GRAD_NORM=1.0
CHECKPOINT_EVERY_OPT_STEPS=5000
EXPECTED_PARAMS=568_234_882


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def evaluate(model,ds,device):
 dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);model.eval();losses=[];correct=[0,0];total=[0,0]
 with torch.no_grad():
  for b in dl:
   idx=int(b.pop('target_index_id').item());b.pop('endpoint_ts_utc_ns',None);b={k:v.to(device,non_blocking=True) for k,v in b.items()}
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):o=model(b['lower'],b['ctx15']);loss=teacher_loss(o,b)
   losses.append(float(loss));p=int(o['action_logits'].argmax(-1).item());correct[idx]+=int(p==int(b['action'].item()));total[idx]+=1
 return {'loss_mean':float(np.mean(losses)) if losses else None,'NIFTY_direction_accuracy':correct[0]/max(total[0],1),'NIFTY_rows':total[0],'SENSEX_direction_accuracy':correct[1]/max(total[1],1),'SENSEX_rows':total[1]}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--worktree',required=True);ap.add_argument('--preflight',required=True);ap.add_argument('--preregistration',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 root=Path(a.root);work=Path(a.worktree);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 pf=json.load(open(a.preflight));pre=json.load(open(a.preregistration))
 if pf.get('status')!='PASS_READY_FOR_HEAVY_OPTIMIZER':raise SystemExit('STRICT_R3_PREFLIGHT_REQUIRED')
 if pre.get('status')!='FROZEN_PRE_OPTIMIZER' or pre.get('optimizer_steps_at_freeze')!=0:raise SystemExit('FROZEN_TRAINING_PREREGISTRATION_REQUIRED')
 b=root/'R3_FULL_MARKET_V2';contract=b/'train_contract/R3_TRAIN_ONLY_CONTRACT.npz'
 # Verify frozen development authority before the first optimizer step. 2025 files are
 # intentionally NOT opened or hashed here; their hashes are checked only after the
 # final development checkpoint is frozen.
 expected_dev={
  work/'configs/teacher_research.json':pre['teacher_config_sha256'],
  Path(a.preflight):pre['strict_preflight_sha256'],
  contract:pre['train_only_contract_sha256'],
  b/'teacher_inputs/nifty_train_teacher_input.npz':pre['development_inputs']['nifty'],
  b/'teacher_inputs/sensex_train_teacher_input.npz':pre['development_inputs']['sensex'],
  b/'targets/nifty_train_targets.npz':pre['development_targets']['nifty'],
  b/'targets/sensex_train_targets.npz':pre['development_targets']['sensex'],
 }
 for path,expected in expected_dev.items():
  if sha(path)!=expected:raise SystemExit(f'FROZEN_DEVELOPMENT_SHA_MISMATCH:{path}')
 if not torch.cuda.is_available():raise SystemExit('CUDA_HEAVY_RUNTIME_REQUIRED_NO_CPU_FALLBACK')
 seed_all(SEED);device=torch.device('cuda')
 tr=R3DualChronologicalDataset(b/'teacher_inputs/nifty_train_teacher_input.npz',b/'targets/nifty_train_targets.npz',b/'teacher_inputs/sensex_train_teacher_input.npz',b/'targets/sensex_train_targets.npz',contract,seq_len=384,step=1)
 # Validation is instantiated only after development training completes; no validation metric controls training.
 cfg=BrainConfig.load(work/'configs/teacher_research.json');model=MarketBrain(cfg).to(device)
 pr=parameter_report(model)
 if pr['parameters']!=EXPECTED_PARAMS:raise RuntimeError('TEACHER_PARAMETER_COUNT_CHANGED')
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
 use_bf16=torch.cuda.is_bf16_supported();scaler=torch.amp.GradScaler('cuda',enabled=not use_bf16)
 dl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)
 model.train();opt.zero_grad(set_to_none=True);t0=time.time();opt_steps=0;micro=0;running=[];last_ts=-1
 for bidx,batch in enumerate(dl,1):
  ts=int(batch.pop('endpoint_ts_utc_ns').item());batch.pop('target_index_id',None)
  if ts<last_ts:raise RuntimeError('TRAINING_CHRONOLOGY_REWIND')
  last_ts=ts
  batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16 if use_bf16 else torch.float16):
   o=model(batch['lower'],batch['ctx15']);loss=teacher_loss(o,batch)/GRAD_ACCUM
  scaler.scale(loss).backward();micro+=1;running.append(float(loss.detach())*GRAD_ACCUM)
  if micro==GRAD_ACCUM or bidx==len(dl):
   scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);micro=0;opt_steps+=1
   if opt_steps%CHECKPOINT_EVERY_OPT_STEPS==0:
    torch.save({'state_dict':model.state_dict(),'config':cfg.__dict__,'optimizer_step':opt_steps,'endpoint_ts_utc_ns':last_ts,'classification':'CRASH_RECOVERY_ONLY_NOT_MODEL_SELECTION'},out/f'recovery_step_{opt_steps}.pt')
 # Freeze final development checkpoint BEFORE opening 2025 validation artifacts.
 final=out/'teacher_final_development.pt';torch.save({'state_dict':model.state_dict(),'config':cfg.__dict__,'optimizer_steps':opt_steps,'training_passes':1,'last_development_endpoint_ts_utc_ns':last_ts},final);final_sha=sha(final)
 # Only now may the 2025 validation files be opened. First verify their preregistered
 # hashes, then instantiate the read-only validation dataset.
 expected_val={
  b/'teacher_inputs/nifty_validation_teacher_input.npz':pre['validation_inputs']['nifty'],
  b/'teacher_inputs/sensex_validation_teacher_input.npz':pre['validation_inputs']['sensex'],
  b/'targets/nifty_validation_targets.npz':pre['validation_targets']['nifty'],
  b/'targets/sensex_validation_targets.npz':pre['validation_targets']['sensex'],
 }
 for path,expected in expected_val.items():
  if sha(path)!=expected:raise RuntimeError(f'FROZEN_VALIDATION_SHA_MISMATCH:{path}')
 va=R3DualChronologicalDataset(b/'teacher_inputs/nifty_validation_teacher_input.npz',b/'targets/nifty_validation_targets.npz',b/'teacher_inputs/sensex_validation_teacher_input.npz',b/'targets/sensex_validation_targets.npz',contract,seq_len=384,step=1)
 metrics=evaluate(model,va,device)
 report={'format':'VARDHANI_R3_FULL_TEACHER_TRAINING_REPORT_V1','status':'TRAINED_FINAL_DEVELOPMENT_CHECKPOINT_THEN_2025_READ_ONLY_EVALUATED','parameter_report':pr,'training_passes':1,'optimizer_steps':opt_steps,'development_sequences':len(tr),'2025_validation_sequences':len(va),'2025_optimizer_steps':0,'checkpoint_selected_on_2025':False,'2026_opened':False,'final_checkpoint':final.name,'final_checkpoint_sha256':final_sha,'train_loss_mean':float(np.mean(running)) if running else None,'validation_2025':metrics,'elapsed_seconds':time.time()-t0,'frozen_r2_modified':False,'economic_promotion':False,'prospective_authority':'NONE'}
 json.dump(report,open(out/'R3_FULL_TEACHER_TRAINING_REPORT_V1.json','w'),indent=2);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
