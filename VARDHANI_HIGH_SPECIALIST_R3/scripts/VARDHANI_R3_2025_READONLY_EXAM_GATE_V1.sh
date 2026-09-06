#!/usr/bin/env bash
set -euo pipefail

VALIDATION_ROOT="${1:?validation root required}"
WORKTREE="${2:?V7.1 source/worktree required}"
DEV_CHECKPOINT="${3:?frozen development checkpoint required}"
DEV_REPORT="${4:?frozen development training report required}"
OUT_JSON="${5:?2025 exam output path required}"
EXPECTED_PARAMS=1457069501
EXPECTED_STEPS=66360

python - <<'PY' "$DEV_CHECKPOINT" "$DEV_REPORT" "$EXPECTED_STEPS"
import sys, json, hashlib
from pathlib import Path
ck=Path(sys.argv[1]); rp=Path(sys.argv[2]); expected_steps=int(sys.argv[3])
if not ck.exists(): raise SystemExit("FROZEN_DEVELOPMENT_CHECKPOINT_MISSING")
if not rp.exists(): raise SystemExit("FROZEN_DEVELOPMENT_REPORT_MISSING")
r=json.load(open(rp))
if r.get("status")!="DEVELOPMENT_TRAINED_AND_FROZEN_2025_NOT_OPENED": raise SystemExit("FROZEN_DEVELOPMENT_REPORT_REQUIRED")
if r.get("optimizer_steps")!=expected_steps: raise SystemExit(f"DEVELOPMENT_OPTIMIZER_STEP_COUNT_INVALID:{r.get('optimizer_steps')}")
if r.get("2025_opened") is not False: raise SystemExit("DEVELOPMENT_BOUNDARY_VIOLATION_2025_ALREADY_OPENED")
if r.get("2025_optimizer_steps")!=0: raise SystemExit("DEVELOPMENT_BOUNDARY_VIOLATION_2025_OPTIMIZER_STEPS")
if r.get("checkpoint_selected_on_2025") is not False: raise SystemExit("DEVELOPMENT_BOUNDARY_VIOLATION_CHECKPOINT_SELECTION")
if r.get("2026_opened") is not False: raise SystemExit("SEALED_2026_BOUNDARY_VIOLATION")
h=hashlib.sha256()
with ck.open("rb") as f:
    for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
if r.get("final_checkpoint_sha256")!=h.hexdigest(): raise SystemExit("FROZEN_DEVELOPMENT_CHECKPOINT_SHA_MISMATCH")
print("FROZEN_DEVELOPMENT_CHECKPOINT_GATE_PASS")
PY

python - <<'PY' "$VALIDATION_ROOT"
import sys, hashlib
from pathlib import Path
root=Path(sys.argv[1])
expected={
"R3_FULL_MARKET_V2/teacher_inputs/nifty_validation_teacher_input.npz":"5e05107e1de66713772c32b29960e0e65fa2e6b8c20ff2fe455701c7e3035f83",
"R3_FULL_MARKET_V2/teacher_inputs/sensex_validation_teacher_input.npz":"4245aed9ea796599f10d1ad20bb2e643c14ef90b571232d9744ee0e1d47f3b86",
"R3_FULL_MARKET_V2/targets/nifty_validation_targets.npz":"ca2e6479b4ce084a39f4e373f62d741532902a7c0a142fa3563e25d46b278979",
"R3_FULL_MARKET_V2/targets/sensex_validation_targets.npz":"753287d3567aa651171405cfbf719c8df9e48c9b15c5d06e94499a37ef8bb705",
}
for rel,want in expected.items():
    p=root/rel
    if not p.exists(): raise SystemExit(f"VALIDATION_ASSET_MISSING:{rel}")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    got=h.hexdigest()
    if got!=want: raise SystemExit(f"VALIDATION_SHA_MISMATCH:{rel}:{got}")
print("REGISTERED_2025_VALIDATION_HASHES_PASS")
PY

command -v nvidia-smi >/dev/null 2>&1 || { echo "NVIDIA_SMI_NOT_FOUND"; exit 12; }
nvidia-smi

PYTHONPATH="$WORKTREE" python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA_EVALUATION_RUNTIME_REQUIRED")
print("gpu",torch.cuda.get_device_name(0))
print("bf16_supported",torch.cuda.is_bf16_supported())
PY

python - <<'PY' "$WORKTREE"
import sys
from pathlib import Path
p=Path(sys.argv[1])/"vardhani_brain/r3_evaluate_student_2025.py"
s=p.read_text()
assert "torch.optim" not in s
assert "model.eval()" in s
assert "torch.no_grad()" in s
assert "FROZEN_DEVELOPMENT_REPORT_REQUIRED" in s
print("READONLY_EVALUATOR_STATIC_GUARD_PASS")
PY

PYTHONPATH="$WORKTREE" python -m vardhani_brain.r3_evaluate_student_2025 \
  --validation-root "$VALIDATION_ROOT" \
  --worktree "$WORKTREE" \
  --development-checkpoint "$DEV_CHECKPOINT" \
  --development-report "$DEV_REPORT" \
  --out "$OUT_JSON"

python - <<'PY' "$OUT_JSON" "$DEV_CHECKPOINT"
import sys, json, hashlib
from pathlib import Path
op=Path(sys.argv[1]); ck=Path(sys.argv[2])
if not op.exists(): raise SystemExit("2025_EXAM_REPORT_MISSING")
r=json.load(open(op))
assert r["format"]=="VARDHANI_R3_2025_READONLY_EXAM_V1"
assert r["status"]=="2025_READONLY_EXAM_COMPLETE_NO_FIT"
assert r["optimizer_created"] is False
assert r["optimizer_steps_2025"]==0
assert r["checkpoint_selection_on_2025"] is False
assert r["2026_opened"] is False
assert r["economic_promotion"] is False
assert r["prospective_authority"]=="NONE"
h=hashlib.sha256()
with ck.open("rb") as f:
    for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
assert r["checkpoint_sha256"]==h.hexdigest()
print("VARDHANI_2025_READONLY_EXAM_ACCEPTED")
PY
