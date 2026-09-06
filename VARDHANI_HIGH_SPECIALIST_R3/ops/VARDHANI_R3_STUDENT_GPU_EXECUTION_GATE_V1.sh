#!/usr/bin/env bash
set -euo pipefail

# VARDHANI R3 — frozen student-learning execution gate.
# This wrapper DOES NOT alter the frozen teacher curriculum, student architecture,
# optimizer hyperparameters, chronology, or validation boundaries.

PART_DIR="${1:-$PWD}"
RUN_ROOT="${2:-$HOME/vardhani_r3_student_v7}"
BUNDLE="$RUN_ROOT/VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip"
EXPECTED_SHA="64d3beea1198ef2c19f67b05c6b7627e53a9bf23d7d0693391ce74a5877dfcdc"
EXPECTED_SEQUENCES=530877
EXPECTED_OPT_STEPS=66360

mkdir -p "$RUN_ROOT"
rm -f "$BUNDLE"

parts=(
  "VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip.part00"
  "VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip.part01"
  "VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip.part02"
  "VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip.part03"
  "VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1.zip.part04"
)

for p in "${parts[@]}"; do
  test -f "$PART_DIR/$p" || { echo "MISSING_PART:$PART_DIR/$p"; exit 10; }
done

cat "${parts[@]/#/$PART_DIR/}" > "$BUNDLE"
actual="$(sha256sum "$BUNDLE" | awk '{print $1}')"
test "$actual" = "$EXPECTED_SHA" || { echo "BUNDLE_SHA_MISMATCH:$actual"; exit 11; }

python -m zipfile -t "$BUNDLE"

EXTRACT="$RUN_ROOT/extracted"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
python -m zipfile -e "$BUNDLE" "$EXTRACT"

python - <<'PY' "$EXTRACT" "$EXPECTED_SEQUENCES"
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
expected=int(sys.argv[2])
m=json.load(open(root/"VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_MANIFEST_V1.json"))
assert m["status"]=="READY_FOR_STUDENT_OPTIMIZER_DEVELOPMENT_ONLY"
assert m["teacher"]=="GPT-5.6 Sol (ChatGPT)"
assert m["student"]=="VARDHANI Master AI Brain"
assert m["development_sequences"]==expected
assert m["years_present_for_training"]==[2022,2023,2024]
assert m["validation_2025_data_present"] is False
assert m["validation_named_payloads"]==[]
assert m["sealed_2026_opened"] is False
assert m["student_optimizer_steps"]==0
print("DEVELOPMENT_BUNDLE_BOUNDARY_PASS")
PY

command -v nvidia-smi >/dev/null 2>&1 || { echo "NVIDIA_SMI_NOT_FOUND"; exit 12; }
nvidia-smi

PYTHONPATH="$EXTRACT/source" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA_STUDENT_RUNTIME_REQUIRED_NO_CPU_FALLBACK")
print("gpu", torch.cuda.get_device_name(0))
print("bf16_supported", torch.cuda.is_bf16_supported())
PY

bash "$EXTRACT/source/scripts/run_student_development.sh" \
  "$EXTRACT" "$EXTRACT/student_output" \
  2>&1 | tee "$RUN_ROOT/student_training_console.log"

python - <<'PY' "$EXTRACT" "$EXPECTED_SEQUENCES" "$EXPECTED_OPT_STEPS"
import json, sys, hashlib
from pathlib import Path
root=Path(sys.argv[1])
expected_seq=int(sys.argv[2])
expected_steps=int(sys.argv[3])
out=root/"student_output"
rp=out/"VARDHANI_R3_STUDENT_DEVELOPMENT_TRAINING_REPORT_V1.json"
assert rp.exists(), "FINAL_TRAINING_REPORT_MISSING"
r=json.load(open(rp))
assert r["status"]=="DEVELOPMENT_TRAINED_AND_FROZEN_2025_NOT_OPENED"
assert r["teacher"]=="GPT-5.6 Sol (ChatGPT)"
assert r["student"]=="VARDHANI Master AI Brain"
assert r["parameter_report"]["parameters"]==1457069501
assert r["training_passes"]==1
assert r["development_sequences"]==expected_seq
assert r["optimizer_steps"]==expected_steps, (r["optimizer_steps"], expected_steps)
assert r["2025_opened"] is False
assert r["2025_optimizer_steps"]==0
assert r["checkpoint_selected_on_2025"] is False
assert r["2026_opened"] is False
assert r["real_orders_enabled"] is False
assert r["economic_promotion"] is False
ck=out/r["final_checkpoint"]
assert ck.exists(), "FINAL_CHECKPOINT_MISSING"
h=hashlib.sha256()
with ck.open("rb") as f:
    for b in iter(lambda:f.read(8*1024*1024),b""):
        h.update(b)
actual=h.hexdigest()
assert actual==r["final_checkpoint_sha256"], (actual,r["final_checkpoint_sha256"])
gate={
    "status":"PASS_STUDENT_DEVELOPMENT_FROZEN_2025_NOT_OPENED",
    "optimizer_steps":r["optimizer_steps"],
    "development_sequences":r["development_sequences"],
    "final_checkpoint":r["final_checkpoint"],
    "final_checkpoint_sha256":actual,
    "2025_opened":False,
    "2026_opened":False,
    "economic_edge_claimed":False
}
gp=out/"VARDHANI_R3_STUDENT_POSTRUN_ACCEPTANCE_GATE_V1.json"
gp.write_text(json.dumps(gate,indent=2)+"\n")
print(json.dumps(gate,indent=2))
PY

echo "VARDHANI_STUDENT_DEVELOPMENT_RUN_ACCEPTED"
