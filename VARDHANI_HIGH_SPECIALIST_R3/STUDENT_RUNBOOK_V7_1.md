# VARDHANI R3 — Student Learning Runbook V7.1

This runbook is operational only. It does not modify methodology, teacher curriculum, student architecture, training corpus, chronology, loss weights, optimizer hyperparameters, the 15m authority rule, the 2025 boundary, or the 2026 seal.

## Frozen authority

- Teacher: GPT-5.6 Sol (ChatGPT)
- Student: VARDHANI Master AI Brain
- Teacher curriculum: `VARDHANI_R3_GPT56SOL_TEACHER_CURRICULUM_FREEZE_V1`
- Student parameters: 1,457,069,501
- Development sequences: 530,877
- Gradient accumulation: 8
- Expected optimizer steps: 66,360
- Student optimizer steps before launch: 0
- Development years: 2022, 2023, 2024 only
- 2025: absent from the development bundle and unopened during development training
- 2026: sealed
- Real orders: disabled

## Authoritative hashes

- Development bundle V1.1 SHA256: `e50c5d95c28bfa9747ffc3598ab794d773ef7801e8fde3f15722ee73a26ecba5`
- Source V7.1 SHA256: `4d3a07d4d13ab74a1099d45bbe8afa2f84b44bdf3c18e0b4489a5c8562a47cdc`
- Execution gate V1.1 SHA256: `82bb6eb97785973b3aa45a8b8ae350ca11d49f4b0220436b62845defe6c60dee`

## Required transport files

Place these five files in one directory:

- `VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1_1.zip.part00`
- `VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1_1.zip.part01`
- `VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1_1.zip.part02`
- `VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1_1.zip.part03`
- `VARDHANI_R3_STUDENT_DEVELOPMENT_BUNDLE_V1_1.zip.part04`

Also place `VARDHANI_R3_STUDENT_GPU_EXECUTION_GATE_V1_1.sh` in a convenient location.

## GPU prerequisites

Use one CUDA-capable GPU environment with persistent storage. Before launch, verify:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')
print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

Do not continue if CUDA is unavailable.

## One-command frozen launch

From the directory containing the five bundle parts:

```bash
bash /path/to/VARDHANI_R3_STUDENT_GPU_EXECUTION_GATE_V1_1.sh "$PWD" "$HOME/vardhani_r3_student_v7_1"
```

The execution gate must:

1. Require all five parts.
2. Reassemble the bundle.
3. Verify SHA256 `e50c5d95c28bfa9747ffc3598ab794d773ef7801e8fde3f15722ee73a26ecba5`.
4. Pass ZIP integrity.
5. Verify the development-only manifest.
6. Prove training years are exactly 2022–2024.
7. Prove 2025 payloads are absent.
8. Prove 2026 remains sealed.
9. Verify CUDA and refuse CPU fallback.
10. Run the frozen 31/31 student-learning preflight.
11. Run exactly one chronological development pass.
12. Preserve exact gradient accumulation, including the final 5-microbatch tail.
13. Preserve Python/NumPy/Torch CPU/CUDA RNG state in crash-recovery checkpoints.
14. Freeze the final development checkpoint.
15. Require exactly 66,360 optimizer steps.
16. Verify the final checkpoint SHA.
17. Produce the post-run acceptance gate.

## Fail-closed conditions

Stop immediately if any of these occur:

- bundle or part hash mismatch
- ZIP integrity failure
- CUDA unavailable
- CPU fallback attempted
- teacher identity differs from GPT-5.6 Sol
- student identity differs from VARDHANI Master AI Brain
- parameter count differs from 1,457,069,501
- training years differ from 2022–2024
- any 2025 development payload is present or opened
- any 2026 payload is opened
- chronology is shuffled or rewound
- loss weights or optimizer hyperparameters differ from preregistration
- final optimizer-step count differs from 66,360
- final checkpoint SHA does not verify
- real broker orders are enabled

## Crash recovery

Resume only from a crash-recovery checkpoint created by this frozen run. Resume must preserve chronological progress and restore Python, NumPy, Torch CPU and CUDA RNG states. Do not manually skip, repeat, reshuffle, or retune sequences after an interruption.

## Successful development-run acceptance

A run is accepted only if the generated post-run gate reports:

- `PASS_STUDENT_DEVELOPMENT_FROZEN_2025_NOT_OPENED`
- optimizer steps = 66,360
- development sequences = 530,877
- one training pass
- 2025 opened = false
- 2025 optimizer steps = 0
- checkpoint selected on 2025 = false
- 2026 opened = false
- real orders enabled = false
- economic promotion = false
- final checkpoint SHA verified

## After successful development freeze

Do not train again and do not retune the development model. The next phase is the separate read-only 2025 examination using the frozen final development checkpoint. The 2025 result may pass or fail; it must not be used to alter the frozen checkpoint, thresholds, normalization, teacher curriculum, architecture, or optimizer recipe.

No profitability claim follows from successful training or implementation integrity alone.