# R3 GitHub migration integrity note

The earlier repository object at `releases/VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT.zip` is **rejected / non-authoritative** because its GitHub blob size did not match the verified local artifact. Do not train from or redistribute that object.

Current authoritative source artifact:
- file: `VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT_V3.zip`
- SHA256: `fe52081eec7fa5cedfc798b76d2c1a7ff7eeb52003e38f26d21df2ba7fbb526e`
- contains corrected chronology guard, hash guards, deferred-2025 guards, trainer tests, and isolated `r3_training_core.py` so the heavy trainer does not import legacy positional-target code.

Current authoritative heavy execution artifact:
- file: `VARDHANI_R3_HEAVY_TRAINING_BUNDLE_V3.zip`
- SHA256: `289319c2c3aa60079722e6c9c274f53c9d516231c10738a01eb92d9cc62f3d13`
- local dry-run result: expected `CUDA_HEAVY_RUNTIME_REQUIRED_NO_CPU_FALLBACK`
- optimizer steps created by dry-run: `0`

Current strict preflight: `PASS_READY_FOR_HEAVY_OPTIMIZER`.
Frozen R2 modified: `false`.
Real broker orders: `disabled`.
Economic edge claimed: `false`.
Prospective authority: `NONE`.

Full teacher execution requires CUDA-class infrastructure. The current chat runtime has no usable Runpod API key/CLI credential, so no paid GPU resource was created and no optimizer work was fabricated.
