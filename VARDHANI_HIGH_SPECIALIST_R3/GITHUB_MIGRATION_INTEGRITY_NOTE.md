# R3 GitHub migration integrity note

The earlier repository object at `releases/VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT.zip` is **rejected / non-authoritative** because its GitHub blob size did not match the verified local artifact. It has been removed. Do not train from or redistribute that object.

Current authoritative source artifact:
- file: `VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT_V3.zip`
- bytes: `80725`
- SHA256: `29d043ea274da94416d26c0c09e2b6b62f905747f06134afc4d83139ab2340a0`
- ZIP integrity: PASS
- internal manifest verification: 56/56 files match declared size and SHA256
- trainer SHA256: `e43a4f424391b44a892aa53f09aa74149a37bc65fae382a94eb27ebc02f91f87`
- isolated training-core SHA256: `59b33d0ba98a7e01d7c1562f9d270b64a623bce879602921d62e6291b8b791aa`

Current authoritative heavy execution artifact:
- file: `VARDHANI_R3_HEAVY_TRAINING_BUNDLE_V3.zip`
- bytes: `270582645`
- SHA256: `289319c2c3aa60079722e6c9c274f53c9d516231c10738a01eb92d9cc62f3d13`
- ZIP integrity: PASS
- embedded trainer/training-core hashes match the authoritative worktree
- local dry-run result: expected `CUDA_HEAVY_RUNTIME_REQUIRED_NO_CPU_FALLBACK`
- optimizer steps created by dry-run: `0`

Current strict preflight: `PASS_READY_FOR_HEAVY_OPTIMIZER`.
Frozen R2 modified: `false`.
2025 optimizer/fitting: `0 / none`.
2026 training exposure: `false`.
Real broker orders: `disabled`.
Economic edge claimed: `false`.
Prospective authority: `NONE`.

Full teacher execution requires CUDA-class infrastructure. Runpod is connected at the ChatGPT product level, but this chat runtime still exposes neither Runpod infrastructure actions nor a local `RUNPOD_API_KEY`/`~/.runpod/config.toml`, so no paid GPU resource was created and no optimizer work was fabricated.
