# R3 GitHub migration integrity note

The binary object currently stored at `releases/VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT.zip` is **NOT AUTHORITATIVE** because its GitHub blob size does not match the verified local artifact. Do not train from or redistribute that repository binary.

Authoritative verified source artifact:
- file: `VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT.zip`
- bytes: 50629
- SHA256: `0080e6436c77f127923aac97a4573d6daf0a121748a382e467843e99dce0a6b6`

The authoritative text-file inventory and per-file SHA256 values are in `SOURCE_MANIFEST.json`.

Current strict preflight: `PASS_READY_FOR_HEAVY_OPTIMIZER`.
Full teacher optimizer steps: `0`.
Frozen R2 modified: `false`.
Real broker orders: `disabled`.

Heavy teacher execution requires CUDA-class infrastructure; CPU fallback is intentionally blocked.
