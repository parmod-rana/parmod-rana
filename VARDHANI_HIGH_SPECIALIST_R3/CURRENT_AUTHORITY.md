# Current authoritative VARDHANI High-Specialist R3 source

## Binary snapshot authority
The exact verified V3 source snapshot is:

`VARDHANI_HIGH_SPECIALIST_R3_SOURCE_SNAPSHOT_V3.zip`

SHA256: `29d043ea274da94416d26c0c09e2b6b62f905747f06134afc4d83139ab2340a0`
Expected size: `80,725 bytes`.

**Do not use the ZIP currently stored under GitHub `releases/` as an authoritative binary.** Post-write verification showed the connector-stored blob is only `15,008 bytes`; it is transport-truncated and therefore fails the exact-byte authority contract. It remains present only as evidence of the failed transport attempt.

The exact V3 artifact retained outside that truncated GitHub blob is the authority until a file-native GitHub upload path can preserve all 80,725 bytes.

## V3 reason
V3 supersedes V2 because V2 packaging omitted transitive Python runtime modules and failed the exact heavy-bundle import dry-run. V3 contains the complete `vardhani_brain` Python module set. Architecture, dataset hashes, chronology, preregistered optimizer schedule, 15m non-veto, and evidence standards are unchanged.

Strict teacher preflight: `PASS_READY_FOR_HEAVY_OPTIMIZER`.
Teacher optimizer steps: `0`.
Frozen R2 modified: `false`.
Real broker orders: `DISABLED`.
2025 fitting/checkpoint selection: `FORBIDDEN`.
2026 historical exposure: `FORBIDDEN`.
Economic/prospective promotion: `NONE`.
