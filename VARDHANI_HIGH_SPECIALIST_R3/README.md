# VARDHANI HIGH-SPECIALIST R3 MARKET BRAIN

Authoritative R3 challenger source for NIFTY/SENSEX market intelligence. Frozen R2 remains untouched and real broker orders remain disabled.

## Governance
- Development: 2022-01-01 through 2024-12-31 only.
- Validation: 2025 read-only; no optimizer, normalization, target-stat, regime-stat, or checkpoint-selection fitting.
- 2026: sealed from historical R3 training and validation fitting.
- 15m is context only; it has no mechanical action-veto path.
- No fabricated D30/depth, bid/ask spread, microprice, Greeks, SENSEX options, index volume/OI, or full-market PCR from incomplete option data.

## Authoritative anchors
- V0.2 core ZIP SHA256: `16d99ab9df59662c9ee3ba80b7b353a20c7411c72f3ae6c74757f6480452c510`
- Raw archive SHA256: `bd7df469f6e7d95bee62a7c51d794a9119478cbc3c95b1e68debcafb4adc5b20`
- Teacher: 568,234,882 parameters, 12 specialists, off-device only.
- Android hard asset ceiling: 1 GiB.
- Student ladder: 66,144,640 → 167,884,160 → 244,322,944 parameters; use the smallest student passing every fidelity gate.

## Current execution state
The six-artifact Full-Market V2 governance layer, exact timestamp targets, 128/32 R3 teacher-input contract, development-only normalization/regime contract, 15m non-veto checks, genuine sparse NIFTY option masks, and future-mutation causality audits all pass.

Strict teacher preflight status: `PASS_READY_FOR_HEAVY_OPTIMIZER`.

Full teacher optimizer steps remain `0` because the current local runtime is CPU-only and below the minimum memory class for the 568M AdamW run. The CUDA trainer deliberately refuses CPU fallback. No economic edge or prospective promotion is claimed.

Large market-data `.npz` artifacts and teacher checkpoints are intentionally not committed to GitHub; their SHA-anchored manifests live with the execution artifacts.
