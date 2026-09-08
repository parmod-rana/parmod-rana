# GLOBAL CRYPTO HUNTER — Microstructure Intelligence Checkpoint

Frozen checkpoint time: 2026-09-08 05:01 UTC

## Authority

- Project: `GLOBAL_CRYPTO_HUNTER_DYNAMIC_SURVIVAL_BRAIN`
- Active shadow remains **Gen3C / R1F-GEN3C-TEMPORAL-ENSEMBLE / SHADOW_PAPER_ONLY**.
- Microstructure research has **no authority** to modify or replace Gen3C.
- Real-money authority: **FALSE**.
- Gen3D, Gen3E, Gen3F_V2 and Gen3G remain rejected; do not retune them from their observed results.

## Frozen 4-symbol experience bank — COMPLETE

- Genuine OKX historical module-6 SPOT 50-level tick-by-tick order books.
- Symbols: BAT-USDT, ZRX-USDT, ATOM-USDT, BCH-USDT.
- 32 fixed dates per symbol = 128 symbol-days.
- Resolution: completed causal 15-minute states.
- Rows: 12,288 total = 9,216 teacher + 3,072 historical evaluation.
- Full bank successful merge-recovery run: `34187443705`.
- Artifact ID: `10040995477`.
- Artifact ZIP SHA-256: `26319c2f8e67dba6fe2b52f1277831e72943876a931eaf653309b2f02b2bc90b`.
- State SHA-256:
  - all: `c3651b318c7a52ff3ffd7a1e4249e76c052aaa275e16a2100f3f87b791f5cffd`
  - teacher: `74d76d3250186ec3b20a0ff578297994cafd4ab9b07b82806b1a1f1559c732c2`
  - historical evaluation: `d6467309c6312f4a7481bba5117762e9b323ba1237ac30771921decee1230a55`
- Integrity: 128/128 pairs, unique chronology, teacher/evaluation disjoint, finite-mid 1.0, no forward returns/model/trade during bank construction.
- The exact bank artifact is permanently committed under `frozen_experience/`.

## Microstructure Teacher V1.1 — COMPLETE AND PRESERVED

- Frozen source commit: `a76b6234478f406b7d3bb80bec0e9795489d035c`.
- Teacher module blob: `69188ed4c1d3932907ca7d0e249c5a1f581c8659`.
- Causal validation run: `34186770762` — PASS, 5/5 tests.
- Teacher execution run: `34187582373` — SUCCESS.
- Teacher result artifact ID: `10041050467`.
- Artifact ZIP SHA-256: `741e33409ab47e7e9a04203813fed34f6842af504a2855647cc2165df4a07530`.
- `teacher_result.json` SHA-256: `3248e9801bcef7df954a557f3081cd99f5b70e5f71ee38f9de1edb297a1b945b`.
- Stable relationship CSV SHA-256: `ab3aee2065485f783db46c283976696153c30c18a01269e64b1e830b1c5d5478`.
- Exact teacher artifact has been permanently preserved through workflow run `34188484866`.

### V1.1 accepted candidate knowledge

Seven feature/horizon relationships passed the frozen teacher cross-regime + cross-symbol criteria:

1. 60m `depth_notional_10_last` — NEGATIVE.
2. 60m `top_ofi_norm_mean` — NEGATIVE.
3. 240m `top_ofi_norm_std` — POSITIVE.
4. 240m `qty_imb_50_std` — POSITIVE.
5. 240m `top_ofi_abs_max` — POSITIVE.
6. 240m `depth_notional_10_last` — NEGATIVE.
7. 240m `qty_imb_20_mean` — NEGATIVE.

Five teacher trust candidates were frozen for breadth confirmation:

- 15m `spread_bps_mean` — HIGH_IS_BAD.
- 15m `depth_notional_50_mean` — LOW_IS_BAD.
- 15m `capture_latency_ms_mean` — HIGH_IS_BAD.
- 60m `spread_bps_mean` — HIGH_IS_BAD.
- 240m `depth_notional_50_mean` — LOW_IS_BAD.

No fixed model family passed teacher stability. OFI as a complete incremental block did not pass. Therefore V1.1 provides **candidate transferable state/trust knowledge, not a profitable trading model**.

Authoritative accepted knowledge file:
`crypto_survival_training/MICROSTRUCTURE_ACCEPTED_KNOWLEDGE_V1.json`

Semantic constraint file:
`crypto_survival_training/MICROSTRUCTURE_KNOWLEDGE_SEMANTICS_V1.json`

Key semantic separation frozen before breadth result:
- normalized OFI/imbalance variables = possible market-state information;
- spread and absolute depth = liquidity/trust context unless separately proven directional;
- `capture_latency_ms_mean` = measurement/data-quality only, never current directional alpha;
- absolute depth is cross-symbol scale-sensitive and cannot be normalized retroactively inside the current test.

## 20-symbol historical transfer confirmation — ACTIVE DATA BUILD

Study: `MICROSTRUCTURE_TRANSFER_BREADTH_CONFIRMATION_V1`.

Frozen non-teacher symbols: BTC, ETH, SOL, XRP, ADA, DOGE, LINK, AVAX, DOT, LTC, TRX, UNI, AAVE, NEAR, ETC, FIL, CRV, SUSHI, ALGO, XLM versus USDT.

Frozen dates: 2025-07-15, 2025-08-15, 2025-11-15, 2025-12-15, 2026-03-15, 2026-04-15, 2026-06-15, 2026-07-15.

- Required pairs: 20 × 8 = **160**.
- Data-only workflow run: `34188108465`.
- Frozen data source commit: `72dd299a245e43da9ba5b81d738e81d9b3fd5694`.
- Builder blob: `e17d22dd22f348db3f6d3adad921f771dedb4960`.
- Merger blob: `01d42d949d4a5ff86f2ce39dfd83e36862a0ef6e`.
- Missing-pair policy: fail closed; never substitute a symbol/date and never impute.
- At checkpoint time, **40/160 compact pair artifacts had completed**. This is a progress snapshot only. Do **not** infer the run completed from this file; always query run `34188108465` for the current state.

Audited example: BTC-USDT 2026-06-15 processed 4,562,529 genuine TBT updates into exactly 96 completed 15-minute states; finite-mid fraction 1.0; pair state SHA `c1f44baa4e7d95aa51630e03053862bd1b1f51f2c3dce65726cf98135c42e3b3`.

## Exact-grid gate — VALIDATED, MANDATORY BEFORE EVALUATION

The data build itself predates an additional integrity audit discovered before evaluation exposure. Therefore the merged bank must separately prove exactly 96 states per pair and 15,360 rows total.

- Auditor: `crypto_survival_training/audit_microstructure_transfer_bank_grid.py`.
- Validated source commit: `4662c1782e4b420b8d46be280ec067b143a746de`.
- Auditor blob: `03c9a2b55e3a462230ecbce5da47d5d6947e34a8`.
- Validation run: `34188910425` — PASS.
- Synthetic tests prove:
  - exact 20×8×96 grid passes;
  - one missing state fails;
  - one shifted timestamp fails.
- Mandatory execution workflow exists but remains dormant until the 160-pair merge artifact ID and states SHA are known.

## Frozen breadth evaluator — VALIDATED, DORMANT

- Evaluator: `crypto_survival_training/run_microstructure_transfer_confirmation.py`.
- Validated source commit: `b54ea322c35ebef8638c501bb0798900e37a3146`.
- Evaluator blob: `5c2dcc179f377c4dd7af86ad05cfb334153cdc86`.
- Validation run: `34188424674` — PASS.
- Relationship gate: >=12/20 symbols same teacher sign + median abs Spearman >=0.02 + >=3/4 frozen regimes same sign.
- Trust gate: >=12/20 symbols and >=3/4 folds with bad/good forecast-error ratio >=1.10 using immutable teacher thresholds.
- No feature, symbol, sign, threshold or model reselection permitted.
- Evaluation execution workflow exists but remains dormant.
- **Do not create `MICROSTRUCTURE_BREADTH_EVALUATE.json` until the 160-pair merge and mandatory exact-grid audit both pass.**

## Fresh-forward evidence — PRE-REGISTERED BEFORE BREADTH RESULT

Files:
- `MICROSTRUCTURE_FRESH_FORWARD_PROTOCOL_V1.json`
- `MICROSTRUCTURE_FRESH_FORWARD_HOLDOUT_V1.json`

Frozen future holdout dates:
- P1: 2026-09-15, 2026-10-01
- P2: 2026-10-15, 2026-11-01
- P3: 2026-11-15, 2026-12-01
- P4: 2026-12-15, 2027-01-01

Only historically breadth-confirmed knowledge may later be qualified on these dates. No failed historical candidate can be resurrected from future outcomes. No date/symbol replacement or threshold retuning is allowed.

Current OKX exact-live-feed note frozen in protocol: `books50-l2-tbt` is the preferred same-domain live channel; current OKX documentation requires login/VIP4 and uses `seqId/prevSeqId` continuity. The checksum field must not be used for integrity after its June 2026 deprecation. A slower public `books` feed is explicitly a degraded domain and cannot directly qualify 10ms-TBT historical knowledge.

## Mandatory continuation sequence

1. Query run `34188108465`; continue the exact 160-pair build without changing symbols/dates/features.
2. If any pair fails, repair execution mechanics only and recover that exact frozen pair; do not replace it.
3. Fail-closed merge all 160 pairs.
4. Record merged artifact ID, exact states SHA-256 and row count.
5. Run the validated exact-grid auditor; require 160 pairs × 96 states = 15,360 exact-grid states.
6. Only after grid PASS, create one-time frozen confirmation authorization and execute the validated evaluator once.
7. Preserve the exact confirmation result artifact.
8. Accept only candidates that pass the already frozen breadth gates; failed candidates stay failed.
9. Do not define Gen3H from post-hoc result tweaking. Any future integration architecture must be written and frozen after knowledge acceptance but before its own economic evaluation.
10. Fresh-forward holdout qualification remains mandatory before any future integration can gain authority.

No economic edge is claimed at this checkpoint. The project is deliberately building a more experienced and better-calibrated market mind without manufacturing profitability.
