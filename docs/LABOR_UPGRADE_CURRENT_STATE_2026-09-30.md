# Labor upgrade current state — 2026-09-30

This is a factual coordination receipt. It records materialized inputs and
published implementation state; it does not claim approved-mode welfare science.

## C1/L1

- Official release: `indec-eph-labor-state-52ca6bcb586f2b0b`.
- Official coverage: 1060/1064 required cells.
- The four unavailable official cells are NEA (`noreste`), 2019-Q3, across the four principal indicators.
- Local receipt/artifact root: `/home/matias/data/labor-state-releases/`.

## L1B

- Centerline bounded backward-fill overlay: `indec-eph-labor-context-completion-backward_fill-2212ec7a74aa7f16`.
- Sensitivity bounded forward-fill overlay: `indec-eph-labor-context-completion-forward_fill-71229ee29569dc24`.
- Official parent remains immutable and incomplete; overlays are separate derived contracts.

## L2

- Release: `eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39`.
- Coverage: 37 exact quarters, 2017-Q1 through 2026-Q1; 1,869,620 person rows.
- Panel audit headline: 1,172,374 candidate links; 1,073,713 demographically consistent; 87,642 conflicts; 10,976 component-key-only; 182,852 repeated-household candidates.
- Monetary centerline is the transparent quarter midpoint (Q1 February, Q2 May, Q3 August, Q4 November).
- The conversion parent remains candidate; bounded commissioning is allowed, approved-mode scientific freeze is not claimed.

## L3A

- Donor release: `eph-cpv2010-semantic-plane-2024q3-v2`.
- 469,172 rows; donor vintage 2010.
- This is an explicit donor-vintage handoff and carries no transport authorization.
- Local artifact root: `/home/matias/data/l3a-cpv2010-donor-labor-2024q3`.

## L3B

- `P0_LONG`: `eph-longitudinal-composition-p0_long-7b8fdc0ec1f2a553`.
- `P1R_NOLAB_LONG`: `eph-longitudinal-composition-p1r_nolab_long-ed30aa112c9b7d31`.
- Both contain 1,869,620 rows over all 37 periods, preserve exact C2 identity, exclude current individual labor state, and have zero unresolved unexpected/impossible code cells.
- Historical special handling is governed by `longitudinal_specials_v1.json`; recognized special values become feature-level canonical nulls without dropping C2 rows.
- Receipt: `/home/matias/data/l3b-longitudinal-composition-2017-2026/L3B_RECEIPT.json`.

## L4

- C4/C4B runtime and explicit L1B intake are published; canonical real L10 configs are available at `configs/longitudinal/l10_real_p0.yaml` and `configs/longitudinal/l10_real_p1r.yaml`.
- Preflight contracts and tests are green.
- The first full real L10 attempt on the pre-C6 runtime terminated under practical memory pressure before emitting a run bundle because that path materialized the full C2 person and composition surfaces as Python lists.
- C6 resource-safe execution code is now merged on `main` at `ca3fe0fbf93d89a85a1d10b4498ff28366a890b2`. Hosted CI proved lint/compile, synthetic restartability, numerical parity with the C4B L10 path, existing longitudinal regressions, and the full repository suite.
- C6 introduces one streamed C2↔C5 model-plane materialization, typed memory-mapped arrays, persistent deterministic fold IDs, sequential outer-fold execution and atomic restartable checkpoints.
- C6 real acceptance is **PASS**: the P1R model plane has 1,869,620 rows, is 459 MiB on disk, used 0.64 GiB peak RSS to build and about 2.03 GiB peak RSS to fit, with zero swap events and restartable five-fold execution.
- Gate A is **COMPLETE_BOUNDED_COMMISSIONING**. Matched P1R arms show national-only context is immaterial while national-plus-regional deviations improve held-out welfare error.
- Gate B is **COMPLETE_DESCRIPTIVE_EVIDENCE**: 827,793 eligible pairs (571,984 at 1Q; 255,809 at 3Q). Receipt: `/home/matias/data/l4-gate-b-20261001/GATE_B_REAL_RECEIPT.json`.
- C6/Gate-A artifacts and receipts: `/home/matias/data/l4-c6-20260930/`.


## Frozen baseline

Unless new evidence specifically breaks one of these contracts, the next execution work
should treat the following as settled:

1. observed EPH measurement window is `2017-Q1..2026-Q1`;
2. `empleoARG` owns current official aggregate labor context; official observations and completion overlays remain distinct artifacts;
3. `eph-censo-aligner` owns canonical longitudinal composition and EPH↔Census semantic recodes;
4. `P1R_NOLAB_LONG` is the richer centerline composition candidate and `P0_LONG` is the matched baseline;
5. L10 never consumes current person `CONDACT`;
6. donor labor has an explicit observation clock/vintage;
7. repeated EPH evidence supports short-gap labor research, not CPV-2010→current identification;
8. explicit time corrections use nested household-safe OOF evidence;
9. 2020-Q2 and 2024-Q1/Q2 remain measured but do not estimate ordinary year/quarter structure;
10. KL projection is an L12-only sensitivity under an explicit compatible universe;
11. longitudinal runtime is measurement-mode, not forecast/nowcast;
12. reviewed historical survey-special values may become feature-level canonical nulls without dropping C2 rows.

## Next development frontier

Gate B closes the descriptive short-gap panel question without promoting a model.
The next work package is a matched L11 experiment on the exact Gate-B-supported
rows and grouped folds. L12 remains pending justification/design and must use
OOF transition probabilities; neither result may be extrapolated to the
CPV-2010 donor clock. The monetary conversion parent remains candidate.
