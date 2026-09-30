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
- The first full real L10 attempt terminated under practical memory pressure before emitting a run bundle because the runtime materializes the full C2 person and composition surfaces as Python lists.
- Current status is `BLOCKED_RESOURCE`; no L10/L11/L12 scientific result bundle exists.
- Local receipt: `/home/matias/data/L4_RECEIPT.json`.
