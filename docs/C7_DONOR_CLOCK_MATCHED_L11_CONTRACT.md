# C7 frozen real-panel input contract (R0)

Status: **cloud contract/fixture repair only**. C7A must still implement the
real resource-safe panel plane; C7B is local commissioning. This contract is
not a claim that L11 has already been fitted on real EPH and is not a second
person-linkage system.

## Authority and selection

Inputs must bind exact immutable artifacts, without modifying them:

- L2 `research.eph-longitudinal-analysis-frame/v1`: manifest.json, persons.csv,
  panel_links.csv and qa.json;
- Baseline Gate-B `research.encuestador-gate-b-panel-evidence/v1` with selection
  `l2_consistent_1q3q_age14_reviewed_labor_v1` and `exclude_exceptional=false`;
- Accepted C6 P1R model plane with `P1R_NOLAB_LONG` composition;
- Target-quarter national/regional official labor context, separately binding
  official L1 and explicit L1B completion parents when required.

The implemented R0 helper is `longitudinal_c7_contract.C7PairProjector`.
It calls the SAME Gate-B `_select_reason` on L2-audited links and prefetched
original person rows. It does not reconstruct component matching, demographic
consistency, period arithmetic, or reviewed labor classifications.
The projector tracks target-row uniqueness across ALL link batches, not just
within a chunk. It preserves Gate-B's exclusive exclusion ordering.

**Reconciliation boundary before extra income filtering:**

- Original candidate links: 1,172,374 on frozen real baseline.
- Gate-B eligible: 827,793 = 571,984 (1Q) + 255,809 (3Q).
- Gate-B excluded: 344,581.
- Distinct eligible later observations: 827,793.

These are local real Gate-B receipt values, NOT fixture findings. C7A must
fail closed if baseline counts, first-exclusion reasons, source manifest/file
hashes, policy ID, eligible target identities or gap counts differ from
the bound Gate-B release. Do not publish a row-level panel dataset. Private
C7 modeling artifacts may hold an exactly audited pair-to-row index.

Only THEN apply an additional valid later P47T_real restriction:
`p47t_value_status=positive` with finite value strictly above zero, or
`p47t_value_status=zero` with finite value exactly zero. Preserve genuine
zeros; reject missing/negative/special income. Failure of this welfare filter
does NOT retroactively change Gate-B transition eligibility. Earlier-wave
income validity is not an L11 selection requirement.

The common monetary reference and candidate-conversion-parent caveat survive.

## Clock, group and private pair-row schema

    pair_id                         stale_observation_row_id + "->" + target_observation_row_id
    stale_observation_row_id        exact earlier C2 row ID
    target_observation_row_id       exact later C2 row ID
    person_linkage_candidate_id    L2 candidate key; NOT a permanent person identifier
    panel_household_id             same at both observations
    stale_period                   observed earlier quarter
    target_period                  observed later quarter
    elapsed_quarters               exactly 1 or 3
    stale_labor_state              earlier reviewed EPH class 1/2/3
    target_current_labor_state     later observed label; NEVER terminal input
    target_real_income             valid later P47T_real including zero
    target_region_id               later geography for official context
    target_monetary_reference_period common real-income reference

The model's canonical P1R composition comes from the earlier
`stale_observation_row_id`, NOT from the later person row. Target-quarter
national/regional labor context, target-period time and elapsed gap are shared
by both matched arms. Later observed individual labor is reserved as an
evaluation label (or future L12 transition-training target), never an L11 or
L12 terminal external feature. Target-quarter individual composition does
not enter the primary donor-faithful model.

Freeze one pooled 1Q/3Q matched comparison:

- **C7-0:** earlier P1R composition + elapsed gap + target official
  national/regional labor context + target-period explicit time;
- **C7-1 (L11):** exactly C7-0 plus actually observed earlier labor state;
- both: identical panel-supported valid-target rows, group/fold vector,
  hurdle family, hyperparameters and nested household-safe OOF time layer.

C7-0 is a **panel-matched historical-input L10 analogue**, NOT identical to
Gate-A's full-population contemporary-X L10.

Use `panel_household_id` as group fold. Any repeated person candidate's pairs
must remain in that household group across all observations. Match the C6
deterministic household hash/fold policy when sharing group keys. Persist pair
and source row identity hashes, fold hash, original C6 release/hash, and the
extra income-exclusion ledger.

**Households:** eligible linked persons are a selected subset of the later
household. Never call the sum of their observed/predicted incomes full
household income without reviewed complete-membership verification.
Person-level comparison is primary; selected-panel-group metrics must be
unambiguously labeled.

## Interpretation and follow-on limits

C7 tests the incremental held-out welfare value of earlier observed EPH labor
over supported 1Q/3Q gaps. Report paired fold deltas and strata by gap and
earlier E/U/I. Do not fit separate per-gap models or tune many estimators
before the primary comparison.

Neither C7 nor Gate B proves individual labor transport from CPV-2010 to a
2024/2026 target, or CPV-2022 readiness. Historical September Q8 used donor
Census-2010 CONDACT in a same-clock EPH-2024-trained model: keep Q8 as a
research sensitivity, not an L11/L12 donor-clock validation.

Later C8 (L12) requires training-fold-only transition baselines, properly
nested multiclass OOF inputs, fixed E/U/I ordering, preserved categorical
feature handling and a distinct transition/welfare acceptance gate. KL
anchoring, broad tuning and Census scoring remain out of C7 scope.

## Repair proof and local handoff

Cloud R0 fixture proof: `tests/test_longitudinal_c7_contract.py`.
R0 has not run on private local real/Census data.

To close R0 locally:
1. Pull a clean main containing this contract/test.
2. Inspect source Gate-B `gate_b_receipt.json` within
   `gate-b-panel-evidence-c15bac5f1a6cbeb7`, plus local
   `GATE_B_REAL_RECEIPT.json` and immutable L2 `manifest.json`.
3. Confirm source IDs/hashes and real eligible-by-gap counts reconcile with
   the numbers above; reject drift, do not regenerate L2/Gate B.
4. Write a small `R0_C7_CONTRACT_ACCEPTANCE.md` documenting provenance,
   earlier-input clock and readiness for the subsequent C7A implementation.
5. Do NOT fit L11/L12 or score Census during R0.

Real C7 implementation and execution require separately accepted C7A/C7B.
