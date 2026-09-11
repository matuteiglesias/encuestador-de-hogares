# Current Development Directive — 2026-09-11

**Status:** active development overlay  
**Applies after:** `13_IMPLEMENTATION_STATUS_20260910.md`  
**Purpose:** tell a fresh implementation agent what to do next without re-running already-green seed milestones.

The original `00`–`12` documents remain the scientific/design contract. They should not be read as an instruction to replay the sprint from M0. The implementation has crossed the real-EPH gateway, and the exact CPV-2010 -> 2024 Census producer integration is now green. This file is the current pull order.

## Current baseline

Treat the following as already landed on `main` unless current code/tests prove otherwise:

- household-safe outer OOF primitives;
- HGB classifier/regressor adapters, including Gamma positive-amount support;
- direct and one-layer lean hurdle execution;
- real Q3-2024 EPH source intake and governed runs;
- direct/oracle/deployable comparison machinery;
- run bundles, CLI and real-EPH acceptance workflow;
- expanded welfare diagnostics: oracle-presence, positive-amount, low-tail, rank/transition, complete-vs-incomplete selection and fold stability;
- experimental nested empirical-residual predictive distribution with the hard rule that calibration residuals come only from inner household OOF predictions inside the outer-training partition.

Do not recreate these primitives under new names. Integrate or extend them only where current governed execution does not yet emit their evidence.

## Exact real inputs

The first bounded real integration is pinned to:

```text
EPH
  repository: matuteiglesias/microdatos-EPH-INDEC
  contract: publicdata.eph-microdata@1
  release: eph-2024-q3-3b6a7a15c4af

Census frame
  repository: matuteiglesias/samplerCensoARG
  contract: research.census-frame/v1
  release: arg-cpv2010-frame-ee6ada167c2d6429
  vintage: 2010
  status: producer validation green

Census sample
  repository: matuteiglesias/samplerCensoARG
  contract: research.census-target-year-sample/v2
  release: census-sample-2024-0839713eafea8d1b
  target year: 2024
  materialization: full-payload
  households: 141863
  persons: 469172
  dwellings: 141621
  checker: PASS
```

The sampling universe is governed explicitly. Department `94021` is the sole donor-only exclusion (17 donor persons), with zero reassignment and no mutation of the source/frame. `analysis_weight` and generic model weights are unset; inverse-probability quantities remain audit/design metadata only.

See `contracts/real_input_binding_2024q3_census2024.yaml`.

These identities are parents, not hints. Do not silently substitute another quarter, sample, v1 sampler artifact, locally reconstructed sample, or sibling-repository path.

## Current dependency order

### D1 — Keep governance truthful

- keep `SYSTEM.yaml` synchronized with the real runtime and exact pinned inputs;
- keep sampler v2 as the only active Census intake contract;
- remove or quarantine transitional v1 runtime/test/CI paths rather than maintaining two active meanings of "Census sample";
- keep legacy files as archaeology only when useful for reconstruction.

Exit: a fresh agent cannot plausibly infer that `research.census-target-year-sample/v1` is an active supported input.

### D2 — Integrate the landed science-diagnostics slice

The diagnostics module existing is not sufficient. Governed P0/P1 run evidence must actually expose the diagnostics needed for adjudication.

Required evidence, where scientifically applicable:

- ordinary person and household point risk;
- presence diagnostics;
- positive-amount-head diagnostics;
- explicitly non-deployable oracle-presence / target-status upper-bound diagnostics;
- low-tail CDF bias;
- household rank association;
- quintile/decile transition diagnostics;
- complete-vs-incomplete household selection counts and declared evaluation cohort;
- person/household fold stability.

Do not let diagnostic-only oracle information enter deployable training or scoring.

Exit: one run/comparison bundle can answer the above without notebook-side recomputation.

### D3 — Keep predictive distribution experimental

`predictive_distribution.py` is an experimental primitive, not a production uncertainty framework.

Hard boundary:

```text
outer train
   -> inner household OOF predictions
   -> calibration residuals
   -> fit outer point model on outer train
outer test
   -> evaluation only
```

Forbidden:

- calibration residuals from outer-test households;
- global OOF residual reuse across an outer test boundary;
- claiming independent person residual draws form a coherent household posterior;
- allowing this primitive to block P0/P1 point-prediction qualification.

### D4 — Sampler-v2 intake is a consumer gate, not a production blocker

The exact Census producer release is already materialized and producer-validated. Do not rebuild or resample CPV-2010 as part of encuestador development.

The active consumer is `research.census-target-year-sample/v2`. The encuestador intake gate must validate, when the local release is available to the run:

- exact release identity `census-sample-2024-0839713eafea8d1b`;
- exact parent frame `arg-cpv2010-frame-ee6ada167c2d6429`, donor vintage 2010 and target year 2024;
- household selection unit and person target-mass semantics;
- complete membership assertion;
- artifact hashes;
- selection/design metadata kept distinct from model or poverty weights;
- full-payload presence for `persona`, `hogar`, `vivienda`.

Producer evidence already records PASS for the complete sampler checker plus independent identity/FK/membership checks. Consumer validation is a custody check, not permission to score.

### D5 — Real 23-concept semantic review in `eph-censo-aligner`

This is the active bottleneck. Review the exact historical 23-concept bridge against the pinned EPH release and the validated CPV-2010 full-payload sample:

```text
IX_TOT
P02 P03 CONDACT
V01
H05 H06 H07 H08 H09 H10 H11 H12 H16 H15 PROP H14 H13
P07 P08 P09 P10 P05
```

The review is intentionally broader than the first approved model plane. Review all 23 so ambiguity is visible, but approve only source-backed concepts justified by the active P0/P1 design.

The real Census payload already confirms structural availability:

- `persona.parquet`: `P02 P03 P05 P07 P08 P09 P10 CONDACT`;
- `hogar.parquet`: `H05 H06 H07 H08 H09 H10 H11 H12 H13 H14 H15 H16 PROP`;
- `vivienda.parquet`: `V01`;
- `IX_TOT`: deterministic from complete household membership / household-person relation, not a raw field in the sample.

`AGLO_rk` and `Reg_rk` remain forbidden external predictors. No geography-derived substitute should be invented for this first bounded plane.

For every concept record:

1. exact EPH source field(s) in `eph-2024-q3-3b6a7a15c4af`;
2. exact Census source table/field(s) in `census-sample-2024-0839713eafea8d1b` or its pinned frame parent;
3. question/definition evidence where available;
4. universe/reference-period differences;
5. observed category support on both exact releases;
6. directional recode/derivation;
7. information loss/ambiguity;
8. semantic class;
9. reviewer status;
10. whether encuestador may use it as an external predictor for the target welfare period.

Semantic comparability does not decide temporal admissibility. A concept may be semantically approved but still require `donor_vintage_proxy`, `target_period_latent`, another transport-time role, or exclusion in encuestador.

Exit: a versioned, fail-closed approved feature plane exists for the exact releases.

### D6 — P0 vs approved-P1 Gamma on EPH

Only after the approved real semantic plane exists, run the decisive EPH comparison before any Census scoring.

Definitions:

- **P0**: direct hurdle-Gamma using only the approved external/common feature plane.
- **P1**: the smallest approved one-layer latent extension justified by the reviewed plane; no historical stage survives merely because it existed in RFC1–RFC4.

Freeze between P0 and P1:

- exact EPH release and cohort;
- terminal welfare target and Gamma formulation;
- outer household folds;
- feature semantics available to both arms;
- estimator policy except where the architecture itself requires the latent head;
- weighting policy (`none` under the current bounded policy unless explicitly changed through a separate scientific decision).

Adjudicate on governed evidence, not one headline MAE. At minimum inspect terminal person/household risk, low tail, rank/transitions, fold stability, selection cohort, latent reconstruction/calibration and oracle/deployable gap.

P1 is promoted only if deployable evidence improves the terminal welfare objective enough to justify its extra learned state. Oracle gain alone is not promotion evidence.

### D7 — Census scoring only after D5 + D6

Do not score the Census sample merely because its production and intake are green.

Required parents/gates:

```text
exact sampler-v2 release valid
+ exact semantic feature plane approved
+ transport-time roles resolved
+ P0/P1 EPH adjudication complete
+ promoted model frozen
+ scoring rows preserve exact sample_person_id/sample_household_id
```

Then execute one bounded scoring pass and emit the household-welfare handoff with exact lineage.

## Current bottleneck

The current bottleneck is **real semantic approval of the exact EPH/CPV feature plane**, followed immediately by the P0/P1 Gamma decision. Census production, sampler architecture, additional estimator families, deeper cascades, anchors, broad uncertainty machinery and generalized multi-vintage frameworks are subordinate to that path.

## Agent behavior

A fresh agent should:

1. inspect `SYSTEM.yaml`, this directive, `13_IMPLEMENTATION_STATUS_20260910.md`, and the exact current code before proposing work;
2. reuse landed primitives instead of creating parallel implementations;
3. keep changes in dependency-sized slices with attributable tests;
4. report an exact blocker when an upstream artifact is unavailable rather than fabricating substitute data;
5. stop before downstream Census scoring if any semantic or scientific gate remains unresolved.
