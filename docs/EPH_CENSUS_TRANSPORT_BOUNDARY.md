# EPH -> Census transport boundary

## Decision

This repository is being revived as the **survey-to-Census welfare inference instrument**.

The historical project mixed four concerns that are now governed elsewhere:

1. EPH preprocessing;
2. EPH/Census semantic alignment;
3. Census sampling/projection;
4. staged statistical inference of Census-side characteristics and income.

Only the fourth concern is the durable scientific kernel of this repository.

The intended future responsibility is therefore:

```text
approved EPH analysis artifact
        +
approved EPH/Census semantic feature contract
        +
exact Census sample / population-frame identity
        +
reviewed monetary-reference utilities
                |
                v
      staged transport model
                |
                v
      Census welfare release
                |
                v
            Poverty v2
```

The current branch records this boundary and the candidate deployment DAG. It does **not** approve a real EPH/Census inference run or reactivate the old cron/training machinery.

## What was scientifically valuable in the historical system

The historical training code used four waves:

```text
Census-observable / Census-shaped predictors
        -> RFC1: CAT_OCUP, CAT_INAC, CH07
        -> RFC2: income-presence indicators
        -> RFC3: detailed labour variables
        -> RFC4: monetary outcomes including P47T
```

That is a real scientific idea: when the final EPH income model needs variables not observed in Census, learn those variables from the common information plane and propagate the learned outputs downstream.

The old implementation, however, trained later waves on the **observed** upstream labels. At Census scoring time those same variables are only predictions. A revived implementation must therefore train every downstream stage on out-of-fold predictions from prior stages. This is a hard scientific invariant, not an optimization detail.

The historical four-wave grouping is preserved as evidence, not frozen as the future architecture. The revived model should be expressed as a dependency DAG and may split, merge or remove legacy waves after real validation.

## Crosswalk evidence recovered

The historical preprocessing and the current `eph-censo-aligner` agree on the following candidate naming plane. All mappings remain **pending real-vintage methodological review**; code resemblance is not semantic approval.

| EPH source | Census-facing concept | Raw Census location | Important rule/status |
|---|---|---|---|
| `IX_TOT` | `IX_TOT` | candidate derivation from `HOGAR.TOTPERS` | legacy clips to 0..8; raw mapping still must be proven |
| `CH04` | `P02` | `PERSONA.P02` | candidate shared observable |
| `CH06` | `P03` | `PERSONA.P03` | candidate shared observable; legacy clips negative age |
| `ESTADO` | `CONDACT` | `PERSONA.CONDACT` | candidate shared observable; EPH age<14 override requires review |
| `AGLOMERADO` | `AGLOMERADO` | no direct CPV field | must be derived through governed geography, not guessed |
| `IV1` | `V01` | `VIVIENDA.V01` | reverse Census->EPH value recode is separately governed |
| `IV3` | `H05` | `HOGAR.H05` | candidate shared observable |
| `IV4` | `H06` | `HOGAR.H06` | reverse value recode exists |
| `IV5` | `H07` | `HOGAR.H07` | candidate shared observable |
| `IV6` | `H08` | `HOGAR.H08` | candidate shared observable |
| `IV7` | `H09` | `HOGAR.H09` | reverse value collapse exists |
| `IV8` | `H10` | `HOGAR.H10` | candidate shared observable |
| `IV10` | `H11` | `HOGAR.H11` | EPH 3 -> Census 2 collapse in legacy/current aligner |
| `IV11` | `H12` | `HOGAR.H12` | candidate shared observable |
| `II1` | `H16` | `HOGAR.H16` | reverse Census-side clip recorded |
| `II2` | `H15` | `HOGAR.H15` | candidate shared observable |
| `II7` | `PROP` | `HOGAR.PROP` | EPH 7/8/9 -> Census 6 collapse |
| `II8` | `H14` | `HOGAR.H14` | reverse value recode exists |
| `II9` | `H13` | `HOGAR.H13` | forward and reverse many-to-one rules differ; never invert mechanically |
| `CH09` | `P07` | `PERSONA.P07` | forward/reverse recodes differ |
| `CH10` | `P08` | `PERSONA.P08` | candidate shared observable |
| `CH12` | `P09` | `PERSONA.P09` | legacy 99 -> 0 before alignment |
| `CH13` | `P10` | `PERSONA.P10` | conditional zeroing based on education universe |
| `CH15` | `P05` | `PERSONA.P05` | legacy 1/2/3 -> 1, 4/5 -> 2, 9 -> 0 |

The precise directional recodes are machine-readable in `contracts/deployment_dag.yaml` and remain subject to the aligner's mapping-review gate.

## Feature classes recovered from the historical cascade

### Candidate shared/derived inputs

The old `x_cols1` mostly consisted of the crosswalk plane above. Two exceptions are important:

- `AGLO_rk`;
- `Reg_rk`.

Both were calculated from group means of `P47T` and therefore contain target-derived information. They are now classified as **`research_only`** and forbidden as external Census deployment inputs. Their historical presence is useful evidence of why explicit feature authority is needed.

`AGLOMERADO` itself is not treated as a raw Census observable. If it is useful in transport, it must be a separately governed geography-derived feature.

### Stage 1 — person/status variables

Historical targets:

- `CAT_OCUP`;
- `CAT_INAC`;
- `CH07`.

These are **`stage_target`** candidates: observed in EPH, not assumed to be available as external Census columns. A future stage may learn them from approved shared/derived inputs.

### Stage 2 — income-presence hurdle variables

Historical targets:

- `INGRESO = P47T > 100`;
- `INGRESO_NLB = T_VI > 100`;
- `INGRESO_JUB = V2_M > 100`;
- `INGRESO_SBS = V5_M > 100`.

These are not ordinary predictors. In the current EPH-only model they are correctly treated as leakage when observed. In a transport model they may be legitimate **predicted hurdle-stage outputs**, but only if they are produced out-of-fold for downstream training and shown to improve real transport evidence. They are candidates, not commitments.

### Stage 3 — labour detail

Historical targets:

- `PP07G1`;
- `PP07G_59`;
- `PP07I`;
- `PP07J`;
- `PP07K`.

These remain candidate learned intermediates. Their exact role should be justified by dependency/ablation evidence rather than by historical grouping alone.

### Stage 4 — monetary outcomes

The historical final regressor predicted:

- `P21`;
- `P47T`;
- `PP08D1`;
- `TOT_P12`;
- `T_VI`;
- `V12_M`;
- `V2_M`;
- `V3_M`;
- `V5_M`.

The first revived scope should be narrower: **produce one defensible welfare concept sufficient for Poverty v2**, likely derived from `P47T`, rather than re-create nine outputs for compatibility.

## Relationship with `income-modeling-eph`

`income-modeling-eph` should be allowed to become a pure EPH scientific instrument.

Its current baseline uses observed `CAT_INAC`, `CAT_OCUP`, `CH07` and `PP07G_59` as predictors. Those same variables are stage targets in a Census deployment design. Therefore the EPH flagship can be an excellent EPH model while being intentionally non-deployable on Census rows.

That is not a defect. It is a separation of scientific questions:

```text
income-modeling-eph
  -> What predicts income well inside EPH?

encuestador-de-hogares
  -> What can be transported from EPH to an exact Census-derived frame under a defensible information plane?
```

The transport repo may consume a neutral versioned EPH artifact currently produced upstream, but it must not import the `income-modeling-eph` runtime or force its research feature choices to become Census constraints.

## Semantic alignment boundary

`eph-censo-aligner` owns:

- variable correspondence rules;
- direction-specific value recodes;
- ambiguity/loss reporting;
- classification of features as `shared_observable`, `derived_shared`, `stage_target`, `unsupported` or `research_only`.

It explicitly does **not** establish statistical transport validity.

The current aligner supports only synthetic `fixture-v1`; real EPH/Census vintages remain pending review. Therefore this repository may preserve the historical mapping as candidate evidence but must refuse a real promoted model until an exact alignment release is approved.

## Census frame boundary

`samplerCensoARG` owns Census sample identity, household membership, inclusion/sampling semantics, weights and exact frame lineage.

This repository must never resample or silently project that frame while doing inference. It scores the exact declared frame it receives.

If an inference-ready aligned Census feature table is materialized, it must preserve exact sampler person/household IDs and reference the parent sample/frame release. Fuzzy or positional matching is forbidden.

## Monetary boundary

The historical preprocessing directly downloaded `IPC-Argentina`, normalized nine monetary fields to the January-2016 index reference and rounded them. That behaviour is valuable lineage evidence but is no longer acceptable as hidden model preprocessing.

`IPC-Argentina` is the intended monetary-semantics/conversion authority. A revived transport model must declare:

- training-target currency and price reference;
- exact monetary-conversion release;
- model target transform;
- inverse/retransformation rule;
- final welfare currency and price reference.

Poverty must receive a resolved linear welfare concept, not a model-native `log10` prediction whose inverse transform is inferred downstream.

## First target contracts

### Transport model

Candidate family:

```text
research.eph-census-transport-model/v1
```

Minimum semantics:

- exact EPH training release;
- exact alignment release;
- exact deployment DAG;
- fold assignment / out-of-fold policy;
- stage estimator identities;
- metrics and subgroup diagnostics;
- Census-support/domain-shift diagnostics;
- monetary target semantics;
- hashes and limitations.

### Welfare handoff

Candidate family:

```text
research.household-welfare/v1
```

Minimum conceptual fields:

```text
sample_household_id
welfare_period
welfare_amount
currency
price_reference
welfare_concept
estimation_status
transport_model_release_id
```

Person-level predictions can remain internal/diagnostic or be published as a separately governed restricted artifact. Poverty should consume household welfare semantics rather than raw model outputs.

## Promotion gates before real inference

1. Approve one exact EPH and one exact Census vintage in `eph-censo-aligner`.
2. Materialize an exact Census feature frame keyed to the sampler namespace.
3. Remove `AGLO_rk`/`Reg_rk` from deployment inputs or replace them with independently governed, non-target-derived information.
4. Implement honest out-of-fold intermediate predictions for all downstream learned stages.
5. Re-evaluate the four historical waves as a dependency DAG; preserve only stages that add evidence-backed value.
6. Add support/domain-shift and subgroup diagnostics between EPH and Census feature distributions.
7. Bind monetary training and scoring to exact `IPC-Argentina` conversion/reference releases.
8. Approve the person-to-household welfare construction.
9. Produce a fully synthetic/fixture end-to-end transport-model + household-welfare release before touching a real poverty run.

## Non-goals of this boundary packet

- no real Census inference;
- no model retraining;
- no resurrection of historical cron/update jobs;
- no approval of real-vintage semantic mappings;
- no claim that RFC1-RFC4 are the optimal modern model;
- no transfer of Poverty, IPC, geography, Census sampling or EPH-only modeling authority into this repository.
