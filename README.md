# encuestador-de-hogares

**Status:** active-bounded scientific revival  
**Role:** survey-to-Census welfare inference

This repository is the statistical bridge between target-period EPH evidence and an exact Census-derived household sample.

Its modern mission is:

> Given target-period EPH evidence, an approved EPH/Census semantic feature plane, one exact Census sample/scoring frame, and explicit monetary semantics, infer a declared target-period household welfare quantity for those exact sample households, with auditable assumptions, diagnostics and lineage.

```text
neutral EPH observation frame
        +
approved EPH/Census semantic feature plane
        +
exact Census sample + aligned scoring frame
        +
explicit monetary semantics
        +
transport study specification
                 |
                 v
       direct / hurdle / staged transport
                 |
                 v
       qualified transport model
                 |
                 v
        exact Census sample scoring
                 |
                 v
       resolved household welfare
                 |
                 v
             Poverty v2
```

## Longitudinal measurement mode — L10/L11/L12

The active runtime now has a real-parent longitudinal measurement path for
observed EPH quarters `2017-Q1..2026-Q1`. Its scientific contracts are proven
against materialized L1/L1B/L2/L3A/L3B parents; it is **not** a forecasting or
nowcasting surface.

The centerline is a person-level hurdle over common-reference real `P47T`,
with exact C1 labor context and an explicit year/quarter/shock correction.
`2020-Q2`, `2024-Q1`, and `2024-Q2` remain measurable but are excluded
from ordinary recurring seasonality and structural-year estimation.

The arms are deliberately distinct:

- **L10:** shared composition + current official aggregate labor context +
  explicit time layer;
- **L11:** L10 plus an actually earlier observed EPH labor state on supported
  repeated-wave pairs;
- **L12:** L10 plus leakage-safe OOF target-period labor probabilities learned
  from stale state/panel evidence, with optional explicit KL/moment anchoring.

Current EPH labor state is never copied and called stale donor state. Official
regional labor rates are context, not row-level unemployment probabilities.
Fixtures cannot promote L11/L12.

### Current real commissioning state

All upstream real-data/semantic prerequisites for L4 Gate A are now available:

- official aggregate labor parent + explicit bounded 2019-Q3 NEA completion overlay;
- real 37-quarter EPH frame with 1,869,620 person-period rows;
- real `P0_LONG` and `P1R_NOLAB_LONG` canonical composition releases over all 37 quarters;
- explicit CPV-2010 donor-labor handoff for later donor-aware research;
- canonical real L10 configs and nested household-safe OOF time-layer semantics.

The first full real L10 attempt did **not** emit a scientific result bundle. It was
terminated by practical memory pressure because the current data plane materializes the
full EPH and composition surfaces as Python `list[dict]` objects before fold execution.

Current status:

```text
L1/L1B/L2/L3A/L3B     complete
C4/C4B scientific runtime  complete
L4 preflight               green
L4 Gate A result           not yet produced
development blocker        resource-safe execution architecture
```

The C6 resource-safe execution plane is now merged: it streams C2/C5 into one typed
model plane, persists deterministic folds, executes one outer fold at a time, and
checkpoints results for restart. Hosted CI proves synthetic numerical parity with C4B.
The remaining frontier is the **real 1.87M-row C6 acceptance run**; no L10 scientific
result is claimed before that local gate passes. See
`docs/C6_RESOURCE_SAFE_LONGITUDINAL_EXECUTION.md` and
`docs/LABOR_UPGRADE_CURRENT_STATE_2026-09-30.md`.

See `contracts/longitudinal_runtime.yaml` and
`docs/LONGITUDINAL_WELFARE_C4_IMPLEMENTATION.md`.

## What this repository owns

- the transport training population and eligibility contract;
- direct, hurdle and staged/DAG transport model science;
- honest household/group-aware out-of-fold intermediate predictions;
- EPH survey-weight policy for transport fitting/calibration/evaluation;
- the distinction between semantically comparable Census variables and scientifically admissible target-period inputs;
- optional, explicit target-period aggregate calibration inside the transport study;
- support/domain-shift, cascade, subgroup, tail and ablation diagnostics;
- scoring one exact Census sample namespace;
- inverse transforms and monetary-reference resolution through an exact `IPC-Argentina` release;
- construction of the declared household welfare concept;
- governed transport-model and household-welfare releases.

## What it does not own

- raw EPH acquisition or the neutral EPH observation-frame producer;
- EPH-only income-model research;
- semantic EPH↔Census mapping authority;
- Census sample construction or target-year department sampling;
- Census geography;
- price-index methodology;
- poverty lines, adult equivalence, poverty classification or FGT estimation.

The neighboring authorities are intentionally separate:

```text
income-modeling-eph    -> EPH-only income science
samplerCensoARG        -> exact Census sample identity/design
eph-censo-aligner      -> semantic variable alignment
IPC-Argentina          -> monetary semantics/conversion
indice-pobreza-UBA     -> poverty method and estimation
```

## What does it ask for?

A modern transport run consumes five kinds of governed evidence:

1. a **neutral EPH training frame** with exact person/household identity, survey-design fields, periods and candidate transport targets;
2. an **approved semantic feature plane** from `eph-censo-aligner`;
3. an **exact Census sample and aligned scoring frame** preserving `sample_person_id` and `sample_household_id`;
4. an exact **monetary-reference/conversion release** from `IPC-Argentina`;
5. a **transport study specification** declaring training population, welfare period, temporal-role assumptions, model family/DAG, fold policy, weighting policy, optional aggregate anchors and terminal welfare concept.

It does **not** consume the flagship model from `income-modeling-eph`. It consumes EPH evidence and owns a different scientific question.

## What does it return?

Two external products define the modern system boundary:

```text
artifact:research.eph-census-transport-model@1
artifact:research.household-welfare@1
```

The model release records the scientific transport claim: exact parents, cohort, temporal assumptions, folds, weighting, fitted estimators, OOF evidence, support diagnostics, ablations, monetary semantics and limitations.

The household-welfare release is the clean downstream handoff to Poverty. Its conceptual row is:

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

Person-level stage predictions remain internal/restricted audit state by default. Poverty should not need to understand classifiers, RFC stages or log transforms.

## The key scientific distinction: semantic alignment is not temporal transport

A Census variable may mean the same thing as an EPH variable and still be stale for the welfare period.

For example, a condition-of-activity value observed in Census 2010 is not automatically an observed condition-of-activity value for 2024. The historical project already encountered this problem: one quarterly prediction notebook changed Census `CONDACT` counts to match a quarter-specific unemployment target before the first classifier.

The implementation was artisanal, but the scientific distinction survives. The modern system classifies every deployable Census feature not only by semantic class, but also by transport-time role, such as:

- `donor_vintage_proxy`;
- `target_period_latent`;
- `deterministic_target_period_derived`;
- `target_period_anchor`;
- `time_stable_or_invariant`;
- `forbidden_temporal_input`.

Any target-period calibration is optional and explicit. It must never silently rewrite a donor Census value and call it observed current data.

## Model families

The historical RFC1→RFC4 cascade is scientific evidence, not the new architecture.

Every modern study begins with a mandatory direct baseline:

```text
approved common/donor information -> terminal welfare
```

Then it may compare:

```text
direct model
hurdle / two-part welfare model
staged dependency DAG
```

A learned intermediate stage survives only if honest OOF evidence shows useful **final-welfare** value or improves calibration/support robustness. Historical membership in RFC1/RFC2/RFC3 is not sufficient.

## Time and identity

Every consequential run must keep these clocks separate:

```text
eph_training_period
census_frame_vintage
sampling_target_period
welfare_period
monetary_reference_period
```

The same exact annual Census sample can be scored at several welfare periods. Those repeated outputs are synthetic snapshots on stable donor IDs, not observed longitudinal records.

Exact sampler identity is never changed during inference. Positional or fuzzy joins are forbidden.

## Weight semantics

These are different quantities and must remain different:

```text
EPH survey / expansion weight
!= Census sample selection probability
!= donor-frame inverse-probability quantity
!= Poverty analysis weight
```

The encuestador decides only how EPH survey weights enter transport fitting/calibration/evaluation. It does not reinterpret sampler probabilities as training weights or invent the final Poverty estimand.

## First modern proof

No real Census inference should run before a deterministic synthetic fixture proves:

- neutral EPH person/household identity and explicit EPH survey weights;
- exact Census sample identity and separate selection metadata;
- one approved synthetic semantic feature plane;
- a direct baseline plus at least one hurdle/staged candidate;
- household-aware OOF;
- explicit temporal role for every Census input;
- explicit/no-hidden calibration policy;
- exact model and monetary lineage;
- exact Census scoring coverage;
- complete person→household accounting;
- one linear `research.household-welfare@1` release.

## Current commissioning status

The thin 2024-Q3 / CPV-2010 source-separation surface (D-1) is closed as
`diagnostic_only` after the categorical-canonicalization/missingness refresh. The
stable/shared tier remains near-random source separation, while target-period and
research-only tiers add systematic separation.

This result does **not** promote a welfare transport model and does not create transport
weights. Current cross-ecosystem adjudication and rerun triggers live in
`matuteiglesias/indice-pobreza-UBA/science/commissioning/registry.json`.

## Current documents

Start here:

- [`docs/LABOR_UPGRADE_CURRENT_STATE_2026-09-30.md`](docs/LABOR_UPGRADE_CURRENT_STATE_2026-09-30.md) — current real longitudinal program state and acceptance frontier;
- [`docs/C6_RESOURCE_SAFE_LONGITUDINAL_EXECUTION.md`](docs/C6_RESOURCE_SAFE_LONGITUDINAL_EXECUTION.md) — resource-safe real-scale execution contract and local acceptance gate;
- [`docs/LABOR_UPGRADE_EXECUTION_DAG.md`](docs/LABOR_UPGRADE_EXECUTION_DAG.md) — current execution/dependency frontier;
- [`docs/FUNCTIONAL_CONTRACT.md`](docs/FUNCTIONAL_CONTRACT.md) — what the system asks for, does, evaluates and returns;
- [`contracts/functional_interface.yaml`](contracts/functional_interface.yaml) — machine-readable target interface;
- [`contracts/deployment_dag.yaml`](contracts/deployment_dag.yaml) — recovered variable/stage archaeology and candidate deployment DAG;
- [`docs/EPH_CENSUS_TRANSPORT_BOUNDARY.md`](docs/EPH_CENSUS_TRANSPORT_BOUNDARY.md) — revival boundary and promotion gates;
- [`SYSTEM.yaml`](SYSTEM.yaml) — repository authority;
- [`LIFECYCLE.md`](LIFECYCLE.md) — active-bounded lifecycle and real-run stop conditions;
- [`docs/HISTORICAL_README.md`](docs/HISTORICAL_README.md) — preserved historical project description.

## Historical assets

The legacy Random Forest models, EPH/Census preparation code, notebooks, figures and serialized artifacts remain valuable evidence. They are not automatically current releases.

In particular, the old project preserved three durable ideas that the modern system is testing rather than blindly inheriting:

1. learn EPH-only states from a common EPH/Census information plane;
2. propagate those learned states toward income/welfare;
3. adapt inference to a target period rather than pretending Census-2010 states are all current.

The modern architecture keeps those ideas while moving preprocessing, sampling, semantic mapping, monetary authority and poverty measurement into their proper neighboring systems.
