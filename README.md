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

## Current documents

Start here:

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
