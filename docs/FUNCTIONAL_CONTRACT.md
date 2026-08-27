# Functional contract: what the revived encuestador asks for and returns

## Mission

The revived `encuestador-de-hogares` is the **survey-to-Census welfare transport instrument**.

Its job is not to preprocess EPH, sample Census, define semantic mappings, build a price index, or calculate poverty. Its job is narrower and scientifically stronger:

> Given target-period EPH evidence, an approved common EPH/Census information plane, and one exact Census-derived household sample, infer a declared target-period household welfare quantity for those exact sample households, with explicit assumptions, diagnostics and lineage.

The shortest useful picture is:

```text
neutral EPH observation frame
        +
approved EPH/Census semantic feature plane
        +
exact Census sample + aligned Census scoring frame
        +
explicit monetary semantics
        +
transport study specification
                 |
                 v
       statistical transport study
     direct / hurdle / staged DAG
                 |
                 v
       qualified transport model
                 |
                 v
       exact Census sample scoring
                 |
                 v
     resolved person-level welfare
                 |
                 v
     declared household welfare
                 |
                 v
             Poverty v2
```

This repository therefore owns **statistical transport and welfare resolution**, not the surrounding source authorities.

## The scientific object is not "a Census income model"

The historical project taught a useful idea: EPH contains target variables and rich current socioeconomic states that Census either does not observe or observes under a different information/time regime. One can learn relationships on EPH and use the common information plane to impute missing states and income on Census-derived units.

The modern formulation is more precise:

```text
EPH target-period survey population
        |
        | observed target states + welfare
        |
        +---- common semantic information ----+
                                               |
                                               v
                                    Census donor/sample units
                                    observed at frame vintage
                                               |
                                               v
                                    target-period latent states
                                               |
                                               v
                                      target-period welfare
```

The model is a **transport study under partial observability and temporal mismatch**. It is not a claim that Census observed the predicted state or income.

## What the encuestador asks for

The repository should eventually expose two explicit operations: **qualify a transport model** and **score an exact Census sample**.

### 1. Neutral EPH training evidence

The transport study needs an EPH artifact that preserves the survey as evidence, not an EPH-only modeling dataset that has already chosen somebody else's cohort.

Minimum required semantics include:

- exact person identity;
- exact household identity;
- period/quarter identity;
- native or explicitly canonical source variables;
- survey/design/expansion fields;
- variables required to define candidate intermediate states and terminal welfare targets;
- exact monetary lineage for monetary targets, or enough lineage to create an approved monetary view.

It should **not** receive a frame that has already silently imposed:

- `INGRESO == 1`;
- `P47T > 0`;
- an EPH-only feature contract;
- target-derived geography ranks;
- Census-shaped aliases as if they were native EPH semantics;
- random-person split assignments.

Those are study decisions, not neutral source evidence.

### 2. An approved semantic feature plane

`eph-censo-aligner` remains the authority for what EPH and Census variables mean relative to each other.

The encuestador asks for a release that tells it:

- exact EPH and Census vintages;
- canonical feature names;
- directional recodes;
- deterministic derived features;
- category losses and ambiguity;
- which concepts are `shared_observable`, `derived_shared`, `stage_target`, `unsupported` or `research_only`.

The encuestador does **not** decide that two source variables are semantically equivalent merely because a historical script renamed one into the other.

### 3. An exact Census scoring population

`samplerCensoARG` supplies the population to score.

The encuestador asks for:

- exact `sample_person_id`;
- exact `sample_household_id`;
- person-to-household membership;
- exact Census frame vintage;
- sampling target period;
- department/sample lineage;
- selection/design metadata;
- exact coverage and checksums.

The encuestador does not resample, modify sample probabilities, or create another population frame while scoring.

The same exact annual sample may legitimately be scored for several welfare periods, for example multiple quarters, provided each run declares its own EPH evidence, temporal assumptions and monetary reference. Repeated predictions on the same Census donor IDs are **synthetic repeated snapshots**, not observed longitudinal records.

### 4. An aligned Census scoring frame

The Census sample also needs the approved transport features materialized for its exact persons/households.

The preferred boundary is that the aligner owns the semantic rules and can materialize or govern an aligned release. The encuestador should not import the aligner's Python runtime or reconstruct source recodes independently.

The scoring frame must preserve the exact sampler namespace.

### 5. Monetary semantics

The encuestador asks `IPC-Argentina` for an exact monetary-reference/conversion release.

It needs enough information to declare:

```text
training target nominal/reference semantics
        -> model target transform
        -> inverse/retransformation
        -> output welfare reference
```

The downstream welfare release must be linear and interpretable. Poverty never receives an unexplained `log10(P47T)` prediction.

### 6. A transport study specification

This is the encuestador's main scientific configuration. It defines:

- EPH training window;
- welfare period;
- training population / eligibility;
- feature temporal roles;
- direct, hurdle or staged model family;
- DAG dependencies;
- household/group fold policy;
- EPH survey-weight use for fitting/calibration/evaluation;
- optional target-period aggregate anchors;
- terminal welfare concept candidate;
- model comparison and promotion criteria.

The historical RFC1-RFC4 arrangement is one candidate design, not the interface.

## A new distinction: semantic comparability is not temporal admissibility

This round exposed a missing layer in the previous design.

A variable can be semantically comparable between EPH and Census and still be scientifically problematic as a predictor for a later welfare period.

For example, historical Census and EPH both have a concept related to activity condition. That does **not** make the Census-2010 value an observed activity condition for 2024.

The old quarterly prediction notebook already contained this intuition. Before RFC1 scoring it changed Census `CONDACT` counts to reproduce a quarter-specific unemployment target. The implementation was artisanal and mutated rows directly, but the scientific point matters: some Census-side states were treated as stale and in need of target-period updating.

The modern encuestador therefore owns a second classification, orthogonal to the aligner's semantic class:

```text
semantic class                transport-time role
--------------                -------------------
shared_observable      +      donor_vintage_proxy
shared_observable      +      time_stable_or_invariant
shared_observable      +      target_period_latent
shared_observable      +      forbidden_temporal_input
stage_target           +      target_period_latent
aggregate source       +      target_period_anchor
```

The proposed temporal-role vocabulary is:

- **`donor_vintage_proxy`** — use the Census-vintage observation only as an explicit proxy under a declared stability assumption;
- **`target_period_latent`** — the target-period state is not treated as observed and must be learned/updated if needed;
- **`deterministic_target_period_derived`** — a reviewed deterministic rule creates the target-period value;
- **`target_period_anchor`** — aggregate target-period evidence constrains/calibrates a latent state but does not turn it into a row-level observation;
- **`time_stable_or_invariant`** — approved as usable across the frame/welfare clock gap;
- **`forbidden_temporal_input`** — semantically mappable but not defensible in the target-period transport design.

This belongs to transport science, not semantic alignment.

## Target-period calibration is optional transport science, not sampling

The historical system adjusted unemployment before inference. That operation should not return to the sampler: sampling decides which Census households/persons exist in the scoring frame.

If modern evidence supports aggregate calibration of a target-period state, it belongs inside the transport study because it changes an **inferred state**, not sample identity.

The default should be **no hidden calibration**.

An allowed target-period anchor must be explicit:

```text
anchor_release_id
concept
period
population/universe
value
method
which latent node it constrains
calibration algorithm
pre/post diagnostic
```

The encuestador must never silently rewrite a donor Census field and then call it observed target-period data.

## What is learned

The minimum scientific comparison should not begin with the four historical waves. It should begin with a simple benchmark.

### Model family A — direct baseline

```text
approved shared / donor-proxy information
                 -> terminal welfare
```

This baseline is mandatory because it tells us whether staged complexity is buying anything.

### Model family B — hurdle / two-part welfare

```text
shared information
        -> income presence / relevant welfare state
        -> positive amount conditional on state
```

This is a natural candidate because the current EPH income study conditions away zero/nonpositive incomes, while the Census welfare problem cannot know that condition in advance.

### Model family C — staged transport DAG

```text
shared / derived inputs
        -> learned state A
        -> learned state B
        -> ...
        -> terminal welfare
```

Every learned intermediate used downstream is generated out-of-fold during downstream training. Once exact EPH household identity is available, folds must be household/group aware.

A historical stage survives only if it adds useful **final-welfare** evidence. Predicting an intermediate variable accurately is not enough by itself.

## What gets evaluated

EPH predictive metrics are necessary but not sufficient.

A promoted transport model should carry at least four evidence layers.

### 1. Within-EPH predictive evidence

- honest OOF error;
- calibration where relevant;
- subgroup error;
- tails/distribution shape;
- household-aware split evidence;
- survey-weighted and/or unweighted metrics according to the declared policy.

### 2. Stage-value evidence

For every intermediate stage:

- does it improve terminal welfare error?
- does it improve calibration/tails?
- is the improvement stable across folds/periods/subgroups?
- does it worsen error propagation?

### 3. EPH -> Census support evidence

- category coverage;
- overlap of shared/derived features;
- extrapolation flags;
- geography/period/subgroup support;
- donor-vintage versus target-period limitations.

### 4. Temporal transport evidence

- which Census variables are donor proxies;
- which states are refreshed/latent;
- which aggregate anchors are used, if any;
- sensitivity to target-period assumptions;
- exact distinction among frame vintage, sample target period and welfare period.

A model can have excellent EPH RMSE and still fail promotion because it has poor Census support or indefensible temporal assumptions.

## What the encuestador returns

There are two canonical external products.

### A. Transport model release

```text
research.eph-census-transport-model@1
```

It should package or reference:

- exact EPH training release;
- exact semantic feature-plane release;
- exact Census frame/scoring release used for support qualification;
- training population contract;
- welfare period / training window;
- feature temporal-role specification;
- fold/group policy;
- EPH weighting policy;
- DAG/model family and fitted estimators;
- optional calibration-anchor releases and algorithms;
- OOF metrics and ablations;
- support/domain-shift diagnostics;
- monetary target semantics;
- limitations, hashes and reproducibility metadata.

The model release is the scientific claim about a transport design.

### B. Household welfare release

```text
research.household-welfare@1
```

This is the downstream handoff to Poverty.

Minimum conceptual row:

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

with release-level lineage to:

- exact Census sample;
- exact aligned feature plane;
- exact EPH training evidence;
- exact monetary conversion;
- exact person-to-household aggregation rule.

The amount must be a **linear household-level monetary welfare concept**. Adult equivalence and poverty thresholds remain downstream in Poverty v2.

## Person-level outputs are audit state, not the downstream contract

The internal scoring process may produce:

```text
sample_person_id
predicted intermediate states/probabilities
terminal person-income prediction
support/extrapolation flags
prediction status
```

Those outputs are scientifically useful for diagnostics and person-to-household accounting. They can be retained as a restricted/internal artifact.

But Poverty should not have to understand RFC stages, classifier outputs or model-native transforms. Its contract starts at resolved household welfare.

## Household welfare is a scientific decision, not `groupby().sum()` by convention

The historical system predicted `P47T` plus several monetary components at the person level. The most obvious poverty-facing construction is to sum predicted person total income within the exact Census household.

That is a **candidate**, not a silently approved rule.

The first real implementation should compare and document possible terminal designs such as:

1. predict person total income (`P47T`-like concept) and sum to household total;
2. train a direct household total-income target where defensible;
3. use a justified hybrid if evidence supports it.

The chosen construction must account for every household member and report missing/invalid person predictions. No household welfare row may appear merely because some members happened to score successfully.

## Weight boundaries

Four concepts must remain separate:

```text
EPH survey / expansion weight
        !=
Census sample selection probability
        !=
Census donor-frame inverse-probability quantity
        !=
Poverty analysis weight
```

The encuestador owns how EPH survey weights affect **transport fitting/calibration/evaluation**.

It does not reinterpret sampler weights as EPH weights, and it does not decide the final Poverty estimand.

A generic `weight` column is therefore forbidden in the modern interface.

## Clock boundaries

Every real run must state at least:

```text
eph_training_period
census_frame_vintage
sampling_target_period
welfare_period
monetary_reference_period
```

Example:

```text
eph_training_period     = 2024Q1-2024Q4
census_frame_vintage    = 2010
sampling_target_period  = 2024
welfare_period          = 2024Q3
monetary_reference      = 2024Q3 ARS
```

The exact choices are scientific inputs, not naming conventions.

## The sampler and encuestador now have a very clean relationship

```text
samplerCensoARG
    asks: which Census households/persons are in the sample?
    returns: exact immutable sample identity + sampling metadata

encuestador-de-hogares
    asks: what target-period states/welfare can be inferred for those exact units?
    returns: qualified transport model + household welfare
```

The sampler changes **who is represented** through its declared household design. The encuestador changes **what is inferred about those units** through a declared statistical transport design.

Those are different scientific operations.

## The income-modeling and encuestador relationship is also clean

```text
income-modeling-eph
    EPH-only income science
    may use any legitimate EPH-only feature
    may study a positive-income cohort

neutral EPH analysis frame
    reusable observation layer
                |
                +------------------+
                                   |
                                   v
encuestador-de-hogares
    independent transport cohort
    independent target/hurdle/DAG
    independent weighting/fold policy
```

The encuestador does not consume the EPH flagship model. It consumes EPH evidence.

This is important: the statistical relation between EPH and Census is the encuestador's contribution. A strong EPH-only model and a strong EPH->Census transport model are different scientific objects.

## First modern end-to-end proof

Before any real Census inference, the repo should prove this exact interface synthetically:

```text
synthetic neutral EPH frame
  exact household/person IDs
  explicit survey weight
  zero + positive income persons
             +
synthetic approved feature plane
             +
synthetic Census household sample
  exact sampler IDs
  distinct selection probability
  donor frame vintage != welfare period
             +
exact synthetic monetary conversion
             |
             v
transport qualification
  direct baseline
  hurdle/staged candidate
  household-aware OOF
  explicit temporal roles
             |
             v
transport-model release
             |
             v
exact Census scoring
             |
             v
person audit state
             |
             v
household-welfare release
             |
             v
contract verification
```

The fixture should deliberately make EPH survey weights, Census selection probabilities and Poverty-facing analysis semantics numerically different so accidental conflation is caught by tests.

## Current major open decisions

The architecture is now clear enough that the remaining unknowns are real scientific decisions rather than repo-boundary ambiguity:

- first real EPH training window and Census scoring vintage;
- temporal role of every candidate Census input;
- whether/how target-period aggregate anchors are used;
- direct vs hurdle vs staged final design;
- representation of intermediate predictions (hard class, probability, draw, etc.);
- exact EPH survey-weight policy;
- approved person-to-household welfare construction;
- transport promotion thresholds under support and temporal shift;
- uncertainty representation beyond the first deterministic release.

Those are the problems the revived encuestador is supposed to solve.

## Machine-readable companion

The target interface is recorded in:

```text
contracts/functional_interface.yaml
```

`contracts/deployment_dag.yaml` remains the variable/stage archaeology and candidate DAG. The functional interface answers a different question: **what does the system require, what science does it perform, and what exactly does it return?**
