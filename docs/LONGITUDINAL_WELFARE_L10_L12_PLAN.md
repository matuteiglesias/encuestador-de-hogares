# Longitudinal welfare measurement, L10-L12 — implementation and commissioning plan

Status: implementation-ready design, 2026-09-29.

This document is the coordination front door for the new **measurement-mode** longitudinal welfare study.

No forecasting or nowcasting is in scope.

## Scientific objective

Measure welfare over:

```text
2017-Q1 .. 2026-Q1
```

using one pooled longitudinal transport framework that explicitly separates:

```text
slow cross-sectional composition
+
observed current aggregate labor context
+
year level
+
quarter seasonality
+
residual heterogeneity
```

and then asks whether stale donor-vintage individual labor state adds defensible information.

The current donor is CPV-2010. CPV-2022 is a later donor swap under the same architecture.

## Upstream plan documents

Required parallel work:

- `matuteiglesias/empleoARG/docs/CANONICAL_LABOR_STATE_2017_CURRENT_PLAN.md`
- `matuteiglesias/income-modeling-eph/docs/LONGITUDINAL_EPH_2017_2026_INPUT_PLAN.md`
- `matuteiglesias/eph-censo-aligner/docs/DONOR_LABOR_STATE_TRANSPORT_PLAN.md`

The canonical population/donor-frame work remains separate in `samplerCensoARG`.

## Target model family

Keep the person-level hurdle formulation as centerline:

```text
P(P47T > 0 | inputs)
+
E(P47T_real | P47T > 0, inputs)
```

with linear-real monetary semantics after exact deflation.

The positive-amount head remains the primary error-reservoir focus; do not revive the historical nine-monetary-output cascade.

## Labor context input

Consume the immutable official aggregate labor release from `empleoARG`.

First stable context:

```text
national activity rate
national unemployment rate
national subemployment rate

regional deviation from national activity
regional deviation from national unemployment
regional deviation from national subemployment
```

for the person's EPH/Census region and welfare quarter.

Employment rate remains available as product evidence/sensitivity but is not required in the minimal vector because of near-redundancy with activity/unemployment definitions.

Do not treat a regional unemployment rate as a person's unemployment probability.

## Explicit time layer

Do not rely only on raw tree splits over `ANO4` and `TRIMESTRE`.

Preferred first implementation is a nonlinear cross-sectional base plus low-dimensional time correction.

For the hurdle-presence head, conceptually:

```text
logit(p_final)
  = logit(p_base(X, labor_context))
  + year_effect[ANO4]
  + quarter_effect[TRIMESTRE]
  + exceptional_period_effect
```

For the positive real-income head, conceptually:

```text
mu_final
  = mu_base(X, labor_context)
    * exp(year_effect[ANO4] + quarter_effect[TRIMESTRE] + exceptional_period_effect)
```

Exact estimator implementation is a commissioning choice; it must preserve the interpretation and be fitted leakage-safely inside the training evidence of each evaluation fold.

Year effects are measurement-period effects, not forecast parameters.

For partial 2026:

```text
year_effect_2026
quarters_observed_for_year = 1
```

is legitimate because 2026-Q1 is observed. Persist its partial-year support explicitly. It updates when later observed EPH quarters arrive.

## Exceptional shock periods

2020-Q2, 2024-Q1 and 2024-Q2 must remain visible in the actual historical measurement, but they must not teach the system that their shocks are ordinary seasonality or ordinary time levels.

Default:

- keep all three quarters and their observed targets;
- flag `2020-Q2` as `pandemic_fieldwork_regime`;
- flag `2024-Q1` and `2024-Q2` as `2024_h1_macroeconomic_shock`;
- exclude all three from estimation of recurring quarter seasonality;
- exclude all three from estimation of the ordinary/structural time level;
- permit dedicated period/shock corrections estimated from their observed EPH evidence so the realized 2020-Q2 and 2024-H1 dips remain in measurement outputs;
- estimate the ordinary 2024 year effect from non-exceptional 2024 quarters where available, rather than allowing Q1/Q2 to drag the ordinary level;
- always publish structural-fit diagnostics with and without exceptional-quarter participation in the ordinary-time component.

## L10 / L11 / L12 arms

Names are intentionally moved away from the older L1-L4 labor-bridge nomenclature.

### L10 — aggregate-current-labor baseline

Inputs:

```text
approved cross-sectional shared/derived features
+ observed official regional/national labor context
+ explicit year/quarter measurement layer
```

No person-level donor labor feature.

This is the first model that must become fully runnable.

### L11 — raw donor-labor proxy sensitivity

Adds the **explicit donor-vintage Census labor observation**.

For CPV-2010 this is a stale-proxy scientific sensitivity, not automatically a promotion candidate.

Critical training rule:

> Current EPH `CONDACT_t` may not simply be copied and called donor `CONDACT`.

A valid EPH-side analogue requires repeated-wave/panel construction, using an earlier observed labor state as the stale proxy for a later observed welfare period.

Commission L11 first over observed EPH panel gaps to measure how quickly raw labor-state usefulness degrades with elapsed quarters.

Because EPH 2-2-2 follows a dwelling for only about 1.5 years, this evidence does not identify 2010→2026 persistence. Long-gap CPV-2010 use must remain explicitly unsupported/sensitivity-only unless a further persistence model is justified.

### L12 — donor-informed latent current labor

Target object:

```text
P(current labor state_t |
  donor labor state,
  approved composition,
  current aggregate labor context,
  elapsed time)
```

Then, if a person-level probability surface is retained, optionally calibrate its aggregate class moments to the exact official target-period labor margins using the existing KL / moment-projection design.

The aggregate anchor is useful here because it reconciles heterogeneous individual probabilities to known current margins. It is not needed by L10.

Rules:

- EPH downstream welfare training consumes only OOF latent probabilities;
- preserve raw and anchored probabilities separately;
- never hard-flip donor labels and call them current truth;
- exact anchor universe/denominator must match the official labor rate;
- report lambda/KL/probability displacement and downstream welfare delta.

### Persistence limitation

The observed EPH panel only supports short gaps. Do not extrapolate a fitted persistence coefficient dozens of quarters and call it identified.

For the current CPV-2010 donor, acceptable outcomes include:

- L12 donor contribution shrinks effectively to zero at long gap, reducing toward L10;
- L12 remains a bounded sensitivity, not promoted;
- the architecture waits for CPV-2022 before donor-informed labor becomes consequential.

This is a scientific result, not a failure.

## Monetary target

Consume an exact `IPC-Argentina` conversion release.

Train positive-income amount on `P47T_real` under one frozen common price reference.

Persist nominal and real semantics. Poverty receives a resolved linear welfare quantity with an explicit price reference; it never infers how to invert a model-native scale.

## Evaluation

### Household-safe OOF

Primary model comparison must group all repeated waves of the same household/panel group together as strongly as the proven identity permits.

No random-person leakage.

### Measurement-mode validation

Primary question: conditional on an observed quarter, can the model represent the cross-sectional welfare distribution and household aggregation?

Evaluate person and household:

- MAE/RMSE/R2;
- dispersion ratio;
- rank/Spearman;
- deciles and tails;
- hurdle presence diagnostics;
- positive-amount diagnostics;
- residual-distribution poverty-threshold diagnostics.

### Temporal sensitivities

Also run:

- leave-period / blocked-period diagnostics where scientifically interpretable;
- partial-year year-effect reconstruction, e.g. estimate a year's level using only Q1 then compare after Q2-Q4 become available for historical years;
- with/without exceptional-period participation in the ordinary structural fit;
- labor-context family ablation;
- L10 vs L11 vs L12 under matched folds.

Do not turn these into forecast claims.

## Cloud work packet — C4

Implement the reusable longitudinal experiment machinery before expensive real runs.

Deliver:

1. artifact consumers for longitudinal EPH frame and official labor-state release;
2. deterministic period × region labor-feature join;
3. explicit time-layer abstraction;
4. exceptional-period policy for 2020-Q2 and 2024-Q1/Q2;
5. L10 config/runtime;
6. panel-pair builder + L11 stale-proxy diagnostics;
7. L12 transition-model interface;
8. KL/moment anchor integration only behind L12;
9. immutable run metadata binding all parent release IDs;
10. synthetic tests for leakage, clocks, time corrections, panel grouping and anchor behavior;
11. comparison/report commands.

Do not refit or alter the frozen 2024-Q3 commissioning evidence in place.

## Local work packet — L4

After upstream real artifacts and C4 are green:

### Gate A — L10 real longitudinal commissioning

- build 2017-Q1..2026-Q1 merged model frame;
- verify monetary conversions;
- verify labor context coverage;
- run pooled hurdle model;
- fit/diagnose year + quarter time layer;
- produce special measurements and structural sensitivities for 2020-Q2 and 2024-Q1/Q2;
- persist OOF person/household evidence.

This gate does not wait for L11/L12.

### Gate B — labor persistence/panel evidence

- construct repeated-wave EPH panel pairs;
- quantify `CONDACT` transition matrices by elapsed-quarter gap;
- measure incremental welfare value of stale raw labor state;
- measure whether aggregate current labor context explains transition drift;
- do not infer unobserved long-horizon persistence.

### Gate C — L11

Run the stale-proxy arm only under elapsed gaps supported by EPH panel evidence. For CPV-2010 target scoring, label any raw-donor application as a stress/sensitivity unless a separate long-gap justification exists.

### Gate D — L12

Fit donor-informed latent labor on the supported panel design; evaluate raw vs anchored probabilities and terminal welfare.

Adjudicate whether donor labor earns retention. It is acceptable for the CPV-2010 result to collapse to L10.

### Gate E — exact Census research scoring

Only after semantic plane + donor sample gates are green:

- score one exact CPV-2010 sample;
- bind donor vintage, target year/quarter and elapsed time;
- produce research-only household welfare releases;
- no Census outcome-validation claim.

## Promotion/adjudication

The first stable longitudinal transport release should be the **simplest arm that survives evidence**.

Do not promote L12 merely because it is richer.

A learned labor layer survives only if it demonstrates at least one of:

- matched OOF terminal welfare gain;
- material distribution/tail improvement;
- improved robustness to current labor-state shifts;

without unacceptable calibration displacement or support failure.

If L11/L12 add no defensible value for CPV-2010, promote L10 and revisit donor labor with CPV-2022.

## Cloud/local DAG

```text
C1 empleoARG canonical labor product ──┐
                                      ├──> C4 encuestador runtime
C2 longitudinal EPH input surface ────┤          │
                                      │          v
C3 donor labor semantic handoff ──────┘       L4 Gate A (L10)
                                                 │
                                   C2 panel ids ─┴─> L4 Gate B
                                                        │
                                                 L11 / L12
                                                        │
                                  exact Census sample + semantic plane
                                                        │
                                                research scoring
                                                        │
                                      household-welfare release
                                                        │
                                      indice-pobreza-UBA intake
```

C1 and C2 are independent and should start in parallel.

C3 can run in parallel and must not block L10.

C4 can implement against fixtures while C1-C3 are in flight.

Heavy 37-quarter materialization/training belongs to the local agent.

## Downstream poverty boundary

No new poverty methodology is required for this work.

`indice-pobreza-UBA` should consume only an immutable household-welfare release with:

- welfare period;
- currency/real price reference;
- model release ID;
- donor/sample lineage;
- estimation/support status;
- uncertainty representation when justified.

Do not add predictive-model implementation to the poverty repo.

## Definition of done for this program

1. official labor context exists through the latest official quarter;
2. longitudinal EPH evidence exists 2017-Q1..2026-Q1;
3. L10 is fully commissioned under household-safe OOF;
4. 2020-Q2 and 2024-Q1/Q2 are measured but isolated from ordinary seasonal/time-level learning;
5. donor labor is explicitly vintage-qualified;
6. L11/L12 are adjudicated using honest panel evidence rather than copied current labels;
7. the selected arm can score an exact Census sample and emit an auditable household-welfare release;
8. no nowcast/forecast claim has been introduced.
