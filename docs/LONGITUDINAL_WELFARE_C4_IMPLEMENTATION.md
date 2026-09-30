# C4 implementation — longitudinal welfare L10/L11/L12 runtime

Status: fixture-first cloud implementation.

This document records the executable implementation of
`docs/LONGITUDINAL_WELFARE_L10_L12_PLAN.md`. It does not commission the real
37-quarter model and does not alter the frozen 2024-Q3 science bundles.

## Parent artifacts

The longitudinal runtime consumes immutable parents rather than reproducing
their producer logic:

- `research.eph-longitudinal-analysis-frame/v1` from `income-modeling-eph`;
- `publicdata.indec-eph-labor-state/v1` from `empleoARG`;
- `research.eph-census-donor-labor-handoff/v1` from
  `eph-censo-aligner` when a real donor handoff is needed.

The run parent record binds release IDs and parent manifest SHA-256 values. The
monetary conversion release/reference are inherited from the C2 longitudinal
EPH parent; this runtime does not silently re-deflate P47T.

## Composition-plane custody

Scientific composition is feature-plane-agnostic. A scientific config declares:

```text
composition_source: canonical_parent
composition_parent_contract: research.eph-longitudinal-composition-plane/v1
feature_profile_id: <named upstream profile>
```

The runtime then resolves the declared profile from one immutable composition
parent, verifies its manifest/artifact hashes, and joins it to C2 by exact
one-to-one `row_id`. Missing, extra, duplicated, or fuzzy identities fail
closed. EPH↔Census recoding remains upstream in `eph-censo-aligner`.

Every run binds the composition release ID, manifest SHA-256, profile ID,
profile-spec SHA-256, and resolved feature/categorical lists.

The historical cloud fixture list:

```text
CH04 CH06 CH09 CH10 CH12 CH13 CH15 IX_TOT
```

is now explicitly named `RAW_C2_8VAR_TEST_FIXTURE_V1` under
`fixture.raw-c2-composition-inline/v1`. It is synthetic-test infrastructure,
not an adjudicated L10 scientific profile. CLI execution requires the explicit
`--allow-fixture-composition` switch to use it.

C4B supports a synthetic C5-format parent now; no real C5 materialization or
profile winner is asserted by this PR.

## Labor-context join

For every person-period row, the runtime performs an exact period × region join
to the official C1 surface. No interpolation, carry-forward, smoothing or
nearest-period substitution is available.

The initial six-dimensional vector is:

```text
national activity
national unemployment
national subemployment
regional activity - national activity
regional unemployment - national unemployment
regional subemployment - national subemployment
```

These are contextual covariates. They are never interpreted as an individual's
unemployment/activity probabilities.

## Explicit time layer

The nonlinear hurdle base does not receive raw year or quarter as ordinary tree
features. For every outer fold, the time layer is estimated from **inner
household/panel-safe OOF base hurdle predictions** over the outer-training
population. In-sample base predictions are not residual evidence.

After the time layer is frozen, the base hurdle is refit on all outer-training
rows, predicts the outer holdout, and the frozen time correction is applied.
Outer-holdout targets and current labor labels never fit the time layer.
L11/L12 matched-L10 baselines use the identical discipline.

Presence:

```text
logit(p_final)
  = logit(p_base)
  + year_effect
  + quarter_effect
  + exceptional_period_effect
```

Positive real amount:

```text
mu_final
  = mu_base * exp(year_effect + quarter_effect + exceptional_period_effect)
```

The frozen exceptional periods are:

```text
2020-Q2  pandemic_fieldwork_regime
2024-Q1  2024_h1_macroeconomic_shock
2024-Q2  2024_h1_macroeconomic_shock
```

Exceptional rows remain observed and measurable but are excluded from ordinary
year and recurring-quarter estimation. Dedicated exceptional-period residual
corrections are then fitted from training-fold observations. In particular,
ordinary 2024 is learned from non-exceptional 2024 evidence when available.

Every fitted time layer records observed quarters per year and
`partial_year=true` when fewer than four observed quarters support that year.
This is measurement-support metadata, not a forecast mechanism.

## L10

L10 is independently runnable from C2 + C1:

```text
shared composition
+ exact current aggregate labor context
+ explicit time layer
-> hurdle welfare
```

No individual donor labor state and no aggregate anchor enter L10.

## L11

L11 first creates honest repeated-wave EPH pairs. The stale field is copied only
from an actually earlier observed EPH wave, and target welfare/current labor come
from a later observed wave.

Default supported gaps are one and three quarters, matching the expected short
EPH repeat pattern. Rows whose source labor state is a reviewed unavailable
special state are excluded from the labor-pair universe; unknown codes fail.

The terminal design is:

```text
L10 inputs + stale_labor_state
```

The fold group remains the panel household, so repeated waves/pairs for one
household cannot cross outer folds. L11 diagnostics report transition counts by
elapsed-quarter gap and explicitly state that these data do not identify a
CPV-2010→2026 persistence path.

Current `ESTADO_t`/CONDACT_t is never copied and relabeled as stale state.

## L12

L12 uses the same honest panel pairs, but the stale state is an input only to a
transition model:

```text
P(current labor_t |
  stale labor_{t-gap},
  composition_t,
  current aggregate labor context_t,
  elapsed gap)
```

For every terminal outer fold:

1. the transition model's meta-training probabilities are themselves generated
   by inner household-safe cross-fitting restricted to the outer-training
   population;
2. outer-holdout transition probabilities come from a transition model fitted
   only on outer-training rows;
3. the welfare hurdle receives the probability vector, never observed current
   labor truth.

Thus the L12 terminal head never trains on a current labor label for the row it
is predicting.

Raw transition probabilities are always retained.

### Optional L12 anchor

The implemented anchor is a generic categorical KL/moment projection using
exponential tilting:

```text
q*_ik ∝ q_ik exp(lambda_k)
```

It minimizes weighted categorical KL subject to explicit target class moments
and reports:

- raw/target/anchored moments;
- lambda;
- KL displacement;
- mean/median/max probability displacement;
- constraint residual.

The anchor path is rejected for L10/L11.

The runtime deliberately does **not** infer an anchor universe merely from a C1
rate label. An anchored L12 run must provide an explicit
`universe_contract`; the caller is responsible for proving that the modeled
labor-state population and official-rate denominator are compatible. This
prevents aggregate context from being silently reinterpreted as individual
probability truth.

## Run evidence

`research.encuestador-longitudinal-run/v1` bundles contain:

- resolved arm configuration;
- exact parent metadata and hashes;
- household/panel fold manifest;
- person OOF welfare components;
- transition raw/terminal-input probabilities for L12;
- time-layer fits per outer fold;
- panel diagnostics;
- anchor diagnostics;
- person and household metrics;
- explicit `measurement_mode=true` and
  `forecasting_authorized=false`.

Comparison output is descriptive and carries
`promotion_authorized=false`. Fixtures cannot promote L11/L12.

## CLI

Validate configs:

```bash
encuestador longitudinal-validate configs/longitudinal/l10.yaml
encuestador longitudinal-validate configs/longitudinal/l11.yaml
encuestador longitudinal-validate configs/longitudinal/l12.yaml
```

Local real L10 commissioning starts with:

```bash
encuestador longitudinal-run /path/to/adjudicated-l10-config.yaml \
  --eph-release-root /path/to/C2-release \
  --labor-release-root /path/to/C1-release \
  --composition-release-root /path/to/C5-composition-release \
  --output-root /path/to/runs
```

Anchored L12 additionally accepts an explicit anchor JSON with schema
`research.encuestador-labor-moment-anchors/v1`.

Comparison/report surfaces:

```bash
encuestador longitudinal-compare RUN_A RUN_B --output comparison.json
encuestador longitudinal-report RUN_A
```

## Existing September result remains frozen

C4 does not rewrite the September 2024-Q3 result: observed/true labor state had
material oracle welfare value while the earlier reconstruction from non-labor
shared covariates captured essentially none of that gain. L11/L12 ask a
different question—whether **actually stale** donor/panel labor information
adds defensible value after explicit current context and time controls.

## Local L4 handoff

The evidence order remains:

1. **Gate A — L10:** materialize C1/C2 plus an adjudicated C5 composition
   profile, verify all 37 quarters, exact composition row identity, monetary
   reference, labor-context coverage, household-safe OOF and inner-OOF
   time-layer behavior.
2. **Gate B — panel evidence:** build real repeated-wave pairs and quantify
   transitions/value by observed gap only.
3. **Gate C — L11:** run stale raw labor only where Gate B supports the gap.
4. **Gate D — L12:** evaluate OOF transition probabilities and, separately,
   raw versus anchored probability surfaces under a proven anchor universe.
5. **Gate E — Census research scoring:** only after exact C3 donor/sample
   materialization and the preceding gates.

C1/C2/C3 implementation contracts are now merged, but their full local real
materializations/commissioning receipts remain prerequisites at the
corresponding gates. No cloud fixture result is a model-promotion result.
