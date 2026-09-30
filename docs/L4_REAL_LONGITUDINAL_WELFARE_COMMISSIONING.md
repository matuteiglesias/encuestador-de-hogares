# L4 — real longitudinal welfare and labor commissioning

Status: local/heavy execution packet, 2026-09-29.

## Purpose

Commission the real measurement-mode L10/L11/L12 evidence after cloud contracts are merged and real upstream releases exist.

No forecast or nowcast is authorized.

## Prerequisites

Required before Gate A:

1. L1 real `publicdata.indec-eph-labor-state/v1`;
2. L2 real `research.eph-longitudinal-analysis-frame/v1`;
3. L3B real `research.eph-longitudinal-composition-plane/v1`;
4. merged C4/C4B runtime;
5. one frozen common-real monetary timing rule from L2.

L3A donor labor is not required until Census scoring.

## Gate A — L10 real commissioning

### A0 — profile comparison

Run matched household-safe OOF L10 using at least:

```text
P0_LONG
P1R_NOLAB_LONG
```

Both use the same:

- rows;
- outer folds;
- real-income target;
- labor-context release;
- time-layer policy;
- estimator family/hyperparameters.

Do not include current person labor state in either profile.

Adjudicate the composition profile from matched evidence. Persist both.

### A1 — labor-context ablation

On the selected composition profile compare:

```text
composition + time only
composition + time + national labor state
composition + time + national + regional-deviation labor state
```

Initial labor state:

```text
activity
unemployment
subemployment
```

Employment rate remains a sensitivity, not an automatic fourth centerline feature.

This adjudicates whether observed aggregate labor context adds terminal welfare information beyond explicit time state.

### A2 — time layer

Use the C4B nested discipline:

- base predictions for time-layer estimation are inner household-safe OOF inside each outer-training population;
- outer holdout is untouched;
- partial-year support is persisted.

Inspect year effects and quarter effects for stability.

Explicitly check 2026-Q1:

```text
quarters_observed_for_year = 1
partial_year = true
```

### A3 — exceptional periods

Produce actual measurement outputs for:

- 2020-Q2;
- 2024-Q1;
- 2024-Q2.

Also produce structural sensitivity excluding these periods from ordinary year/quarter learning.

Verify the dips remain visible through dedicated exceptional effects.

### A4 — acceptance evidence

At person and household level persist:

- MAE / RMSE / R2 where applicable;
- dispersion ratio;
- rank/Spearman;
- deciles/tails;
- presence diagnostics;
- positive-amount diagnostics;
- threshold/distribution diagnostics;
- fold-wise paired deltas;
- period-level residual summaries.

Gate A selects a stabilized **L10** configuration but does not yet claim Census transport validity.

## Gate B — repeated-wave labor persistence evidence

Using the real L2 panel audit and C4 panel-pair builder:

1. construct only credible short-horizon person-link pairs;
2. report transition matrices for supported elapsed gaps, especially 1 and 3 quarters;
3. stratify transition behavior by starting labor state and broad composition;
4. quantify how stale observed labor state changes terminal welfare performance beyond L10;
5. test whether current aggregate labor context improves transition prediction;
6. quantify linkage/conflict/attrition support.

Do not extrapolate short-gap persistence to 2010→2026.

Gate B defines the support boundary for L11/L12.

## Gate C — L11 stale-state sensitivity

Run L11 only on Gate-B-supported gaps.

Compare to matched L10 on identical target rows/folds.

Report:

- overall welfare delta;
- positive-amount delta;
- distribution/tail delta;
- performance by elapsed gap;
- performance by stale labor class.

L11 is an empirical stale-proxy sensitivity, not automatically a Census model.

## Gate D — L12 latent current labor

Fit the donor-informed transition probabilities under nested household-safe OOF.

Evaluate:

```text
L10
L12 raw probabilities
L12 anchored probabilities
```

### Anchor target construction

For a CONDACT-like three-class state, official aggregate moments must be derived under an explicit matching universe.

From compatible official activity rate `a` and unemployment rate `u`:

```text
P(unemployed) = a * u
P(employed)   = a * (1 - u)
P(inactive)   = 1 - a
```

after converting percentages to proportions and only when C1 denominator/universe semantics prove the rates are compatible with the modeled anchor universe.

Do not use subemployment as a CONDACT class moment.

Persist raw and anchored probabilities separately plus:

- target/raw/anchored moments;
- lambda;
- KL displacement;
- probability displacement;
- constraint residual;
- downstream welfare delta.

If the universe does not match, anchored L12 does not run.

## Gate E — CPV-2010 research scoring

Prerequisites:

- selected L10/L11/L12 arm from real EPH evidence;
- L3A real donor-labor handoff if the selected arm uses donor state;
- exact CPV-2010 Census sample/scoring plane;
- same canonical composition profile identity as the selected EPH model.

Start with one bounded exact target-period scoring proof before scaling.

Bind:

```text
census donor vintage
composition profile
donor labor vintage
welfare period
elapsed donor gap
model release
labor-context release
monetary reference
support status
```

For CPV-2010, L11/L12 donor persistence may remain unsupported at long horizons. It is acceptable for the promoted scoring arm to be L10.

No Census outcome-validation claim is authorized.

## Promotion rule

Prefer the simplest arm supported by matched evidence.

L11/L12 survive only if they add material terminal/distributional value without support or calibration failure.

A valid outcome is:

```text
promote L10 now
retain L11/L12 as short-gap research
revisit donor-informed labor under CPV-2022
```

## Final receipt

Persist a program receipt containing:

- exact upstream release IDs/hashes;
- selected composition profile;
- selected labor-context profile;
- selected time-layer policy;
- L10 evidence;
- Gate-B panel evidence;
- L11/L12 comparison;
- anchor diagnostics;
- selected/promoted arm and reason;
- CPV-2010 scoring status;
- unresolved support limitations.

## Non-goals

- no forecasting;
- no nowcasting;
- no causal interpretation of labor-context coefficients;
- no poverty methodology change.
