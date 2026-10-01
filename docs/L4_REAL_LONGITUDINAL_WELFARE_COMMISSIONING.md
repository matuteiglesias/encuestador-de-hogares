# L4 — real longitudinal welfare and labor commissioning

Status at 2026-10-01: **C6 real resource acceptance PASS; Gate A COMPLETE_BOUNDED_COMMISSIONING; Gate B COMPLETE_DESCRIPTIVE_EVIDENCE.** The 2017-Q1–2026-Q1 L2 source, P0/P1R L3B composition planes, official L1/L1B context and C6 resource-safe L10 plane are available. The historic pre-C6 RAM failure below is resolved, not an active blocker. R0 freezes the C7 input/identity contract; the next real model experiment is Gate C (L11). L12 and Census donor-clock scoring remain separate subsequent gates.

The Gate-B real baseline has 1,172,374 audited candidate links and 827,793 eligible 1Q/3Q transitions: 571,984 (1Q), 255,809 (3Q). This establishes *descriptive* short-gap evidence, not L11/L12 model gains. Source receipt: `data/l4-gate-b-20261001/GATE_B_REAL_RECEIPT.json`; original diagnostic release `gate-b-panel-evidence-c15bac5f1a6cbeb7`. Consult `docs/LABOR_UPGRADE_CURRENT_STATE_2026-09-30.md`, `docs/C7_DONOR_CLOCK_MATCHED_L11_CONTRACT.md` and `docs/LABOR_CONDACT_HISTORICAL_REPAIR_20261001.md` for the current execution boundary.

## Purpose

Commission the real measurement-mode L10/L11/L12 evidence after cloud contracts are merged and real upstream releases exist.

No forecast or nowcast is authorized.

## Prerequisites

Required before Gate A:

1. L1 real `publicdata.indec-eph-labor-state/v1`; if the parent has the documented 2019-Q3 NEA four-cell official gap, also require the explicit L1B completion overlay defined in `matuteiglesias/empleoARG/docs/L1B_BOUNDED_LABOR_CONTEXT_COMPLETION.md`;
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

If L1B is used, bind the official C1 release and the completion-overlay release separately in run metadata. The overlay must not be relabeled as official labor data.

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

For the documented 2019-Q3 NEA gap, use the L1B backward-fill overlay as centerline and run at least one matched forward-fill sensitivity. The purpose is only to show whether this one-quarter derived context matters materially.

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

## Gate B — repeated-wave descriptive evidence (COMPLETE)

The exact L2 `panel_links.csv` audit and existing Gate-B selector establish
one- and three-quarter observed labor transitions and later-income associations.
It uses a 14+ reviewed E/U/I universe, preserves missing income separately
from zero, and counts excluded links by exclusive first reason. It does NOT
fit an L11 or L12 model, estimate incremental held-out welfare error, or
prove 2010→2024/2026 donor persistence.

Real baseline: 1,172,374 audited links; 827,793 eligible pairs;
344,581 exclusions. The exceptional-period exclusion sensitivity has
719,097 eligible pairs and retains the broad transition pattern.

All downstream pairs must delegate to the exact Gate-B selector and
reconcile candidate, exclusion, 1Q/3Q and distinct-later-row totals before
the *additional* valid-target-income filter. No new ad hoc linkage method.

## Gate C — L11 donor-clock stale-state experiment (NEXT)

Use the exact selection/provenance contract in
`docs/C7_DONOR_CLOCK_MATCHED_L11_CONTRACT.md`.

The real EPH experiment is pooled on Gate-B-supported 1Q/3Q pairs,
restricted further only for a valid later-income target. Take P1R
composition from the **earlier** observation, not the target observation.
Use official national/regional labor context at the **target** quarter,
explicit target time and elapsed gap in BOTH arms:

- C7-0: donor-X/historical-input L10 analogue, without donor labor;
- C7-1 (L11): exactly C7-0 plus actually observed earlier labor.

Run identical target rows, panel-household-safe folds, hurdle parameters
and nested OOF time-layer policy. C7-0 is not numerically identical to
Gate-A whole-population L10, which uses current composition.

Primary endpoint: matched held-out person welfare deltas, including
presence and positive-amount metrics, horizon-specific and prior
E/U/I diagnostic strata, plus fold-wise uncertainty. Do not present
partial-panel member income sums as full household income.

Cloud C7A implements resource-safe private C6/Gate-B pair plane and
restartable paired runs; local C7B independently verifies real source
reconciliation, memory and the scientific result. No CPV scoring or
long-horizon transport authorization follows automatically.

## Gate D — L12 latent current labor (CONDITIONAL)

Do not begin expensive L12 terminal execution before inspecting the
accepted C7 real matched receipt. Reuse existing C4 science, C7 resource-safe
plane, the same outer group folds and strictly nested OOF components.

First test *transition prediction* on the observed 1Q/3Q horizon:

- T0 empirical P(later E/U/I | earlier E/U/I, gap), fitted within each
  outer training population, not imported from full-population Gate B;
- T1 multiclass conditional model adding earlier P1R composition and
  target-quarter official labor context.

Report multiclass log-loss, Brier, classwise support/calibration and
stratification by 1Q/3Q. Then conditionally compare L12's raw ordered
E/U/I probability inputs with matched C7-0 and C7-1 welfare on identical
rows/folds. Never make true target-period labor an external terminal input.
Do not let probability-column append silently disable categorical metadata
for existing composition predictors. See the historical Q4 method audit in
`docs/LABOR_CONDACT_HISTORICAL_REPAIR_20261001.md`.

Moment/KL anchoring is a **deferred L12 sensitivity**, not the default
centerline and not a requirement for the first C8 run. Before ever
attempting it, verify the modeled E/U/I universe matches the official
activity/unemployment denominator. Only under that condition, with rates
expressed as proportions:

    P(unemployed) = a * u
    P(employed)   = a * (1 - u)
    P(inactive)   = 1 - a

Do not treat subemployment as an additional CONDACT class. Persist
raw/anchored moments and displacement if a later separately reviewed
anchor experiment is undertaken.

## Gate E — separately authorized Census donor-clock research scoring

Prerequisites:

- completed matched L10/L11 evidence and any separately accepted L12 evidence;
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

For CPV-2010, observed 1Q/3Q EPH evidence does NOT establish 2010-to-target donor persistence. CPV-2022 requires its own exact governed policy/sample/geography and cannot silently reuse CPV-2010 semantics. It is valid to retain historical-input L11/L12 as short-gap EPH research and use L10 for bounded Census scoring.

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
- no poverty methodology change;
- no unvalidated source-clock substitution of donor CONDACT for current labor.
