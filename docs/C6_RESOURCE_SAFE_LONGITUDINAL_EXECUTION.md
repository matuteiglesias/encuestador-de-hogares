# C6 — resource-safe longitudinal execution plane

Status: cloud implementation + synthetic parity gate; real 1.87M-row acceptance remains local.

## Mission

C6 changes the **data and execution representation only**. It does not change the
longitudinal scientific estimand, model family, fold policy, time layer, exceptional
period policy, labor semantics, or composition authority already reviewed in C4/C4B.

The motivating real failure was operational:

```text
C2 persons.csv -> list[dict] (~1.87M)
C5 composition.csv -> list[dict] (~1.87M)
two hash maps + joined dict copies
        ↓
NumPy matrices
        ↓
nested folds
```

The first real L10 attempt exhausted practical RAM/swap before producing a run bundle.

C6 replaces that path with:

```text
immutable C2 + C5 + L1/L1B parents
        ↓
one streaming lockstep C2↔C5 identity join
        ↓
research.encuestador-longitudinal-model-plane/v1
        ↓
memory-mapped typed arrays + persistent deterministic fold IDs
        ↓
one outer fold at a time
        ↓
atomic fold checkpoints
        ↓
restartable final L10 C6 bundle
```

## Scientific invariants preserved

C6 must preserve all of the following:

1. exact C2 row identity and order;
2. canonical named C5 profile identity;
3. no semantic recoding inside `encuestador-de-hogares`;
4. official aggregate labor rates are contextual features, never person probabilities;
5. deterministic `panel_household_grouped_longitudinal_v1` outer folds;
6. explicit nested household-safe OOF base predictions inside every outer-training fold;
7. outer holdout targets never fit the base model or time layer;
8. explicit year / recurring-quarter / exceptional-period time layer;
9. 2020-Q2 and 2024-Q1/Q2 remain measured but do not estimate ordinary time structure;
10. measurement mode only; no forecast/nowcast authorization;
11. same HGB hurdle estimator semantics and float64 feature values as C4B;
12. no L10/L11/L12 promotion from fixture evidence.

The synthetic C6 test compares C6 L10 predictions against the existing C4B
`execute_longitudinal_arm` path on the same rows/folds and requires numerical parity.

## Model-plane contract

The first real C6 plane is intentionally L10-oriented:

`research.encuestador-longitudinal-model-plane/v1`

It is content-addressed by:

- exact C2 parent;
- exact C5 profile release;
- official L1 parent and optional explicit L1B completion parent;
- config digest;
- exact C2 row identity sequence;
- deterministic fold vector.

Artifacts:

```text
features.npy          float64 [rows, features]
target.npy            float64 [rows]
period_codes.npy      int16   [rows]
fold_ids.npy          uint8   [rows]
household_codes.npy   int32   [rows]
row_ids.txt           exact UTF-8 C2 row sequence
manifest.json
```

The feature matrix is deliberately float64 rather than float32. At the real scale this
remains only a few hundred MB and removes a needless numerical-drift axis relative to the
existing C4B `float()` matrix semantics. The major memory reduction comes from removing
millions of Python dictionaries, two giant identity hash maps, repeated joined row copies,
and whole-run fold objects.

## One exact join

L3B already proves that the real C5 profile preserves the exact C2 identity sequence.
C6 therefore streams C2 and C5 **in lockstep** and fails immediately if the row IDs
diverge. It does not build:

```text
dict[row_id -> C2 row]
dict[row_id -> C5 row]
```

Labor context remains a small exact period/geography lookup and is appended during the
same streaming pass.

## Persistent fold plane

Every row receives its deterministic outer fold directly from
`panel_household_id` using the unchanged SHA-256 fold policy.

The vector is persisted once in `fold_ids.npy`.

C6 comparisons bind:

- `row_identity_sequence_sha256`;
- `fold_ids_sha256`.

Therefore P0/P1R and labor-context ablations cannot silently change cohort or folds.

## Gate-A labor ablation

The same model plane stores all six C1 labor-context columns.

The C6 runner selects one of three feature views without rebuilding parents:

```text
none
  composition only

national
  composition
  + national activity
  + national unemployment
  + national subemployment

national_regional
  composition
  + national rates
  + six-region deviations
```

This directly supports the intended L10 A1 matched ablation.

## Restartable execution

Each outer fold is an independent atomic checkpoint:

```text
checkpoints/fold_0/
...
checkpoints/fold_4/
```

Each checkpoint binds:

- model-plane release and manifest hash;
- config digest;
- labor mode;
- fold policy/hash;
- outer-fold index;
- holdout row indices;
- corrected presence and positive-amount predictions;
- fitted time-layer report;
- peak/current RSS observed after the fold.

A process interruption after fold 2 must not invalidate folds 0–2. Rerunning the same
command validates and reuses existing exact checkpoints.

Finalization occurs only after all expected folds validate.

## CLI

Materialize one exact model plane:

```bash
encuestador longitudinal-c6-plane \
  configs/longitudinal/l10_real_p1r.yaml \
  --eph-release-root /path/to/eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39 \
  --labor-release-root /path/to/indec-eph-labor-state-52ca6bcb586f2b0b \
  --labor-completion-root /path/to/indec-eph-labor-context-completion-backward_fill-2212ec7a74aa7f16 \
  --composition-release-root /path/to/eph-longitudinal-composition-p1r_nolab_long-ed30aa112c9b7d31 \
  --output-root /path/to/c6/model-planes
```

Run one outer fold as a resource smoke test:

```bash
encuestador longitudinal-c6-run \
  configs/longitudinal/l10_real_p1r.yaml \
  --model-plane-root /path/to/model-plane \
  --labor-mode national_regional \
  --outer-fold 0 \
  --output-root /path/to/c6/runs
```

Resume/finish all folds:

```bash
encuestador longitudinal-c6-run \
  configs/longitudinal/l10_real_p1r.yaml \
  --model-plane-root /path/to/model-plane \
  --labor-mode national_regional \
  --output-root /path/to/c6/runs
```

Run matched labor ablations against the same plane:

```bash
for mode in none national national_regional; do
  encuestador longitudinal-c6-run \
    configs/longitudinal/l10_real_p1r.yaml \
    --model-plane-root /path/to/model-plane \
    --labor-mode "$mode" \
    --output-root /path/to/c6/runs
done
```

Compare exact-cohort/fold runs:

```bash
encuestador longitudinal-c6-compare \
  /path/to/run-none \
  /path/to/run-national \
  /path/to/run-national-regional
```

## First-release scope

Cloud C6 intentionally stops at **real L10 execution**.

It does not yet replace the row-oriented L11/L12 pair builder. That is deliberate:
Gate B must first inspect real short-gap panel evidence after L10 is operational.
The same compact execution primitives can then be extended to L11/L12 without
prematurely redesigning panel science.

## Real acceptance gate

The local gate must prove, on the real P1R plane:

```text
rows = 1,869,620
profile = P1R_NOLAB_LONG
folds = 5
measurement_mode = true
forecasting_authorized = false
all outer folds checkpoint independently
process restart reuses completed folds
all final OOF predictions finite
peak RSS < 10 GiB
no sustained swap thrashing
exact row identity sequence preserved
exact fold hash preserved
nested time-layer evidence remains inner household-safe OOF
```

Then repeat Gate-A matched runs needed for:

1. P0_LONG vs P1R_NOLAB_LONG;
2. selected profile: labor `none` vs `national` vs `national_regional`.

No scientific arm is selected merely because C6 runs successfully.

## Failure interpretation

If C6 still exceeds the resource gate, inspect:

- estimator-internal HGB allocations;
- dense per-fold training copies;
- thread/process multiplication;
- swap trajectory;
- whether one fold alone exceeds the budget.

Do **not** respond by weakening:

- row coverage;
- fold isolation;
- nested OOF;
- time-layer semantics;
- profile identity;
- labor clocks.

The allowed fix surface is computational representation/execution only.

## Definition of done

C6 is complete when:

- cloud CI proves model-plane contracts, restartability and synthetic C4B parity;
- the real P1R model plane materializes;
- one real outer-fold smoke completes within the resource budget;
- interruption/resume is demonstrated;
- full five-fold P1R L10 completes under the budget;
- P0 and labor-context Gate-A matched comparisons can run without changing
  cohort/fold identity;
- exact receipts are persisted;
- the implementation is merged before scientific L4 Gate A is interpreted.
