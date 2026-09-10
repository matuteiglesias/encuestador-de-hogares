# Test and Integration Matrix

## 1. Testing philosophy

The active scientific runtime must be testable at several levels:

```text
pure math / unit
scientific invariant
synthetic integration
real-data bounded smoke
comparison-level evidence
```

A green unit-test suite is necessary but not sufficient. The system must also prove that the scientific semantics survive composition.

## 2. Test classes

### T0 — Pure mathematical/unit tests

Fast, deterministic, no repository data dependency.

Examples:

- Bernoulli KL projection reaches a target moment;
- logit-shift projection preserves ordering;
- probability validator rejects invalid rows;
- DAG validator rejects cycles;
- household aggregation sums complete member predictions;
- metric functions reproduce hand-computed examples.

### T1 — Scientific invariant tests

Synthetic fixtures designed specifically to trigger forbidden behavior.

Examples:

- household member leakage across folds;
- observed intermediate inserted into a deployable downstream feature matrix;
- `PONDERA` inserted as a model feature;
- Census selection probability used as EPH training weight;
- category probability class order lost/reordered;
- incomplete household silently accepted;
- anchor applied to mismatched universe;
- donor field mutated by anchor implementation.

### T2 — Synthetic end-to-end tests

Small deterministic EPH-like data with known dependency structure.

Must exercise:

- direct architecture;
- lean architecture;
- generic DAG execution;
- HGB classifier probabilities;
- terminal model;
- household aggregation;
- evaluator;
- run packaging.

A separate fixture can make `Z` genuinely useful so the oracle/cascade decomposition has a predictable sign.

### T3 — Real EPH bounded smoke

Runs on one small/pinned real EPH slice or fixture derived through an approved local test path.

Purpose:

- schema compatibility;
- categorical dtypes;
- realistic missingness;
- multiclass support;
- zero/positive-income handling;
- household identities;
- runtime/memory sanity.

This test may be optional/skipped when the external data root is unavailable in CI, but the skip reason must be explicit.

### T4 — Scientific comparison integration

Produces real OOF runs and paired comparisons. This may be a Make target rather than every-commit CI.

Examples:

```text
direct vs lean HGB
hurdle-log vs hurdle-Gamma
raw vs calibrated probability
raw vs employment-anchored
HGB vs RF robustness
```

## 3. Identity and split matrix

| Test | Expected behavior |
|---|---|
| same household, several persons | all rows receive same outer fold |
| same row population + deterministic config | identical fold manifest digest |
| shuffled input row order | row IDs recover correct fold/prediction mapping |
| duplicate person ID | fail validation |
| missing household ID | fail before cross-fitting |
| empty fold | fail |
| comparison with different split digest | reject as matched comparison unless explicitly allowed |

## 4. Prediction artifact matrix

### Regression

Test:

- correct row count;
- stable row IDs;
- finite policy;
- task/target metadata;
- point shape `(n,)`;
- quantile maps consistent with row count when present.

### Binary classification

Test:

- probability shape `(n,2)` or explicit accepted binary convention;
- rows sum to one;
- class labels persisted;
- predicted class corresponds to probability argmax where applicable;
- raw/calibrated provenance distinct.

### Multiclass classification

Test:

- shape `(n,K)`;
- all configured classes accounted for;
- fold with missing training class fails or follows a declared policy rather than silently changing output dimension;
- class order remains identical across fold artifacts and final combined OOF artifact.

## 5. Cross-fit matrix

For each outer fold `k` prove:

```text
fit row groups ∩ validation row groups = empty
```

For learned stage `Z` consumed downstream prove:

```text
downstream training feature for row i == OOF prediction for row i
```

and not:

```text
observed Z_i
full-fit prediction generated using row i
```

Useful adversarial test:

Create a synthetic target containing a unique household identifier signal. A leaky person-random split should appear unrealistically strong while grouped OOF does not. This demonstrates why the invariant matters.

## 6. DAG tests

| Case | Expected |
|---|---|
| direct graph | executes with no learned latent node |
| one-layer graph | upstream probabilities become terminal features |
| two/deeper-layer graph | topological ordering correct |
| cycle A→B→A | fail validation |
| unknown upstream node | fail |
| duplicated node ID | fail |
| unavailable target | fail before fit |
| observed intermediate in deployable feature declaration | fail |
| forbidden external feature | fail |

## 7. Estimator tests

### HGB classifier

Test binary and multiclass fixtures with:

- numerical + categorical inputs;
- missing values where supported;
- stable `predict_proba` classes;
- deterministic-enough behavior under fixed seed/config;
- explicit early-stopping policy.

### HGB Gamma regressor

Test:

- positive target fits;
- zero/negative target rejected before sklearn call with a clear scientific error;
- predictions positive/finite;
- output remains on declared linear scale.

### HGB log-target regressor

Test:

- transform eligibility;
- explicit inverse-resolution metadata;
- no raw exponentiation path exists without declared policy.

### Random Forest

Test only the governed challenger pipeline:

- categorical encoding stable train/score;
- unseen-category policy explicit;
- same prediction artifact interface.

## 8. Hurdle tests

Synthetic cases:

### All positive

Presence model should handle or fail under an explicit single-class policy; positive amount model runs.

### Mixed zero/positive

Verify:

\[
\widehat Y_i=\widehat p_i\widehat\mu_i^+.
\]

### All zero

Terminal config should fail or return an explicitly defined degenerate result; never attempt Gamma fit on an empty positive population.

### Missing target

Eligibility/missingness policy must be explicit and counted.

## 9. Oracle/cascade diagnostic tests

Construct synthetic data where:

### Case A — `Z` irrelevant given `X`

Expected:

```text
oracle gain ~ 0
cascade gain ~ 0
```

### Case B — `Z` highly relevant and reconstructable

Expected:

```text
oracle gain > 0
cascade gain > 0
capture ratio meaningfully positive
```

### Case C — `Z` relevant but not reconstructable from `X`

Expected:

```text
oracle gain > 0
stage reconstruction weak
cascade gain ~ 0
```

### Case D — noisy representation hurts finite model

Expected:

```text
cascade gain < 0
```

The evaluator must report these cases without imposing a false `[0,1]` bound on capture ratio.

## 10. Calibration tests

When calibration is disabled:

- raw probabilities pass through unchanged;
- calibration diagnostics can still run.

When calibration is enabled:

- inner split uses outer-training households only;
- validation household never appears in calibration fitting;
- calibrated probabilities remain valid;
- raw probabilities retained;
- class order unchanged;
- calibration metadata records method and split policy.

For multiclass temperature calibration, test that probability simplex is preserved and predicted ranking/argmax behavior follows the chosen method's contract.

## 11. Distributional metric tests

Use tiny fixed vectors to verify:

- MAE;
- RMSE;
- bias;
- R²;
- SD ratio;
- quantile differences;
- observed-decile grouping;
- predicted-decile grouping.

Edge policies must be explicit for:

- zero observed variance;
- repeated quantile boundaries;
- tiny sample with fewer unique values than requested bins.

## 12. Household tests

Construct households of different sizes.

Prove:

\[
H_h=\sum_i Y_{hi}
\]

and:

\[
\widehat H_h=\sum_i\widehat Y_{hi}.
\]

Test:

- one invalid member marks household incomplete;
- duplicate member IDs fail;
- member order does not change result;
- household-size slices correct;
- bootstrap resamples whole households.

## 13. Cluster bootstrap tests

For a fixed prediction table and fixed bootstrap seed:

- replicate household samples deterministically;
- all sampled persons of a selected household move together;
- paired architecture differences use same resample indices;
- interval output contains requested quantiles;
- sample count and seed persisted.

Do not refit expensive models inside the bootstrap unless a specific experiment declares model-fit uncertainty as the target. The default science comparison bootstrap can operate on honest OOF predictions.

## 14. Anchor projection tests

### Exact feasible binary target

Projection aggregate equals target within tolerance.

### No-op

When target equals raw aggregate:

\[
\lambda\approx0
\]

and adjusted probabilities approximately equal raw probabilities.

### Upward target

`lambda > 0` and all non-degenerate probabilities increase.

### Downward target

`lambda < 0` and all non-degenerate probabilities decrease.

### Ordering

If `q_i < q_j`, then after a common finite logit shift:

\[
q_i^\star<q_j^\star.
\]

### Extreme probabilities

Clipping/numerical policy tested and reported.

### Impossible/malformed target

Rates outside `[0,1]`, zero total aggregation measure, missing universe or NaN values fail clearly.

### Raw preservation

Input artifact remains unchanged after projection.

## 15. Employment-universe tests

Construct fixtures distinguishing:

```text
all adults
working-age population
active population
```

Ensure the unemployment rate adapter uses the configured denominator and fails when required state/universe fields are absent.

A target defined over active persons must never be applied over all persons merely to make the code run.

## 16. Run artifact tests

For a completed synthetic run verify presence/content of:

```text
resolved config
manifest
split identity
prediction references
metrics
engineering acceptance
limitations
code/environment metadata
```

Test that changing a consequential resolved config field changes the run identity/hash input.

Test that a completed run is not overwritten silently.

## 17. CLI/Make tests

At minimum:

```text
validate good experiment -> 0
validate bad experiment -> nonzero
run synthetic experiment -> run directory
compare compatible runs -> comparison artifact
compare incompatible runs -> clear failure
report run -> human-readable summary
```

Make targets should simply invoke the CLI with visible parameters.

## 18. Real EPH acceptance run

The first bounded real-data proof should report:

```text
source/release identity
rows
households
periods
feature count
class prevalences
positive-income prevalence
fold counts
weight-use audit
```

Then produce direct and lean OOF results on the same split manifest.

At minimum inspect:

- person linear MAE/RMSE/bias;
- SD ratio;
- decile errors;
- household MAE/bias;
- intermediate log loss / calibration;
- oracle gain;
- cascade gain.

If any of these cannot be computed, the run report must say why.

## 19. CI layering

Suggested CI split:

### Every PR

```text
format/lint if configured
unit tests
scientific invariant tests
small synthetic end-to-end
```

### Optional/manual/scheduled

```text
real EPH bounded smoke, when data available
heavier architecture comparison
RF robustness
anchor time-series experiments
```

Do not require large private/local datasets for basic PR correctness.

## 20. Integration close checklist

Before declaring the sprint green, prove from the active runtime—not notebooks—that:

- direct and lean experiments resolve declaratively;
- both use identical household folds;
- HGB intermediate probabilities are typed and evaluated;
- the terminal full-population model handles zeros explicitly;
- real EPH OOF predictions exist;
- household aggregation exists;
- oracle/cascade diagnostics exist;
- run artifacts record lineage;
- optional anchor can be enabled without mutating raw source state;
- legacy output reproduction was not required to achieve any of the above.
