# Implementation Milestones

## 1. Principle

The sprint should advance through vertical slices that each leave the repository more truthful and usable.

Do not begin by creating every target directory and interface. Prefer the smallest slice that can be tested end to end.

The dependency order below is intentional.

## M0 — Freeze the active scientific surface

### Goal

Establish a clean baseline before refactoring.

### Work

- run current tests;
- identify current active imports of `transport_proof.py`, `census_sample_intake.py`, config helpers and legacy training files;
- record current `main` commit and test state;
- add/adjust docs if needed so legacy training/artifacts are unmistakably archaeological;
- identify current dependency versions and packaging entry points;
- identify the smallest real EPH data release/interface that can support subsequent work.

### Do not

- reproduce old serialized model outputs;
- rerun historical notebooks;
- preserve old output parity as a contract.

### Exit

```text
baseline documented
tests known
real-EPH candidate input identified
no ambiguity about active vs legacy path
```

## M1 — Reusable identity, fold and prediction primitives

### Goal

Extract the most trustworthy parts of the synthetic proof into reusable modules.

### Work

Implement/test:

- row/person/household identity helper;
- `FoldManifest` or equivalent;
- household-grouped fold assignment;
- `PredictionArtifact`;
- probability validation and class metadata;
- person-to-household aggregation primitive.

Refactor current synthetic proof only as needed to use these primitives.

### Acceptance

- every person gets one fold;
- no household crosses folds;
- fold assignment is stable under deterministic policy;
- prediction row IDs align exactly;
- probability artifacts validate;
- incomplete household membership is never hidden.

### Exit demonstration

A synthetic direct prediction can be represented and aggregated through the new primitives.

## M2 — Generic cross-fit and estimator adapter

### Goal

Make cross-fitting independent from a particular estimator or stage.

### Work

Implement:

```text
crossfit_predict(...)
fit_full_and_score(...)
```

plus estimator protocol.

Add HGB classifier/regressor adapters first.

HGB classifier requirements:

- binary and multiclass;
- stable class order;
- probability outputs;
- categorical-feature declaration;
- no hidden outer splitting.

HGB regressor requirements:

- squared-error mode;
- Gamma mode for positive targets;
- explicit `early_stopping=false` default in seed configs unless group-safe validation is supplied.

### Acceptance

Synthetic tests prove:

- held-out rows are never fitted by their own fold model;
- categorical OOF probabilities are complete and valid;
- full-fit scoring path works separately;
- feature schemas reach HGB correctly.

## M3 — Declarative experiment resolver and generic DAG

### Goal

Replace the hard-coded four-stage assumption with a validated DAG runtime.

### Work

Implement:

- config fragment loading;
- resolved experiment config;
- schema/semantic validation;
- generic node spec;
- topological execution;
- upstream prediction feature expansion;
- direct architecture;
- lean architecture;
- optional deep/historical-shaped config.

### Acceptance

- cycles fail;
- invalid/missing dependencies fail;
- forbidden observed-stage substitution fails;
- direct has zero learned upstream nodes;
- lean executes one learned layer;
- deep config can be represented without historical code imports;
- all candidates can share one fold manifest.

### Important

Historical deep config is a curiosity/reference hypothesis. It may be synthetically runnable if cheap, but old-output parity is irrelevant.

## M4 — Terminal hurdle and positive-income alternatives

### Goal

Represent the full person welfare population, including zero income.

### Work

Implement terminal formulations:

1. hurdle + positive log-income HGB;
2. hurdle + positive Gamma HGB.

Required outputs:

```text
p_positive
positive_amount_prediction
unconditional_expected_income
transform/retransformation metadata
```

### Acceptance

- zero/positive eligibility is explicit;
- positive-only regressor never trains on zero target rows in Gamma mode;
- combined unconditional point estimate exists for every eligible person;
- log formulation uses an explicit linear-scale resolution method;
- architecture comparison can freeze the terminal formulation.

## M5 — Oracle and cascade evaluator

### Goal

Make the scientific triangle measurable.

### Work

For each candidate latent state/block implement matched terminal evaluation for:

```text
X
X + Z_true
X + Z_oof
```

Compute:

```text
oracle gain
deployable cascade gain
capture ratio when meaningful
```

Also add stage reconstruction/calibration diagnostics.

### Acceptance

- oracle features are available only inside diagnostic evaluation;
- deployable graph remains leakage-safe;
- direct/oracle/cascade use identical outer folds;
- evaluator can distinguish low relevance from poor reconstructability.

## M6 — Real EPH execution

### Goal

Reach the first scientifically meaningful real-data run.

This is the key milestone of the sprint.

### Work

- bind one neutral/pinned real EPH frame;
- preserve person and household IDs;
- declare training period and cohort;
- define admissible feature candidate set for EPH-side science;
- preserve survey fields but do not use them in fit/evaluation under current policy;
- run a bounded real direct HGB experiment;
- run lean HGB experiment on the exact same split manifest;
- if cheap/stable, run deep candidate.

### Required evidence

- rows/households/fold counts;
- target prevalence/positive-income rate;
- direct OOF metrics;
- latent OOF calibration/reconstruction metrics;
- lean terminal OOF metrics;
- person and household predictions;
- no weight leakage.

### Stop condition

If real EPH cannot run because of a concrete source/interface problem, record the exact blocker and make the smallest upstream repair. Do not continue building optional anchor/report sophistication while the core real path is blocked.

## M7 — Distributional and household evaluator

### Goal

Move beyond average fit.

### Work

Add:

- linear-scale R²/MAE/RMSE/bias;
- SD ratio;
- predicted vs observed quantiles;
- error by observed/predicted income decile;
- tail bias/compression;
- household MAE/RMSE/bias;
- household dispersion/quantiles;
- error by household size;
- paired household-cluster bootstrap for candidate differences.

### Acceptance

A comparison report can answer whether an apparent gain survives household-level uncertainty and whether it improves or worsens low-tail/distributional behavior.

## M8 — Probability calibration option

### Goal

Calibrate only where evidence requires it.

### Work

- reliability diagnostics always available;
- add grouped inner calibration support;
- implement at least one binary method and one multiclass-suitable method if justified by current sklearn support;
- compare raw versus calibrated probability artifacts;
- measure terminal effect.

### Acceptance

No calibration row/fold leakage; class order preserved; raw outputs retained; calibration can be disabled cleanly.

### Note

This milestone can move earlier if raw HGB probabilities are clearly problematic for the lean stage, but it should not block the first real EPH run.

## M9 — Aggregate employment anchor

### Goal

Allow target-period aggregate evidence to condition a latent labor state.

### Work

Implement:

- anchor release schema;
- population-universe matcher;
- binary KL/moment projection;
- raw + anchored prediction artifacts;
- `lambda`, KL and displacement diagnostics;
- downstream raw/anchored welfare comparison;
- synthetic tests;
- one historical/real period demonstration if the external release is already available and governed.

### Acceptance

- aggregate constraint satisfied to tolerance;
- donor fields unchanged;
- raw result inspectable;
- external universe explicit;
- downstream terminal effects measured;
- anchor match is not automatically labeled scientific improvement.

## M10 — Bounded uncertainty v1

### Goal

Make uncertainty explicit without overbuilding.

### Work

- household-cluster bootstrap confidence intervals for metrics/differences;
- optional HGB quantile prediction path;
- anchor uncertainty metadata/draw interface;
- explicit limitations around household predictive covariance.

### Acceptance

No false claim of a full joint household posterior. All reported uncertainty quantities state what randomness/uncertainty they represent.

## M11 — Run packaging, CLI and Make

### Goal

Make the scientific system usable without notebooks.

### CLI surface

At minimum:

```text
encuestador validate <experiment>
encuestador run <experiment>
encuestador compare <runs...>
encuestador report <run-or-comparison>
```

Optional later:

```text
encuestador reproduce <run>
encuestador score <qualified-model> <scoring-frame>
```

### Make targets

Suggested:

```text
make test
make validate EXP=...
make run EXP=...
make compare GROUP=...
make report RUN=...
make accept RUN=...
```

Make targets wrap declared CLI semantics; they must not add hidden config changes.

### Run packaging

Emit the contract in `08_RUN_ARTIFACT_CONTRACT.md`.

## M12 — Integration and sprint close

### Goal

Prove the system as one scientific workflow.

### Required final comparison

At minimum, on real EPH:

```text
direct HGB vs lean HGB
```

under one fixed terminal formulation.

Strong close additionally includes:

```text
oracle diagnostics
log-vs-Gamma terminal comparison
selected RF robustness rerun
employment anchor demonstration
```

only if the core path is already green.

### Final deliverables

- active configs;
- tests green;
- reproducible run bundle(s);
- comparison summary;
- implementation status against `02_CURRENT_TO_TARGET.md`;
- explicit remaining reds/yellows;
- no claim that unresolved items are complete.

## Dependency summary

```text
M0
 |
 v
M1 -> M2 -> M3 -> M4 -> M5 -> M6
                              |
                              v
                             M7
                              |
                  +-----------+-----------+
                  v                       v
                 M8                      M10
                  |
                  v
                 M9
                  \                       /
                   +--------- M11 --------+
                              |
                              v
                             M12
```

M6 is the critical scientific gateway. Optional sophistication should not outrun it.
