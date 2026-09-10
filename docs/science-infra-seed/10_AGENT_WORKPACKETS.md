# Agent Work Packets

## 1. Purpose

This document turns the sprint into bounded, mergeable work packets for coding agents.

The failure mode to avoid is several agents independently “modernizing the repo” and producing incompatible abstractions.

Agents should work from the scientific contracts first, then the code.

## 2. Global rules for every agent

Before editing:

1. read `00_SPRINT_README.md`;
2. read `01_SCIENTIFIC_CONTRACT.md`;
3. read the work packet assigned here;
4. inspect the current implementation paths named by the packet;
5. run the smallest relevant baseline tests.

Every agent must:

- preserve exact person/household identity semantics;
- preserve household-grouped OOF requirements;
- keep observed stage labels out of deployable downstream training;
- keep EPH weights / Census selection probabilities / poverty weights semantically separate;
- avoid notebook-only implementations;
- add or update tests for every new invariant;
- report exact files changed and exact commands run;
- leave an explicit handoff note for downstream packets;
- avoid broad unrelated cleanup.

Every agent must not:

- rebuild legacy output parity;
- preserve historical serialized models as an API;
- copy large datasets/models into Git;
- silently alter estimands or cohort definitions;
- invent target-period calibration;
- add many estimator families;
- weaken validation because a real dataset is inconvenient.

## 3. Interface-change rule

A work packet may add the smallest interface it needs inside its owned area.

It may **not silently redesign an upstream interface owned by another packet**.

If an upstream API is insufficient:

1. document the mismatch;
2. propose the smallest compatible extension;
3. add a focused test demonstrating why it is needed;
4. coordinate through the integration packet rather than forking the architecture.

## 4. Work Packet A — Core contracts, identities and cross-fitting

### Dependency

Start after M0 baseline census.

### Primary ownership

```text
src/encuestador/splits/
src/encuestador/models/base.py    # prediction/protocol types only
src/encuestador/aggregation/
related tests
```

### Inspect first

```text
src/encuestador/transport_proof.py
contracts/functional_interface.yaml
contracts/model_specs/historical_staged_v1.json
```

### Tasks

1. Introduce stable row/group identity helpers.
2. Implement `FoldManifest` or equivalent.
3. Extract household-grouped fold generation.
4. Implement typed `PredictionArtifact` with:
   - task;
   - target;
   - row IDs;
   - point predictions;
   - probability matrix;
   - class order;
   - optional quantiles;
   - provenance metadata.
5. Implement generic cross-fit skeleton/protocol if it can be done without estimator-specific policy; otherwise expose the primitives needed by Packet B.
6. Extract/implement household sum aggregation with explicit incomplete-member status.
7. Add tests for leakage, alignment, completeness and probability validation.

### Must preserve

For every household `h`:

\[
|\{fold(i):i\in h\}|=1.
\]

### Deliverable

A small reusable core capable of representing OOF predictions and aggregating them to households without knowing anything about HGB or cascade stage names.

### Handoff

Document the exact protocol Packet B must implement for estimators and the exact prediction shape Packet C should consume.

## 5. Work Packet B — sklearn estimator and terminal model layer

### Dependency

Packet A prediction/fold interfaces stable enough to consume.

### Primary ownership

```text
src/encuestador/models/sklearn_estimators.py
src/encuestador/models/terminal.py
src/encuestador/models/calibration.py   # raw hooks / optional initial methods
related configs/tests
```

### Tasks

1. Implement HGB classifier adapter:
   - binary/multiclass;
   - explicit categorical feature handling;
   - stable class order;
   - `predict_proba` output;
   - no hidden outer CV.
2. Implement HGB regressor adapter:
   - squared error;
   - Gamma loss;
   - `early_stopping=false` in seed configs unless group-safe validation supplied.
3. Implement governed RF challenger adapters with appropriate categorical preprocessing.
4. Implement hurdle terminal model:
   - probability of positive income;
   - positive amount model;
   - unconditional expected person income.
5. Implement both positive-amount formulations:
   - log-target squared error with explicit inverse-resolution policy;
   - Gamma positive linear-income target.
6. Ensure model outputs are `PredictionArtifact`s rather than bare arrays.
7. Add focused tests including class-order stability, zero/positive eligibility, positivity of Gamma predictions and categorical schema routing.

### Do not

- perform architecture selection;
- perform broad hyperparameter tuning;
- implement a model zoo;
- choose fold assignments internally.

### Deliverable

A strong nonlinear estimator/terminal layer usable by any DAG architecture.

## 6. Work Packet C — Declarative experiment and DAG runtime

### Dependency

Packet A core interfaces; Packet B estimator protocol may be developed in parallel if interface contract is agreed.

### Primary ownership

```text
src/encuestador/experiments/
src/encuestador/dag/
configs/architectures/
configs/splits/
configs/experiments/
config schema/tests
```

### Tasks

1. Implement config fragment loading and canonical resolution.
2. Emit `resolved_config.yaml`-equivalent object before fitting.
3. Define generic node spec.
4. Validate DAG using standard-library graph utilities unless stronger dependency is necessary.
5. Implement training-time OOF graph execution.
6. Implement full-fit scoring graph execution interface.
7. Expand categorical probability artifacts into stable named upstream features.
8. Add architecture configs:
   - `direct_v1`;
   - `lean_labor_v1`;
   - `deep_candidate_v1` or a clearly dormant historical-shaped config.
9. Ensure one fold manifest can be injected/reused across candidate architectures.
10. Fail on cycles, invalid references, forbidden features and observed-stage substitution.

### Legacy rule

The deep/historical-shaped config is **not** required to reproduce old RFC binaries, preprocessing quirks or outputs. It may remain disabled/dormant until explicitly selected.

### Deliverable

One common executor where architecture is data/config rather than hard-coded Python control flow.

## 7. Work Packet D — Scientific evaluator and oracle diagnostics

### Dependency

Typed prediction artifacts from A; experiment runner contract from C; estimator outputs from B.

### Primary ownership

```text
src/encuestador/evaluation/
configs/evaluation/
report/metric schemas
related tests
```

### Tasks

1. Classification metrics and calibration diagnostics.
2. Person regression metrics on linear scale.
3. Distributional diagnostics:
   - SD ratio;
   - quantile agreement;
   - error by observed/predicted decile;
   - low/high-tail bias.
4. Household metrics:
   - MAE/RMSE/bias;
   - dispersion/quantiles;
   - household-size slices.
5. Oracle/deployable decomposition:
   - `X`;
   - `X + Z_true`;
   - `X + Z_oof`.
6. Compute oracle gain, cascade gain and capture ratio where defined.
7. Implement paired household-cluster bootstrap for metric differences.
8. Make subgroup evaluation configurable.
9. Produce machine-readable evaluation output independent of fitted estimator class.

### Critical invariant

Oracle evaluation must not weaken ordinary DAG feature validation. It is a diagnostic branch only.

### Deliverable

An evaluator capable of deciding *why* a learned stage helps or fails, not merely reporting headline R².

## 8. Work Packet E — Real EPH integration

### Dependency

A + B + C minimally functional; D basic metrics available.

### Primary ownership

```text
src/encuestador/data/eph.py
configs/data/
configs/features/
real-data smoke/integration tests
run manifests tied to exact source release
```

### Tasks

1. Inspect available neutral/current EPH source interfaces.
2. Select the smallest source-backed EPH frame that preserves:
   - `CODUSU`;
   - `NRO_HOGAR`;
   - person/component identity;
   - year/quarter;
   - required candidate features/targets;
   - design fields for audit.
3. Pin the release/path/content identity.
4. Declare the surveyor training population separately from the positive-income thesis cohort.
5. Ensure zeros/no-income cases are represented for the hurdle terminal model.
6. Build the candidate shared/deployable feature schema for **EPH-side model science** without waiting for complete real Census semantic approval.
7. Run a bounded direct HGB real experiment.
8. Run lean HGB on the exact same folds.
9. Run oracle diagnostics for the lean latent block.
10. Run deep candidate only if cheap/stable and explicitly selected.
11. Verify no `PONDERA`-family field enters model features/weights under the current policy.

### Stop rule

If blocked, produce:

```text
BLOCKER
exact file/interface
why it blocks scientific execution
smallest repair
exact command to retry
```

Do not compensate for a blocked real path by polishing optional modules.

### Deliverable

The first truthful real-EPH architecture comparison produced by the new engine.

## 9. Work Packet F — Aggregate employment anchor

### Dependency

Real/raw latent predictions and stable prediction artifact interface. Can implement synthetic math earlier, but integration waits for core runtime.

### Primary ownership

```text
src/encuestador/anchors/
configs/anchors/
anchor release schema/tests
```

### Tasks

1. Define versioned anchor release schema.
2. Implement generic moment-projection interface.
3. Implement binary Bernoulli KL/logit-shift projection.
4. Validate anchor population universe.
5. Preserve raw and anchored artifacts separately.
6. Emit:
   - target/raw/anchored aggregate;
   - residual;
   - `lambda`;
   - KL displacement;
   - individual probability-displacement statistics.
7. Implement first employment/unemployment adapter.
8. Wire downstream re-evaluation of welfare predictions after the anchor.
9. Add synthetic tests and one bounded real/historical-period example if a governed external release is available.

### Do not

- overwrite donor row labels;
- implement random hard-label flipping as the canonical method;
- use sample weights merely because they exist;
- treat the anchor as scientific improvement by construction.

### Deliverable

A transparent conditional-inference intervention that can be turned on/off in experiment config.

## 10. Work Packet G — Run artifacts, CLI, Make and integration

### Dependency

Starts lightly after C; completes after E/D and optionally F.

### Primary ownership

```text
src/encuestador/artifacts/
src/encuestador/cli.py
Makefile
run/report integration
```

### Tasks

1. Implement run directory/manifest writer.
2. Record code/environment/input/config/fold identities.
3. Add engineering acceptance artifact.
4. Add comparison artifact.
5. Implement CLI:
   - `validate`;
   - `run`;
   - `compare`;
   - `report`.
6. Add Make targets wrapping CLI.
7. Ensure Make/CLI add no hidden science defaults.
8. Produce final sprint demonstration runs/comparisons.
9. Update `02_CURRENT_TO_TARGET.md` status in a follow-up implementation report rather than rewriting seed intent.

### Deliverable

A scientist can run and compare experiments without opening a notebook.

## 11. Suggested concurrency

The safest sequencing is:

```text
A core primitives
|
+--> B models ------------------+
|                               |
+--> C experiment/DAG ----------+--> E real EPH
|                               |       |
+--> D evaluator (after types) -+       +--> F anchor
                                        |
C early -----------------------------> G packaging
                                        |
                                        v
                                  final integration
```

A, B, C and D should not all edit the same general-purpose module. Keep ownership boundaries explicit.

## 12. Merge/PR expectations

Each work packet should be a small number of coherent commits or one focused PR.

A packet report should state:

```text
WHAT CHANGED
FILES OWNED/TOUCHED
TESTS RUN
SCIENTIFIC INVARIANTS PROVED
KNOWN LIMITATIONS
HANDOFF / NEXT DEPENDENCY
```

Avoid prose-only “done” reports.

## 13. Integration authority

Packet G/integration owns conflict resolution across interfaces.

When two packets disagree, resolve by this priority:

1. `01_SCIENTIFIC_CONTRACT.md`;
2. current machine-readable repository contracts where compatible;
3. `03_EXPERIMENT_MODEL.md` / `04_RUNTIME_ARCHITECTURE.md`;
4. smallest interface that enables real EPH evidence;
5. implementation convenience.

Implementation convenience comes last.

## 14. Sprint stop conditions

Agents should stop and surface rather than improvise if they encounter:

- ambiguous target/cohort semantics;
- missing household identity;
- a feature that appears target-derived/leaky;
- an anchor universe that cannot be matched;
- a proposed calibration split that breaks group isolation;
- an inverse transform whose welfare estimand is unclear;
- a need to use a design weight not authorized by the experiment;
- a real input that does not match its declared release/schema.

The correct response is a bounded blocker report and smallest repair, not silent scientific drift.
