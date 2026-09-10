# Runtime Architecture

## 1. Design goal

The active runtime should become a small scientific execution engine with explicit interfaces between:

- experiment resolution;
- data loading;
- fold assignment;
- estimator fitting;
- prediction representation;
- DAG execution;
- terminal welfare construction;
- evaluation;
- aggregate anchors;
- uncertainty;
- run packaging.

The design should generalize the useful mechanics in `src/encuestador/transport_proof.py` without preserving its hard-coded four-stage shape.

## 2. Recommended package shape

A reasonable target is:

```text
src/encuestador/
  experiments/
    schema.py
    resolve.py
    runner.py

  data/
    contracts.py
    eph.py

  splits/
    grouped.py
    crossfit.py

  models/
    base.py
    sklearn_estimators.py
    calibration.py
    terminal.py

  dag/
    spec.py
    executor.py

  aggregation/
    household.py

  evaluation/
    classification.py
    regression.py
    distribution.py
    household.py
    cascade.py

  anchors/
    base.py
    moment_projection.py
    employment.py

  uncertainty/
    bootstrap.py
    quantiles.py

  artifacts/
    runs.py
    manifests.py

  cli.py
```

Do not split further until the code actually demands it.

## 3. Core runtime types

### 3.1 Row identity

Every prediction artifact must carry row identity independently of row position.

At minimum for EPH:

```text
CODUSU
NRO_HOGAR
COMPONENTE
```

At Census scoring time use exact `sample_person_id` / `sample_household_id` as defined by the approved upstream contract.

Prediction joins by position are forbidden.

### 3.2 FoldManifest

A fold manifest should contain:

```python
@dataclass(frozen=True)
class FoldManifest:
    strategy: str
    group_columns: tuple[str, ...]
    n_splits: int
    seed: int | None
    row_id: ArrayLike
    fold_id: ArrayLike
```

Required invariants:

- every training row receives exactly one outer fold;
- all members of a group receive the same fold;
- every expected fold is nonempty;
- manifest digest is stable for a fixed row population/config;
- one manifest is reused across compared candidate architectures.

### 3.3 PredictionArtifact

A canonical prediction object should distinguish output semantics.

Illustrative shape:

```python
@dataclass
class PredictionArtifact:
    task: Literal["binary", "multiclass", "regression"]
    target: str
    row_ids: np.ndarray
    point: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    classes: tuple[Any, ...] | None = None
    quantiles: dict[float, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Required invariants:

- probability rows sum to one within tolerance;
- class order is persisted;
- finite-value rules are explicit;
- row order is never the only identity;
- OOF/full-score provenance is encoded in metadata.

A categorical probability vector and a hard class prediction must never be represented as an unlabeled numeric array.

### 3.4 FittedEstimator

Estimator adapters should expose a common interface:

```python
fit(X, y, feature_schema, fit_context) -> FittedEstimator
predict(model, X, row_ids, prediction_context) -> PredictionArtifact
```

`fit_context` may include:

- seed;
- sample weights, normally absent initially;
- categorical feature declaration;
- task/loss;
- calibration split if applicable.

Estimator code must not choose outer folds.

## 4. Split and cross-fit layer

The cross-fit API should own the pattern:

\[
\widehat y_i^{OOF}=f_{-fold(i)}(X_i).
\]

Illustrative interface:

```python
crossfit_predict(
    estimator_spec,
    X,
    y,
    fold_manifest,
    feature_schema,
) -> PredictionArtifact
```

For each outer fold:

1. fit only on rows outside the fold;
2. apply any inner calibration logic using training rows only;
3. predict the held-out fold;
4. place predictions back by stable row ID;
5. validate completeness.

At the end, optionally fit a final full-data estimator for scoring external rows.

The cross-fit layer should work identically for terminal models and learned intermediate stages.

## 5. DAG specification

Each node needs enough information to be executable without knowing the historical RFC numbering.

Illustrative structure:

```python
@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    targets: tuple[str, ...]
    task_by_target: Mapping[str, str]
    base_inputs: tuple[str, ...]
    upstream_nodes: tuple[str, ...]
    estimator_role: str
    output_representation: str
```

A node may have several parallel targets, but each target should remain separately evaluable.

Use the standard library's DAG utilities such as `graphlib.TopologicalSorter` unless a stronger graph package becomes genuinely necessary.

Validation must reject:

- cycles;
- missing node references;
- duplicate node IDs;
- duplicate output target names when ambiguous;
- observed-stage target injection into deployable node features;
- forbidden external features;
- architecture nodes with no usable inputs.

## 6. DAG execution semantics

During OOF training, downstream node `j` receives:

\[
W_j=[X_j,\widehat Z_1^{OOF},\ldots,\widehat Z_{j-1}^{OOF}].
\]

During final scoring, it receives:

\[
W_j^{score}=[X_j^{score},\widehat Z_1^{full},\ldots,\widehat Z_{j-1}^{full}].
\]

The executor therefore needs two related modes:

```text
crossfit training graph
full-fit scoring graph
```

These must use the same node definitions and feature semantics.

## 7. Representation expansion

When a categorical node outputs probabilities:

```text
CAT_OCUP probabilities with classes [0, 1, 2, 3]
```

the downstream design matrix should receive stable named columns, e.g.:

```text
latent.CAT_OCUP.p[0]
latent.CAT_OCUP.p[1]
latent.CAT_OCUP.p[2]
latent.CAT_OCUP.p[3]
```

Do not collapse to `argmax` unless an experiment explicitly requests hard-state representation.

Class labels and generated feature names must be persisted in the run manifest.

## 8. Feature schema

The runtime needs a lightweight schema per input column:

```text
name
dtype_role: continuous | categorical | binary | identifier | audit
semantic_class
temporal_role
allowed_as_external_input
missingness_policy
```

HGB categorical support should receive pandas categorical columns or the equivalent explicit categorical feature mask. Random Forest should use a governed encoding pipeline rather than ordinal integer codes for unordered categories.

## 9. Terminal model API

Terminal welfare formulations may themselves contain multiple estimators.

A `TerminalModel` interface should conceptually expose:

```python
fit_oof(...)-> TerminalPredictionArtifact
fit_full(...)-> FittedTerminalModel
predict(...)-> TerminalPredictionArtifact
```

For a hurdle model, the artifact may carry:

```text
p_positive
positive_amount_prediction
unconditional_expected_income
```

For a log-target model it should also carry transformation metadata and the explicit linear-scale resolution method.

The rest of the runtime should consume `unconditional_expected_income` as the person-level point welfare candidate rather than knowing the internals of the hurdle.

## 10. Oracle evaluation path

The core DAG executor should not accept observed intermediate labels as ordinary upstream predictions.

Oracle evaluation should instead be implemented as a separate evaluator/helper that constructs matched terminal design matrices:

```text
base:       X
oracle:     X + Z_true
cascade:    X + Z_oof
```

This avoids weakening the central deployable invariant.

## 11. Anchor path

Anchors should act on prediction artifacts after unconstrained latent prediction and before downstream consumers that are declared to depend on the anchored state.

Conceptually:

```python
raw = PredictionArtifact(... probabilities=q ...)
anchored, diagnostics = anchor.apply(raw, release, population_context)
```

The returned artifact must preserve a link to the raw one and declare:

```text
intervention=aggregate_anchor
anchor_release_id
projection_method
constraint_residual
```

No anchor implementation may mutate raw source columns in place.

## 12. Evaluation API

Evaluation functions should consume truth plus typed prediction artifacts, not fitted sklearn estimators.

This keeps evaluation independent from model family.

Examples:

```python
evaluate_classification(y_true, prediction_artifact)
evaluate_regression(y_true, prediction_artifact)
evaluate_households(person_truth, person_predictions, membership)
evaluate_cascade(base, oracle, cascade)
```

## 13. Run orchestration

`experiments.runner` should perform roughly:

```text
resolve config
load/validate data
materialize fold manifest
execute requested architecture OOF
fit terminal head OOF
run diagnostics
optionally apply anchors in declared experiment path
aggregate households
run uncertainty/evaluation
package artifacts
return RunResult
```

A separate scoring command can later:

```text
fit full approved model
load exact scoring frame
execute scoring DAG
aggregate welfare
emit household-welfare release
```

Do not conflate scientific qualification with Census scoring in the first implementation.

## 14. Legacy handling

The active runtime must not import historical training scripts.

Legacy code may remain where it is initially, but new modules and documentation should make its status unmistakable. A later cleanup may move it beneath an explicit `legacy/` namespace or archive directory.

No runtime adapter is required for historical serialized models.

## 15. Migration rule

Refactor only what is needed to establish the active interfaces.

`transport_proof.py` may continue to exist as a synthetic fixture driver while its reusable mechanics are extracted. Numerical parity with its basic synthetic behavior is useful during refactoring where cheap, but historical model/output parity is not a goal.

Once the generic path has its own equivalent synthetic tests, duplicated helper logic can be retired deliberately.
