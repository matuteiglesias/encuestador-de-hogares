# Experiment Model and Configuration Semantics

## 1. Objective

The experiment layer must let us vary one scientific axis while holding the others fixed.

The primary failure mode to avoid is an experiment that simultaneously changes:

- cascade depth;
- feature set;
- target transform;
- estimator family;
- split policy;
- calibration;
- anchor behavior;
- evaluation rules.

Such a run may produce numbers but cannot answer a clean scientific question.

## 2. Canonical experiment decomposition

Every resolved experiment is:

\[
\mathcal E=(D,X,G,T,M,S,C,A,U,V).
\]

### `D` — data

Declares:

- source release identity;
- training years/quarters;
- row unit;
- training population/cohort;
- household/person identity columns;
- target eligibility rules;
- exact monetary reference when applicable.

### `X` — feature plane

Declares the approved candidate inputs and their semantic types:

```text
continuous
categorical
binary
identifier/audit-only
forbidden
```

The feature plane also carries temporal roles and deployment admissibility.

### `G` — inference graph

Declares the learned dependency structure:

- nodes;
- targets;
- dependencies;
- learned intermediate representations;
- terminal node(s).

`G` must not encode estimator hyperparameters.

### `T` — terminal welfare formulation

Examples:

- positive-log regression;
- positive Gamma regression;
- hurdle + positive-log amount;
- hurdle + Gamma amount;
- direct household target in a future experiment.

This is deliberately distinct from cascade depth.

### `M` — estimators

Declares estimator families and parameters by node/task role.

Initial families:

- HGB primary;
- Random Forest robustness challenger.

### `S` — splitting/cross-fitting

Declares:

- grouping columns;
- outer fold strategy;
- random seed;
- number of folds;
- optional grouped inner calibration split;
- optional out-of-time sensitivity split.

### `C` — calibration

Declares whether stage probabilities are:

- raw;
- sigmoid calibrated;
- isotonic calibrated;
- temperature calibrated;
- other future governed method.

Calibration is a separate transform with its own fitted state and diagnostics.

### `A` — aggregate anchors

Declares zero or more external aggregate constraints and their releases.

Default:

```text
none
```

Anchors must not be implied by architecture.

### `U` — uncertainty

Declares optional uncertainty mechanisms such as:

- categorical probabilities;
- quantile fits;
- household-cluster bootstrap;
- anchor-value uncertainty propagation.

### `V` — evaluation

Declares metrics, diagnostic slices and promotion comparisons.

## 3. Configuration hierarchy

Recommended configuration structure:

```text
configs/
  experiments/
  data/
  features/
  architectures/
  terminal/
  estimators/
  splits/
  calibration/
  anchors/
  uncertainty/
  evaluation/
```

An experiment file should mostly compose named components rather than duplicate them.

Example:

```yaml
experiment:
  id: lean_labor_hgb_gamma_2024_2025_v1
  seed: 42

data: data/eph_2024_2025_transport_v1.yaml
features: features/census_deployable_candidate_v1.yaml
architecture: architectures/lean_labor_v1.yaml
terminal: terminal/hurdle_gamma_v1.yaml
estimators: estimators/hgb_v1.yaml
splits: splits/household_grouped_5fold_v1.yaml
calibration: calibration/raw_v1.yaml
anchors: null
uncertainty: uncertainty/core_v1.yaml
evaluation: evaluation/welfare_core_v1.yaml
```

## 4. Resolution

The runtime must produce a single immutable **resolved config** before fitting.

Resolution should:

1. load referenced fragments;
2. apply explicit experiment overrides;
3. validate schemas and references;
4. canonicalize defaults;
5. materialize the exact effective configuration;
6. compute a stable config digest;
7. fail before fitting on invalid science.

The resolved config, not the collection of source YAML fragments, is the run authority.

## 5. Allowed overrides

Experiment-local overrides are useful but must remain explicit.

Good:

```yaml
estimators:
  terminal_positive:
    params:
      max_iter: 300
```

Bad:

```yaml
magic_mode: fast
```

when `fast` silently changes folds, targets, features and estimator capacity.

No convenience flag may bundle scientifically consequential changes without expanding into the resolved config.

## 6. Architecture configs

### 6.1 Direct

```yaml
architecture:
  id: direct_v1
  nodes: []
```

A direct experiment may still have a multi-part terminal head such as hurdle + positive amount. This does not make it a cascade.

### 6.2 Lean labor

Initial hypothesis:

```yaml
architecture:
  id: lean_labor_v1
  nodes:
    labor_state:
      kind: parallel_categorical_latents
      targets:
        - CAT_OCUP
        - CAT_INAC
        - CH07
      inputs:
        - shared_observables
      output_representation: probabilities
```

The exact target membership is not sacred. Ablations may remove weak states.

### 6.3 Deep candidate

A deeper DAG may encode historically inspired target groups. It exists to test whether additional decomposition helps.

It is not an active compatibility contract.

Do not attach requirements such as:

- old RF estimator family;
- old serialized file naming;
- old exact preprocessing;
- old Census outputs;
- old stage prediction values.

## 7. Oracle configurations

Oracle runs are evaluation variants, not deployment architectures.

A clean design is:

```yaml
evaluation:
  oracle:
    enabled: true
    target_blocks:
      - labor_state
```

The evaluator then fits matched terminal models with:

```text
X
X + Z_true
X + Z_oof
```

under identical outer evaluation folds.

Observed intermediate labels must never leak into the deployable path merely because oracle evaluation is enabled.

## 8. Terminal configurations

Suggested first terminal formulations:

### `hurdle_log_v1`

```text
HGBClassifier -> P(Y > 0 | W)
HGBRegressor  -> E[log10(Y) | Y > 0, W]
explicit retransformation policy
```

### `hurdle_gamma_v1`

```text
HGBClassifier -> P(Y > 0 | W)
HGBRegressor(loss="gamma") -> E[Y | Y > 0, W]
E[Y | W] = P(Y > 0 | W) * E[Y | Y > 0, W]
```

When comparing architectures, freeze one terminal config.

When comparing terminal formulations, freeze the architecture.

## 9. Estimator configs

Estimator configs should declare only estimator mechanics.

Example:

```yaml
family: hist_gradient_boosting
categorical:
  estimator: HistGradientBoostingClassifier
  params:
    learning_rate: 0.05
    max_iter: 200
    early_stopping: false

continuous:
  estimator: HistGradientBoostingRegressor
  params:
    learning_rate: 0.05
    max_iter: 200
    early_stopping: false
```

Task-specific loss may be defined in the terminal config or node config, but the resolved config must make the final estimator call unambiguous.

## 10. Split configs

Initial authority:

```yaml
strategy: household_grouped
n_splits: 5
group_columns:
  - CODUSU
  - NRO_HOGAR
seed: 42
```

The fold assignment must be generated once per experiment and reused by every architecture/node being compared.

Do not independently ask each sklearn model to create its own folds.

## 11. Calibration configs

Default:

```yaml
method: none
```

If enabled, calibration must declare grouped inner-split semantics.

Example:

```yaml
method: temperature
inner_split:
  strategy: household_grouped
  n_splits: 3
```

Calibration should be enabled because diagnostics justify it, not by default ritual.

## 12. Anchor configs

Example:

```yaml
anchor:
  id: unemployment_rate_2025q2
  concept: unemployment_rate
  release: anchors/releases/unemployment_2025q2.yaml
  target_node: labor_state.CONDACT
  projection: binary_kl_moment
  mode: expected_moment
```

The anchor release must define its population universe. A rate over economically active persons is not interchangeable with a rate over the full adult population.

## 13. Evaluation configs

The evaluation profile should identify:

- node-level classification diagnostics;
- oracle comparisons;
- person-income metrics;
- distributional diagnostics;
- household metrics;
- bootstrap settings;
- optional poverty-threshold diagnostic inputs.

Metrics must not be hard-coded into estimator adapters.

## 14. Config identity and comparison groups

Every experiment should optionally declare a scientific comparison group:

```yaml
comparison:
  group: cascade_depth_hgb_gamma_v1
  axis: architecture
```

Then `compare` can reject apples-to-oranges runs unless explicitly forced.

For an architecture comparison, the comparison contract should assert equality of:

```text
data
feature plane
terminal formulation
estimator family/settings
split manifest
calibration policy
anchor policy
uncertainty policy
evaluation profile
```

except for the architecture field.

Equivalent contracts should exist for terminal-formulation and estimator-robustness comparisons.

## 15. No hidden defaults

Defaults may exist for ergonomics, but after resolution every consequential field must appear explicitly in `resolved_config.yaml`.

A future reader should never need to know what the Python code happened to default to on 2026-09-10 in order to interpret a run.
