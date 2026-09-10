# Current-to-Target Implementation Map

## 1. Purpose

This document converts the repository's current implementation state into explicit sprint acceptance criteria.

The point is not to reward the existence of files. A light becomes green only when the capability is usable through the active modern path and is covered by tests or run evidence.

Historical code and artifacts are not part of the active implementation baseline.

## 2. Existing assets worth reusing

The current repository already contains several strong ingredients.

### 2.1 Scientific boundary and machine-readable contract

`README.md`, `SYSTEM.yaml`, `docs/FUNCTIONAL_CONTRACT.md`, `contracts/functional_interface.yaml` and `contracts/deployment_dag.yaml` already establish:

- transport ownership;
- direct/hurdle/staged comparison;
- household-aware OOF semantics;
- separation of EPH weights, Census sample probabilities and poverty weights;
- optional explicit aggregate calibration;
- exact Census identity preservation;
- household welfare as the downstream handoff.

These remain authoritative unless explicitly amended by the sprint.

### 2.2 Synthetic OOF transport kernel

`src/encuestador/transport_proof.py` currently proves several important mechanics:

- deterministic household-grouped fold assignment;
- no household crossing folds;
- OOF intermediate generation;
- full-model scoring path distinct from OOF training path;
- direct and staged terminal predictions;
- person-to-household sum aggregation;
- refusal to hide invalid household members;
- basic person and household error metrics;
- explicit non-use of survey/sampler design weights.

This code is a useful prototype, but it is hard-coded to one synthetic four-stage setting and a simple Ridge implementation.

### 2.3 Historical evidence

Legacy files such as `train.py`, `entrenar_modelos.py`, historical notebooks and serialized RFC artifacts show earlier scientific ideas, including Random Forest stages, probability use and unemployment adjustment.

They are archaeology only.

No active implementation must preserve their API, predictions, serialized model format or numerical outputs.

## 3. Light matrix

| Capability | Current state | Sprint target | Green criterion |
|---|---|---|---|
| Scientific repository boundary | GREEN | retain | active docs/contracts remain coherent and tests do not violate ownership boundaries |
| Declarative experiments | YELLOW | composable YAML-driven runs | one resolved experiment config fully identifies data, DAG, terminal head, estimator, split, calibration, anchor, uncertainty and evaluation choices |
| DAG execution | YELLOW | generic acyclic graph runtime | direct, lean and deep configs execute through the same engine; cycles and invalid dependencies fail validation |
| Household OOF | GREEN in synthetic proof | generic primitive | reusable cross-fit API works for all stage types and proves no household leakage |
| OOF prediction artifacts | YELLOW | typed artifacts | categorical probabilities, regression points and optional quantiles retain target/class metadata and row identity |
| Direct baseline | GREEN in proof | first-class architecture | direct config uses exact same folds/data/evaluator as cascade candidates |
| Lean cascade | RED | one-layer candidate | executable config using one parallel learned latent layer; stage membership can be ablated |
| Deep/historical-shaped candidate | YELLOW | optional curiosity/reference architecture | config can express a deeper DAG, but active sprint does not require historical output parity or model reproduction |
| Oracle diagnostics | RED | explicit diagnostic mode | evaluator can compare `X`, `X + Z_true`, and `X + Z_oof` without weakening deployable leakage rules |
| HGB classifiers | RED | primary categorical estimator | native categorical-capable HGB adapter emits stable class-labelled `predict_proba` outputs under grouped OOF |
| HGB regressors | RED | primary continuous estimator | HGB adapter supports squared-error/log-target and Gamma positive-income variants |
| Random Forest | historical only | one robustness challenger | governed adapter and preprocessing path can rerun selected experiments without becoming a model zoo |
| Probability calibration | RED | diagnostic + optional intervention | reliability/log-loss/Brier diagnostics exist; calibration can be enabled with group-safe inner splitting |
| Terminal hurdle | YELLOW conceptually | explicit terminal formulation | positive-income probability and positive-income amount are modeled separately and combined on linear scale |
| Real EPH runner | RED | active scientific path | one pinned real EPH frame runs through the same experiment engine and produces valid OOF evidence |
| Feature-role enforcement | YELLOW/GREEN in contracts | runtime validation | forbidden or non-admissible deployment inputs fail before fitting |
| Weight separation | GREEN contract/proof | runtime invariant | no EPH/Census design weight enters features or fitting unless explicitly declared by a future experiment |
| Household aggregation | GREEN in proof | reusable module | complete-member linear sum with exact identity and failure accounting |
| Distributional evaluation | RED | active evaluator | SD ratio, decile bias, quantile agreement, tail compression and linear-scale errors emitted automatically |
| Stage evaluation | RED | active evaluator | calibration, reconstruction, oracle gain, deployable gain and capture diagnostics available by stage |
| Household evaluation | YELLOW | strong evaluator | MAE/RMSE/bias plus distributional and cluster-aware uncertainty metrics |
| Poverty-aware diagnostics | RED/YELLOW | bounded diagnostic support | optional threshold-distance and classification diagnostics are possible without making this repo owner of poverty methodology |
| Aggregate anchors | YELLOW contract only | explicit projection engine | generic moment projection + employment adapter produce raw/anchored outputs and diagnostics without donor mutation |
| Anchor uncertainty | RED | optional v1 | anchor value may carry uncertainty metadata and be propagated in simulations/resampling when enabled |
| Predictive uncertainty | RED | bounded v1 | stage probabilities, optional quantile predictions and household-cluster bootstrap scientific CIs supported |
| Immutable run bundle | YELLOW | governed runs | every run records resolved config, code/data identities, split manifest, metrics, diagnostics and limitations |
| CLI | RED | stable command surface | `validate`, `run`, `compare`, `report` work without notebooks |
| Makefile | RED | human-facing shortcuts | stable Make targets wrap CLI and tests; targets do not encode hidden science |
| Tests | YELLOW | layered suite | unit, invariant, synthetic integration and bounded real-data smoke tests cover active path |
| Legacy separation | RED/YELLOW | unmistakable status | active docs and imports do not present old training scripts/artifacts as current scientific execution surfaces |

## 4. What must not block the sprint

The following are explicitly not prerequisites for turning the core scientific infrastructure green:

- reproducing any historical serialized RF model;
- reproducing historical notebook plots;
- preserving numerical parity with old Census predictions;
- maintaining old model file formats;
- recreating the artisanal historical unemployment mutation;
- running the historical architecture successfully on real data;
- tuning many estimator families;
- solving complete Bayesian household predictive uncertainty;
- obtaining a real Census semantic-plane approval before real-EPH model science can proceed.

## 5. Active-path priorities

The highest-value dependency chain is:

```text
household split + typed prediction artifacts
        |
        v
generic experiment/DAG runner
        |
        v
HGB classifier/regressor adapters
        |
        v
direct + lean + deep candidate configs
        |
        v
oracle/cascade evaluator
        |
        v
real EPH experiment
```

Only once this works should the sprint spend substantial time on:

```text
aggregate anchors
uncertainty extensions
run comparison/report polish
```

## 6. Red-light triage principle

Not every red item deserves equal work.

A red item is **critical** if it blocks real, honest architecture comparison.

Critical reds:

- experiment/DAG runtime;
- typed probabilities;
- HGB adapters;
- lean architecture;
- oracle diagnostics;
- real EPH execution;
- strong evaluation.

A red item is **secondary** if it enriches an already truthful experiment.

Secondary reds:

- aggregate anchors;
- quantile/predictive uncertainty;
- poverty-threshold diagnostics;
- polished reports.

Agents should preserve this ordering.

## 7. Green-state demonstration

The sprint should end with at least one compact demonstration directory or report showing:

1. the exact resolved experiment config;
2. household fold counts and leakage check;
3. direct and lean HGB OOF metrics on real EPH;
4. at least one intermediate categorical probability diagnostic;
5. oracle vs OOF terminal-value decomposition for the lean state;
6. person-income distributional diagnostics;
7. household-level aggregation/error diagnostics;
8. optional deep candidate result if cheap enough;
9. optional anchor result if the core path is already green;
10. full run lineage and non-claims.

The proof of green is evidence generated by the active runtime, not the existence of planned modules.
