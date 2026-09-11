# Science Infrastructure Sprint — Seed Bundle

**Repository:** `encuestador-de-hogares`  
**Bundle status:** implementation seed / agent work contract  
**Date:** 2026-09-10  
**Active path:** modern scientific transport infrastructure  

## Purpose

This bundle defines the implementation sprint that turns `encuestador-de-hogares` from a well-specified but partly hard-coded transport proof into a reusable scientific inference laboratory.

The goal is not to preserve the historical implementation. The historical RFC cascade, notebooks, serialized models, figures and produced outputs are **archaeological evidence only**. They may remain available for curiosity and reconstruction, but no sprint acceptance criterion depends on reproducing them. Historical configuration concepts may be retained when they are useful as candidate experimental settings.

The sprint succeeds when the repository can run reproducible, household-safe, real-EPH experiments that compare a small set of scientifically meaningful welfare-inference architectures under common machinery, produce auditable diagnostics, and optionally condition latent-state predictions on governed external aggregate evidence.

## Current execution overlay

The seed was written before the September 10 real-EPH integration sprint completed. Do **not** replay the original milestone sequence mechanically.

For current work, read the design contract in `00`–`12`, then use these overlays to determine what is already green and what should be pulled next:

- `13_IMPLEMENTATION_STATUS_20260910.md` — evidence-backed capability ledger after the real-EPH integration close;
- `14_CURRENT_DEVELOPMENT_DIRECTIVE_20260911.md` — current dependency order, exact input pins, hard gates and agent guidance.

When an older milestone says to build a capability that `13` records as green and current `main` confirms is present, reuse the landed implementation rather than creating a parallel version. When code, `13`, and `14` disagree, surface the mismatch and resolve it explicitly.

## Core scientific object

An experiment is conceptually:

\[
\mathcal E=(D,X,G,T,M,S,C,A,U,V)
\]

where:

- `D`: exact data release and cohort;
- `X`: admissible observable feature plane;
- `G`: inference graph / cascade architecture;
- `T`: terminal welfare formulation;
- `M`: estimator family and parameters;
- `S`: split / cross-fit policy;
- `C`: probability-calibration policy;
- `A`: optional aggregate anchors;
- `U`: uncertainty policy;
- `V`: evaluation profile.

These dimensions must remain separable. In particular, cascade depth and terminal income formulation must not be changed simultaneously when the scientific question is architecture.

## Primary architecture comparison

The first active comparison is deliberately small:

1. **Direct** — no learned intermediate state between shared observables and terminal welfare.
2. **Lean labor** — one parallel learned latent layer containing only empirically useful labor/income-state representations.
3. **Historical-depth candidate** — a deeper staged graph retained as a scientific hypothesis and curiosity, not as privileged legacy behavior.

The historical-depth config may be available, but it is not required to match old model binaries or old output files.

## Primary estimator policy

The default nonlinear tabular family is scikit-learn HistGradientBoosting:

- `HistGradientBoostingClassifier` for categorical latent states, emitting probability vectors;
- `HistGradientBoostingRegressor` for continuous terminal quantities;
- positive-income experiments should support both a log-income squared-error formulation and a Gamma-loss formulation on positive linear income;
- Random Forest is the single robustness challenger, not a parallel model zoo.

Linear estimators may remain useful as synthetic-test helpers or local diagnostics, but they are not a required scientific contender for this sprint.

## Non-negotiable split semantics

Every learned intermediate consumed downstream during training must be generated out-of-fold under a household-aware grouping policy.

Default household identity:

```text
CODUSU + NRO_HOGAR
```

No member of a household may appear in both training and validation portions of the same outer fold. Observed intermediate labels may never be substituted for predicted intermediates in deployable evidence.

Oracle experiments are allowed only as explicitly labeled diagnostics.

## External employment information

Aggregate employment/unemployment time series are permitted as an explicit second source of information. They must never be smuggled into donor rows as if they were observed person-level facts.

The active design is:

```text
micro model                    aggregate evidence
p(Z | X)             +              A_t
        \                          /
         ---- governed projection ----
                     |
                     v
               p*(Z | X, A_t)
                     |
                     v
                welfare model
```

The first implementation target is a moment-constrained information projection, with the binary case reducing to an additive logit shift chosen to satisfy the aggregate moment.

## Sprint order

The original implementation order was intentional:

```text
1. freeze active scientific contracts
2. extract reusable split / OOF / aggregation primitives
3. build generic experiment + DAG runtime
4. add HGB estimator adapters and probability artifacts
5. implement direct / lean / deep configs + oracle diagnostics
6. execute the same machinery on real EPH
7. strengthen evaluation
8. add governed aggregate-anchor projection
9. add bounded uncertainty support
10. finalize run artifacts, CLI, Make targets and acceptance gates
```

This remains useful history for understanding dependencies, but the current repository has already crossed several of these milestones. `14_CURRENT_DEVELOPMENT_DIRECTIVE_20260911.md` is authoritative for the next pull order.

## Definition of done

A strong sprint endpoint is:

> One common experiment engine can resolve a declarative experiment, generate household-safe OOF predictions on real EPH, execute direct / lean / deep candidate graphs using HGB, emit categorical probability artifacts with calibration diagnostics, compare oracle versus deployable intermediate value, aggregate person predictions to households, optionally apply a governed employment anchor, and package the run with exact configuration, lineage and evaluation evidence.

An engineering-successful run does not imply scientific promotion. A candidate may execute correctly and still lose to the direct baseline.

## Document map

Read in this order for current development:

1. `01_SCIENTIFIC_CONTRACT.md` — estimand, information boundaries, scientific invariants.
2. `05_MODEL_AND_CASCADE_POLICY.md` — HGB/RF, hurdle, Gamma/log, probabilities, architecture policy.
3. `06_EVALUATION_AND_ACCEPTANCE.md` — scientific diagnostics and acceptance rules.
4. `12_NON_GOALS_AND_GUARDRAILS.md` — explicit scope controls.
5. `13_IMPLEMENTATION_STATUS_20260910.md` — what actually became green/red on real EPH.
6. `14_CURRENT_DEVELOPMENT_DIRECTIVE_20260911.md` — what to do now and in what order.

Use the remaining documents as detailed references:

- `02_CURRENT_TO_TARGET.md` — original implementation gap map and green criteria;
- `03_EXPERIMENT_MODEL.md` — experiment composition and config semantics;
- `04_RUNTIME_ARCHITECTURE.md` — target Python architecture and runtime interfaces;
- `07_ANCHORS_AND_UNCERTAINTY.md` — aggregate constraints and bounded uncertainty;
- `08_RUN_ARTIFACT_CONTRACT.md` — immutable run evidence and storage rules;
- `09_IMPLEMENTATION_MILESTONES.md` — original dependency-ordered vertical slices;
- `10_AGENT_WORKPACKETS.md` — bounded agent ownership and handoffs;
- `11_TEST_AND_INTEGRATION_MATRIX.md` — tests required at each layer.

## Working rule for agents

When the bundle and current code disagree, do not silently reinterpret the science. Surface the mismatch in the PR or implementation report. Small interface changes are allowed when they preserve the scientific semantics documented here; hidden changes to estimands, folds, feature admissibility, weights, temporal roles or anchor meaning are not.

For current execution, `13` and `14` are overlays over the original seed: they may mark original milestone work as already complete, narrow the current critical path, and pin exact producer releases. They do not relax the scientific invariants in `01`–`12`.
