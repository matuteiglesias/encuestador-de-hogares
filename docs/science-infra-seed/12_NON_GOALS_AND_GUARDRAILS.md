# Non-Goals and Guardrails

## 1. Why this document exists

This sprint is broad enough that agents can easily spend large amounts of effort on work that sounds reasonable but does not improve the core scientific inference path.

This document defines explicit boundaries.

When in doubt, prefer the smallest implementation that gets honest real-EPH architecture evidence sooner.

## 2. Legacy is archaeology, not an acceptance target

The repository contains historically valuable code, notebooks, serialized Random Forest models, generated figures and output artifacts.

They may help answer questions such as:

- what variables were historically grouped together;
- what scientific intuition motivated the original cascade;
- whether probability outputs or target-period employment adjustment were previously attempted;
- what configurations might be amusing to rerun later.

They do **not** impose requirements on the new runtime.

The sprint does not need to:

- reproduce historical predictions;
- reproduce historical metrics;
- preserve serialized model compatibility;
- preserve old model filenames;
- preserve old preprocessing quirks;
- preserve notebook execution;
- preserve the RFC1→RFC4 control flow;
- keep old generated artifacts alive;
- prove parity with old Census outputs.

Historical/deep configuration may remain as a dormant candidate experiment, explicitly selected only when someone is curious.

## 3. Do not build a compatibility layer for its own sake

Avoid adapters whose only purpose is to make old files appear current.

Examples of work that should be rejected unless a concrete active requirement emerges:

```text
legacy model deserializer abstraction
legacy output converter
RFC artifact registry
historical notebook runner
old file-name compatibility shim
```

A short archaeology note is preferable to weeks of compatibility code.

## 4. Do not create a model zoo

The active estimator policy is narrow:

```text
HGB primary
Random Forest one robustness challenger
```

Do not add during the first sprint:

- XGBoost/LightGBM/CatBoost merely because they exist;
- neural networks;
- SVMs;
- kNN;
- ExtraTrees as a third tree contender;
- broad stacking/ensembling;
- AutoML;
- exhaustive hyperparameter optimization.

A new estimator family requires a scientific reason tied to an observed failure mode.

## 5. Do not conflate architecture and terminal formulation

The following are separate axes:

```text
direct vs lean vs deep
hurdle-log vs hurdle-Gamma
HGB vs RF
raw vs calibrated probabilities
raw vs anchored latent states
```

Do not create a config named `best_model_v2` that changes all of these at once.

Every comparison should identify what changed.

## 6. Do not privilege cascade complexity

A deep model is not more scientific because it contains more stages.

If:

\[
R_{direct}\le R_{lean}\le R_{deep}
\]

then direct may be the correct answer.

If lean and deep are statistically indistinguishable in terminal welfare performance, prefer lean.

Intermediate-stage accuracy cannot justify complexity by itself.

## 7. Do not optimize intermediate tasks independently of welfare value

A latent classifier with excellent F1 but no oracle or terminal cascade value is not a useful stage.

Stage retention must consider:

```text
oracle relevance
reconstructability
probability quality
deployable terminal gain
complexity cost
```

Do not tune `CAT_OCUP` to perfection if its OOF probabilities do not improve final welfare inference.

## 8. Do not use observed intermediate labels in deployable downstream training

Observed `Z_true` is allowed only in explicitly labeled oracle diagnostics.

Deployable downstream training must consume:

\[
\widehat Z^{OOF}
\]

for training rows.

No debug shortcut may substitute `Z_true` and then report the resulting terminal metric as cascade performance.

## 9. Do not weaken household grouping

Once household identity is available, random-person splits are not the active acceptance design.

Do not:

- use `train_test_split` on persons for headline metrics;
- allow sklearn internal CV to override the outer household split;
- let calibration folds leak household members;
- refit a downstream stage on full-data upstream predictions and call them OOF.

Random-person results may exist only as clearly named historical/sensitivity references.

## 10. Do not use survey/design weights implicitly

Current sprint policy:

```text
PONDERA-family fields preserved for audit
fit weight = none
calibration weight = none
evaluation weight = none
```

unless a later explicit experiment changes the contract.

Census selection probabilities are design lineage, not ML features or EPH weights.

Never use a generic `weight` argument whose semantics are ambiguous.

## 11. Do not smuggle temporal updates into features

A Census-2010 state is not automatically a 2024/2025 state.

Do not overwrite donor variables with target-period estimates and then call them observed.

If new aggregate target-period information exists, use the explicit anchor path.

Raw donor/source state and model-updated latent state must remain distinguishable.

## 12. Do not let employment anchoring become a hidden trick

The employment anchor is allowed precisely because it is explicit.

Do not:

- randomly flip person labels to hit the aggregate target as the canonical method;
- erase the raw model result;
- report only the post-anchor employment series;
- treat exact aggregate matching as proof of better welfare prediction;
- use an external rate whose denominator/universe differs from the modeled population without an explicit mapping;
- hide a very large `lambda`/KL displacement.

The anchor must make model-vs-reality disagreement **more visible**, not less.

## 13. Do not wait for perfect Census deployment semantics before doing EPH model science

The first scientific question can be answered on real EPH using a candidate deployment-admissible/common feature plane and household-safe OOF.

Do not block:

```text
direct vs lean welfare evidence
oracle value
probability calibration
distributional error
household aggregation behavior
```

on completion of every real-vintage Census semantic mapping.

However, do not claim that EPH success proves Census transport validity.

## 14. Do not make this repository own upstream authorities

The repo does not need to absorb:

- raw EPH acquisition;
- the canonical EPH/Census mapping authority;
- Census sample construction;
- Census geography;
- IPC methodology;
- poverty-line methodology.

Consume those through explicit interfaces/releases.

If an upstream interface is missing, make the smallest necessary contract contribution there rather than duplicating authority here.

## 15. Do not turn evaluation into poverty methodology

This repo may use externally supplied household thresholds to study errors near a decision boundary.

It must not become the authority for:

- adult equivalence;
- poverty lines;
- official poor/nonpoor classification;
- FGT estimands;
- publication methodology.

Those remain downstream responsibilities.

## 16. Do not claim a full household posterior prematurely

The sprint should make uncertainty more honest, not more impressive.

Allowed bounded-v1 claims include:

- categorical probability uncertainty;
- conditional quantile predictions;
- household-cluster bootstrap intervals for evaluation metrics;
- anchor uncertainty sensitivity.

Do not combine independent person intervals and present them as a coherent household predictive distribution without modeling covariance.

Do not call separately fitted quantiles a posterior distribution.

## 17. Do not overbuild DAG infrastructure

The initial architecture graph is small.

Use simple validated DAG machinery. Standard-library `graphlib` is sufficient unless an actual requirement emerges.

Do not add graph databases, workflow engines, distributed schedulers or orchestration frameworks for this sprint.

The challenge is scientific semantics, not graph scale.

## 18. Do not overbuild config infrastructure

YAML composition/resolution needs to be reliable, but it does not need a custom configuration language.

Prefer:

```text
small explicit YAML fragments
schema validation
resolved config snapshot
```

Avoid:

- recursive template metaprogramming;
- hidden environment-dependent overrides;
- config inheritance chains that are hard to inspect;
- magic profile names that alter many scientific dimensions.

## 19. Do not store large generated state in Git

Do not commit:

- full EPH datasets;
- full Census data;
- large OOF prediction matrices for every run;
- fitted HGB/RF models for routine experiments;
- bootstrap sample dumps;
- large repeated figure caches.

Commit small fixtures, configs, manifests, summaries and meaningful compact diagnostics.

Use a configured external run/artifact root for heavy generated state.

## 20. Do not make notebooks authoritative

Notebooks may be used for exploration and plotting.

Anything required for scientific reproduction must be callable through Python modules/CLI/Make and expressed in resolved run configuration.

A notebook cell is never the only implementation of:

- a cohort rule;
- a feature transform;
- an anchor;
- a metric;
- a split;
- a welfare aggregation;
- an inverse transform.

## 21. Do not silently repair invalid data

Fail clearly on:

- duplicate person IDs;
- missing household identity;
- impossible category mappings;
- invalid Gamma targets;
- mismatched prediction lengths;
- unmatched anchor universe;
- non-finite probabilities;
- incomplete household membership;
- missing required monetary semantics.

If an explicit repair policy is scientifically defensible, encode it in config and report its effect.

## 22. Do not prematurely hard-code promotion thresholds

The framework should emit the evidence required for promotion.

It should not initially decree rules such as:

```text
cascade must improve R² by 0.01
Brier must be below 0.1
anchor KL must be below X
```

before real runs reveal meaningful scales.

Engineering invariants may have hard thresholds. Scientific preference should initially be evidence-led and reviewable.

## 23. Do not confuse reproducibility with historical preservation

Reproducibility means a current experiment is fully identified and repeatable from its code/config/input releases.

It does not mean carrying every artifact ever produced by previous incarnations of the project.

New run immutability applies prospectively to the active runtime.

## 24. Priority test

When an agent proposes work, ask:

> Does this help us obtain or trust the real-EPH direct-vs-lean welfare comparison?

If yes, it is likely core.

If no, ask:

> Does it directly enable the explicit employment anchor, uncertainty honesty or reproducibility after the core is working?

If still no, defer it.

## 25. Sprint success boundary

A successful sprint does **not** require every imaginable red light in the long-term research program to be solved.

It does require the active implementation table's core reds/yellows to become genuinely green:

```text
declarative experiment
household-safe OOF
typed latent probabilities
HGB estimator layer
direct + lean architecture
oracle/cascade diagnostics
full-population terminal welfare
real EPH execution
distributional + household evaluation
run lineage
```

A strong stretch close adds:

```text
employment anchor
bounded uncertainty
RF robustness
```

without sacrificing the truthfulness of the core.
