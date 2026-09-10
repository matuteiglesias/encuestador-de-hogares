# Model and Cascade Policy

## 1. Scope

This sprint intentionally avoids a large model zoo.

The scientific question is primarily whether representation/cascade structure improves welfare inference under a strong nonlinear tabular learner, not which of fifteen estimator families wins a leaderboard.

The active policy is:

- HistGradientBoosting (HGB) is the primary estimator family;
- Random Forest (RF) is the sole robustness challenger;
- linear estimators may remain as synthetic-test helpers or interpretive references, but are not required contenders;
- architecture comparisons must be performed under fixed estimator/terminal settings before estimator robustness is tested.

## 2. Why HGB is the primary family

HGB is well matched to the problem because the data are:

- tabular;
- moderately/highly sized;
- nonlinear;
- interaction-rich;
- mixed continuous/categorical;
- incomplete in places;
- evaluated repeatedly under cross-fitting.

The current Spisso evidence already indicates that nonlinear boosting materially outperforms OLS/Ridge/Lasso while retaining a stable performance region rather than requiring a fragile hyperparameter optimum.

The implementation should therefore optimize for **stable, interpretable experiment comparisons**, not maximal HGB tuning.

## 3. Categorical latent states

Use:

```text
HistGradientBoostingClassifier
```

for categorical stage targets.

The canonical downstream artifact is the full class-probability vector:

\[
\widehat{\mathbf p}_i
=
(\widehat P(Z_i=1\mid X_i),\ldots,\widehat P(Z_i=K\mid X_i)).
\]

Do not use only:

\[
\arg\max_k \widehat p_{ik}
\]

as the default learned representation.

Probability vectors preserve uncertainty and allow downstream models to use near-boundary information that hard classes destroy.

### Required outputs per categorical target

At minimum persist:

```text
row identity
class order
probability vector
argmax class for convenience
oof/full-score provenance
raw/calibrated status
```

## 4. Calibration policy

`predict_proba` is not automatically assumed calibrated.

For every important categorical stage evaluate:

- log loss;
- Brier score where meaningful;
- reliability/calibration curves;
- class prevalence;
- confusion matrix;
- balanced accuracy;
- macro-F1 for multiclass imbalance.

Default experiment setting:

```text
calibration = none
```

If probability calibration is visibly inadequate and terminal cascade value is sensitive to it, enable a calibrated experiment.

Candidate methods:

- sigmoid for binary tasks;
- isotonic when sample support is sufficient;
- temperature scaling for multiclass probability confidence.

Any fitted calibration must use only the corresponding outer-training data and preserve household grouping in inner splits.

## 5. HGB early stopping

Initial architecture comparisons should set:

```text
early_stopping = false
```

unless the implementation provides an explicit household-safe validation set.

Reason: sklearn's automatic internal validation split is not the scientific outer fold authority and may split households.

The first experiments should prefer a fixed, stable capacity region so that architecture—not hidden internal stopping—is the main manipulated variable.

A reasonable starting neighborhood based on prior work is:

```text
learning_rate: about 0.05
max_iter: about 200
max_leaf_nodes: about 31–63
min_samples_leaf: about 50–100
early_stopping: false
```

These are seeds, not sacred values.

## 6. Terminal welfare formulation

The full welfare population contains zero-income people. Therefore a positive-income regression alone is insufficient.

The primary terminal design should support a hurdle:

\[
D_i=1(Y_i>0).
\]

Estimate:

\[
p_i=P(D_i=1\mid W_i)
\]

and

\[
\mu_i^+=E[Y_i\mid Y_i>0,W_i].
\]

Then:

\[
\widehat E[Y_i\mid W_i]=\widehat p_i\widehat\mu_i^+.
\]

`W_i` is the architecture-dependent terminal feature set.

This decomposition handles the point mass at zero explicitly rather than asking a single continuous loss to represent both structural zeros and positive heavy-tailed income.

## 7. Positive-income regression: two HGB formulations

### 7.1 Log-income squared error

Historical benchmark:

\[
L_i=\log_{10}(Y_i),\qquad Y_i>0.
\]

Fit HGB under squared error on `L`.

Advantages:

- direct continuity with prior thesis evidence;
- scale stabilization;
- established performance benchmark.

Required caveat:

\[
10^{E[L\mid X]}\neq E[Y\mid X]
\]

in general.

Any linear-scale point estimate must declare the retransformation method.

### 7.2 Gamma loss on positive linear income

Fit HGB regressor with Gamma loss for strictly positive `Y`.

This directly models a positive conditional mean under a log-link-like geometry and emits predictions on the linear income scale.

Advantages:

- no naive post-hoc exponentiation;
- loss geometry compatible with positive skew and scale-dependent variance;
- directly relevant to household sums.

It is not assumed that income is literally Gamma distributed. It is an alternative predictive objective.

## 8. How to compare log versus Gamma

Freeze:

```text
data
feature plane
architecture
fold manifest
hurdle presence model
estimator capacity neighborhood
anchor policy
evaluation profile
```

Change only the positive-income objective/resolution method.

Judge using common downstream metrics:

\[
MAE(Y),\quad RMSE(Y),\quad Bias(Y),
\]

\[
SD(\widehat Y)/SD(Y),
\]

decile-conditioned bias, tail compression, household MAE and household distributional agreement.

Do not select a formulation solely because it optimizes its native training-space loss.

## 9. Architecture policy

### 9.1 Direct `G0`

No learned intermediate states:

```text
shared observables
      |
      v
terminal hurdle/amount model
```

This is mandatory.

### 9.2 Lean latent `G1`

One parallel latent layer:

```text
                    -> CAT_OCUP probabilities
shared observables  -> CAT_INAC probabilities
                    -> CH07 probabilities
                    -> optional income-presence-related state
                           |
                           v
                terminal welfare head
```

The initial list is a hypothesis, not a permanent contract.

The thesis suggests labor-state information is likely valuable, but each target must earn retention through oracle relevance, reconstructability and terminal cascade gain.

### 9.3 Deep candidate `Gdeep`

A multi-layer graph may include historically inspired labor, income-presence and job-detail states.

Its purpose is to answer:

> Does deeper supervised decomposition add final-welfare value beyond the lean representation?

It has no compatibility obligations to historical artifacts.

If it loses, simplify.

## 10. Stage-selection diagnostics

For each candidate latent `Z` calculate matched terminal risks:

\[
R_0=R(X),
\]

\[
R_{oracle}=R(X,Z^{true}),
\]

\[
R_{cascade}=R(X,\widehat Z^{OOF}).
\]

Then:

\[
\Delta_{oracle}=R_0-R_{oracle},
\]

\[
\Delta_{cascade}=R_0-R_{cascade},
\]

\[
\eta=\frac{\Delta_{cascade}}{\Delta_{oracle}}
\]

when the denominator is meaningfully nonzero.

Interpretation:

- small oracle gain: state has little terminal value;
- large oracle gain + weak reconstruction: scientifically valuable but inaccessible from current `X`;
- large oracle gain + strong OOF gain: strong latent candidate;
- negative deployable gain: cascade hurts and should not be retained merely because the intermediate classifier looks good.

## 11. Random Forest robustness policy

Use RF only after the HGB architecture result is established.

Purpose:

- test whether the sign/ranking of architecture effects depends on boosting-specific inductive bias;
- retain continuity with the original surveyor intuition;
- provide a bagging/randomized-tree contrast.

RF experiments should use governed categorical preprocessing, normally one-hot encoding for unordered categorical features unless the installed sklearn version provides an approved native alternative.

Do not spend the sprint performing broad RF hyperparameter optimization.

A robustness result such as:

```text
HGB: lean > direct
RF:  lean > direct
```

is more scientifically valuable than finding a third estimator with marginally better headline R².

## 12. Multioutput policy

Prefer one separately evaluable estimator per intermediate target initially.

Even when an estimator supports multioutput, stage targets such as `CAT_OCUP`, `CAT_INAC`, `CH07` should retain distinct:

- classes;
- calibration;
- OOF metrics;
- oracle value;
- cascade contribution.

If target dependence becomes scientifically interesting, express it deliberately in the DAG rather than hiding it inside estimator convenience.

## 13. Quantiles and continuous uncertainty

HGB quantile loss may later estimate conditional quantiles:

\[
Q_\tau(Y\mid X).
\]

Suggested exploratory levels:

```text
0.10
0.50
0.90
```

This is useful for conditional spread and tail diagnostics but is not automatically a coherent full predictive distribution.

Do not block the first real-EPH architecture experiments on quantile support.

## 14. Model-selection order

Use this order:

```text
A. HGB + one frozen terminal formulation
   direct vs lean vs deep

B. within best/interesting architecture
   log-positive vs Gamma-positive terminal formulation

C. selected architecture/formulation
   raw vs calibrated latent probabilities if needed

D. selected comparisons
   HGB vs RF robustness
```

This ordering keeps causal interpretation of experiment differences clean.

## 15. What not to optimize yet

Do not spend the first sprint on:

- exhaustive HGB grids;
- many boosting libraries;
- neural networks;
- SVMs;
- kNN;
- ExtraTrees as an additional challenger;
- ensemble stacking across model families;
- feature-selection tournaments;
- automated hyperparameter search.

The dominant uncertainty today is **scientific architecture and transportability**, not the last few basis points of estimator tuning.
