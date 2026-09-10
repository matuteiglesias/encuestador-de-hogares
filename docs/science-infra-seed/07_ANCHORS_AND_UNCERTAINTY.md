# Aggregate Anchors and Uncertainty

## 1. Purpose

The framework must allow external target-period aggregate evidence to condition latent-state predictions without pretending that the aggregate source provides person-level observations.

Employment/unemployment is the first concrete use case.

The anchor system is optional. The unconstrained model remains a valid and inspectable baseline.

## 2. Statistical framing

Let `Z_i` be a latent categorical state for person `i`, and let the micro model emit:

\[
q_i(z)=P(Z_i=z\mid X_i).
\]

Let external evidence `A_t` constrain one or more aggregate moments.

The scientific problem becomes:

\[
p(Z,Y\mid X,A_t)
\]

rather than only:

\[
p(Z,Y\mid X).
\]

The anchor is therefore an explicit conditioning/intervention layer between raw latent prediction and downstream welfare prediction.

## 3. Binary expected-moment projection

Suppose `Z_i=1` represents an event such as unemployment within the anchor's declared universe.

The raw model gives:

\[
q_i=P(Z_i=1\mid X_i).
\]

The external target is a weighted aggregate rate `r`:

\[
\frac{\sum_i c_i q_i^\star}{\sum_i c_i}=r.
\]

Here `c_i` represents the aggregation measure defined by the anchor universe. It is not automatically an EPH fitting weight or Census sampling weight.

Choose adjusted probabilities closest to the raw model under Bernoulli KL divergence:

\[
\min_{q_i^\star}
\sum_i c_i
D_{KL}(Bern(q_i^\star)\Vert Bern(q_i))
\]

subject to the aggregate moment.

The solution has form:

\[
logit(q_i^\star)=logit(q_i)+\lambda,
\]

where `lambda` is chosen so the constraint holds.

This preserves rank ordering under a common logit shift while updating overall prevalence.

## 4. Numerical implementation

Implement the binary projection with a robust one-dimensional root solver.

Define:

\[
f(\lambda)=
\frac{\sum_i c_i\sigma(logit(q_i)+\lambda)}{\sum_i c_i}-r.
\]

`f(lambda)` is monotone increasing.

A stable implementation can:

1. clip raw probabilities to `[eps, 1-eps]` for numerical logit operations;
2. bracket the root over an expanding finite interval;
3. solve with a robust scalar method such as bisection or Brent;
4. verify the final aggregate residual against a declared tolerance;
5. preserve both raw and adjusted probabilities.

Avoid a custom unconstrained optimizer when a monotone scalar root is sufficient.

## 5. Multi-class extension

For a multi-class state:

\[
q_{ik}=P(Z_i=k\mid X_i),
\quad \sum_k q_{ik}=1.
\]

A future anchor may constrain one or several aggregate class shares.

The generic projection can be framed as:

\[
Q^\star=
\arg\min_Q D_{KL}(Q\Vert Q_0)
\]

subject to:

\[
E_Q[g_j(Z)]=a_j,\quad j=1,\ldots,m.
\]

The first sprint does not need a fully generic convex-optimization framework if the employment use case is binary or can be reduced to a simple categorical margin. Design interfaces so this extension remains possible.

## 6. Expected-moment versus exact-count anchors

These are different interventions.

### Expected-moment

Constrain:

\[
\sum_i c_i q_i^\star = target.
\]

Individual states remain probabilistic.

This is the preferred first mode.

### Exact realized count

Require a generated synthetic realization to contain exactly a configured count.

This introduces dependence across assignments and is stronger than an expectation constraint.

Do not implement exact-count assignment as the default just because it is easy to explain.

## 7. Employment anchor semantics

An employment/unemployment release must explicitly define:

```text
concept
period
geography
numerator universe
denominator universe
value
source/release identity
revision status
uncertainty, if any
```

Examples of distinct concepts that must not be mixed:

```text
unemployment rate among economically active population
employment rate among working-age population
activity rate among working-age population
share unemployed among all adults
```

An anchor adapter is responsible for constructing the exact model-side population mask/moment that corresponds to the source definition.

If the source universe cannot be matched defensibly, the anchor must fail closed rather than silently approximate it.

## 8. Raw versus anchored branches

Every anchored experiment must retain:

```text
raw latent prediction
anchored latent prediction
raw terminal welfare
anchored terminal welfare
```

when the downstream structure allows them to be recomputed reasonably.

This supports matched diagnostics:

\[
\Delta \widehat Y_i=\widehat Y_i^{anchor}-\widehat Y_i^{raw}.
\]

The framework should make it impossible to inspect only the anchored result and forget the underlying model discrepancy.

## 9. Required anchor diagnostics

Persist:

### Aggregate fit

\[
a_{raw},\quad a_{target},\quad a_{anchor}.
\]

and residuals:

\[
d_{raw}=a_{raw}-a_{target},
\]

\[
d_{anchor}=a_{anchor}-a_{target}.
\]

### Intervention magnitude

- `lambda` or equivalent dual parameter;
- total KL displacement;
- mean absolute probability displacement;
- median displacement;
- P90/P95/P99 displacement;
- maximum displacement;
- fraction moving by more than configured thresholds;
- raw/anchored probability histograms.

### Downstream effect

- mean/median `Delta Y`;
- changes by raw latent probability region;
- changes in person-income distribution;
- changes in household-income metrics;
- changes in threshold-relevant diagnostics if configured.

## 10. Interpretation of lambda

The dual parameter is itself scientifically useful.

If:

\[
\lambda\approx0,
\]

the raw micro model already agrees with the external margin.

Large positive or negative `lambda` means the anchor is correcting a substantial aggregate disagreement.

A time series of `lambda_t` may reveal periods where transport from donor/EPH information is especially poor.

Useful time-series diagnostics include:

```text
external rate
raw model aggregate rate
anchored aggregate rate
raw-minus-external discrepancy
lambda_t
KL displacement
change in predicted household welfare
```

The post-anchor rate matching the target is expected by construction and is not itself evidence that the model improved.

## 11. Anchor uncertainty

An external aggregate may itself be estimated rather than exact.

Represent uncertainty explicitly, for example through:

```text
point estimate + standard error
confidence interval
bootstrap draws
posterior/sample draws from source release
```

If anchor uncertainty is enabled, replicate `b` may use:

\[
r^{(b)}\sim p(r)
\]

then solve:

\[
\lambda^{(b)}
\]

and produce:

\[
q_i^{\star(b)}.
\]

Do not invent a probability distribution from a published interval without documenting the approximation.

## 12. Types of uncertainty in the system

Keep at least these concepts distinct.

### Aleatoric outcome uncertainty

\[
Var(Y\mid X,Z)>0.
\]

### Latent-state uncertainty

\[
p(Z\mid X).
\]

### Parameter/model uncertainty

Finite-data uncertainty in fitted estimators.

### Structural uncertainty

Direct versus lean versus deep architecture; log versus Gamma terminal formulation; HGB versus RF.

### Transport uncertainty

EPH relationship may not transfer exactly to Census-side population/support.

### Anchor uncertainty

External aggregate evidence may itself be estimated/revised.

The first sprint does not need to combine all of these into one scalar predictive interval.

## 13. Bounded uncertainty v1

The first green implementation should be modest and truthful.

Recommended v1:

1. preserve categorical probability vectors;
2. evaluate probability calibration;
3. support optional HGB conditional quantile fits for positive income;
4. use household-cluster bootstrap for uncertainty in scientific metrics and architecture differences;
5. allow anchor releases to carry uncertainty metadata;
6. do not claim a complete joint predictive distribution of household income.

This is enough to make uncertainty visible without blocking real EPH experiments.

## 14. Household prediction uncertainty

For household `h`:

\[
H_h=\sum_i Y_{hi}.
\]

Even if person-level conditional variances were available:

\[
Var(H_h)=\sum_iVar(Y_{hi})+2\sum_{i<j}Cov(Y_{hi},Y_{hj}).
\]

Household covariance terms cannot be assumed zero.

Therefore:

- do not combine independent person intervals into a household interval as if errors were independent;
- use household-level residual evidence, clustered resampling or a future joint model if a household predictive distribution is required;
- clearly distinguish uncertainty in **evaluation metrics** from uncertainty in a specific household's latent welfare.

## 15. Quantile predictions

HGB quantile models may estimate:

\[
Q_\tau(Y\mid W)
\]

for selected `tau`.

Use them initially for:

- spread diagnostics;
- lower/upper conditional behavior;
- interval-coverage experiments.

Do not automatically treat separately fitted quantiles as a coherent density. Check monotonicity/crossing if several quantiles are emitted.

## 16. Interaction with cascade probabilities

A probabilistic latent state can conceptually propagate through:

\[
p(Y\mid X)=\sum_z p(Y\mid X,z)P(z\mid X).
\]

The first implementation may instead pass probability vectors as features into a flexible terminal learner. This is a finite-sample supervised representation strategy, not exact probabilistic marginalization.

The runtime and docs should name the implementation honestly.

If a future experiment implements explicit mixture/marginalization, it should be a separate terminal model type.

## 17. Prohibited anchor behavior

Do not:

- overwrite donor `CONDACT` and call it target-period observed data;
- randomly flip hard labels merely to hit a margin when a probability projection is available;
- use Census selection probabilities as anchor weights by default;
- apply an anchor whose universe differs materially from the modeled population without declaring a mapping;
- evaluate only post-anchor agreement;
- hide raw predictions;
- let anchor code mutate source dataframes in place;
- treat anchor success as scientific promotion by construction.

## 18. Acceptance

Anchor implementation is green when:

- the binary projection satisfies known synthetic constraints to tolerance;
- rank order is preserved under the common binary logit shift, subject to ties;
- probabilities remain valid;
- extreme targets fail or clip only under explicit policy;
- raw and anchored artifacts are both preserved;
- the exact anchor release/universe is recorded;
- intervention diagnostics are emitted;
- downstream effects can be compared using identical evaluation machinery;
- no donor observation is relabeled as current truth.

Uncertainty v1 is green when it supports the bounded mechanisms above with explicit limitations and household-aware resampling.
