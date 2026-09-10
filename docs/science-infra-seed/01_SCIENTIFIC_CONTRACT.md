# Scientific Contract

## 1. Scientific question

The repository owns a statistical transport problem:

> Given a governed EPH training population, an admissible EPH/Census feature plane, one exact Census-derived scoring population, explicit welfare-period semantics and optional external aggregate evidence, what household welfare quantity can be inferred, with what predictive error, support limitations and uncertainty?

The output is model-based welfare inference. It is not an official Census income observation and it is not poverty measurement itself.

## 2. Units and terminal estimand

The primary modeling unit is the person. The primary downstream welfare unit is the household.

Let person `i` belong to household `h`.

\[
Y_{hi}=\text{person-level total income candidate}
\]

and

\[
H_h=\sum_{i\in h}Y_{hi}.
\]

The default candidate terminal person target is `P47T` or a precisely governed successor with equivalent declared semantics.

The household output consumed downstream must be a linear monetary quantity with explicit currency and price reference. Log-scale or model-native predictions are never the downstream welfare artifact.

## 3. Information sets

Define:

- `X`: information genuinely admissible at deployment for the target scoring frame;
- `Z`: intermediate state observed in EPH but not necessarily available as current person-level information on the Census side;
- `A_t`: optional aggregate target-period evidence such as an employment/unemployment margin;
- `Y`: terminal person income;
- `H`: household welfare constructed from person-level predictions.

The system must preserve the distinction between:

\[
X,
\quad Z^{true},
\quad \widehat Z^{OOF},
\quad \widehat Z^{score},
\quad A_t.
\]

These are not interchangeable.

## 4. Direct, oracle and deployable cascade risks

The direct predictor estimates:

\[
m_0(X)=E[Y\mid X].
\]

An oracle diagnostic may use an observed intermediate state:

\[
m_\star(X,Z)=E[Y\mid X,Z].
\]

The deployable cascade may use only learned intermediate representations:

\[
X\rightarrow \widehat Z\rightarrow Y.
\]

Under squared loss, the oracle value of `Z` is connected to the law of total variance:

\[
R_0-R_\star
=
E\left[\operatorname{Var}(E[Y\mid X,Z]\mid X)\right].
\]

Under ideal probabilistic log loss the analogous quantity is conditional mutual information:

\[
H(Y\mid X)-H(Y\mid X,Z)=I(Y;Z\mid X).
\]

A learned intermediate `\widehat Z=f(X)` does not create Shannon information conditional on `X`:

\[
I(Y;\widehat Z\mid X)=0.
\]

Therefore, if a cascade using `X, \widehat Z` outperforms a direct model on the same `X`, the gain must be interpreted as improved finite-sample representation, inductive bias or use of auxiliary supervision, not as newly created information.

## 5. Stage-retention principle

No intermediate state survives because it is historically present or easy to predict.

For each candidate state `Z_j`, distinguish:

### Oracle relevance

\[
\Delta^{(j)}_{oracle}=R(X)-R(X,Z_j^{true}).
\]

### Reconstructability

Measured through honest OOF predictive quality of

\[
p(Z_j\mid X).
\]

### Deployable cascade value

\[
\Delta^{(j)}_{cascade}=R(X)-R(X,\widehat Z_j^{OOF}).
\]

A useful descriptive capture ratio is:

\[
\eta_j=
\frac{R(X)-R(X,\widehat Z_j^{OOF})}
     {R(X)-R(X,Z_j^{true})}.
\]

`eta` is diagnostic, not guaranteed to remain in `[0,1]` under finite-sample estimation.

A candidate intermediate is attractive when it is both terminally relevant and reconstructable.

## 6. Probability semantics for categorical stages

For categorical `Z` with `K` classes, the desired learned artifact is:

\[
\widehat{\mathbf p}(X)
=
(\widehat P(Z=1\mid X),\ldots,\widehat P(Z=K\mid X)).
\]

Hard classes are not sufficient as the canonical downstream representation.

If a terminal conditional model exists,

\[
p(Y\mid X,Z),
\]

then the correct marginalization is

\[
p(Y\mid X)=\sum_z p(Y\mid X,z)P(z\mid X).
\]

Probability calibration therefore matters scientifically, not cosmetically.

## 7. Continuous intermediate states

For continuous `Z`, the full object is `p(Z|X)`, not only a point prediction. The exact marginalization is:

\[
p(Y\mid X)=\int p(Y\mid X,z)p(z\mid X)\,dz.
\]

The first implementation may use point predictions for continuous intermediates when justified, but the prediction artifact type must not make future distributional representations impossible.

## 8. Full-population income and zeros

The positive-income thesis cohort is not the complete welfare population.

The default full-population formulation should support a hurdle or two-part decomposition:

\[
D_i=1(Y_i>0),
\]

\[
p_i=P(D_i=1\mid W_i),
\]

\[
\mu_i^+=E[Y_i\mid Y_i>0,W_i],
\]

and therefore

\[
E[Y_i\mid W_i]=p_i\mu_i^+.
\]

`W_i` may equal `X_i` or include learned latent-state representations depending on the architecture.

The hurdle formulation is a terminal welfare formulation, not by itself a cascade stage.

## 9. Log-income and retransformation

If a model is fitted to

\[
L=\log_{10}(Y),
\]

the downstream linear prediction may not be naively interpreted as

\[
E[Y\mid X]=10^{E[L\mid X]}.
\]

Any inverse transform must explicitly account for the target estimand and retransformation bias.

For comparison, a positive-income Gamma-loss HGB with log link may emit a positive mean estimate directly on the linear scale. The sprint must compare downstream welfare behavior rather than choosing between these formulations solely on native model loss.

## 10. Household grouping and OOF invariants

Default group identity:

```text
CODUSU + NRO_HOGAR
```

Every outer validation fold must hold out entire households.

For any learned intermediate consumed by a downstream model during training:

\[
\widehat Z_i^{train}
=
\widehat f^{-fold(i)}(X_i).
\]

Observed `Z_i` may not replace `\widehat Z_i^{OOF}` in deployable training evidence.

For final scoring, fit the stage on the full approved training set and produce:

\[
\widehat Z_i^{score}=\widehat f_{full}(X_i^{score}).
\]

## 11. Weight semantics

The following quantities are distinct:

\[
\text{EPH survey weight}
\neq
\text{Census selection probability}
\neq
\text{donor inverse probability}
\neq
\text{poverty analysis weight}.
\]

For the initial scientific sprint, `PONDERA`-family EPH weights remain preserved for audit but are not used in fitting, calibration or evaluation unless a later explicit experiment changes that policy.

Census sample probabilities are never ML features or EPH fitting weights.

## 12. Temporal semantics

The repository must keep these clocks distinct:

```text
eph_training_period
census_frame_vintage
sampling_target_period
welfare_period
monetary_reference_period
```

A semantically aligned Census variable is not automatically a current welfare-period observation.

Every deployable Census-side feature must have an explicit temporal role such as:

- `donor_vintage_proxy`;
- `target_period_latent`;
- `deterministic_target_period_derived`;
- `target_period_anchor`;
- `time_stable_or_invariant`;
- `forbidden_temporal_input`.

## 13. Aggregate target-period evidence

An external series `A_t` may be introduced as new evidence.

This changes the conditional inference problem from

\[
p(Z,Y\mid X)
\]

to

\[
p(Z,Y\mid X,A_t).
\]

Such conditioning is allowed only when:

1. the anchor release is versioned and its universe is explicit;
2. the transformation is explicit and diagnosed;
3. person-level donor fields are not silently rewritten and relabeled as observations;
4. raw and anchored outputs remain separately inspectable;
5. terminal predictive effect is evaluated independently from constraint fidelity.

## 14. Household uncertainty

Point expectations aggregate linearly:

\[
E[H_h]=\sum_i E[Y_{hi}].
\]

Uncertainty does not:

\[
\operatorname{Var}(H_h)
=
\sum_i\operatorname{Var}(Y_{hi})
+
2\sum_{i<j}\operatorname{Cov}(Y_{hi},Y_{hj}).
\]

Household members are not assumed independent. The sprint's first uncertainty implementation may be bounded and frequentist, but it must preserve household clustering in resampling and avoid claiming independent-person predictive intervals as household uncertainty.

## 15. Engineering acceptance versus scientific promotion

An experiment may be technically valid and scientifically inferior.

Engineering acceptance asks:

> Did the declared experiment execute truthfully, without leakage, semantic drift or missing lineage?

Scientific promotion asks:

> Did the candidate improve final welfare inference or robustness under the evaluation profile?

The framework must support:

```text
ENGINEERING: PASS
SCIENTIFIC: NOT PROMOTED
```

without treating this as failure.

## 16. Non-claims

The system does not claim that:

- EPH OOF validation proves Census transport accuracy;
- an aggregate employment anchor makes latent person states observed;
- improved aggregate realism guarantees improved household welfare prediction;
- semantic feature alignment proves temporal validity;
- a sophisticated cascade must outperform a direct learner;
- historical architecture deserves preservation because it existed;
- a single point prediction represents the full uncertainty relevant to poverty analysis.
