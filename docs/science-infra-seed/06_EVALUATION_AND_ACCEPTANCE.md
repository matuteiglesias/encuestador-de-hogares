# Evaluation and Acceptance

## 1. Evaluation is a first-class subsystem

The repository should optimize for trustworthy scientific comparison, not merely estimator execution.

The evaluator must answer:

1. Can each intermediate state be reconstructed honestly?
2. Does the true intermediate state contain terminal welfare signal beyond deployable observables?
3. How much of that signal survives through OOF reconstruction?
4. Does the final person-income model reproduce the relevant distribution, not only average fit?
5. Do errors improve or worsen after household aggregation?
6. Does an external anchor improve realism only, or also terminal welfare accuracy?
7. Are differences large enough to survive household-cluster uncertainty?

## 2. Comparison discipline

Every scientific comparison must declare the axis being changed.

### Architecture comparison

Change:

```text
G: direct -> lean -> deep
```

Hold fixed:

```text
D, X, T, M, S, C, A, U, V
```

### Terminal formulation comparison

Change:

```text
T: hurdle-log -> hurdle-Gamma
```

Hold architecture and all other axes fixed.

### Estimator robustness comparison

Change:

```text
M: HGB -> RF
```

only after the architecture comparison is understood.

### Anchor comparison

Change:

```text
A: none -> governed employment anchor
```

using the same fitted/unconstrained base experiment where practical.

The comparison command should detect incompatible run contracts and refuse misleading comparison unless explicitly overridden.

## 3. Intermediate categorical diagnostics

For each categorical `Z` record:

### Basic support

- row count;
- class count;
- class prevalence;
- missing/invalid target count;
- per-fold class support.

### Classification quality

- confusion matrix;
- balanced accuracy;
- macro-F1;
- ordinary accuracy as descriptive only;
- per-class recall/precision where useful.

### Probabilistic quality

Log loss:

\[
LL=-\frac{1}{n}\sum_i\log \widehat p_{i,y_i}.
\]

For binary states, Brier score:

\[
BS=\frac{1}{n}\sum_i(\widehat p_i-y_i)^2.
\]

For multiclass states, use multiclass Brier or an equivalent explicitly defined decomposition.

Also emit:

- reliability/calibration curves;
- expected calibration error only if its binning definition is persisted;
- probability histograms by true class;
- raw versus calibrated metrics when calibration is enabled.

Do not treat high accuracy as evidence that probability vectors are trustworthy.

## 4. Oracle / reconstructability / cascade triangle

For a candidate intermediate `Z`, evaluate matched terminal models under the same outer folds.

### Direct

\[
M_0:X\rightarrow Y.
\]

### Oracle

\[
M_\star:(X,Z^{true})\rightarrow Y.
\]

### Deployable

\[
M_C:(X,\widehat Z^{OOF})\rightarrow Y.
\]

Calculate risk using one or more terminal metrics, especially linear-scale MAE and household MAE.

Define:

\[
\Delta_{oracle}=R_0-R_\star,
\]

\[
\Delta_{cascade}=R_0-R_C,
\]

and when meaningful:

\[
\eta=\frac{\Delta_{cascade}}{\Delta_{oracle}}.
\]

Also retain the intermediate's own predictive metrics so the failure mode can be classified:

```text
low oracle value -> irrelevant intermediate
high oracle value + poor reconstruction -> inaccessible useful state
high oracle value + good reconstruction + low cascade gain -> terminal model/representation mismatch
high oracle value + good reconstruction + positive cascade gain -> useful learned representation
```

## 5. Person-income point metrics

For final linear income prediction `\widehat Y` compute:

\[
MAE=E|\widehat Y-Y|,
\]

\[
RMSE=\sqrt{E[(\widehat Y-Y)^2]},
\]

\[
Bias=E[\widehat Y-Y],
\]

and, where interpretable,

\[
R^2=1-\frac{\sum_i(Y_i-\widehat Y_i)^2}{\sum_i(Y_i-\bar Y)^2}.
\]

If a log-target model exists, native log-space metrics may also be reported, but model selection for welfare use must include linear-scale metrics.

## 6. Distributional diagnostics

The thesis evidence shows that average fit can coexist with strong distributional compression.

Every terminal experiment should therefore report:

### Dispersion ratio

\[
\rho_{sd}=\frac{SD(\widehat Y)}{SD(Y)}.
\]

### Quantile agreement

For a configured grid `q`:

\[
Q_q(\widehat Y)-Q_q(Y).
\]

### Error by observed-income decile

For decile `D_j`:

\[
Bias_j=E[\widehat Y-Y\mid Y\in D_j],
\]

\[
MAE_j=E[|\widehat Y-Y|\mid Y\in D_j].
\]

### Error by predicted-income decile

This complements observed-decile analysis and can expose calibration/ranking differences.

### Tail compression

Record at least:

- low-tail mean/median bias;
- high-tail mean/median bias;
- observed versus predicted P10/P50/P90 or richer quantile set;
- fraction of predicted values outside plausible/declared support.

## 7. Hurdle diagnostics

For a hurdle terminal model report both components separately.

### Presence model

\[
p_i=P(Y_i>0\mid W_i).
\]

Evaluate classification/calibration as above.

### Positive amount model

Evaluate only the eligible `Y>0` population with clear sample counts and then evaluate the combined unconditional prediction:

\[
\widehat Y_i=\widehat p_i\widehat\mu_i^+.
\]

Do not allow a good positive-amount fit to hide a poor zero/nonzero model.

## 8. Household aggregation diagnostics

Let:

\[
H_h=\sum_{i\in h}Y_i,
\quad
\widehat H_h=\sum_{i\in h}\widehat Y_i.
\]

Report:

- household MAE;
- household RMSE;
- household mean bias;
- household median absolute error;
- household dispersion ratio;
- quantile agreement;
- error by household size;
- error by observed household-income decile;
- complete/incomplete household count;
- member-level failure accounting.

The evaluator must refuse to silently compute household metrics on incomplete membership unless an experiment explicitly defines another policy.

## 9. Poverty-relevant diagnostics without owning poverty methodology

If a governed household threshold `B_h` is provided solely for evaluation, the repo may compute diagnostics such as:

\[
1(H_h<B_h)
\quad\text{vs}\quad
1(\widehat H_h<B_h).
\]

Useful metrics:

- sensitivity/recall among truly below-threshold households;
- specificity;
- balanced accuracy;
- false-poor and false-nonpoor rates;
- error as a function of distance to threshold;
- calibration of `P(H_h<B_h)` if a predictive distribution is available.

The repository must not redefine official poverty lines, equivalence scales or FGT methodology.

## 10. Subgroup diagnostics

At minimum make subgroup evaluation configurable for dimensions such as:

- year/quarter;
- region/geography where semantically valid;
- sex;
- age bands;
- education level;
- household size;
- labor-state group;
- model-support/extrapolation flags.

Subgroups are for diagnosing heterogeneous error, not automatically for optimizing subgroup-specific models.

## 11. Temporal robustness

The default development design is household-grouped OOF.

Once the core path works, add sensitivity designs such as:

- train on earlier period, evaluate later period;
- leave-one-year-out;
- train 2024, evaluate 2025 and vice versa when data support it.

Temporal robustness should not replace household grouping. It answers a different question.

## 12. Cluster bootstrap

Scientific uncertainty in headline performance comparisons should respect household dependence.

Bootstrap households, not persons.

For replicate `b`:

1. sample households with replacement;
2. include all persons belonging to selected households;
3. recompute the metric from already generated OOF predictions when valid;
4. retain paired run comparison when comparing two architectures.

For paired architecture differences:

\[
\Delta_b=R_b(M_a)-R_b(M_b).
\]

Report empirical intervals for `Delta`, not only separate intervals for each model.

This is cheap once predictions exist and directly answers whether a performance difference is stable.

## 13. Anchor diagnostics

For a constrained state record both external fidelity and downstream consequences.

### Constraint fidelity

Before:

\[
d_{raw}=\widehat a_{raw}-a_t.
\]

After:

\[
d_{anchor}=\widehat a_{anchor}-a_t.
\]

Also record:

- Lagrange multiplier / logit shift `lambda`;
- KL displacement;
- mean/median/max individual probability movement;
- quantiles of probability movement;
- number/fraction of rows moving more than configured thresholds.

### Terminal effect

Report:

\[
\Delta MAE_{person},
\]

\[
\Delta MAE_{household},
\]

changes in distribution/tails, and threshold diagnostics if available.

A perfect aggregate match does not count as improved welfare validity by itself.

## 14. Engineering acceptance

An experiment receives `ENGINEERING: PASS` only if all applicable invariants hold.

Minimum checks:

- resolved configuration validates;
- input identities/releases are recorded;
- every row belongs to exactly one outer fold;
- no household crosses outer folds;
- OOF predictions are complete;
- categorical probability rows are valid and class-labelled;
- observed intermediate labels were not substituted in deployable path;
- forbidden/audit-only fields did not enter features;
- no undeclared sample weight entered fit/calibration/evaluation;
- transformations and monetary scale are explicit;
- household membership accounting is complete;
- anchor constraints, if enabled, are explicitly diagnosed;
- run artifact manifest is complete;
- metrics are finite or failures are explicitly represented.

## 15. Scientific promotion

Scientific promotion is a separate judgement layer.

The framework should emit evidence, not hard-code premature global thresholds.

A candidate architecture is worth promotion when evidence supports claims such as:

- positive terminal gain over direct baseline;
- gain survives paired household bootstrap uncertainty;
- no unacceptable worsening of lower-tail or household behavior;
- latent probability calibration is credible enough for downstream use;
- gain persists under at least one sensible robustness split/model family;
- complexity is justified by final-welfare value.

If the lean model performs as well as the deep model within uncertainty, prefer the lean model.

If the direct model wins, accept that result.

## 16. Machine-readable acceptance

Each run should eventually emit something like:

```json
{
  "engineering": {
    "status": "pass",
    "checks": {...}
  },
  "science": {
    "status": "candidate|promoted|not_promoted|undetermined",
    "comparison_group": "...",
    "evidence": {...},
    "limitations": [...]
  }
}
```

No automated rule should label a scientifically inferior but correctly executed model as an engineering failure.
