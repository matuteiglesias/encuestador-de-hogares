# Telescope C upstream transport evidence

This directory prepares the **model-side evidence** for Telescope C. It belongs in
`encuestador-de-hogares` because it fits/scores the frozen P1-R welfare model and
diagnoses EPH→Census support. It does **not** build poverty lines, classify poverty,
or invent downstream Census analysis weights.

## Scientific contract

For each outer fold `f`:

1. fit the exact frozen P1-R hurdle-Gamma model on eligible EPH persons with
   `fold != f`;
2. reproduce the held-out EPH OOF person prediction for `fold == f`;
3. score **all** Census P1 persons with that same fitted model `M_-f`.

This yields a literal matched-model transport surface:

```text
M_-f : held-out EPH fold f  <->  all Census persons
```

The poverty repository can then pair each `M_-f` with the already-governed
nested residual ECDF `G_-f`.

## Support evidence

A separate five-fold, household-safe EPH-vs-Census classifier is fit only as a
diagnostic. Domain priors are balanced by sample weights so its probability has
an interpretable equal-prior source/target meaning.

It emits:

- cross-fitted `target_probability_equal_prior`;
- hard marginal `support_weak` for Census persons using the eligible EPH model
  training support;
- cross-fitted AUC and score quantiles.

This diagnostic is **not** a production transport weight and is not used to
refit the income model.

## Outputs

```text
eph_matched_person_scores.parquet
census_matched_person_scores.parquet
eph_support_scores.parquet
census_support_scores.parquet
model_reproduction.json
support_summary.json
manifest.json
```

The matched Census score file is long-form with one row per
`(outer_fold, Census person)`.

Large real outputs remain local.

## Explicit non-goals

Telescope C upstream does not:

- alter P1-R features or hyperparameters;
- fit P2;
- condition or replace the Q7 residual ECDF;
- create poverty estimates;
- create generic Census analysis weights;
- perform domain adaptation;
- claim that the Census-derived target-year sample is observed 2024 household
  microdata.
