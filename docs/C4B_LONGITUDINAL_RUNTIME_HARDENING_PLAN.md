# C4B — longitudinal runtime hardening before merge

Status: cloud follow-up packet for PR #33, 2026-09-29.

Work on branch:

```text
feat/longitudinal-welfare-l10-l12-c4
```

This packet does not redesign C4. It closes two scientific/configuration gaps found during review.

Authoritative background:

- `docs/LONGITUDINAL_WELFARE_L10_L12_PLAN.md`
- `docs/LONGITUDINAL_WELFARE_C4_IMPLEMENTATION.md`
- PR #33 review comment dated 2026-09-29.

## Finding 1 — feature plane must be explicit

The current fixture configs use:

```text
CH04 CH06 CH09 CH10 CH12 CH13 CH15 IX_TOT
```

That list must not be silently promoted as the scientific L10 centerline.

It is neither the historical nine-feature P0 baseline (it omits CH07) nor the richer P1-R information plane that previously improved welfare prediction.

The longitudinal program is adding a governed composition-plane producer in `eph-censo-aligner` under:

```text
research.eph-longitudinal-composition-plane/v1
```

with named profiles such as `P0_LONG` and `P1R_NOLAB_LONG`.

### Required runtime change

Make C4 explicitly feature-plane-agnostic.

A scientific run config must be able to declare:

```text
composition_source
composition_parent_contract
feature_profile_id
```

and, for a canonical composition parent, join it by exact `row_id` to the C2 longitudinal observations.

Bind the composition release ID + manifest hash + profile ID in the run manifest.

No fuzzy joins.

Keep the existing raw-C2 minimal feature list only as an explicitly named fixture/testing profile. It must be impossible to mistake it for an adjudicated longitudinal scientific default.

The runtime must not implement EPH↔Census recoding itself.

### Compatibility

C4B may implement and test the consumer against a synthetic C5 fixture before C5 is merged.

L10 must remain runnable in fixture mode without C5 real data.

## Finding 2 — time layer must be fit on training-side OOF base predictions

Current outer-fold logic is holdout-safe, but the time correction is estimated from base-model predictions made on the same outer-training rows used to fit the base model.

For the explicit temporal layer, that is not strong enough.

### Required estimator flow

For each outer fold:

```text
outer-training rows
    |
    +--> inner household/panel-safe cross-fitting of base hurdle
             |
             --> OOF p_positive and positive_amount on every outer-training row
                      |
                      --> fit year + quarter + exceptional-period time layer

then:

fit base hurdle on all outer-training rows
    |
    --> predict outer holdout
            |
            --> apply frozen time layer from training-side OOF evidence
```

The time layer may never use:

- outer-holdout targets;
- outer-holdout current labor labels;
- in-sample base predictions as its primary residual evidence.

Apply the same rule to matched L10 baselines inside L11/L12 comparisons.

Reuse existing household-safe crossfit primitives where possible.

## Exceptional periods

Do not change the current policy:

```text
2020-Q2  pandemic_fieldwork_regime
2024-Q1  2024_h1_macroeconomic_shock
2024-Q2  2024_h1_macroeconomic_shock
```

All three remain measurable and excluded from ordinary year/quarter estimation.

Dedicated shock effects are fit only from training evidence.

## Tests

Add tests proving:

1. a canonical composition-plane parent is joined one-to-one by exact row ID;
2. parent/profile identity is persisted;
3. fixture/raw composition cannot masquerade as an adjudicated scientific profile;
4. time-layer training predictions are inner-OOF with respect to the outer-training population;
5. panel households do not cross inner or outer folds;
6. outer-holdout target perturbation cannot change the fitted time layer;
7. exceptional-period behavior remains unchanged;
8. L11/L12 matched L10 comparisons use the same corrected time-layer discipline.

## PR #33 completion rule

C4B is complete when the new scientific fixtures pass and the PR receipt explicitly states the composition-plane and nested-time-layer semantics.

Do not fix unrelated Telescope-B baseline files inside PR #33. That baseline test issue has a separate packet.

## Non-goals

- no real 37-quarter fitting;
- no real C5 materialization;
- no feature-profile scientific winner;
- no nowcast/forecast;
- no change to frozen 2024-Q3 evidence.
