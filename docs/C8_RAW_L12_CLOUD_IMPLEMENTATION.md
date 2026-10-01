# C8 cloud implementation — raw, short-gap latent-labor L12

Status: cloud implementation + fixture validation only. C8 has NOT been run
on the user's local immutable real 2017-Q1..2026-Q1 L2 or C7 panel.
C7B's real matched experiment and local R0 acceptance remain independent
prerequisites for downstream C8 welfare. Neither stage authorizes Census
2010/2022 income inference or scientific arm promotion.

## Reused contracts

The source plane is exactly the immutable C7 private
`research.encuestador-c7-donor-clock-panel-plane/v1`, built from audited Gate B
and the accepted C6 P1R model plane. Donor-wave composition, donor-wave E/U/I,
later-quarter official aggregate labor/time, 1Q/3Q elapsed gap, observed
later E/U/I training label and later real P47T are already frozen there.
C8 NEVER redoes individual linkage or loads raw 1.87M-row L2 dictionaries.

C8 uses `configs/longitudinal/c8_real_p1r.yaml` (L12, no anchoring)
and preserves the C7 group-fold and later-valid-income row/target identities.
Only person-level and explicitly **selected-panel-household** metrics are
permitted. This is research short-gap measurement; do not rename the
selected linked-person aggregate as complete household income.

## Stage 1: independent transition evidence, no welfare fitting

Executable `longitudinal-c8-transition` in
`src/encuestador/longitudinal_c8.py`. Two specifications:

- T0: P(later E/U/I | earlier E/U/I, gap) from the **outer-training** sample
  only, with disclosed mild Dirichlet/global-train prior smoothing
  (strength 3). Full-population Gate-B matrix is NEVER used as a prediction
  feature or fold baseline.
- T1: conditional multiclass HGB using the C7 donor-X features, donor
  E/U/I, 1Q/3Q elapsed gap, and target official national/regional labor
  context. Reuses the C4 factory/estimator configuration.

Each outer holdout is scored by transition estimators fitted only on the
other household-group folds. Inside that outer training population, T1's
meta-training probabilities are cross-fitted across the remaining folds.
All 3 probability columns have stable class ordering:
E=1, U=2, I=3, finite, nonnegative and normalized.

Store private atomic, hash-validated per-fold checkpoints and release
`research.encuestador-c8-raw-transition/v1`:

- `t0_oof.npy` and `t1_oof.npy`, shape (C7_valid_income_rows, 3);
- `transition_metrics.json`: multiclass log-loss, 3-class summed Brier,
  accuracy, support/recall and ten-bin calibration by class, fold and
  gap × previous labor class;
- `transition_manifest.json` with exact C7 plane/config/source identities,
  stage-1 OOF training and restartability proof;
- per-fold `t1_train_oof.npy` needed by the downstream cascade.

This is intentionally a **selected valid-income research cohort**. Comparing
it with the full 827,793 Gate-B descriptive pairs requires documenting the
extra income-validity selection, not calling the populations identical.

The local agent reviews T0 vs T1 evidence before explicitly commissioning
stage 2. A poor transition result is a reason to stop/defer expensive
welfare modeling, not a mathematical proof that all L12 specifications fail.

## Stage 2: raw conditional-probability welfare, manually invoked

Executable `longitudinal-c8-welfare` in
`src/encuestador/longitudinal_c8_run.py`. It fails closed unless both
completed source releases are present and hash-verified:

1. C8 Stage 1 raw transition release (same C7 panel/config/identity);
2. C7B completed matched `research.encuestador-c7-matched-l11-run/v1`
   with the same panel, row/fold/target, composition profile, hurdle
   estimator parameters and group-fold policy.

C8-1 terminal input:

    earlier-wave canonical P1R composition
    later-quarter national/regional aggregate labor context
    elapsed quarters (1 or 3)
    T1 OOF P(current E), P(current U), P(current I)

The terminal model does NOT receive actual later ESTADO/CONDACT, nor does it
directly receive earlier state after it has been transformed into T1
probabilities. The latter is a deliberate model-family contrast versus
direct-stale-state C7-1, not a free claim of more informational content.

**Strict nested time layer:** the outer training population receives OOF
T1 probabilities. For each inner time-layer holdout, recompute conditional
transition crossfit strictly inside its own fit population; train terminal
hurdle on those meta-OOF probabilities and score the inner holdout using
transition fit without its actual later E/U/I. This is computationally
more expensive than reusing one global OOF table, but prevents the
September-Q4 ambiguity where validation-fold labor can indirectly inform
terminal-fit predictions. The class probabilities are NUMERIC columns;
original P1R categorical positions are preserved (no `cats=[]` regression).

Both stages use sequential atomic restartable outer-fold checkpoints. Stage 2
outputs complete person/presence/positive-amount/selected-group metrics and
matched paired deltas versus C7-0 and C7-1, by outer fold and 1Q/3Q ×
earlier E/U/I.

No KL anchoring, official moment targeting, broad hyperparameter sweeps,
tenure/probation analysis, source classifier reruns, separate per-gap fits,
Census scoring, or long-gap extrapolation were added.

## Verification boundary

Synthetic fixture tests:
`tests/test_longitudinal_c8.py`, including train-only T0, adversarial OOF
labor-label perturbation, class normalization/order, source-hash drift,
checkpoint resume, fixed original categorical positions, nested time-layer
labor isolation, a complete stage-2 C7 matched comparison and tampered
parent refusal.

Operational commands and acceptance are in
`docs/C8_LOCAL_LOOP_CLOSURE.md`. No cloud synthetic result substitutes
for real scientific commissioning.
