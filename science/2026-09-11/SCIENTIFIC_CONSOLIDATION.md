# Scientific Consolidation — 2026-09-11

## Executive adjudication

The semantic-infrastructure phase is complete enough for science. The exact real plane `eph-cpv2010-semantic-plane-2024q3-v1` has materialized locally over 47,564 EPH persons and 469,172 Census persons with a passing final support report. The active P1-R plane is the 21-field approved subset; `H11` and `H16` are rejected. Temporal reconstruction is explicitly deferred for this sprint.

The strongest result that is already fully evidenced is Q1: **the current direct hurdle-Gamma model is primarily limited by positive-income magnitude, not by the zero/positive classifier.** Its conditional positive-amount prediction has R² 0.232 and only 0.465 observed dispersion. Household aggregation does not repair this: household dispersion is 0.543, with bottom-decile overprediction of about ARS 398k and top-decile underprediction of about ARS 1.424m. Household ordering is useful but far from exact (Spearman 0.618; mean absolute decile displacement 1.90 deciles).

The central unanswered hinge is Q2. No immutable P1-R EPH run bundle was found in the accessible evidence surface. Therefore no claim is made yet about whether Census-compatible information repairs the conditional-income frontier.

## Question ledger

| Question | Hypothesis / scientific target | Experiment / evidence | Result | Answer | Confidence | Next consequence |
|---|---|---|---|---|---|---|
| Q1 — What is broken? | Positive-amount magnitude is the dominant failure, not participation | Existing exact direct Gamma OOF bundle + derived rank diagnostics | Presence balanced accuracy 0.872; positive R² 0.232, dispersion 0.465; household R² 0.288, dispersion 0.543, Spearman 0.618; bottom/top decile bias +398k / -1.424m | **YES** | **HIGH** | Treat amount compression as the baseline failure to be challenged by P1-R/P2 |
| Q2 — Does P1-R fix it? | Real shared Census-compatible information restores missing conditional-income information | Exact paired P0 vs P1-R, same folds/Gamma/HGB | Not yet run; no persisted P1-R run bundle found | **NOT YET RUN** | HIGH on status | This is the next experiment |
| Q3 — Which family matters? | Any P1-R gain is concentrated in a small information family | Leave-one-family-out P1-R ablation | Gated on Q2 material gain | **NOT YET RUN** | HIGH on status | Run only if Q2 changes regime |
| Q4 — Does labor matter / reconstruct? | True labor state has welfare signal but current reconstruction is weak | Existing minimal-plane ESTADO oracle triangle; P1-R A/B/C still pending | Old oracle person-MAE gain ~21.3k, deployable gain -85; capture -0.004 | **PARTIALLY** | MEDIUM-HIGH | Re-run exact labor oracle only on the Q2-selected plane |
| Q5 — Where is information ceiling? | P2 distinguishes transportable-information loss from model/formulation limit | P0 vs P1-R vs bounded EPH-only P2 | P2 contract identified; matched run absent | **NOT YET RUN** | HIGH on status | Run after/with Q2, no feature fishing |
| Q6 — Is amount still dominant after richer information? | Amount remains dominant after best plane | Baseline oracle-presence + nondeployable observed-positive-amount upper bound; richer-plane repeat pending | Baseline person MAE: 199k -> 171k with oracle presence -> 64.8k in amount upper-bound diagnostic | **PARTIALLY** | HIGH baseline; inconclusive post-P1/P2 | Repeat after information-plane selection |
| Q7 — Do distributions solve threshold estimation? | Probability-of-low-income can be calibrated even with compressed E[Y\|X] | Leakage-safe nested distribution + threshold grid | Primitive exists; no real threshold-grid result found | **NOT YET RUN** | HIGH on status | Run after mean frontier is understood |
| Q8 — First Census commissioning | Selected EPH model can score exact Census P1 matrix | `census_p1.parquet` after Q1-Q7 adjudication | Deliberately not started | **NOT YET RUN** | HIGH on status | Commission only after model/plane selection |
| Q9 — What did sprint establish? | We can locate the bottleneck and close the data boundary | Current evidence set | Baseline failure mode + real semantic plane are established; information frontier is not | **PARTIALLY** | HIGH | Q2 supplies the missing hinge |
| Q10 — Next decision | Freeze science, not infrastructure roadmap | This consolidation | Q2 is first; later work conditional | **PARTIALLY** | HIGH | See final three moves |

## Q1 — What is actually broken?

**ANSWER:** The current direct hurdle-Gamma model is mainly limited by **positive-income amount prediction and associated distributional compression**. Presence is not perfect, but it is not the principal failure. Household aggregation mostly inherits the amount compression and makes the tail asymmetry economically larger. Rank/order contains real signal but is only moderate.

**CONFIDENCE:** HIGH.

**EVIDENCE:**

Baseline run: `real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c`

Exact population:
- 47,564 persons; 41,821 terminal-eligible.
- 25,209 positive; 16,612 zero; 5,688 `-9`; 55 true missing.
- 16,650 households; 12,568 complete observed-income households.
- no survey/design weights.

Presence:
- accuracy **0.8883**
- balanced accuracy **0.8724**
- macro-F1 **0.8804**
- log loss **0.2758**
- Brier **0.1665**
- ECE **0.0052**
- positive recall **0.9501**, zero recall **0.7947**

Conditional positive amount:
- R² **0.2316**
- observed SD **579,074**
- predicted SD **269,147**
- dispersion ratio **0.4648**
- observed-low-tail mean bias **+272,863**
- observed-high-tail mean bias **-910,453**
- observed-decile-1 bias **+272,863**
- observed-decile-10 bias **-931,973**

Complete households:
- R² **0.2881**
- observed SD **901,757**
- predicted SD **489,292**
- dispersion ratio **0.5426**
- Spearman **0.6178**
- mean absolute decile displacement **1.901**
- exact same decile **20.3%**
- within one decile **50.3%**
- bottom-decile bias **+397,564**
- top-decile bias **-1,423,714**

Fold stability rules out a single-split accident:
- positive-amount R²: **0.191–0.252**
- positive dispersion: **0.458–0.473**
- household R²: **0.232–0.321**
- household dispersion: **0.532–0.552**
- household Spearman: **0.609–0.633**
- bottom-tail household bias: about **+375k to +416k**
- top-tail household bias: about **-1.33m to -1.51m**

Direct vs lean also rules out the first cascade as a solution:
- direct Gamma person MAE/RMSE: **198,969 / 414,089**
- lean Gamma person MAE/RMSE: **199,054 / 415,156**
- direct Gamma household MAE/RMSE: **449,640 / 760,856**
- lean Gamma household MAE/RMSE: **449,897 / 762,890**

**WHAT THIS RULES OUT:**
1. “The classifier is basically the whole problem.” No: presence is substantially stronger than the amount head, and amount compression is severe.
2. “Aggregation itself creates the failure from an otherwise good person model.” No: positive-amount compression is already obvious before aggregation.
3. “The model cannot rank at all.” No: household Spearman ~0.62 is meaningful, but ranking is not accurate enough to rescue the badly compressed scale.
4. “The lean ESTADO cascade already solves the missing information.” No: it is essentially tied/slightly worse than direct Gamma.

## Q2 — Does P1-R repair the problem?

**ANSWER: NOT YET RUN.**

The final local semantic plane is ready:
- release `eph-cpv2010-semantic-plane-2024q3-v1`
- P1-R = 21 fields
- final `support_report.status = pass`
- unresolved fields = none
- rejected = `H11`, `H16`
- temporal reconstruction deliberately deferred.

No immutable EPH P1-R run or prediction bundle was found in the connected evidence surface. Therefore it would be scientifically dishonest to infer its effect from the semantic plane or from the historical lean run.

The exact experiment is frozen in `experiment_matrix.json`. The critical requirement is to reuse the baseline household FoldManifest **byte-for-byte** and change only the information plane. The adjudication is based on paired deltas in positive-amount R²/dispersion and household R²/dispersion/rank/tails, not headline MAE alone.

## Q3 — Which P1-R information family works?

**ANSWER: NOT YET RUN, correctly gated.**

Run only if Q2 shows a material regime change. The predeclared leave-one-family-out groups are composition, demographics, education, labor, and housing. If Q2 is null, a broad ablation exercise would be low-value and should be skipped.

## Q4 — Does labor state matter, and can it be reconstructed?

**ANSWER: PARTIALLY.**

The existing minimal-plane experiment gives a useful prior answer:
- direct Gamma person MAE: ~198,969
- true-ESTADO oracle person MAE: ~177,693
- deployable lean person MAE: ~199,054
- oracle gain: ~21,277 MAE
- deployable gain: -85
- capture ratio: -0.004 (RMSE capture ~-0.059)

The ESTADO OOF stage had accuracy 0.8359 but balanced accuracy only 0.5365 and macro-F1 0.5314; its ECE was only ~0.00238. That combination says **labor state contains downstream welfare signal, but the current external plane did not reconstruct the class structure usefully enough**. Calibration alone was not the bottleneck.

This does not settle the requested P1-R experiment. If P1-R itself contains enough new information to reconstruct labor state, the conclusion can change. Therefore A/B/C on `P1-R minus CONDACT` remains the decisive version.

## Q5 — Information ceiling versus model ceiling

**ANSWER: NOT YET RUN.**

P2 no longer needs invention. `income-modeling-eph/configs/feature_contract.yaml` gives a bounded source:
- allowed families: demographic, education, labor, housing/household;
- add the declared household-pyramid block as the richer sensitivity ceiling;
- explicitly exclude P47T/logP47T, direct/transformed income components, income indicators, sample indicators, identifiers, weights, target-derived information and geography ranks.

Important: reuse the feature *contract*, not that repository's historical positive-only log-income cohort or target formulation. P2 here must use the exact same Q3-2024 hurdle-Gamma, eligible population and household folds as P0/P1-R.

## Q6 — Is positive amount the dominant error reservoir?

**ANSWER: PARTIALLY — strongly YES at baseline, not yet adjudicated after P1-R/P2.**

Using the persisted direct-Gamma OOF rows:
- baseline person MAE **198,969**, R² **0.382**, dispersion **0.595**
- true-presence + fitted amount (nondeployable oracle-presence diagnostic): MAE **170,923**, R² **0.440**, dispersion **0.648**
- target-status-conditioned observed-positive-amount upper bound: MAE **64,776**, R² **0.933**, dispersion **0.916**

At household level:
- baseline MAE **449,640**, R² **0.288**, dispersion **0.543**
- oracle presence: MAE **437,782**, R² **0.315**, dispersion **0.581**
- amount upper bound: MAE **132,338**, R² **0.939**, dispersion **0.929**

The amount-upper-bound diagnostic is explicitly **NON-DEPLOYABLE and NON-CAUSAL**. It does not tell us that an amount model can achieve this result. It does show that the amount side contains a much larger error reservoir than zero/positive presence in the baseline.

## Q7 — Predictive distributions

**ANSWER: NOT YET RUN.**

The repository has the leakage-safe nested empirical-residual primitive, but no persisted real threshold-grid result was found. Do not run it until Q2/Q5 establish the mean-model frontier. Then test poverty-adjacent thresholds using observed prevalence, point prevalence, mean predicted probability, Brier, calibration and prevalence error.

## Q8 — Census commissioning

**ANSWER: NOT YET RUN by design.**

The exact `census_p1.parquet` has 469,172 rows locally, but commissioning should not preempt Q2-Q7. When it starts, it is a research commissioning run, not an official 2024 poverty estimate. Large `IX_TOT` values must remain untouched; valid value, training-support overlap and transport universe are separate concepts.

## Current repository/evidence state

- `encuestador-de-hogares` main: `d216cf3...` — direct/lean real Gamma bundles exist; science-diagnostics and nested distribution primitives are landed.
- `eph-censo-aligner` main: `4e01f4b...` — executable real alignment merged.
- `income-modeling-eph` main: `774c290...` — governed income-study cohort and explicit feature contract available for P2 definition.
- `microdatos-EPH-INDEC` master: `492bc92...` — source/acquisition authority for the exact Q3 release.

One reproducibility drift is recorded, but it is not a reason to reopen semantic science: current `eph-censo-aligner` GitHub main still contains `IX_TOT max=40` in the committed review policy, while the successful local materialization explicitly fixed the collective-dwelling case and passed support without clipping. Capture that local implementation fix before any future regeneration of the plane; do not reinterpret the semantics.

## Emerging scientific picture

The evidence already rejects a simplistic “classification problem” story. The current model knows a substantial amount about who has positive income and has moderate household ranking ability, but it compresses the conditional-income distribution drastically. That compression survives aggregation and appears as economically large bottom-overprediction/top-underprediction.

The old labor oracle says missing labor-state information can matter, but the first cascade failed to reconstruct enough of that state to create downstream value. This makes the new P1-R experiment unusually informative: if the full real Census-compatible plane materially restores amount dispersion and household tails, the bottleneck was missing transportable information. If it does not, P2 tells us whether the missing information exists only in richer EPH observables. If neither P1-R nor P2 changes the regime, the remaining frontier is the conditional-mean/model formulation or irreducible heterogeneity, and Q7 becomes central.

## At most three next scientific moves

1. **Run Q2: exact paired P0 vs P1-R hurdle-Gamma.** Reuse the existing household folds exactly. This is the scientific hinge; do not tune anything else first.
2. **Locate the information ceiling conditional on Q2.** Run bounded P2 under identical folds; if P1-R gains materially, add the small family ablation and P1-R labor A/B/C rather than a broad feature-selection program.
3. **After the mean frontier is known, test the nested predictive distribution on a threshold grid.** If it recovers calibrated low-tail prevalence where the mean remains compressed, promote a probability-of-poverty interface; only then commission the selected model on the exact Census matrix.
