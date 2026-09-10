# Science Infrastructure Sprint — Implementation Status

**Date:** 2026-09-10  
**Status basis:** original science-infra seed capability criteria, not implementation milestone labels  
**Real evidence head:** `d02036e1a2b2023f6db66c37de66778580111dfd`  
**Real acceptance workflow:** `Real EPH 2024-Q3 acceptance`, run `34540912580`  
**Evidence artifact:** `real-eph-2024q3-science-evidence`, artifact `10177301359`

This is a follow-up implementation ledger. It does not rewrite the seed intent in `00`–`12`.

## Capability ledger

| Capability | Status | Evidence / exact blocker |
|---|---|---|
| Household-safe OOF | GREEN | reusable `FoldManifest`; household-grouped outer folds; direct and lean real runs share fold counts and exact manifest semantics |
| HGB adapters | GREEN | classifier probabilities, squared-error and Gamma regressor adapters; no hidden early stopping; real HGB execution passed |
| Direct HGB architecture | GREEN | governed real Q3 direct hurdle runs completed for Gamma and log formulations |
| Lean probability cascade | GREEN | one-layer `ESTADO` probability cascade executes with outer-fold-isolated nested latent fitting; real Q3 runs completed |
| Oracle evaluator | GREEN | real lean runs emit matched direct/oracle/deployable evidence and capture ratios under identical outer folds |
| Terminal hurdle | GREEN | explicit `p_positive`, positive-amount prediction and unconditional linear-scale expected income; zero retained, `-9` and true missing unavailable |
| Hurdle-log | GREEN | HGB presence + positive log-income head + declared Duan smearing retransformation; real Q3 direct and lean runs completed |
| Hurdle-Gamma | GREEN | HGB presence + strictly-positive Gamma amount head + linear-scale combination; real Q3 direct and lean runs completed |
| Real EPH intake | GREEN | exact `publicdata.eph-microdata@1` Q3-2024 producer release reproduced and hash-verified; source-faithful individual/household join only |
| Real EPH direct run | GREEN | `real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c` and log counterpart completed |
| Real EPH lean run | GREEN | `real_eph_2024q3_lean_hurdle_gamma_v1-731b59b3bd55b2dc` and log counterpart completed |
| Distributional diagnostics | GREEN | person/household linear errors, dispersion ratio, quantiles, deciles and tail bias persisted in run metrics |
| Household diagnostics | GREEN | complete-observed-income cohort explicit: 12,568 complete / 4,082 unavailable of 16,650 household observations |
| Deployable fitted transport state | GREEN | immutable run stores fitted direct/lean hurdle model; lean terminal is fit on OOF latent meta-features and full-fit latent model is used for scoring |
| Immutable run bundle | GREEN | `research.encuestador-run/v1`; resolved config, hashes, folds, OOF predictions/components, latents, metrics, household rows, model, software identity and limitations; all four real bundles hash-validated |
| CLI | GREEN | installed `encuestador validate`, `run`, `compare`, `report`; exercised by hosted synthetic and real acceptance workflows |
| Real Census frame/sample | BLOCKED | governed CPV-2010 `research.census-frame/v1` followed by `research.census-target-year-sample/v2` has not yet surfaced to this integration run |
| Real semantic feature plane | BLOCKED | `eph-censo-aligner` must review the active minimal CPV-2010/EPH feature plane against the real frame/sample schemas |
| Real Census welfare scoring | BLOCKED | depends only on the preceding real sampler-v2 artifact plus reviewed scoring feature plane; exact sample-person to sample-household scoring machinery is implemented and synthetically tested |

## Real EPH evidence

Pinned release: `eph-2024-q3-3b6a7a15c4af`, period `2024-Q3`.

- persons: **47,564**
- households: **16,650**
- terminal-income eligible: **41,821**
- positive: **25,209**
- zero: **16,612**
- `-9` nonresponse: **5,688**
- true missing: **55**
- fold row counts: **9,831 / 9,543 / 9,425 / 9,260 / 9,505**
- complete observed-income households: **12,568**
- unavailable observed-income households: **4,082**
- fitting/evaluation survey weights: **none**

Exact source evidence:

- source archive SHA256: `fbeacb866777759e2fcbc64e9643734bae29ec6db480dd6d532dc505505b3204`
- source manifest SHA256: `0104a46dc53dfb537a730e6fa323ee5ab0240cc838b66d32459cb2e2c3002f5d`
- individual SHA256: `b435a2f22c72c5d0e281279f27ade8da674c6db6fe6597b620dc0485d9b27260`
- household SHA256: `4fc25e8d9e283be1d9c41bc93d237e6c6ad666dbc80bb33559cbe185d17f2fb5`

## First governed scientific comparison

All architecture comparisons use the same candidate external feature plane and the same five household folds. The lean arm adds only OOF `ESTADO` probabilities.

| Candidate | Person MAE | Person RMSE | Household MAE | Household RMSE |
|---|---:|---:|---:|---:|
| Direct hurdle-Gamma | **198,969** | **414,089** | **449,640** | **760,856** |
| Lean hurdle-Gamma | 199,054 | 415,156 | 449,897 | 762,890 |
| Direct hurdle-log | 201,010 | 414,301 | 456,984 | 762,120 |
| Lean hurdle-log | 200,934 | 415,082 | 457,505 | 764,526 |

The first real evidence does **not** support promoting the lean `ESTADO` cascade. Direct hurdle-Gamma is strongest on all four headline metrics among these four bounded candidates.

The oracle result is scientifically informative. Under Gamma, direct person MAE is `198,969`, oracle MAE is `177,693`, and deployable lean MAE is `199,054`: oracle gain is about **21,277**, while deployable gain is **-85**, for an MAE capture ratio of **-0.004**. RMSE capture ratio is about **-0.059**. Under hurdle-log, MAE capture is only **0.0035** and RMSE capture is **-0.043**.

The `ESTADO` OOF reconstruction itself has accuracy `0.8359`, balanced accuracy `0.5365`, macro-F1 `0.5314`, log loss `0.4243`, multiclass Brier `0.2373`, and expected calibration error about `0.00238`. The main weakness is class reconstruction, not gross probability calibration. The oracle/deployable split therefore suggests that labor state contains useful welfare information but is not reconstructed usefully enough from the current minimal external feature plane.

Distributional diagnostics also show substantial compression remains. For direct Gamma, person unconditional-income dispersion ratio is about `0.595`; household dispersion ratio is about `0.543`. Mean bias in the observed high tail is approximately `-692k` per person and `-1.420m` per complete household, while the low tail is overpredicted. This is evidence for future scientific refinement, not a reason to delay the current integration close.

## Explicit scope at close

The active runtime intentionally supports the sprint-critical direct and one-layer lean execution path. Generic DAG **validation/config representation** exists, but arbitrary deep-DAG execution is not claimed green. Historical-depth execution, probability-calibration intervention, aggregate anchors, cluster bootstrap, RF robustness, and national Censo-2022 work were not allowed to outrun the real-EPH integration gateway and remain deferred.

The real Q3 feature plane is an **EPH-side Census candidate**, not a certified Census deployment plane. Real Census scoring may proceed only after the current CPV-2010 producer artifacts and the corresponding minimal `eph-censo-aligner` semantic review exist.

## Sprint-close conclusion

The original seed's critical real-science gateway is green:

```text
real official EPH
 -> governed direct + lean hurdle experiments
 -> identical household-safe folds
 -> oracle/deployable diagnostics
 -> deployable fitted models
 -> person + complete-household OOF evidence
 -> immutable validated run bundles
 -> CLI comparison/report surface
```

The remaining end-to-end gap is external and named:

```text
CPV-2010 census-frame/v1
 -> sampler-v2 sample
 -> minimal real-vintage eph-censo-aligner approval
 -> existing exact sample-person / sample-household scoring handoff
```
