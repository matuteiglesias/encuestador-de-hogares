# Repository lifecycle

**State:** `candidate`  
**Decision date:** 2026-08-03  
**Review cadence:** consumer-triggered

## Why this state

This repository preserves an ambitious EPH/census modeling workflow for imputing household and individual characteristics. It may contain valuable research ideas and implementation knowledge, but there is no declared current consumer that justifies reviving its data preparation, model training, automation, or dependency surface.

## Revival gate

Do not begin maintenance or modernization until a proposal names:

1. the current consumer;
2. the decision, research output, or system capability they need;
3. why the current EPH, census, poverty, or income-modeling estate does not already satisfy it;
4. the exact variables and geography required;
5. the smallest reproducible experiment that would establish value;
6. the data-access, privacy, bias, and methodological review required.

## Current interpretation

- Existing models, figures, periodic-update instructions, and economic values are historical unless reverified.
- Cron or daily-update language is not evidence that any automation is currently operating.
- Large training sets and serialized models are external artifacts, not evidence of a reproducible current release.
- The repository is not current authority for EPH microdata, census data, price indices, or official population estimates.

## Allowed work while candidate

- preserve the repository and research genealogy;
- document known inputs, outputs, assumptions, and limitations;
- identify reusable concepts without moving code;
- prepare a bounded revival decision memo.

Do not run billable or large-scale jobs, refresh datasets, retrain models, or redesign the architecture merely to see whether the project can be made current.

## Preprocessing authority

This repository is historical evidence for the former `EPHARG_train*` preprocessing lineage. `income-modeling-eph` is the current authority for annual preprocessing and `EPHARG_annual_input_*`; no artifact here should be presented as a current release. See `docs/PREPROCESSING_AUTHORITY_HANDOFF.md` before interpreting or transferring legacy behavior. Legacy schedules and automatic commits must remain dormant.
