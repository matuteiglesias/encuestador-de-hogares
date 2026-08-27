# Repository lifecycle

**State:** `active-bounded`  
**Decision date:** 2026-08-26  
**Review cadence:** active-development

## Revival decision

The previous `candidate` gate has been passed for one bounded purpose: this repository is the intended **EPH -> Census statistical transport and welfare-inference instrument**.

A current consumer now exists: Poverty v2 needs an exact, governed household-welfare release over a Census-derived population frame, while the surrounding estate intentionally keeps the necessary concerns separate:

- `income-modeling-eph` owns EPH-only model science and should not know about Census deployment;
- `eph-censo-aligner` owns semantic EPH/Census mappings and their loss/ambiguity review;
- `samplerCensoARG` owns Census sample/frame identity, household membership and sampling/design semantics;
- `IPC-Argentina` owns monetary-reference/conversion semantics;
- `indice-pobreza-UBA` owns poverty methodology, thresholds, classification and estimation.

The missing capability is the bridge between an approved EPH information plane and an exact Census scoring frame. This repository now owns that bridge.

See:

- `SYSTEM.yaml` for the authority boundary;
- `contracts/deployment_dag.yaml` for the recovered candidate feature/stage graph;
- `docs/EPH_CENSUS_TRANSPORT_BOUNDARY.md` for the scientific decision and promotion gates.

## What is active

Allowed and expected work is limited to the revived transport boundary:

- recover and review the historical staged-classifier/regressor design;
- maintain the deployment DAG and feature classification;
- consume exact versioned EPH/alignment/Census/monetary artifacts rather than sibling runtime code;
- implement honest out-of-fold intermediate training;
- characterize support/domain shift, subgroup error and cascade error propagation;
- package promoted transport-model artifacts;
- score one exact Census feature/sample release;
- resolve model-native transforms and monetary references into a declared welfare concept;
- publish an exact household-welfare handoff with QA, limitations and checksums.

## What remains historical or disabled

The repository is **not** revived as its former monolithic pipeline.

- `preprocesar_datos.py` remains historical preprocessing evidence; current annual EPH preprocessing authority stays upstream.
- Historical EPH/Census renames and recodes are evidence until approved under the current aligner's real-vintage review process.
- Existing serialized models, figures and `EPHARG_train*` files are legacy evidence, not current releases.
- Old cron/daily-update/write-back behaviour remains disabled.
- The legacy four-wave RFC1-RFC4 grouping is evidence, not a frozen modern architecture.
- Large training or Census inference jobs must not run until exact parent releases and promotion gates are satisfied.

## Scientific stop conditions

A real inference run is blocked until all of the following are true:

1. one exact EPH release and one exact Census sample/frame release are pinned;
2. the real-vintage semantic feature plane is reviewed in `eph-censo-aligner`;
3. target-derived `AGLO_rk`/`Reg_rk` are absent from the external deployment input plane unless replaced by independently governed information;
4. every learned intermediate used downstream is generated out-of-fold during training;
5. exact Census person/household identity is preserved with no fuzzy or positional matching;
6. support/domain-shift diagnostics are available;
7. monetary target/reference lineage is resolved through a versioned `IPC-Argentina` conversion artifact;
8. the final person-to-household welfare concept is explicitly approved;
9. a deterministic synthetic end-to-end transport-model + welfare release succeeds first.

## Preprocessing authority remains transferred

This repository is historical evidence for the former `EPHARG_train*` preprocessing lineage. `income-modeling-eph` remains the current producer of the versioned annual EPH preprocessing artifacts during this transition; this repository consumes an artifact and does not regain preprocessing authority. See `docs/PREPROCESSING_AUTHORITY_HANDOFF.md`.
