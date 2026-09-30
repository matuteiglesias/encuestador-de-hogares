# Labor upgrade — current execution DAG

Status: coordination front door, 2026-09-30.

This document tracks the **longitudinal 2017-Q1..2026-Q1 welfare/labor program**.
It must not be confused with the older bounded 2024-Q3 labor-bridge L1–L4 family
retained in the Poverty commissioning registry.

## Completed foundations

Cloud/scientific runtime:

- C1 — canonical official aggregate labor product runtime;
- C2 — 37-quarter longitudinal EPH frame runtime;
- C3 — explicit donor-vintage labor semantics;
- B0 — Telescope-B five-fold fixture repair;
- C4/C4B — L10/L11/L12 runtime, explicit composition custody and nested-OOF time layer;
- C5 — named canonical longitudinal composition profiles.

Real/local gates:

- L1 — real official labor release materialized; 1060/1064 official required cells;
- L1B — explicit bounded completion overlays for the documented 2019-Q3 NEA gap;
- L2 — real 37-quarter EPH frame materialized: 1,869,620 person-period rows;
- L3A — real CPV-2010 donor-labor handoff materialized;
- L3B — real `P0_LONG` and `P1R_NOLAB_LONG` planes materialized over all 37 quarters;
- L4 preflight — exact upstream parent intake and tests green.

Authoritative factual receipt:

`docs/LABOR_UPGRADE_CURRENT_STATE_2026-09-30.md`

## Current blocker

The first full real L10 attempt reached the real input runtime but terminated before a
run bundle was emitted because the current implementation holds the full 1.87M-row EPH
surface and canonical composition surface in Python object-heavy lists.

This is a **resource/execution-plane blocker**, not an upstream data or semantic blocker.

No L10/L11/L12 scientific result has yet been produced by the longitudinal program.

## Current DAG

```text
                         COMPLETE

       L1 official labor  +  L1B bounded completion
                    \        /
                     \      /
              L2 real 37Q EPH frame
                     |
        +------------+-------------+
        |                          |
        v                          v
 L3A donor labor             L3B composition
 CPV-2010 clock          P0_LONG / P1R_NOLAB_LONG
        |                          |
        +------------+-------------+
                     |
               C4/C4B runtime
                     |
               L4 preflight green
                     |
                     v

                    CURRENT

        resource-safe execution plane (C6)
        columnar / bounded / restartable
                     |
                     v
              L4 Gate A — L10
        profile + labor context + time
                     |
              L4 Gate B — panel
                     |
              +------+------+
              |             |
              v             v
           Gate C L11    Gate D L12
              |             |
              +------+------+
                     |
                     v
              arm adjudication
                     |
                     v
             Gate E Census proof
                     |
                     v
       research.household-welfare@1
                     |
                     v
             indice-pobreza-UBA
```

## Execution rule before C6

Do not:

- rerun L1/L2/L3A/L3B merely because L4 exhausted memory;
- move semantic recodes into `encuestador-de-hogares`;
- weaken nested OOF time-layer/fold semantics;
- promote the candidate monetary parent to approved status implicitly;
- claim a longitudinal L10/L11/L12 result from preflight alone.

C6 should change the **data/execution representation**, not the scientific estimand.

## Program completion

The longitudinal labor upgrade completes only after:

1. resource-safe real L10 matched profile/context/time commissioning;
2. short-gap panel persistence evidence is measured;
3. L11/L12 are adjudicated rather than assumed useful;
4. any KL anchor is tied to an explicit compatible universe;
5. the selected arm completes one bounded exact Census research-scoring proof;
6. a household-welfare release is emitted with exact lineage and limitations;
7. no forecast/nowcast claim is introduced.
