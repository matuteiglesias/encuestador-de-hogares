# Labor upgrade — execution DAG after C1/C2/C3

Status: coordination front door, 2026-09-29.

## Completed cloud foundation

Merged:

- C1 — `empleoARG` canonical official labor-state product runtime;
- C2 — `income-modeling-eph` 37-quarter longitudinal EPH frame runtime;
- C3 — `eph-censo-aligner` explicit donor-vintage labor semantics.

Open:

- C4 PR #33 — longitudinal L10/L11/L12 runtime.

## Remaining cloud work

### B0 — baseline test repair

Repo: `encuestador-de-hogares`

Instruction:

`docs/BASELINE_TELESCOPE_B_TEST_REPAIR.md`

Small independent maintenance PR. It is not part of C4 science.

### C4B — harden PR #33

Repo: `encuestador-de-hogares`

Instruction on C4 branch:

`docs/C4B_LONGITUDINAL_RUNTIME_HARDENING_PLAN.md`

Adds:

- explicit canonical composition-plane consumption/profile identity;
- fixture-only classification for the current minimal raw feature list;
- inner-OOF base predictions for year/quarter/shock time-layer estimation.

C4B can proceed against a synthetic C5 parent.

### C5 — longitudinal canonical composition plane

Repo: `eph-censo-aligner`

Instruction:

`docs/C5_LONGITUDINAL_COMPOSITION_PLANE_PLAN.md`

Adds named canonical profiles:

- `P0_LONG`;
- `P1R_NOLAB_LONG`.

C5 and C4B can run in parallel.

## Remaining local work

### L1 — official labor release

Repo: `empleoARG`

Instruction:

`docs/L1_REAL_LABOR_STATE_RELEASE_GATE.md`

Produces the real 2017-Q1..2026-Q2 labor-context release.

### L2 — real longitudinal EPH frame

Repo: `income-modeling-eph`

Instruction:

`docs/L2_REAL_LONGITUDINAL_EPH_COMMISSIONING_GATE.md`

Produces the real 2017-Q1..2026-Q1 EPH frame and adjudicates monetary timing.

L1 and L2 should run in parallel.

### L3A / L3B — semantic real gates

Repo: `eph-censo-aligner`

Instruction:

`docs/L3_REAL_SEMANTIC_MATERIALIZATION_GATES.md`

- L3A: CPV-2010 donor-labor handoff.
- L3B: real 37-quarter canonical composition plane.

L3B depends on C5 + L2.

L3A is independent and is needed only for later donor-aware Census scoring.

### L4 — welfare/labor commissioning

Repo: `encuestador-de-hogares`

Instruction:

`docs/L4_REAL_LONGITUDINAL_WELFARE_COMMISSIONING.md`

Runs:

- Gate A: L10 profile/context/time adjudication;
- Gate B: real panel persistence;
- Gate C: L11;
- Gate D: L12 raw/anchored;
- Gate E: bounded CPV-2010 research scoring.

## DAG

```text
                         CLOUD

              B0 baseline repair
                      |
                      v
                full main pytest green

   C4B encuestador hardening  <----->  C5 aligner composition plane
          |                                |
          |                                |
          +---------------+----------------+
                          |
                       merge C4
                          |
                          v

                         LOCAL

       L1 real labor release      L2 real 37Q EPH frame
               |                         |
               |                         +----------+
               |                                    |
               |                              L3B real canonical
               |                              composition plane
               |                                    |
               +------------------+-----------------+
                                  |
                                  v
                           L4 Gate A — L10
                       profile + labor context + time
                                  |
                                  +-------------------+
                                  |                   |
                                  v                   |
                          L4 Gate B panel             |
                                  |                   |
                           +------+-------+            |
                           |              |            |
                           v              v            |
                        Gate C L11     Gate D L12      |
                           |              |            |
                           +------+-------+            |
                                  |                    |
                                  v                    |
                         arm adjudication              |
                                  |                    |
                                  +--------------------+
                                  |
                     L3A CPV-2010 donor handoff
                     (required only if selected arm
                        uses donor labor state)
                                  |
                                  v
                         L4 Gate E Census proof
                                  |
                                  v
                     household-welfare release
                                  |
                                  v
                        indice-pobreza-UBA
```

## Parallelism

Start immediately in parallel:

```text
Cloud: C4B + C5 + B0
Local: L1 + L2
```

L3A may also run opportunistically if the exact CPV-2010 sample is already available.

Do not start heavy L4 Gate A until C4B/C5 are merged and L1/L2/L3B have immutable real releases.

## Program completion

The labor upgrade is complete when:

1. exact official aggregate labor context is available for every modeled quarter;
2. real 37-quarter EPH welfare evidence uses governed real monetary semantics;
3. composition enters through a named canonical plane rather than raw recodes inside the model repo;
4. L10 is commissioned with aggregate labor + explicit time decomposition;
5. short-gap stale labor value is measured honestly;
6. L11/L12 are adjudicated rather than assumed useful;
7. any KL anchor is tied to an explicit compatible official universe;
8. the selected arm can produce an auditable research Census welfare release;
9. no forecast/nowcast claim is introduced.
