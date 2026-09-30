# Labor upgrade — execution DAG after C1/C2/C3

Status: coordination front door, 2026-09-29.

## Completed cloud foundation

Merged and complete:

- C1 — `empleoARG` canonical official labor-state product runtime;
- C2 — `income-modeling-eph` 37-quarter longitudinal EPH frame runtime;
- C3 — `eph-censo-aligner` explicit donor-vintage labor semantics;
- B0 — Telescope-B baseline five-fold fixture repair;
- C4/C4B — `encuestador-de-hogares` longitudinal L10/L11/L12 runtime, canonical composition-parent consumption, and nested-OOF time-layer hardening;
- C5 — `eph-censo-aligner` longitudinal canonical composition-plane runtime with `P0_LONG` and `P1R_NOLAB_LONG`.

The cloud implementation phase is closed. Remaining substantive work is real-data/local commissioning.

## Cloud receipts

The completed cloud packets remain documented for provenance:

- B0: `docs/BASELINE_TELESCOPE_B_TEST_REPAIR.md`;
- C4B: `docs/C4B_LONGITUDINAL_RUNTIME_HARDENING_PLAN.md`;
- C5: `matuteiglesias/eph-censo-aligner/docs/C5_LONGITUDINAL_COMPOSITION_PLANE_PLAN.md`.

Do not reopen them during local work unless a real-data gate exposes a concrete contract defect.

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
Local: L1 + L2 + L3A (when convenient)
```

Then:

```text
L2 + merged C5 -> L3B
L1 + L2 + L3B + merged C4/C4B -> L4 Gate A
```

Do not start heavy L4 Gate A until L1/L2/L3B have immutable real releases.

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
