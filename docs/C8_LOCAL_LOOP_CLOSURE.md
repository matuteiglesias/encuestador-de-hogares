# Local C8B — close the raw L12 scientific loop (Codex task)

Repository: /home/matias/repos/encuestador-de-hogares
Cloud code: C8 PR/merge receipt; commands below assume main includes C8.
Research mode ONLY. No Census scoring, source mutation or automatic promotion.

## Dependencies, in strict order

The user deferred local R0/C7B while away from the ThinkPad. When local
access resumes, complete these in order:

1. Verify source lineage using the existing
   `docs/C7_DONOR_CLOCK_MATCHED_L11_CONTRACT.md` and original real Gate-B
   receipt. L2 parent:
   `/home/matias/data/eph-longitudinal-2017-2026/releases/eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39`.
   Gate B:
   `/home/matias/data/l4-gate-b-20261001/gate-b-panel-evidence-c15bac5f1a6cbeb7`.
   Confirm source IDs/hashes and 827,793 eligible pairs (571,984 1Q,
   255,809 3Q) BEFORE valid-later-income restriction.
   Do not repeat L2/Gate B materialization.
2. Finish the C7B local task exactly as
   `docs/C7A_RESOURCE_SAFE_L11_EXECUTION.md` directs: materialize private
   C7 donor-X panel from the accepted C6 P1R plane and complete matched
   C7-0/C7-1 outer-fold run. The actual C7 valid-income row count is not
   known a priori. Freeze `C7_REAL_ACCEPTANCE.json` and scientific findings
   before attempting stage 2.
3. Run C8 fixture tests locally. Check available RAM and swap; Stage 2's
   strictly nested time transition is costlier than C7, so start with one
   outer-fold canary. Never run parallel fold processes on a RAM-limited
   ThinkPad.

## Sync and quick checks

    cd /home/matias/repos/encuestador-de-hogares
    git status --short
    git fetch origin
    git switch main
    git pull --ff-only origin main
    PYTHONPATH=src python3 -m pytest -q tests/test_longitudinal_c8.py
    PYTHONPATH=src python3 -m encuestador.cli longitudinal-validate \
      configs/longitudinal/c8_real_p1r.yaml

Preserve local changes if any; do not force-reset or create a replacement
source release. Prefer a clean checkout before commissioning.

Set exact accepted paths from the C7 JSON receipts; do NOT guess the
content-addressed panel/run filenames:

    export C7_PANEL="FULL_DIRECTORY_FROM_C7_PLANE_RECEIPT"
    export C7_MATCHED_RUN="FULL_DIRECTORY_FROM_C7B_COMPLETE_RUN_RECEIPT"
    export C8_ROOT="/home/matias/data/l4-c8-20261001"
    mkdir -p "$C8_ROOT"/transitions "$C8_ROOT"/welfare

C7_PANEL must contain manifest.json, features.npy, fold_ids.npy and
target_labor.npy. C7_MATCHED_RUN must contain run_manifest.json and the
two matched C7 expected-income arrays. A .work directory is NOT a completed
C7 comparator.

## Stage 1 — real transition commissioning only

Optional outer-fold 0 resource and restartability canary:

    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c8-transition \
      configs/longitudinal/c8_real_p1r.yaml \
      --panel-root "$C7_PANEL" \
      --output-root "$C8_ROOT/transitions" --outer-fold 0 \
      2>&1 | tee "$C8_ROOT/transition_fold0.log"

Then complete verified missing outer folds:

    set -o pipefail
    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c8-transition \
      configs/longitudinal/c8_real_p1r.yaml \
      --panel-root "$C7_PANEL" \
      --output-root "$C8_ROOT/transitions" \
      2>&1 | tee "$C8_ROOT/transition_complete.log"

Record the JSON-emitted `root` value as C8_TRANSITION.
Verify `transition_manifest.json` (contract
`research.encuestador-c8-raw-transition/v1`), hashes, all five fold
checkpoints, row/fold identity against C7, E/U/I class order (1,2,3),
finite/normalized q arrays, and source parent custody. The baseline T0 MUST
be estimated within each training fold; do not import full Gate-B population
rates as a feature or comparator. Review `transition_metrics.json`:

- T0 vs T1 held-out multiclass log-loss and three-class Brier;
- per-class support/recall, probability calibration;
- 1Q/3Q and earlier E/U/I strata;
- foldwise consistency and any class-degenerate T1 fallback;
- actual peak RSS, active memory and swap behavior.

Write:
`$C8_ROOT/C8_TRANSITION_REAL_ACCEPTANCE.json` and
`$C8_ROOT/C8_TRANSITION_FINDINGS.md`.

**Explicit stop gate:** report whether conditional T1 offers sufficiently
credible calibration/transition value over T0 to justify the expensive
L12 cascade. If weak/inconsistent, stop, retain stage-1 receipt and state
reason for deferring stage 2. Do not manufacture a preselected improvement
threshold or claim mathematical impossibility.

## Stage 2 — optional explicit raw L12 welfare comparison

Only after stage 1 reviewed AND matched real C7B run accepted. If both
accepted, set C8_TRANSITION to the EXACT completed transition release
directory emitted in the command above, then run fold-0 canary:

    export C8_TRANSITION="FULL_DIRECTORY_FROM_C8_TRANSITION_RECEIPT"

    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c8-welfare \
      configs/longitudinal/c8_real_p1r.yaml \
      --panel-root "$C7_PANEL" \
      --transition-root "$C8_TRANSITION" \
      --c7-run-root "$C7_MATCHED_RUN" \
      --output-root "$C8_ROOT/welfare" --outer-fold 0 \
      2>&1 | tee "$C8_ROOT/welfare_fold0.log"

Inspect peak RSS / swap on the actual host. If safe, complete remaining
five grouped outer-fold checkpoints sequentially:

    set -o pipefail
    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c8-welfare \
      configs/longitudinal/c8_real_p1r.yaml \
      --panel-root "$C7_PANEL" \
      --transition-root "$C8_TRANSITION" \
      --c7-run-root "$C7_MATCHED_RUN" \
      --output-root "$C8_ROOT/welfare" \
      2>&1 | tee "$C8_ROOT/welfare_complete.log"

Stage 2 must FAIL CLOSED if the completed C7 comparator is missing, any
parent SHA mismatches, the real C7 and C8 estimator configurations differ
for the hurdle, or the C7 matched row/fold identities do not match.

Review output `welfare_metrics.json` and `run_manifest.json`:

- C8-1 vs C7-0 and vs C7-1 on EXACT rows, fold IDs and later P47T_real;
- person MAE/RMSE, positive-income presence and positive-amount metrics;
- paired per-fold differences; 1Q/3Q × earlier labor stratification;
- selected-panel-household group sums, explicitly NOT full-household income;
- fixed original P1R categorical metadata; numeric q_E/q_U/q_I;
- no actual later person labor in any terminal features;
- strict inner-time transition nesting and outer holdout isolation;
- Stage 1 empirical T0 comparison preserved as external evidence;
- real resource/swap diagnostics and candidate monetary parent caveat.

Write `C8_WELFARE_REAL_ACCEPTANCE.json` and
`C8_SCIENTIFIC_FINDINGS.md` under C8_ROOT. Include exact source/manifest
SHA-256, Git SHA, model configuration SHA, elapsed-gap support, output
contract IDs, and completed fold checkpoint SHA inventory.

## Forbidden shortcuts

- Do NOT automatically proceed from stage 1 to stage 2.
- Do NOT train T0/T1 on full Gate-B outcomes when scoring any of their rows.
- Do NOT introduce later true E/U/I to the terminal welfare model.
- Do NOT append probabilities while silently removing categorical metadata
  for earlier shared features (old Q4 `cats=[]` failure mode).
- Do NOT pool households across outer train/holdout or silently replace the
  accepted L2 candidate linkage policy.
- Do NOT refit C7-0/C7-1 with different data or parameters just to favor L12.
- Do NOT treat C7 valid-income pairs as a population-weighted national EPH.
- Do NOT perform official-moment/KL anchoring, CPV-2010/2022 scoring or
  donor-gap extrapolation in C8B.
- Do NOT change `indice-pobreza-UBA` promotion registry on fixture evidence.

Exit with a compact receipt: ACCEPTED, HOLD or BLOCKED for transition and,
if attempted, welfare, with precise source hashes, metrics, resource use,
unresolved limitations and smallest following research decision. Raw C8 is
a short-gap EPH research model until a separate Census transport gate is
explicitly commissioned.
