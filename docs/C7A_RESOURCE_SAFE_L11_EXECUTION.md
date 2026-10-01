# C7A — resource-safe, matched historical-donor L11

Status: cloud implementation / synthetic fixture verification only.
No private 1.87M-row real input has been processed by GitHub Actions. R0
real-parent acceptance is deferred to the local agent; C7B owns real-scale
resource and scientific commissioning. C7A does not authorize Census scoring.

## Scope and estimand

Primary C7 contrast, pooled at observed 1Q/3Q gaps:

- C7-0: earlier-wave canonical P1R composition, elapsed gap, later-quarter
  official national/regional context and explicit target-period time;
- C7-1 (L11): **exactly the same** plus actually observed earlier-wave
  three-state individual labor.

C7-0 is a matched historical-input L10 analogue, not the full-population
current-composition Gate-A L10. Both arms use the identical subset of
Gate-B-eligible pairs with valid later P47T_real, fold vector, C4 hurdle
estimator and C6 nested household-safe OOF time correction. Genuine income
zeros remain valid. Invalid earlier income does not exclude the pair.

Actual *later* individual labor is stored only as a private L12 research
label and is NOT a C7 terminal predictor. Target-wave individual composition
also does not enter the fitted donor-X features. The only C7-1 increment is
the earlier labor column. A shorter-horizon association is not evidence of
individual persistence from CPV-2010 to 2024/2026.

## Executable files

- `src/encuestador/longitudinal_c7_contract.py`: R0 selection/provenance.
- `src/encuestador/longitudinal_c7.py`: real source verifier, indexed
  Gate-B replay, exact row-order C2↔C6 join, and immutable compact private
  pair/feature plane.
- `src/encuestador/longitudinal_c7_run.py`: matched C7-0/C7-1 sequential
  OOF fitting, independently verifiable atomic paired checkpoints, point
  and categorical/horizon diagnostics.
- `configs/longitudinal/c7_real_p1r.yaml`: frozen real configuration.
- `tests/test_longitudinal_c7.py`: synthetic parent custody, count/hash/
  fold/donor-clock checks, restartability and numerical C4 parity.

## Local C7B invocation (after R0 local identity inspection)

Use a clean local main containing the merged C7A commit. Do not reconstruct
L2, L3B, official labor context, the accepted C6 P1R model plane or Gate B.

Find the exact accepted P1R C6 model-plane directory (has `manifest.json`,
`features.npy`, `row_ids.txt`; profile `P1R_NOLAB_LONG`) under the already
materialized C6 resource-safe workspace. Verify its source parents, version
and hashes before selecting it. Pass its **directory** to C7, not the C6
run-summary directory.

Materialize the immutable private C7 panel:

    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c7-plane \
      --eph-release-root \
      /home/matias/data/eph-longitudinal-2017-2026/releases/eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39 \
      --gate-b-root \
      /home/matias/data/l4-gate-b-20261001/gate-b-panel-evidence-c15bac5f1a6cbeb7 \
      --c6-plane-root PATH_TO_ACCEPTED_P1R_C6_PLANE \
      --output-root /home/matias/data/l4-c7-20261001/panels

Keep the JSON receipt; read the emitted `root` for the exact immutable
C7 panel ID. Do not infer it from a filename example. The phase prints
progress to stderr. Parent C6 arrays are memmapped, L2 persons and C6
identities are streamed in lockstep, Gate-B links are replayed using exactly
its original selector in 2,000-link batches with SQLite-indexed lookup.
Before the later-income filter, counts, gap support, exclusive exclusion
reasons, period×region support and the full 3×3 transition matrix must
reconcile with Gate B. Source/receipt and C6 hashes/identity are rechecked.
A compact typed array and private row-pair index is then finalized.

Expected on the *frozen real parent* before income restriction:

- audited candidate links: 1,172,374;
- Gate-B eligible 1Q: 571,984;
- Gate-B eligible 3Q: 255,809;
- Gate-B total eligible: 827,793; excluded: 344,581;
- all distinct eligible target observations, no duplicate target.

Actual C7 welfare-row count must equal Gate-B eligible count **minus
additional invalid later P47T_real counts**, which are discovered locally,
not guessed from descriptive group summaries.

Run one restartable outer fold first (optional memory canary):

    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c7-run \
      configs/longitudinal/c7_real_p1r.yaml \
      --panel-root PATH_FROM_C7_PLANE_ROOT \
      --output-root /home/matias/data/l4-c7-20261001/runs \
      --outer-fold 0

Then complete missing folds and finalize the immutable matched run:

    PYTHONPATH=src python3 -u -m encuestador.cli longitudinal-c7-run \
      configs/longitudinal/c7_real_p1r.yaml \
      --panel-root <EMITTED_C7_PANEL_DIRECTORY> \
      --output-root /home/matias/data/l4-c7-20261001/runs

Outer-fold checkpoints are source+config+fold-bound and contain both
C7-0 and C7-1 predictions. A rerun reuses verified completed checkpoints.
The run's final receipt includes full names of the baseline/L11 inputs,
matched fold IDs, same target, paired foldwise MAE deltas, elapsed-gap and
earlier E/U/I strata, presence, positive-amount and resource evidence.
Selected-panel-household group sums are explicitly NOT labeled full
household welfare.

## Local acceptance criteria

1. Verify the real R0 source lineage and Gate-B baseline first; do not
   let the cloud fixture counts substitute for real-parent truth.
2. C7 plane materializes and all source/selection/identity/target checks pass.
3. Fold-0 canary stays within available RAM, shows no swap thrashing and
   checkpoint restart is idempotent. C7 records its peak RSS; local agent
   must verify against real host free memory and the C6 10 GiB envelope.
4. All five grouped folds finish with finite component predictions on the
   same row/target/fold vector; no later individual labor enters either
   terminal model. Earlier composition and later aggregate context are
   verified on actual sampled row IDs.
5. Produce `C7_REAL_ACCEPTANCE.json` and `C7_SCIENTIFIC_FINDINGS.md` with
   actual paired overall/fold/1Q/3Q/prior-state results and candid monetary
   candidate-parent limitation. No automatic scientific promotion.
6. If source, fold, or memory constraints fail, stop and return a bounded
   repair receipt; never force materialize a million-row Python dict and
   never silently expand to Census/CPV2022/L12.

No CPV-2010→2024/2026 long-gap extrapolation is authorized by the run.
