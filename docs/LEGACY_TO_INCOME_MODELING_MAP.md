# Legacy → `income-modeling-eph` map

## Evidence boundary

The target repository was not present under `/workspace`, this repository has no remote configured, and a read-only `git ls-remote https://github.com/matuteiglesias/income-modeling-eph.git HEAD` was blocked by the environment's HTTP tunnel (403). Consequently, only the governing names and authority decision in the work packet can be confirmed here. No equivalence below is inferred from names alone; current implementation details remain **unresolved** pending inspection in the current authority.

| Legacy component | Intended current counterpart | Migration classification | Equivalence classification | Evidence / required next comparison |
|---|---|---|---|---|
| `EPHARG_train_YY.csv` | `EPHARG_annual_input_*` | unresolved | insufficient evidence | Legacy producer and four tracked outputs are inspectable; no current artifacts were accessible. Compare manifests, schemas, stable survey keys, synthetic membership, and monetary reference. |
| `preprocesar_datos.py` annual loop | current annual preprocessing entry point | candidate for extraction | insufficient evidence | Transfer behavior as tests/specification, not code: four-quarter ingestion, census recodes, merge, exclusions, and schema patch. |
| `preprocess.build_training_matrix` | current dataset builder | obsolete or superseded | not equivalent | Legacy extraction omits actual deflation, rank attachment, exclusions/recodes, and region merge details; its comments admit a placeholder. |
| `AGLO_rk`, `Reg_rk` columns/files | current rank columns/process | historical regression oracle | insufficient evidence | Compare qualifying population, yearly grouping scope, tie behavior, reference income, rounding, and pooled agglomerate. |
| January-2016 IPC transform | current monetary normalization | historical regression oracle | insufficient evidence | Compare pinned IPC vintage, quarter timestamp convention, formula, rounding, and units. |
| `V2_M`/`V5_M` fallback sum | current harmonization rules | candidate for extraction | insufficient evidence | Verify current raw schema semantics and missing-component policy before migrating. |
| EPH→Census renames/recodes | current feature contract | candidate for extraction | insufficient evidence | See machine-readable lineage; confirm whether current authority keeps EPH names or normalized census names. |
| 5% replacement sample at `AGLOMERADO=0` | current sample membership | unresolved | insufficient evidence | Determine whether current annual input intentionally contains synthetic pooled rows and how they are flagged. Legacy has no flag. |
| `clf1`–`clf4` staged models | current experiment/model system | obsolete or superseded | insufficient evidence | Preserve serialized models/history only; do not migrate producer ownership. |
| scheduled preprocessing/ranking/training | current release automation | obsolete or superseded | not equivalent | Legacy workflows invoke missing paths and write back to `main`; they must remain dormant. |

## Bounded legacy artifact analysis

Metadata-only inspection was bounded to tracked CSV headers, newline counts, hashes, and code; no full preprocessing/training ran. The four annual files have 189,581 (2022), 199,766 (2023), 198,604 (2024), and 47,950 (partial 2025) data rows. Years 2022–2023 share a 56-column header; 2024–2025 have 62 columns because six `V2_*`/`V5_*` components remain in output. Therefore even adjacent legacy vintages are **not byte-identical** and their schemas are **not equivalent without deterministic projection**. This does not establish current equivalence.

A safe current comparison should (1) use manifest-pinned annual pairs, (2) separate genuine and `AGLOMERADO=0` synthetic rows, (3) join genuine rows only after identifying the current person key, (4) compare raw and normalized monetary fields separately, and (5) report differences rather than assigning intent. Stop if current inputs lack a safe key or monetary reference.
