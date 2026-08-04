# Legacy code disposition

| Code path | Disposition | Exact retained value / action |
|---|---|---|
| `src/encuestador/preprocesar_datos.py` | retain as historical regression oracle | Preserve source-to-output behavior. Extract tests/specifications for merge keys, census recodes, 5% seeded pooled rows, quarter date, CPI equation, exclusions, targets, ranks, and the 2025 fallback—not the script itself. |
| `src/encuestador/preprocess.py` | unsafe or irreproducible | Incomplete attempted extraction: external `eph_align` is unavailable and CPI is a placeholder. Do not treat output as equivalent. |
| `src/encuestador/recalcular_rankings.py` | retain only for methodological notes | It republishes hard-coded years from existing columns; test current authority against the producer formula instead. |
| `src/encuestador/preprocess.py::recompute_ranks/export_ranks` | worth extracting into current authority | If absent, move a deterministic test: filter `CAT_OCUP==3, P47T>=100`, mean by year/geography, global percentile rank behavior as actually intended, three-decimal rounding. Resolve whether ranking must be within year. |
| `src/encuestador/entrenar_modelos.py` | obsolete or superseded | Preserve staged feature lists and artifacts as history; unseeded split, one-tree regressors, and ambiguous overwrite parsing are not a current experiment system. |
| `src/encuestador/config.py`, `io.py` | retain only for methodological notes | Generic, unused scaffolding; no evidence they produced tracked annual artifacts. |
| notebooks `01`, `02`, `12` and checkpoints | retain as historical regression oracle | Consult for genealogy and examples, never as current release entry points. |
| `.github/workflows/*` | unsafe or irreproducible | Keep disabled: obsolete paths, old actions, mutable dependencies, automatic commits/pushes, and chained retraining. |
| `data/training/EPHARG_train_*`, `data/info/*rk*` | retain as historical regression oracle | Hash-pinned evidence for bounded tests; never relabel as current releases. |
| `artifacts/models/clf*`, `notebooks/modelos/*` | retain as historical regression oracle | Serialized historical outputs only; do not load untrusted joblib artifacts or resume generation. |

## Tests to transfer, subject to current-authority review

1. Fixture with duplicate household/person rows verifies exact de-duplication and inner merge cardinality.
2. Fixture covering every recode domain and exclusions (`IV1=9`, category 9, negative/absent income).
3. 2025 fixture covers aggregate-present, all-components-present, and incomplete-components cases for `V2_M`/`V5_M`; the last case requires a deliberate policy decision because legacy silently emits zero.
4. Monetary fixture pins IPC inputs and verifies middle-quarter date, January-2016 scaling, and rounding.
5. Rank fixture covers ties, multiple years, regions, agglomerate zero, and the ambiguity between per-year and global percentile ranking.
6. Sample fixture verifies deterministic replacement sampling and demands an explicit synthetic-row indicator if retained.
