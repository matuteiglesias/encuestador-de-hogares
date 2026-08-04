# Preprocessing authority handoff

## Authority decision

`income-modeling-eph` owns current annual EPH preprocessing and `EPHARG_annual_input_*`. This repository preserves the former `EPHARG_train*` producer and regression evidence only. Nothing here should publish, schedule, train, or commit current releases.

## Confirmed legacy behavior

The producer inner-joined deduplicated household/person records on household-period-agglomerate keys, census-aligned selected fields, attached a manually disambiguated region, appended a deterministic 5% replacement sample as agglomerate zero, normalized nine monetary fields to the January-2016 IPC reference, excluded several invalid/sentinel rows, derived four income-presence targets and two income ranks, and emitted annual CSVs. The July-2025 patch reconstructs missing aggregate `V2_M`/`V5_M` from three components. Full evidence is in [the producer inventory](LEGACY_PREPROCESSING_INVENTORY.md) and [column lineage](legacy_column_lineage.csv).

## Results and limitations

Four tracked annual artifacts and rank/model families are hash-inventoried in [the artifact inventory](legacy_artifact_inventory.csv). Metadata establishes a legacy schema break (component columns appear in 2024–2025) and a partial 2025 file, but no current repository or artifact was accessible. Every legacy-to-current artifact equivalence is therefore **insufficient evidence**, not assumed equivalence. The attempted read-only GitHub access failed with HTTP tunnel 403.

## Current-authority actions

- Compare manifest-pinned annual artifacts using the bounded protocol in [the mapping](LEGACY_TO_INCOME_MODELING_MAP.md).
- Preserve regression fixtures for recodes, merge cardinality, monetary normalization, ranks, sampling, and the 2025 patch.
- Decide whether to retain synthetic agglomerate-zero rows; if yes, add provenance/sample indicators and a stable person key.
- Pin IPC and microdata versions rather than mutable raw URLs.
- Keep raw EPH component columns only if the feature contract explicitly requires them.

## Decisions requiring Matías

1. Is the 5% pooled urban sample intentional in annual inputs, and is replacement sampling acceptable?
2. When only some `V2_*`/`V5_*` components exist, should the aggregate be missing, partial sum, or zero? Legacy uses zero.
3. Are ranks intended within each year? The legacy group result is per-year, but `rank(pct=True)` is applied across all grouped rows in a multi-year frame.
4. Should missing values ever be globally converted to zero, or resolved per variable?
5. Which pinned IPC release defines monetary equivalence, and is January 2016 still the desired reference?
6. What is the stable current person key for comparison, since legacy selected no person sequence field?

## Paths and schedules that remain disabled

Do not run `preprocesar_datos.py`, `entrenar_modelos.py`, `recalcular_rankings.py`, notebook producers, or any `.github/workflows/*` job. Do not restore the old `codigo/routines` path, cron language, write-back commits, or chained retraining. A named future consumer must first pass `LIFECYCLE.md`'s revival gate.
