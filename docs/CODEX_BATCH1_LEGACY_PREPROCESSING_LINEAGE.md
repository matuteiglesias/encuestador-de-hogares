# Codex work packet — Batch 1: recover the legacy EPH preprocessing lineage

## Mission

Use this repository as historical evidence to reconstruct the former EPH preprocessing authority and map it to the current preprocessing authority in `income-modeling-eph`.

The governing decision is:

- this repository was the former preprocessing authority;
- `income-modeling-eph` is now the current preprocessing authority;
- legacy artifacts named `EPHARG_train*` are predecessors of the current `EPHARG_annual_input_*` artifacts;
- this repository should not be revived as a competing producer unless a future named consumer requires it.

This is a lineage and equivalence packet, not a modernization campaign.

## Why this matters

Important preprocessing knowledge may still be encoded here: household/person merging, schema harmonization, regional assignment, monetary adjustment, geographic ranks, sample indicators, and historical EPH schema patches. Batch 2 needs that knowledge transferred or explicitly retired so the current model can depend on a defensible annual-input release.

## Read first

1. Read all applicable `AGENTS.md` files and lifecycle documentation.
2. Read `README.md`, the routines that create `EPHARG_train*`, preprocessing helpers, configuration, data/model path logic, cron or workflow remnants, and relevant notebooks.
3. Inspect commit history around processed training data, rank generation, model updates, and EPH schema changes, including the 2025 `V2_M`/`V5_M` patch.
4. Inspect `income-modeling-eph` read-only, especially its annual preprocessed inputs, preprocessing code, feature contract, dataset builder, and manifests if available.
5. Do not load or copy the full repository's large artifacts unless necessary. Begin with metadata, schemas, sampled rows, hashes, and code inspection.

## Authority and boundaries

This repository is authoritative only for:

- its historical implementation and outputs;
- evidence about how legacy `EPHARG_train*` artifacts were produced;
- historical staged models and transformations.

It is not the current authority for:

- annual EPH preprocessing releases;
- model experiments used by `income-modeling-eph`;
- current price or geographic products;
- poverty outputs;
- unattended retraining.

## Required deliverables

### 1. Legacy producer inventory

Create `docs/LEGACY_PREPROCESSING_INVENTORY.md` documenting:

- every routine that creates or modifies `EPHARG_train*`;
- inputs, outputs, parameters, defaults, and write locations;
- household/person merge logic and keys;
- schema harmonization across years;
- region and agglomerate assignment;
- price deflation/monetary reference;
- geographic rank generation;
- sample/training indicators;
- exclusions, missingness, and duplicate treatment;
- generated models and downstream uses;
- scheduled or automated behavior;
- dependencies on `microdatos-EPH-INDEC`, `IPC-Argentina`, `empleoARG`, Census data, or local sibling paths.

For each statement, cite the code path or artifact evidence inside the document.

### 2. Artifact inventory

Create a machine-readable inventory of legacy artifacts containing, where available:

- path and filename;
- artifact family;
- period/year coverage;
- file size and hash;
- row/column counts;
- schema hash;
- producer script/config;
- commit evidence;
- status: source, intermediate, annual training/preprocessed input, model, rank, report, or unknown;
- whether tracked, externally stored, missing, or too large to inspect.

Do not add large artifacts to Git.

### 3. Legacy-to-current mapping

Create `docs/LEGACY_TO_INCOME_MODELING_MAP.md` mapping old components to current equivalents in `income-modeling-eph`.

At minimum cover:

- `EPHARG_train*` → `EPHARG_annual_input_*`;
- old dataset-building routines → current preprocessing entry points;
- old geographic ranks → current rank columns/process;
- old deflation logic → current monetary normalization;
- old schema patches → current harmonization rules;
- old staged models → current experiment/model system or intentionally retired history.

Use this classification:

```text
migrated and current
migrated with intentional change
semantically equivalent after normalization
candidate for extraction
historical regression oracle
obsolete or superseded
unresolved
```

### 4. Bounded equivalence analysis

Where legacy and current artifacts are both accessible, compare bounded samples and metadata. Record:

- row and column counts;
- key coverage and uniqueness;
- common and renamed columns;
- types and category domains;
- missingness;
- value comparisons for stable keys;
- monetary reference;
- sample membership;
- geography/rank differences.

Classify each pair as:

```text
byte-identical
value-equivalent after deterministic normalization
semantically equivalent with documented changes
not equivalent
insufficient evidence
```

Never claim equivalence from names alone.

### 5. Column lineage extraction

Produce a machine-readable legacy column-lineage table containing:

- legacy column name;
- source variable(s);
- producer routine/function;
- transformation;
- entity level;
- unit/reference period;
- first/last observed vintage;
- current name in `income-modeling-eph`;
- migration status;
- evidence path;
- unresolved semantic questions.

This table is evidence for the current authority; it does not become a new public contract here.

### 6. Code disposition report

Create `docs/LEGACY_CODE_DISPOSITION.md` classifying major code paths as:

- already migrated;
- worth extracting into the current authority;
- retain as historical regression oracle;
- retain only for methodological notes;
- obsolete/superseded;
- unsafe or irreproducible;
- unresolved.

For extraction candidates, name the exact behavior and tests to move. Do not copy code across repositories in this packet.

### 7. Current-authority handoff

Create `docs/PREPROCESSING_AUTHORITY_HANDOFF.md` with a concise handoff to `income-modeling-eph`:

- confirmed legacy producer behavior;
- artifact equivalence results;
- lineage evidence available;
- unique transformations not yet present there;
- decisions requiring Matías;
- regression cases that the current authority should preserve;
- legacy paths and schedules that should remain disabled.

### 8. Documentation correction

Update the README or lifecycle note only as needed to state clearly:

- this is a historical/conditional-revival system;
- current preprocessing authority lives in `income-modeling-eph`;
- existing models/data are versioned legacy evidence, not automatically current;
- users should not resume cron or automated model commits from old instructions.

Do not remove historical methodology from the README merely to shorten it.

## Ordered execution

1. Inventory code, outputs, data paths, and automation.
2. Reconstruct the legacy producer graph.
3. Inventory artifacts without copying large data.
4. Inspect the current authority read-only.
5. Build component and column mappings.
6. Run bounded equivalence comparisons when evidence exists.
7. Classify code disposition.
8. Produce the current-authority handoff.
9. Correct only the misleading current-use documentation.

## Human checkpoints

Stop for review before:

- interpreting an unexplained value difference as intentional;
- deciding that a legacy transformation should be migrated;
- changing or deleting tracked models/data;
- running full preprocessing or model training;
- changing cron, workflow, or automatic commit behavior;
- rewriting Git history;
- declaring legacy artifacts current.

## Non-goals

- No new preprocessing package in this repository.
- No resumed scheduled training.
- No model retraining.
- No bulk data or model commits.
- No refactor of all notebooks and scripts.
- No scientific redesign.
- No edits to `income-modeling-eph` from this branch.
- No deletion of historical evidence without explicit approval.

## Stop conditions

Stop rather than guess when:

- a producer script cannot be tied to an artifact;
- absolute/local paths hide an unavailable dependency;
- legacy/current samples cannot be joined safely;
- monetary references differ without documentation;
- a schema patch changes variable meaning;
- the only way to compare artifacts requires an unbounded or destructive run.

## Acceptance criteria

```text
legacy producer routines and dependencies are inventoried
EPHARG_train artifacts are mapped to current annual-input artifacts with evidence
column-level legacy lineage is machine-readable
bounded equivalence classifications exist for accessible artifact pairs
unique still-relevant behavior is identified for the current authority
legacy code is dispositioned rather than broadly modernized
documentation points users to income-modeling-eph for current preprocessing
no models, large data, schedules, or automatic commits are regenerated
```

## Completion report

The final response and PR description must state:

- code and artifact surfaces inspected;
- legacy/current pairs compared;
- equivalence classifications and material differences;
- lineage coverage;
- extraction candidates and regression oracles;
- unresolved methodological questions;
- documentation corrected;
- confirmation that no full preprocessing, retraining, scheduler, model commit, or large-data mutation occurred.
