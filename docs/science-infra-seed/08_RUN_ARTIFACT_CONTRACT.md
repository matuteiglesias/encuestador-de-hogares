# Run Artifact Contract

## 1. Purpose

A scientific run must be reproducible and interpretable months later without relying on notebook state, memory or implicit code defaults.

The run bundle is evidence. It records what was executed, on which data, with which folds, model semantics, anchors and diagnostics.

It should be possible to answer:

> What exactly generated this welfare estimate or architecture comparison?

from the run artifacts alone plus the pinned source/input releases.

## 2. Run identity

A run ID should be content-bound to consequential inputs.

Conceptually:

\[
run\_id=h(
code\_revision,
resolved\_config,
data\_identities,
feature\_plane,
split\_manifest,
seed
).
\]

The exact hash scheme may evolve, but the run manifest must record every component explicitly.

A timestamp may be included for readability but must not be the sole identity.

## 3. Recommended directory structure

```text
reports/runs/<run_id>/
  manifest.json
  resolved_config.yaml
  acceptance.json
  limitations.md

  inputs/
    data_manifest.json
    feature_manifest.json
    anchor_manifest.json        # when applicable
    monetary_manifest.json      # when applicable

  splits/
    fold_manifest.parquet       # or compact equivalent
    split_summary.json

  predictions/
    person_oof.parquet
    stage_oof/                   # optional separate artifacts
    person_scored.parquet        # only when external scoring performed
    household_oof.parquet

  metrics/
    summary.json
    stages.json
    person.json
    household.json
    comparison.json

  diagnostics/
    calibration/
    distribution/
    cascade/
    anchors/
    uncertainty/

  figures/
    ...
```

Not every run must emit every directory. The manifest must state what exists.

## 4. Resolved config

`resolved_config.yaml` is mandatory.

It must contain the effective values for:

```text
data/cohort
feature plane
architecture
terminal formulation
estimator family and parameters
split policy
calibration policy
anchor policy
uncertainty policy
evaluation profile
random seeds
```

Do not require a future reader to reconstruct defaults from Python source.

## 5. Data manifest

Record source identities without copying large source data into Git.

Suggested fields:

```text
artifact/release id
source repo or provider
path/URI or data-root reference
content hash where available
row count
column count
period coverage
person count
household count
identity columns
cohort filter summary
target missingness/eligibility
```

If the input is a local non-versioned file, record that limitation loudly and compute a content checksum when practical.

## 6. Feature manifest

Persist the final feature list with semantic metadata:

```text
name
dtype role
semantic class
temporal role
node use
external/latent/audit status
missingness summary
category levels or category-hash metadata where useful
```

The manifest should make target leakage and accidental feature drift inspectable.

## 7. Split manifest

Persist enough information to prove the split rather than simply declaring it.

At minimum:

```text
row_id -> fold_id
household/group ID
strategy
n_splits
seed/deterministic method
counts by fold
households by fold
```

Acceptance should verify:

\[
\forall h,\quad |\{fold(i):i\in h\}|=1.
\]

Compared runs should reference the same split-manifest digest when the comparison contract requires identical folds.

## 8. Prediction artifacts

### 8.1 Person OOF

Canonical person OOF artifact should include:

```text
row identity
household identity
truth terminal target where allowed
direct/terminal point prediction
component predictions for hurdle models
support flags
fold id
```

Stage probability artifacts may be separate or namespaced in the same table depending on width.

### 8.2 Categorical stage prediction

For target `Z`, preserve:

```text
row identity
fold id
true Z for evaluation-only EPH artifact
predicted class
p(class_1)
...
p(class_K)
class-order metadata
raw/calibrated status
```

### 8.3 Household OOF

Include:

```text
household ID
member count
truth household income
predicted household income
missing/invalid member count
aggregation status
```

## 9. Metrics

Machine-readable metrics should use stable names and include denominators/sample sizes.

Bad:

```json
{"mae": 0.17}
```

Better:

```json
{
  "metric": "mae",
  "unit": "person",
  "scale": "linear_ars",
  "n": 48621,
  "value": 12345.6
}
```

Metric provenance should indicate whether the value is:

```text
OOF
external scoring with known truth
bootstrap summary
anchor raw/adjusted
oracle diagnostic
```

## 10. Comparison artifact

A comparison should itself be an artifact linking exact runs.

Example fields:

```text
comparison_id
comparison_axis
baseline_run_id
candidate_run_ids
compatibility checks
paired metric differences
cluster bootstrap intervals
scientific interpretation status
```

Do not copy run numbers into a markdown report without a machine-readable comparison record.

## 11. Acceptance artifact

`acceptance.json` should contain separate engineering/science status.

Illustrative structure:

```json
{
  "engineering": {
    "status": "pass",
    "checks": {
      "resolved_config": true,
      "household_leakage": false,
      "oof_complete": true,
      "probability_semantics_valid": true,
      "forbidden_features_absent": true
    }
  },
  "science": {
    "status": "undetermined",
    "comparison_group": "cascade_depth_hgb_gamma_v1",
    "claims_supported": [],
    "limitations": []
  }
}
```

## 12. Limitations

Every consequential run should carry explicit non-claims/limitations.

Examples:

```text
EPH-only qualification; no Census accuracy claim
real Census semantic plane not approved
anchor source is estimated, not exact
Gamma objective is predictive, not a distributional truth claim
household predictive covariance not fully modeled
```

This may be a markdown file for human readability plus structured fields in the manifest.

## 13. Model binaries and storage

Large fitted estimator binaries should not be committed to Git by default.

Preferred patterns:

- local artifact/data root outside Git;
- CI artifact for bounded temporary runs;
- future governed object/artifact storage when promotion requires durable models.

The run manifest may refer to model artifact checksums/paths.

Historical serialized model files already in the repository are not precedent for the active storage design.

## 14. Prediction storage

OOF prediction tables can become large.

Do not commit large repeated prediction matrices to Git merely for reproducibility.

Version-control:

- configs;
- small manifests;
- summary metrics;
- compact diagnostic outputs;
- small fixtures.

Keep large predictions in the configured run/artifact root and record their hashes.

## 15. Figures

Figures are derived artifacts, not scientific authority.

Every important figure should be reproducible from machine-readable run outputs.

Useful initial figures:

```text
calibration/reliability
observed vs predicted quantiles
error by income decile
household observed vs predicted
oracle/direct/cascade risk comparison
external vs raw vs anchored employment time series
anchor lambda/displacement
```

## 16. Reproduction command

The run manifest should provide or support a deterministic command conceptually like:

```text
make reproduce RUN=<run_id>
```

A reproduction attempt must distinguish:

- exact reproduction possible;
- input artifact unavailable;
- environment/version drift;
- stochastic tolerance expected;
- scientific config resolvable but old code unavailable.

For the active sprint, prefer deterministic seeds and pinned package versions sufficient to make repeated runs meaningfully comparable.

## 17. Environment identity

Record at least:

```text
Python version
scikit-learn version
numpy/pandas versions
platform summary
git SHA
dirty working tree status when locally run
```

A dirty-tree run may be permitted for development but cannot silently masquerade as a promoted release.

## 18. Run lifecycle

Suggested run states:

```text
created
running
completed
engineering_failed
engineering_passed
scientifically_promoted
scientifically_not_promoted
superseded
```

These states are metadata; do not mutate scientific outputs in place to change their interpretation.

## 19. Immutability

Once a completed run is assigned an identity, its scientific contents should be treated as immutable.

If code/config/data changes, create a new run ID.

Human annotations may be appended in a separate review record, but should not rewrite the resolved configuration or original metrics.

## 20. Green criterion

Run packaging is green when a fresh real-EPH experiment can be inspected and reproduced without opening a notebook or guessing:

- exact data/cohort;
- exact features;
- exact folds;
- exact architecture;
- exact estimator settings;
- exact anchor/calibration policy;
- person and household outputs;
- scientific diagnostics;
- limitations;
- code/environment identity.
