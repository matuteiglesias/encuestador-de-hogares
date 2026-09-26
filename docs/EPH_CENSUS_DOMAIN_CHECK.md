# Thin EPH↔Census agglomerate domain check

Status: **diagnostic commissioning only**

This runner is the minimal executable bridge between the governed semantic plane,
the governed EPH quarter, and the governed Census→EPH agglomerate handoff.

It intentionally stops before the larger joint-distribution microscope tracked in
issue #28.

## Inputs

```text
semantic plane directory
├── feature_plane_manifest.json
├── eph_p1.parquet
└── census_p1.parquet

governed raw EPH individual quarter
└── CODUSU, NRO_HOGAR, COMPONENTE, ANO4, TRIMESTRE, AGLOMERADO, PONDERA

governed G2 Census geography handoff
├── manifest.json
└── household_geography.parquet
```

The runner performs exact identity joins only:

```text
EPH semantic row_id
  = CODUSU:NRO_HOGAR:COMPONENTE

Census semantic household_id
  = G2 sample_household_id
```

No row-order, fuzzy, nearest-neighbour or spatial join is used.

## Feature tiers

The runner derives feature tiers from the semantic-plane manifest rather than
maintaining a second feature-role registry:

```text
S       = stable/shared
S+T     = stable/shared + target-period-state
S+T+R   = stable/shared + target-period-state + research-only
```

`CONDACT` is excluded by default because it is a current labor target. Consumers
may pass a different explicit exclusion set.

## Two lenses

### Model-support lens

```text
EPH    unit rows
Census unit rows
```

This asks whether Census scoring rows occupy support similar to the rows the model
actually sees.

### Population-composition lens

```text
EPH    PONDERA
Census unit selected persons
```

The Census target-year selected sample is the synthetic population surface.
`design_inverse_probability_weight` is never used as an analysis weight.

## Domain classifier

For every native EPH agglomerate with sufficient rows on both sources, plus
`EPH_TOTAL`, the runner fits one source classifier per feature tier.

Cross-fitting uses source-prefixed household groups:

```text
EPH:<household_id>
CENSUS:<household_id>
```

The fit and evaluation weights balance EPH vs Census source mass only. They do
not reweight demographic composition.

Reported metrics are OOF:

- ROC AUC;
- log loss;
- Brier score;
- mean OOF Census-domain probability by source.

The old September Q8 global in-sample classifier remains historical evidence,
not the primary diagnostic.

## Marginal diagnostics

For every domain, concept and lens:

- categorical concepts: total-variation distance and Census unseen mass;
- numeric concepts: source means, medians, standardized mean difference and
  Census mass outside observed EPH support.

These are deliberately interpretable components, not one aggregate distance.

## Geography boundary

The current G2 handoff is valid only for a Census-2010 donor frame and uses the
governed Census-2010-based EPH radio→agglomerate relation.

The runner records this provenance and sets:

```text
frame_equivalence_assumed = false
```

A future CPV-2022 donor requires its own governed EPH-geography relation before
this same agglomerate diagnostic can be run. The runner fails closed on a 2022
G2 parent rather than silently reusing 2010 geography.

## Outputs

```text
domain_inventory.csv
domain_classifier_oof.csv
marginal_comparison.csv
manifest.json
```

The manifest explicitly states:

```text
status = diagnostic_only
statistical_transport_authorized = false
no_reweighting_or_raking_applied = true
```

## Command

```bash
python science/commissioning/run_eph_census_domain_check.py \
  --semantic-plane /path/to/eph-census-semantic-plane \
  --eph-individual /path/to/usu_individual_t324.txt \
  --census-geography-handoff /path/to/g2-handoff \
  --output /path/to/domain-check
```

Useful optional controls:

```text
--exclude-fields CONDACT
--numeric-fields IX_TOT,P03,H15
--min-source-persons 100
--folds 5
--random-state 42
```

## Non-goals

This runner does not:

- perform IPF/raking;
- estimate density ratios for analysis weighting;
- mutate Census;
- fit labor or welfare models;
- compute pairwise association matrices;
- construct joint-cell cubes;
- decide whether transport is scientifically acceptable.

Those remain separate research questions. Issue #28 tracks the larger
within-agglomerate multivariate microscope if the thin diagnostics justify it.
