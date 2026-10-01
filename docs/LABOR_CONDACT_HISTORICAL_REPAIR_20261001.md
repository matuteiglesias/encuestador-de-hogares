# Historical CONDACT evidence: R0 interpretation repair (2026-10-01)

Status: method audit and scope correction, NOT a rerun, repair of historical
science files, or new L11/L12 outcome. Preserve dated artifacts unchanged.

## Three different questions were previously conflated

1. **Same-clock labor oracle**: does actually observed target-quarter labor
   add income signal on the exact EPH quarter?
2. **Reconstruction**: can current labor be inferred using the remaining
   shared contemporaneous characteristics?
3. **Donor-clock transport**: does actual Census labor observed in donor
   year 2010 or 2022 help estimate income at target year 2024/2026?

Question 1 is not question 3. Nor does poor reconstruction in question 2
show that observed Census donor CONDACT is unavailable or uninformative.

## What the committed September 11 scripts actually did

- Q2 P0 versus P1-R: the full 21-field P1-R includes CONDACT. See
  `science/2026-09-11/experiment_matrix.json` and
  `science/2026-09-11/results/q2_p1r/preflight.json`.
- Q3 leave-one-family-out ablation: labor family consists of CONDACT.
  Positive-amount R2: full 0.270915, minus labor 0.253460.
  Household R2: full 0.352472, minus labor 0.311798;
  household MAE: 416,677.5 versus 438,651.6.
  These are historical within-experiment descriptive OOF contrasts,
  not proof of donor-year causal effects or longitudinal transport.
- Q4 P1-R-minus-CONDACT comparison:
  A without labor household MAE 438,651.6;
  B with true contemporaneous labor 416,677.5;
  C with OOF reconstructed labor probabilities 440,592.9.
  Its negative C result applies to that exact implementation only.
- Q8 Census commissioning explicitly includes CONDACT in F for training
  on contemporary EPH-2024-Q3 and scoring the CPV-2010 donor.
  Census CONDACT was measured in 2010, not 2024. Q8 remains research
  commissioning with material transport caveats, not evidence that a
  correctly vintage-aware L11 or L12 has been validated.
- The later household-grouped OOF D-1 EPH-versus-Census source classifier
  explicitly sets `DEFAULT_EXCLUDE_FIELDS=("CONDACT",)`. Its AUC
  S=0.514802, S+T=0.706180, S+T+R=0.811938 are therefore *without*
  CONDACT and answer a source-discrimination question, not welfare error.
  A later matched age-14+ with/without-CONDACT D-1 sensitivity is optional,
  not a prerequisite for C7.

## Q4 confounding identified from the actual script

Read `science/2026-09-11/run_q4_labor_oracle.py`. In `fitpred`, when
`labor_probs is not None`, it appends the probability columns to the
feature matrix AND resets `cats=[]`. That changes HGB's interpretation of
all original categorical composition features in C, rather than changing
only the labor information. The global five-fold OOF transition predictions
are furthermore prepared before the downstream welfare outer-fold loop.
For a downstream outer training row, its OOF transition model can have
been trained on that outer test population. Thus the intermediate-stage
fit is not fully nested within the downstream outer-train partition.
This is a comparability/leakage-design concern; it is not evidence of a
numerical bug in the old persisted files.

Preserve historical Q4, do not rewrite past outcome metrics. C8 must keep
baseline categorical declarations unchanged when appending ordered labor
probabilities and make transition training probabilities nested inside
the outer welfare training population, with true outer test held out.

## C7 versus C8 and the Census clock

- C7 directly conditions on observed **earlier** EPH labor and **earlier**
  canonical composition for 1Q/3Q, using target-quarter official aggregate
  labor and time, against an exactly matched no-donor-labor reference.
  It is an actual short-horizon donor-state-value experiment, not another
  contemporaneous CONDACT reconstruction.
- C8 first tests 1Q/3Q transition probabilities and then their welfare
  contribution, using properly nested E/U/I OOF probabilities. Raw
  transition evidence precedes any optional moment/KL anchoring.
- Real CPV-2010 or CPV-2022 income scoring is a separate later transport
  gate. A 2010-to-2024/2026 elapsed donor gap is not identified by EPH
  1Q/3Q transitions. Two separate Census marginal distributions also do
  not identify individual longitudinal state transitions.

Frozen selection and local closure are specified in
`docs/C7_DONOR_CLOCK_MATCHED_L11_CONTRACT.md`.
