# Gate B — bounded descriptive EPH panel evidence

Status: cloud B0 fixture implementation; real L2 commissioning is local B1.

## Purpose

This is the small descriptive gate between completed L10 Gate A and later L11/L12
scientific commissioning. It does **not** measure incremental OOF welfare value or
train any new transition/income model.

Authority: immutable L2 research.eph-longitudinal-analysis-frame/v1.
Use panel_links.csv as the existing candidate-link audit. Do not reconstruct
identity from raw CODUSU/COMPONENTE or assert permanent person identity.

## Real local invocation

    PYTHONPATH=src python3 -m encuestador.cli longitudinal-gate-b \
      --eph-release-root \
      /home/matias/data/eph-longitudinal-2017-2026/releases/eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39 \
      --output-root /home/matias/data/l4-gate-b-20261001

Optional bounded exceptional-period robustness produces a separate immutable output:

    PYTHONPATH=src python3 -m encuestador.cli longitudinal-gate-b \
      --eph-release-root <SAME_L2_RELEASE> \
      --output-root /home/matias/data/l4-gate-b-20261001 \
      --exclude-exceptional

The optional variant removes pairs touching 2020-Q2 or 2024-Q1/Q2 and never
replaces the observed baseline.

## Selection and income policy

One exclusive first exclusion reason per link. Eligible links must be:
L2-demographically consistent; strictly earlier to later with gap 1 or 3;
expected 2-2-2 rotation status; both exact row IDs present uniquely; matching
candidate/household/source periods; source CH06 age 14+ at both waves; and
reviewed three-class source labor at both waves.

Labor reuses the existing C4 _observed_labor_state rule exactly:

- 1 employed, 2 unemployed, 3 inactive;
- 0/4 special only; never a fourth class;
- conflicting valid ESTADO/CONDACT rejected, never resolved by precedence;
- all excluded links counted by an exclusive first reason.

P47T_real uses its own value status (zero/positive) and L2 common monetary reference.
Negative source values and missing/invalid incomes are not zeros. An otherwise
valid labor pair remains in the transition matrix when its income is unusable.

The primary output is unweighted descriptive **candidate-pair** evidence, not an
independent unique-person estimate or a survey-population longitudinal transition
estimator. No undeclared panel/survey weighting is introduced.

## Artifact contract

Every immutable gate-b-panel-evidence-<digest> release contains:

- gate_b_receipt.json — exact source and output hashes, exhaustive exclusions,
  1Q/3Q counts, distinct candidates and later observations, monetary caveat;
- panel_support.csv — gap totals and gap x earlier/later period x later region;
- transition_matrix.csv — all 18 cells with counts, prior-class denominators
  and row-conditional probabilities (blank if denominator is zero);
- income_by_prior_state.csv — six gap x earlier-class strata with valid later
  incomes, positive incidence, unconditional/positive means, medians, and
  paired-income change with its own denominator;
- GATE_B_NOTE.md — bounded descriptive interpretation and limitations.

Temporary indexed SQLite person and income projections are removed before
finalization. No public row-level pair file is emitted.

## Local scientific handoff

Inspect support, both transition matrices and income stratification. Choose
supported horizons/cohorts for a later matched panel L10-versus-L11 experiment;
never compare panel L11 error with full-population L10 MAE.

L12 requires a separate OOF transition-probability experiment. Observed 1Q/3Q
EPH persistence does not identify CPV-2010 to 2024/2026 state transitions.
The monetary conversion parent remains candidate. No causal effect, Census
scoring, forecast/nowcast or approved-mode model promotion is claimed.
