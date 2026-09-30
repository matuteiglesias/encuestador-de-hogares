# B0 — bounded Telescope-B baseline test repair

Status: small cloud maintenance packet, 2026-09-29.

## Problem

The repository-wide pytest suite currently has one baseline failure:

```text
tests/test_telescope_b_nested_residuals.py::
test_canonical_fold_policy_matches_repository_contract
```

The fixture contains only two household groups while the canonical policy requires all five outer folds to be populated.

The failing test file and `science/telescope_b/prepare_nested_residuals.py` were verified byte-identical between `main` and C4 PR #33. This is not a C4 regression.

## Task

Repair the smallest incorrect baseline fixture/expectation so the test actually exercises the canonical five-fold contract.

Preferred approach:

- expand the synthetic fixture to enough deterministic household groups that the canonical hash policy populates all five folds;
- preserve the production fold policy;
- preserve the invariant that all five folds are nonempty;
- avoid weakening production validation merely to satisfy a tiny fixture.

If deterministic group IDs need to be selected, generate them transparently in the test rather than hardcoding undocumented production exceptions.

## Verification

Run:

```text
pytest -q tests/test_telescope_b_nested_residuals.py
pytest -q
```

No longitudinal C4 files should be changed.

## Delivery

Use a separate small PR to `main`.

Definition of done: full repository pytest is green on main without changing Telescope-B scientific semantics.
