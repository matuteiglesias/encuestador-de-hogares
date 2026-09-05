# Local rehabilitation safety manifest

Captured before repository replacement on 2026-09-05.

- Baseline checkout: `main` at `e17f4b84332a8f509636085849382dcd130a8169`.
- Canonical remote: `origin` = `git@github.com:matuteiglesias/encuestador-de-hogares.git`.
- Local `main` has no commits absent from `origin/main`; it is nine commits behind the remote-tracking ref.
- No stashes or linked worktrees were present.
- Valuable dirty tracked work was captured in `/tmp/encuestador-dirty-tracked.patch` (excluding intentional deletion of four large training CSVs). SHA-256: `f5c9649b5b7c3c889c1e00a687e5395cbc2390382f6e095881c7332e0b63b984`.
- Untracked source/config work in `notes/` and selected `src/encuestador/` files was captured in `/tmp/encuestador-untracked-source.tgz`. SHA-256: `aac5754752fd0f724fcc8d293da353982b2c58a1d49f6e6611c5027f6fbb2190`.
- Excluded from preservation: `src/encuestador/__pycache__/`, `src/fitted_models_hgbr/`, and the four deleted `data/training/EPHARG_train_2*.csv` files; these are generated/cache/data artifacts.

The patch and archive are temporary safety copies on the same filesystem and must be retained until the compact replacement has been verified and the valuable changes restored.
