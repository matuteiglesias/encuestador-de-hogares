#!/usr/bin/env python3
"""Materialize quarter-specific nested outer-fold residual ECDFs for Telescope B.

This is a parameterized extraction of the already-commissioned Q7 P1-R logic.
It does not change the model, folds, weights, welfare definition, or residual
semantics. It only turns the previously Q3-hard-coded nested calibration step
into a reusable local producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


FEATURES = [
    "IX_TOT", "P02", "P03", "P05", "P07", "P08", "P09", "P10", "CONDACT",
    "V01", "H05", "H06", "H07", "H08", "H09", "H10", "H12", "H13", "H14",
    "H15", "PROP",
]
NUMERIC_FEATURES = ("IX_TOT", "P03", "H15")
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}


class NestedResidualError(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(frame: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise NestedResidualError(f"{label} missing columns: {missing}")


def period_parts(period: str) -> tuple[int, int]:
    try:
        year, quarter = period.upper().split("-Q")
        year, quarter = int(year), int(quarter)
    except (AttributeError, ValueError) as exc:
        raise NestedResidualError("period must be YYYY-Q1..Q4") from exc
    if quarter not in (1, 2, 3, 4):
        raise NestedResidualError("period must be YYYY-Q1..Q4")
    return year, quarter


def categorical_indices() -> list[int]:
    return [FEATURES.index(name) for name in FEATURES if name not in NUMERIC_FEATURES]


def fit_hurdle_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if train.empty or test.empty:
        raise NestedResidualError("nested train/test must be nonempty")
    require(train, [*FEATURES, "y"], "nested training frame")
    require(test, FEATURES, "nested validation frame")

    x = train[FEATURES].to_numpy(float)
    xt = test[FEATURES].to_numpy(float)
    y = train.y.to_numpy(float)
    if np.isinf(x).any() or np.isinf(xt).any():
        raise NestedResidualError("P1 feature matrix contains infinity")
    if not np.isfinite(y).all() or (y < 0).any():
        raise NestedResidualError("P47T target must be finite and nonnegative")

    cats = categorical_indices()
    clf = HistGradientBoostingClassifier(
        **MODEL_PARAMS, categorical_features=cats
    ).fit(x, y > 0)
    positive = y > 0
    if not positive.any():
        raise NestedResidualError("nested training split has no positive incomes")
    reg = HistGradientBoostingRegressor(
        loss="gamma", **MODEL_PARAMS, categorical_features=cats
    ).fit(x[positive], y[positive])
    positive_class = list(clf.classes_).index(True)
    out = clf.predict_proba(xt)[:, positive_class] * reg.predict(xt)
    if not np.isfinite(out).all() or (out < 0).any():
        raise NestedResidualError("nested model produced invalid predictions")
    return out


def assign_inner_folds(households: pd.Series, folds: int = 5) -> pd.Series:
    if folds < 2:
        raise NestedResidualError("inner folds must be >=2")
    unique = sorted(set(households.astype(str)))
    mapping = {household_id: i % folds for i, household_id in enumerate(unique)}
    return households.astype(str).map(mapping).astype(int)


def load_inputs(
    eph_persons_path: Path,
    eph_p1_path: Path,
    fold_manifest_path: Path,
    telescope_a_households_path: Path,
    *,
    period: str,
) -> tuple[pd.DataFrame, set[str], dict]:
    year, quarter = period_parts(period)

    raw = pd.read_csv(
        eph_persons_path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        usecols=["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "P47T"],
        low_memory=False,
    )
    raw["row_id"] = raw.CODUSU + ":" + raw.NRO_HOGAR + ":" + raw.COMPONENTE
    raw["hh"] = raw.CODUSU + "\x1f" + raw.NRO_HOGAR
    raw["full_household_id"] = (
        raw.ANO4 + ":" + raw.TRIMESTRE + ":" + raw.CODUSU + ":" + raw.NRO_HOGAR
    )
    raw["fold_row"] = (
        raw.CODUSU + "\x1f" + raw.NRO_HOGAR + "\x1f" + raw.COMPONENTE + "\x1f"
        + raw.ANO4 + "\x1f" + raw.TRIMESTRE
    )
    y = pd.to_numeric(raw.P47T, errors="coerce")
    raw["y"] = y

    observed_periods = set(
        zip(
            pd.to_numeric(raw.ANO4, errors="coerce").dropna().astype(int),
            pd.to_numeric(raw.TRIMESTRE, errors="coerce").dropna().astype(int),
        )
    )
    if observed_periods != {(year, quarter)}:
        raise NestedResidualError(
            f"EPH person period mismatch: {sorted(observed_periods)} != {(year, quarter)}"
        )

    fold_payload = json.loads(fold_manifest_path.read_text())
    rows = fold_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise NestedResidualError("fold manifest must contain nonempty rows")
    fold_map = {str(item["row_id"]): int(item["fold_id"]) for item in rows}
    raw["outer_fold"] = raw.fold_row.map(fold_map)

    eligible = raw[raw.y.notna() & (raw.y >= 0)].copy()
    if eligible.outer_fold.isna().any():
        raise NestedResidualError("eligible person missing governed outer fold")
    eligible["outer_fold"] = eligible.outer_fold.astype(int)
    if sorted(eligible.outer_fold.unique()) != [0, 1, 2, 3, 4]:
        raise NestedResidualError("expected governed outer folds 0..4")
    if eligible.row_id.duplicated().any():
        raise NestedResidualError("duplicate eligible EPH row_id")
    if (eligible.groupby("hh").outer_fold.nunique() != 1).any():
        raise NestedResidualError("household members cross governed outer folds")

    a = pd.read_parquet(telescope_a_households_path)
    require(a, ["household_id", "P47T_complete", "member_count_records"], "Telescope A households")
    if a.household_id.duplicated().any() or a.empty:
        raise NestedResidualError("Telescope A household identity must be unique and nonempty")
    complete = a.P47T_complete
    if complete.dtype != bool:
        complete = complete.astype(str).str.lower().map({"true": True, "false": False})
    if complete.isna().any() or not complete.all():
        raise NestedResidualError("Telescope B requires the P47T-complete Telescope A cohort")

    complete_full_ids = set(a.household_id.astype(str))
    selected = eligible[eligible.full_household_id.isin(complete_full_ids)].copy()
    if set(selected.full_household_id) != complete_full_ids:
        missing = sorted(complete_full_ids - set(selected.full_household_id))
        raise NestedResidualError(
            f"Telescope A cohort not fully represented in eligible EPH persons: {missing[:20]}"
        )
    counts = selected.groupby("full_household_id").size()
    expected_counts = a.set_index("household_id").member_count_records.astype(int)
    if not counts.reindex(expected_counts.index).equals(expected_counts):
        raise NestedResidualError("Telescope A/EPH member-count mismatch")

    p1 = pd.read_parquet(eph_p1_path)
    require(p1, ["row_id", *FEATURES], "EPH P1 semantic plane")
    if p1.row_id.duplicated().any():
        raise NestedResidualError("duplicate EPH P1 row_id")

    transport = eligible[
        ["row_id", "hh", "full_household_id", "outer_fold", "y"]
    ]
    model = p1.merge(transport, on="row_id", how="inner", validate="one_to_one")
    if len(model) != len(eligible):
        raise NestedResidualError(
            f"eligible EPH/P1 identity mismatch: eligible={len(eligible)} joined={len(model)}"
        )
    if np.isinf(model[FEATURES].to_numpy(float)).any():
        raise NestedResidualError("EPH P1 contains infinite feature values")

    complete_hh = set(
        selected[["full_household_id", "hh"]]
        .drop_duplicates()
        .set_index("full_household_id")
        .loc[sorted(complete_full_ids), "hh"]
        .astype(str)
    )
    return model, complete_hh, {
        "eligible_persons": int(len(eligible)),
        "telescope_a_households": int(len(complete_hh)),
        "telescope_a_persons": int(len(selected)),
        "outer_fold_person_counts": {
            str(k): int(v)
            for k, v in eligible.groupby("outer_fold").size().sort_index().items()
        },
    }


def household_residuals(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    complete_households: set[str],
) -> pd.DataFrame:
    z = validation[["hh", "y"]].copy()
    z["pred"] = predictions
    z = z[z.hh.astype(str).isin(complete_households)]
    grouped = z.groupby("hh", sort=True).agg(
        observed=("y", "sum"),
        point=("pred", "sum"),
        persons=("y", "size"),
    )
    grouped["residual"] = grouped.observed - grouped.point
    return grouped.reset_index()


def nested_outer_residuals(
    model: pd.DataFrame,
    complete_households: set[str],
    *,
    inner_folds: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    output = []
    diagnostics = []
    for outer_fold in range(5):
        train = model[model.outer_fold != outer_fold].copy()
        train["inner_fold"] = assign_inner_folds(train.hh, inner_folds)
        fold_parts = []
        for inner_fold in range(inner_folds):
            fit = train[train.inner_fold != inner_fold]
            validation = train[train.inner_fold == inner_fold]
            pred = fit_hurdle_predict(fit, validation)
            residuals = household_residuals(
                validation, pred, complete_households
            )
            residuals["inner_fold"] = inner_fold
            fold_parts.append(residuals)

        fold_frame = pd.concat(fold_parts, ignore_index=True)
        if fold_frame.hh.duplicated().any():
            raise NestedResidualError(
                f"outer fold {outer_fold} nested calibration duplicates household"
            )
        fold_frame["outer_fold"] = outer_fold
        output.append(fold_frame[["outer_fold", "residual"]])
        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "calibration_households": int(len(fold_frame)),
                "residual_sd": float(np.std(fold_frame.residual.to_numpy(float))),
                "residual_mean": float(fold_frame.residual.mean()),
                "residual_p05": float(fold_frame.residual.quantile(0.05)),
                "residual_median": float(fold_frame.residual.median()),
                "residual_p95": float(fold_frame.residual.quantile(0.95)),
            }
        )

    result = pd.concat(output, ignore_index=True)
    if sorted(result.outer_fold.unique()) != [0, 1, 2, 3, 4]:
        raise NestedResidualError("nested residual output missing outer fold")
    if not np.isfinite(result.residual.to_numpy(float)).all():
        raise NestedResidualError("nested residual output contains non-finite values")
    return result, diagnostics


def run(args: argparse.Namespace) -> dict:
    eph_persons = Path(args.eph_persons).resolve()
    eph_p1 = Path(args.eph_p1).resolve()
    fold_manifest = Path(args.fold_manifest).resolve()
    a_households = Path(args.telescope_a_households).resolve()

    model, complete_households, input_summary = load_inputs(
        eph_persons,
        eph_p1,
        fold_manifest,
        a_households,
        period=args.period,
    )
    residuals, folds = nested_outer_residuals(
        model,
        complete_households,
        inner_folds=args.inner_folds,
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    residual_path = output / "fold_residuals.parquet"
    residuals.to_parquet(residual_path, index=False)

    manifest = {
        "status": "RESEARCH_NESTED_RESIDUAL_EVIDENCE_NOT_OFFICIAL_STATISTICS",
        "period": args.period,
        "method": "Q7 nested household-safe P1-R residual ECDF",
        "features": FEATURES,
        "numeric_features": list(NUMERIC_FEATURES),
        "model_params": MODEL_PARAMS,
        "outer_folds": [0, 1, 2, 3, 4],
        "inner_folds": args.inner_folds,
        "weights": "none",
        "input_summary": input_summary,
        "folds": folds,
        "parents": {
            "eph_persons_sha256": sha256(eph_persons),
            "eph_p1_sha256": sha256(eph_p1),
            "fold_manifest_sha256": sha256(fold_manifest),
            "telescope_a_households_sha256": sha256(a_households),
        },
        "files": {
            "fold_residuals.parquet": {
                "sha256": sha256(residual_path),
                "bytes": residual_path.stat().st_size,
                "rows": int(len(residuals)),
            }
        },
        "scientific_invariants": [
            "same frozen P1-R features and hyperparameters as Q7",
            "outer-training persons only for each outer fold",
            "five deterministic household-safe inner folds",
            "household residual = observed sum(P47T) - nested inner-OOF point sum",
            "no survey weights used in residual calibration",
            "global residual ECDF is not substituted",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--period", required=True)
    p.add_argument("--eph-persons", required=True)
    p.add_argument("--eph-p1", required=True)
    p.add_argument("--fold-manifest", required=True)
    p.add_argument("--telescope-a-households", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inner-folds", type=int, default=5)
    return p


def main() -> None:
    args = parser().parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "period": manifest["period"],
                "rows": manifest["files"]["fold_residuals.parquet"]["rows"],
                "folds": manifest["folds"],
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
