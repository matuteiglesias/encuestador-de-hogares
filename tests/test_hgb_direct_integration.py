from __future__ import annotations

import json

import numpy as np

from encuestador.crossfit import crossfit_predict, fit_full_and_score
from encuestador.estimators import HGBRegressorAdapter
from encuestador.scientific_primitives import (
    aggregate_person_predictions,
    build_fold_manifest,
    person_id,
    regression_metrics,
)
from encuestador.transport_proof import (
    DEFAULT_SEMANTIC_PLANE,
    DEFAULT_SPEC,
    DESIGN_FIELDS,
    make_synthetic_people,
    validate_semantic_plane,
    validate_spec,
)


def _matrix(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )


def _hgb() -> HGBRegressorAdapter:
    return HGBRegressorAdapter(
        loss="squared_error",
        parameters={
            "max_iter": 30,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 8,
            "learning_rate": 0.08,
        },
    )


def test_direct_hgb_runs_end_to_end_on_the_approved_synthetic_feature_plane() -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    plane = json.loads(DEFAULT_SEMANTIC_PLANE.read_text(encoding="utf-8"))
    validate_spec(spec)
    approved = validate_semantic_plane(spec, plane)

    assert not (set(approved) & DESIGN_FIELDS)
    assert "AGLO_rk" not in approved
    assert "Reg_rk" not in approved

    eph_rows = make_synthetic_people(
        approved,
        household_count=40,
        codusu_prefix="EPH-HGB",
        include_census_design=False,
    )
    census_rows = make_synthetic_people(
        approved,
        household_count=16,
        codusu_prefix="CPV-HGB",
        include_census_design=True,
    )

    manifest = build_fold_manifest(eph_rows, n_splits=5)
    x_eph = _matrix(eph_rows, approved)
    y_eph = _matrix(eph_rows, ["P47T"])[:, 0]
    x_census = _matrix(census_rows, approved)

    eph_oof = crossfit_predict(
        x_eph,
        y_eph,
        manifest,
        _hgb,
        target="P47T",
        kind="regression",
        run_id="synthetic-direct-hgb",
        metadata={
            "architecture": "direct",
            "estimator": "hist_gradient_boosting",
            "weight_policy": "none",
        },
    )
    census_score = fit_full_and_score(
        x_eph,
        y_eph,
        x_census,
        tuple(person_id(row) for row in census_rows),
        _hgb,
        target="P47T",
        kind="regression",
        run_id="synthetic-direct-hgb",
        metadata={
            "architecture": "direct",
            "estimator": "hist_gradient_boosting",
            "weight_policy": "none",
        },
    )

    eph_oof.validate_alignment(eph_rows)
    census_score.validate_alignment(census_rows)
    assert eph_oof.fold_ids == manifest.fold_ids
    assert census_score.fold_ids is None
    assert np.isfinite(eph_oof.values).all()
    assert np.isfinite(census_score.values).all()

    eph_person_metrics = regression_metrics(y_eph, eph_oof.values)
    assert eph_person_metrics["mae"] >= 0
    assert eph_person_metrics["rmse"] >= 0

    eph_households = aggregate_person_predictions(
        eph_rows,
        {"direct": eph_oof},
        truth_field="P47T",
    )
    census_households = aggregate_person_predictions(
        census_rows,
        {"direct": census_score},
        truth_field="P47T",
    )

    assert all(row["status"] == "complete" for row in eph_households)
    assert all(row["status"] == "complete" for row in census_households)
    assert sum(row["member_count"] for row in eph_households) == len(eph_rows)
    assert sum(row["member_count"] for row in census_households) == len(census_rows)

    eph_household_truth = np.asarray(
        [row["truth_household_income"] for row in eph_households],
        dtype=float,
    )
    eph_household_prediction = np.asarray(
        [row["direct_household_income"] for row in eph_households],
        dtype=float,
    )
    household_metrics = regression_metrics(
        eph_household_truth,
        eph_household_prediction,
    )
    assert household_metrics["mae"] >= 0
    assert household_metrics["rmse"] >= 0

    assert eph_oof.metadata == {
        "architecture": "direct",
        "estimator": "hist_gradient_boosting",
        "weight_policy": "none",
    }
