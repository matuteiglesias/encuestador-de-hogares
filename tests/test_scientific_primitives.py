from __future__ import annotations

import numpy as np
import pytest

from encuestador.scientific_primitives import (
    PredictionArtifact,
    ScientificPrimitiveError,
    aggregate_person_predictions,
    build_fold_manifest,
    household_id,
    person_id,
    regression_metrics,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 1, "P47T": 100.0},
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 2, "P47T": 200.0},
        {"CODUSU": "B", "NRO_HOGAR": "2", "COMPONENTE": 1, "P47T": 50.0},
        {"CODUSU": "C", "NRO_HOGAR": "3", "COMPONENTE": 1, "P47T": 75.0},
        {"CODUSU": "D", "NRO_HOGAR": "4", "COMPONENTE": 1, "P47T": 125.0},
        {"CODUSU": "E", "NRO_HOGAR": "5", "COMPONENTE": 1, "P47T": 150.0},
        {"CODUSU": "F", "NRO_HOGAR": "6", "COMPONENTE": 1, "P47T": 175.0},
        {"CODUSU": "G", "NRO_HOGAR": "7", "COMPONENTE": 1, "P47T": 225.0},
        {"CODUSU": "H", "NRO_HOGAR": "8", "COMPONENTE": 1, "P47T": 250.0},
        {"CODUSU": "I", "NRO_HOGAR": "9", "COMPONENTE": 1, "P47T": 275.0},
        {"CODUSU": "J", "NRO_HOGAR": "10", "COMPONENTE": 1, "P47T": 300.0},
    ]


def test_identity_helpers_preserve_person_and_household_boundaries() -> None:
    row = _rows()[0]
    assert household_id(row) == "A\x1f1"
    assert person_id(row) == "A\x1f1\x1f1"


def test_fold_manifest_is_deterministic_and_household_safe() -> None:
    rows = _rows()
    first = build_fold_manifest(rows, n_splits=3)
    second = build_fold_manifest(rows, n_splits=3)

    assert first == second
    assert len(first.fold_ids) == len(rows)
    assert set(first.fold_ids) == {0, 1, 2}
    assert first.fold_ids[0] == first.fold_ids[1]

    household_to_folds: dict[str, set[int]] = {}
    for household, fold in zip(first.household_ids, first.fold_ids, strict=True):
        household_to_folds.setdefault(household, set()).add(fold)
    assert all(len(folds) == 1 for folds in household_to_folds.values())

    reordered = build_fold_manifest(list(reversed(rows)), n_splits=3)
    assert first.mapping() == reordered.mapping()


def test_fold_manifest_rejects_duplicate_person_identity() -> None:
    rows = _rows()
    with pytest.raises(ScientificPrimitiveError, match="duplicate_person_identity"):
        build_fold_manifest([rows[0], rows[0], *rows[2:]], n_splits=3)


def test_probability_artifact_requires_explicit_valid_class_semantics() -> None:
    rows = _rows()[:3]
    row_ids = tuple(person_id(row) for row in rows)
    artifact = PredictionArtifact(
        target="CAT_OCUP",
        kind="probability",
        row_ids=row_ids,
        values=np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]),
        source="oof",
        class_labels=("0", "1"),
        fold_ids=(0, 0, 1),
        run_id="synthetic-m1",
        metadata={"estimator": "test-double"},
    )

    artifact.validate_alignment(rows)
    assert artifact.class_labels == ("0", "1")
    assert artifact.values.shape == (3, 2)
    assert artifact.run_id == "synthetic-m1"

    with pytest.raises(
        ScientificPrimitiveError,
        match="probability_rows_must_sum_to_one",
    ):
        PredictionArtifact(
            target="CAT_OCUP",
            kind="probability",
            row_ids=row_ids,
            values=np.asarray([[0.8, 0.3], [0.3, 0.7], [0.6, 0.4]]),
            source="oof",
            class_labels=("0", "1"),
            fold_ids=(0, 0, 1),
        )


def test_prediction_alignment_fails_on_reordered_or_missing_people() -> None:
    rows = _rows()[:3]
    row_ids = tuple(person_id(row) for row in rows)
    artifact = PredictionArtifact(
        target="P47T",
        kind="regression",
        row_ids=row_ids,
        values=np.asarray([90.0, 210.0, 55.0]),
        source="score",
    )

    with pytest.raises(ScientificPrimitiveError, match="prediction_row_order_mismatch"):
        artifact.validate_alignment(list(reversed(rows)))

    with pytest.raises(ScientificPrimitiveError, match="prediction_row_identity_mismatch"):
        artifact.validate_alignment(rows[:-1])


def test_synthetic_direct_prediction_aggregates_only_with_complete_membership() -> None:
    rows = _rows()[:3]
    artifact = PredictionArtifact(
        target="P47T",
        kind="regression",
        row_ids=tuple(person_id(row) for row in rows),
        values=np.asarray([90.0, 210.0, 55.0]),
        source="oof",
        fold_ids=(0, 0, 1),
        run_id="synthetic-direct-m1",
    )

    households = aggregate_person_predictions(
        rows,
        {"direct": artifact},
        truth_field="P47T",
    )

    assert households == [
        {
            "CODUSU": "A",
            "NRO_HOGAR": "1",
            "member_count": 2,
            "truth_household_income": 300.0,
            "direct_household_income": 300.0,
            "status": "complete",
        },
        {
            "CODUSU": "B",
            "NRO_HOGAR": "2",
            "member_count": 1,
            "truth_household_income": 50.0,
            "direct_household_income": 55.0,
            "status": "complete",
        },
    ]

    incomplete = PredictionArtifact(
        target="P47T",
        kind="regression",
        row_ids=tuple(person_id(row) for row in rows[:-1]),
        values=np.asarray([90.0, 210.0]),
        source="score",
    )
    with pytest.raises(ScientificPrimitiveError, match="prediction_row_identity_mismatch"):
        aggregate_person_predictions(rows, {"direct": incomplete})


def test_regression_metric_plumbing_is_linear_scale_and_unweighted() -> None:
    metrics = regression_metrics(
        np.asarray([100.0, 200.0, 50.0]),
        np.asarray([90.0, 210.0, 55.0]),
    )
    assert metrics["mae"] == pytest.approx(25.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(225.0 / 3.0))
    assert metrics["mean_error"] == pytest.approx(5.0 / 3.0)
