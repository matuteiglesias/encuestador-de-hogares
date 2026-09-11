from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from encuestador.cascade import CategoricalLatentSpec, run_one_layer_oof
from encuestador.estimators import HGBClassifierAdapter, HGBRegressorAdapter
from encuestador.evaluation import (
    classification_diagnostics,
    distributional_regression_diagnostics,
    evaluate_cascade_triangle,
    household_prediction_diagnostics,
)
from encuestador.scientific_primitives import PredictionArtifact, build_fold_manifest
from encuestador.transport_proof import (
    DEFAULT_SEMANTIC_PLANE,
    DEFAULT_SPEC,
    make_synthetic_people,
    validate_semantic_plane,
    validate_spec,
)


def test_probability_diagnostics_keep_probabilistic_quality_visible() -> None:
    artifact = PredictionArtifact(
        target="STATE",
        kind="probability",
        row_ids=("a", "b", "c", "d"),
        values=np.asarray(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.65, 0.35],
                [0.10, 0.90],
            ]
        ),
        source="oof",
        class_labels=("0", "1"),
        fold_ids=(0, 1, 0, 1),
    )
    metrics = classification_diagnostics(np.asarray([0, 1, 0, 1]), artifact, reliability_bins=5)

    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["log_loss"] < 1.0
    assert 0.0 <= metrics["multiclass_brier"] < 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["reliability"]["definition"] == "one_vs_rest_equal_width_v1"
    assert len(metrics["reliability"]["bin_edges"]) == 6


def test_distributional_diagnostics_detect_compression_and_decile_bias() -> None:
    truth = np.arange(1.0, 101.0)
    prediction = 50.5 + 0.4 * (truth - 50.5)
    metrics = distributional_regression_diagnostics(truth, prediction)

    assert np.isclose(metrics["dispersion_ratio"], 0.4)
    assert metrics["quantile_agreement"]["0.1"]["difference"] > 0
    assert metrics["quantile_agreement"]["0.9"]["difference"] < 0
    assert metrics["tails"]["low_tail_mean_bias"] > 0
    assert metrics["tails"]["high_tail_mean_bias"] < 0
    assert len(metrics["deciles"]["by_observed_income"]) == 10
    assert sum(item["count"] for item in metrics["deciles"]["by_observed_income"]) == 100


def test_household_diagnostics_account_for_every_person() -> None:
    rows = [
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 1, "P47T": 10.0},
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 2, "P47T": 20.0},
        {"CODUSU": "B", "NRO_HOGAR": "2", "COMPONENTE": 1, "P47T": 40.0},
    ]
    artifact = PredictionArtifact(
        target="P47T",
        kind="regression",
        row_ids=("A\x1f1\x1f1", "A\x1f1\x1f2", "B\x1f2\x1f1"),
        values=np.asarray([12.0, 18.0, 38.0]),
        source="oof",
        fold_ids=(0, 0, 1),
    )
    metrics = household_prediction_diagnostics(rows, {"model": artifact}, truth_field="P47T")

    assert metrics["household_count"] == 2
    assert metrics["person_count_accounted"] == 3
    assert metrics["incomplete_household_count"] == 0
    assert set(metrics["candidates"]["model"]["by_household_size"]) == {"1", "2"}


def _synthetic_frame() -> tuple[list[str], list[dict[str, object]]]:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    plane = json.loads(DEFAULT_SEMANTIC_PLANE.read_text(encoding="utf-8"))
    validate_spec(spec)
    approved = validate_semantic_plane(spec, plane)
    rows = make_synthetic_people(
        approved,
        household_count=36,
        codusu_prefix="EPH-EVAL",
        include_census_design=False,
    )
    return approved, rows


def _matrix(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def _categorical_positions(columns: list[str]) -> tuple[int, ...]:
    continuous = {"IX_TOT", "P03", "H16"}
    return tuple(index for index, name in enumerate(columns) if name not in continuous)


def _classifier_factory(categorical: tuple[int, ...]) -> Callable[[], HGBClassifierAdapter]:
    return lambda: HGBClassifierAdapter(
        categorical_features=categorical,
        parameters={"max_iter": 10, "max_leaf_nodes": 7, "min_samples_leaf": 5},
    )


def _terminal_factory(categorical: tuple[int, ...]) -> Callable[[], HGBRegressorAdapter]:
    return lambda: HGBRegressorAdapter(
        categorical_features=categorical,
        parameters={"max_iter": 15, "max_leaf_nodes": 9, "min_samples_leaf": 5},
    )


def _classes(rows: list[dict[str, object]], target: str) -> tuple[str, ...]:
    return tuple(str(value) for value in sorted({float(row[target]) for row in rows}))


def test_cascade_triangle_matches_folds_and_reports_oracle_reconstruction_evidence() -> None:
    columns, rows = _synthetic_frame()
    x = _matrix(rows, columns)
    y = _matrix(rows, ["P47T"])[:, 0]
    manifest = build_fold_manifest(rows, n_splits=5)
    categorical = _categorical_positions(columns)
    specs = tuple(
        CategoricalLatentSpec(
            target=target,
            class_labels=_classes(rows, target),
            estimator_factory=_classifier_factory(categorical),
        )
        for target in ("CAT_OCUP", "CAT_INAC", "CH07")
    )
    latent_targets = {spec.target: _matrix(rows, [spec.target])[:, 0] for spec in specs}
    lean = run_one_layer_oof(
        x,
        y,
        latent_targets,
        manifest,
        specs,
        _terminal_factory(categorical),
        terminal_target_name="P47T",
        base_feature_names=columns,
        run_id="triangle-test",
    )

    result = evaluate_cascade_triangle(
        x,
        y,
        latent_targets,
        manifest,
        specs,
        _terminal_factory(categorical),
        lean.terminal_artifact,
        lean.latent_artifacts,
        terminal_target_name="P47T",
        person_rows=rows,
        truth_field="P47T",
        run_id="triangle-test",
    )

    assert result.direct.fold_ids == manifest.fold_ids
    assert result.oracle.fold_ids == manifest.fold_ids
    assert result.deployable.fold_ids == manifest.fold_ids
    assert set(result.person_metrics) == {"direct", "oracle", "deployable"}
    assert set(result.gains) == {"mae", "rmse"}
    assert set(result.latent_metrics) == {"CAT_OCUP", "CAT_INAC", "CH07"}
    assert all(metrics["log_loss"] >= 0 for metrics in result.latent_metrics.values())
    assert result.household_metrics is not None
    assert result.household_metrics["person_count_accounted"] == len(rows)
