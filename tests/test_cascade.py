from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from encuestador.cascade import (
    CategoricalLatentSpec,
    fit_one_layer_and_score,
    run_one_layer_oof,
)
from encuestador.crossfit import crossfit_predict
from encuestador.estimators import HGBClassifierAdapter, HGBRegressorAdapter
from encuestador.scientific_primitives import (
    aggregate_person_predictions,
    build_fold_manifest,
    person_id,
    regression_metrics,
)
from encuestador.transport_proof import (
    DEFAULT_SEMANTIC_PLANE,
    DEFAULT_SPEC,
    make_synthetic_people,
    validate_semantic_plane,
    validate_spec,
)


class TrackingClassifier:
    def __init__(self, registry: list[set[int]]) -> None:
        self.registry = registry
        self.classes_ = np.asarray([0, 1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> TrackingClassifier:
        del y
        self.registry.append(set(x[:, 0].astype(int).tolist()))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.full(len(x), 0.6), np.full(len(x), 0.4)]
        )


class TrackingRegressor:
    def __init__(self, registry: list[set[int]]) -> None:
        self.registry = registry
        self.mean = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TrackingRegressor:
        self.registry.append(set(x[:, 0].astype(int).tolist()))
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean)


def _probe_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for household in range(40):
        for component in (1, 2):
            rows.append(
                {
                    "CODUSU": f"P{household // 8}",
                    "NRO_HOGAR": str(household + 1),
                    "COMPONENTE": component,
                }
            )
    return rows


def test_outer_holdout_is_absent_from_every_latent_fit_used_by_that_terminal_fold() -> None:
    rows = _probe_rows()
    manifest = build_fold_manifest(rows, n_splits=4)
    global_folds = manifest.as_array()
    x = np.column_stack([global_folds.astype(float), np.arange(len(rows), dtype=float)])
    latent_y = (np.arange(len(rows)) % 2).astype(int)
    terminal_y = 10.0 + np.arange(len(rows), dtype=float)
    latent_registry: list[set[int]] = []
    terminal_registry: list[set[int]] = []

    result = run_one_layer_oof(
        x,
        terminal_y,
        {"STATE": latent_y},
        manifest,
        (
            CategoricalLatentSpec(
                target="STATE",
                class_labels=("0", "1"),
                estimator_factory=lambda: TrackingClassifier(latent_registry),
            ),
        ),
        lambda: TrackingRegressor(terminal_registry),
        terminal_target_name="P47T",
        base_feature_names=("global_fold_probe", "row_probe"),
    )

    assert np.isfinite(result.terminal_artifact.values).all()
    assert result.terminal_artifact.fold_ids == manifest.fold_ids
    assert result.nesting_policy == "outer_fold_isolated_latent_crossfit_v1"

    calls_per_outer = manifest.n_splits
    assert len(latent_registry) == manifest.n_splits * calls_per_outer
    assert len(terminal_registry) == manifest.n_splits
    all_folds = set(range(manifest.n_splits))
    for outer_index in range(manifest.n_splits):
        chunk = latent_registry[
            outer_index * calls_per_outer : (outer_index + 1) * calls_per_outer
        ]
        common_missing = set.intersection(*(all_folds - fitted for fitted in chunk))
        terminal_missing = all_folds - terminal_registry[outer_index]
        assert len(common_missing) == 1
        assert terminal_missing == common_missing
        assert all(len(fitted) <= manifest.n_splits - 1 for fitted in chunk)


def _approved_synthetic_frame() -> tuple[
    list[str],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    plane = json.loads(DEFAULT_SEMANTIC_PLANE.read_text(encoding="utf-8"))
    validate_spec(spec)
    approved = validate_semantic_plane(spec, plane)
    eph_rows = make_synthetic_people(
        approved,
        household_count=32,
        codusu_prefix="EPH-LEAN",
        include_census_design=False,
    )
    census_rows = make_synthetic_people(
        approved,
        household_count=12,
        codusu_prefix="CPV-LEAN",
        include_census_design=True,
    )
    return approved, eph_rows, census_rows


def _matrix(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )


def _categorical_positions(approved: list[str]) -> tuple[int, ...]:
    continuous = {"IX_TOT", "P03", "H16"}
    return tuple(index for index, name in enumerate(approved) if name not in continuous)


def _classifier_factory(categorical: tuple[int, ...]) -> Callable[[], HGBClassifierAdapter]:
    return lambda: HGBClassifierAdapter(
        categorical_features=categorical,
        parameters={
            "max_iter": 10,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 5,
            "learning_rate": 0.08,
        },
    )


def _regressor_factory(categorical: tuple[int, ...]) -> Callable[[], HGBRegressorAdapter]:
    return lambda: HGBRegressorAdapter(
        loss="squared_error",
        categorical_features=categorical,
        parameters={
            "max_iter": 15,
            "max_leaf_nodes": 9,
            "min_samples_leaf": 5,
            "learning_rate": 0.08,
        },
    )


def _classes(rows: list[dict[str, object]], target: str) -> tuple[str, ...]:
    return tuple(str(value) for value in sorted({float(row[target]) for row in rows}))


def test_direct_and_lean_hgb_share_folds_and_produce_complete_household_welfare() -> None:
    approved, eph_rows, census_rows = _approved_synthetic_frame()
    x_eph = _matrix(eph_rows, approved)
    x_census = _matrix(census_rows, approved)
    y = _matrix(eph_rows, ["P47T"])[:, 0]
    manifest = build_fold_manifest(eph_rows, n_splits=5)
    categorical = _categorical_positions(approved)
    terminal_factory = _regressor_factory(categorical)

    direct = crossfit_predict(
        x_eph,
        y,
        manifest,
        terminal_factory,
        target="P47T",
        run_id="synthetic-direct-vs-lean",
    )
    latent_specs = tuple(
        CategoricalLatentSpec(
            target=target,
            class_labels=_classes(eph_rows, target),
            estimator_factory=_classifier_factory(categorical),
        )
        for target in ("CAT_OCUP", "CAT_INAC", "CH07")
    )
    latent_targets = {
        spec.target: _matrix(eph_rows, [spec.target])[:, 0]
        for spec in latent_specs
    }
    lean = run_one_layer_oof(
        x_eph,
        y,
        latent_targets,
        manifest,
        latent_specs,
        terminal_factory,
        terminal_target_name="P47T",
        base_feature_names=approved,
        run_id="synthetic-direct-vs-lean",
    )

    assert direct.fold_ids == lean.terminal_artifact.fold_ids == manifest.fold_ids
    assert set(lean.latent_artifacts) == {"CAT_OCUP", "CAT_INAC", "CH07"}
    assert all(
        artifact.fold_ids == manifest.fold_ids
        for artifact in lean.latent_artifacts.values()
    )
    assert any(name.startswith("latent.CAT_OCUP.p[") for name in lean.terminal_feature_names)
    assert any(name.startswith("latent.CH07.p[") for name in lean.terminal_feature_names)

    direct_metrics = regression_metrics(y, direct.values)
    lean_metrics = regression_metrics(y, lean.terminal_artifact.values)
    assert direct_metrics["rmse"] >= 0
    assert lean_metrics["rmse"] >= 0

    households = aggregate_person_predictions(
        eph_rows,
        {"direct": direct, "lean": lean.terminal_artifact},
        truth_field="P47T",
    )
    assert all(row["status"] == "complete" for row in households)
    assert sum(row["member_count"] for row in households) == len(eph_rows)

    scored = fit_one_layer_and_score(
        x_eph,
        y,
        latent_targets,
        manifest,
        latent_specs,
        terminal_factory,
        x_census,
        tuple(person_id(row) for row in census_rows),
        terminal_target_name="P47T",
        base_feature_names=approved,
        run_id="synthetic-direct-vs-lean",
    )
    scored.terminal_artifact.validate_alignment(census_rows)
    assert np.isfinite(scored.terminal_artifact.values).all()
    assert scored.terminal_artifact.source == "score"
    census_households = aggregate_person_predictions(
        census_rows,
        {"lean": scored.terminal_artifact},
        truth_field="P47T",
    )
    assert all(row["status"] == "complete" for row in census_households)
