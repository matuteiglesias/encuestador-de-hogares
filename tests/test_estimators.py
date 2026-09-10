from __future__ import annotations

import numpy as np
import pytest

from encuestador.crossfit import crossfit_predict
from encuestador.estimators import (
    EstimatorConfigurationError,
    HGBClassifierAdapter,
    HGBRegressorAdapter,
)
from encuestador.scientific_primitives import build_fold_manifest


def _experiment_frame() -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    features: list[list[float]] = []
    for household in range(48):
        members = 1 + household % 2
        for component in range(1, members + 1):
            rows.append(
                {
                    "CODUSU": f"EPH-{household // 8:02d}",
                    "NRO_HOGAR": str(household + 1),
                    "COMPONENTE": component,
                }
            )
            features.append(
                [
                    float((household + component) % 3),
                    float(18 + (household * 3 + component * 7) % 55),
                    float((household % 7) / 7.0),
                ]
            )
    return rows, np.asarray(features, dtype=float)


def _small_parameters() -> dict[str, object]:
    return {
        "max_iter": 15,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 5,
        "learning_rate": 0.08,
    }


def test_hgb_classifier_crossfit_emits_stable_probabilities() -> None:
    rows, x = _experiment_frame()
    manifest = build_fold_manifest(rows, n_splits=5)
    y = ((x[:, 1] > 38) | (x[:, 0] == 2)).astype(int)

    artifact = crossfit_predict(
        x,
        y,
        manifest,
        lambda: HGBClassifierAdapter(
            categorical_features=(0,),
            parameters=_small_parameters(),
        ),
        target="CAT_OCUP",
        kind="probability",
        class_labels=(0, 1),
        run_id="hgb-classifier-test",
    )

    assert artifact.source == "oof"
    assert artifact.class_labels == ("0", "1")
    assert artifact.values.shape == (len(rows), 2)
    assert np.isfinite(artifact.values).all()
    assert np.all((artifact.values >= 0) & (artifact.values <= 1))
    assert np.allclose(artifact.values.sum(axis=1), 1.0)


def test_hgb_squared_error_regressor_crossfit_is_household_safe() -> None:
    rows, x = _experiment_frame()
    manifest = build_fold_manifest(rows, n_splits=5)
    y = 120.0 + 8.0 * x[:, 1] + 25.0 * x[:, 0] + 40.0 * x[:, 2]

    artifact = crossfit_predict(
        x,
        y,
        manifest,
        lambda: HGBRegressorAdapter(
            loss="squared_error",
            categorical_features=(0,),
            parameters=_small_parameters(),
        ),
        target="P47T",
        run_id="hgb-regression-test",
    )

    assert artifact.fold_ids == manifest.fold_ids
    assert artifact.values.shape == (len(rows),)
    assert np.isfinite(artifact.values).all()


def test_hgb_gamma_supports_strictly_positive_linear_income() -> None:
    rows, x = _experiment_frame()
    manifest = build_fold_manifest(rows, n_splits=5)
    y = np.exp(4.0 + 0.015 * x[:, 1] + 0.08 * x[:, 0])

    artifact = crossfit_predict(
        x,
        y,
        manifest,
        lambda: HGBRegressorAdapter(
            loss="gamma",
            categorical_features=(0,),
            parameters=_small_parameters(),
        ),
        target="positive_income",
    )

    assert np.isfinite(artifact.values).all()
    assert np.all(artifact.values > 0)


def test_gamma_rejects_nonpositive_targets_before_sklearn_fit() -> None:
    adapter = HGBRegressorAdapter(loss="gamma", parameters=_small_parameters())
    x = np.asarray([[0.0], [1.0], [2.0]])
    y = np.asarray([1.0, 0.0, 3.0])

    with pytest.raises(
        EstimatorConfigurationError,
        match="gamma_target_must_be_strictly_positive",
    ):
        adapter.fit(x, y)


def test_hgb_adapters_forbid_hidden_early_stopping() -> None:
    with pytest.raises(
        EstimatorConfigurationError,
        match="hidden_early_stopping_forbidden_without_household_safe_validation",
    ):
        HGBClassifierAdapter(parameters={"early_stopping": True})

    with pytest.raises(
        EstimatorConfigurationError,
        match="hidden_early_stopping_forbidden_without_household_safe_validation",
    ):
        HGBRegressorAdapter(parameters={"early_stopping": "auto"})


def test_categorical_feature_positions_are_explicit_and_validated() -> None:
    with pytest.raises(
        EstimatorConfigurationError,
        match="categorical_feature_index_duplicate",
    ):
        HGBClassifierAdapter(categorical_features=(0, 0))

    adapter = HGBRegressorAdapter(categorical_features=(2,), parameters=_small_parameters())
    with pytest.raises(
        EstimatorConfigurationError,
        match="categorical_feature_index_out_of_range",
    ):
        adapter.fit(np.asarray([[1.0], [2.0]]), np.asarray([2.0, 3.0]))
