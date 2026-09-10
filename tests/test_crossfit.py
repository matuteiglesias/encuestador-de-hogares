from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from encuestador.crossfit import CrossfitError, crossfit_predict, fit_full_and_score
from encuestador.scientific_primitives import build_fold_manifest


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for household in range(12):
        members = 2 if household % 3 == 0 else 1
        for component in range(1, members + 1):
            rows.append(
                {
                    "CODUSU": f"H{household:02d}",
                    "NRO_HOGAR": str(household + 1),
                    "COMPONENTE": component,
                }
            )
    return rows


class LeakageProbeRegressor:
    def __init__(self, registry: list[tuple[set[float], set[float]]]) -> None:
        self.registry = registry
        self.train_keys: set[float] = set()
        self.mean = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> LeakageProbeRegressor:
        self.train_keys = set(x[:, 0].tolist())
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        holdout_keys = set(x[:, 0].tolist())
        self.registry.append((set(self.train_keys), holdout_keys))
        assert self.train_keys.isdisjoint(holdout_keys)
        return np.full(len(x), self.mean)


class FixedProbabilityClassifier:
    def __init__(self, reverse_classes: bool = False) -> None:
        self.reverse_classes = reverse_classes
        self.classes_ = np.asarray([0, 1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> FixedProbabilityClassifier:
        del x, y
        if self.reverse_classes:
            self.classes_ = np.asarray([1, 0])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        base = np.column_stack(
            [
                np.full(len(x), 0.75),
                np.full(len(x), 0.25),
            ]
        )
        return base[:, ::-1] if self.reverse_classes else base


class MissingClassClassifier(FixedProbabilityClassifier):
    def fit(self, x: np.ndarray, y: np.ndarray) -> MissingClassClassifier:
        del x, y
        self.classes_ = np.asarray([0, 2])
        return self


def test_crossfit_never_fits_a_held_out_person_or_household_member() -> None:
    rows = _rows()
    manifest = build_fold_manifest(rows, n_splits=4)
    x = np.column_stack(
        [
            np.arange(len(rows), dtype=float),
            np.linspace(0.0, 1.0, len(rows)),
        ]
    )
    y = 10.0 + 2.0 * x[:, 1]
    registry: list[tuple[set[float], set[float]]] = []

    artifact = crossfit_predict(
        x,
        y,
        manifest,
        lambda: LeakageProbeRegressor(registry),
        target="P47T",
        run_id="crossfit-regression-test",
    )

    assert artifact.source == "oof"
    assert artifact.fold_ids == manifest.fold_ids
    assert artifact.row_ids == manifest.row_ids
    assert np.isfinite(artifact.values).all()
    assert len(registry) == manifest.n_splits
    assert all(train.isdisjoint(holdout) for train, holdout in registry)

    for household in set(manifest.household_ids):
        indices = [
            index
            for index, value in enumerate(manifest.household_ids)
            if value == household
        ]
        assert len({manifest.fold_ids[index] for index in indices}) == 1


def test_probability_crossfit_preserves_declared_class_order() -> None:
    rows = _rows()
    manifest = build_fold_manifest(rows, n_splits=4)
    x = np.column_stack(
        [np.arange(len(rows), dtype=float), np.ones(len(rows), dtype=float)]
    )
    y = np.asarray([index % 2 for index in range(len(rows))])

    artifact = crossfit_predict(
        x,
        y,
        manifest,
        lambda: FixedProbabilityClassifier(reverse_classes=True),
        target="CAT_OCUP",
        kind="probability",
        class_labels=(0, 1),
    )

    assert artifact.class_labels == ("0", "1")
    assert np.allclose(artifact.values[:, 0], 0.75)
    assert np.allclose(artifact.values[:, 1], 0.25)
    assert np.allclose(artifact.values.sum(axis=1), 1.0)


def test_probability_crossfit_fails_if_a_fold_estimator_has_wrong_class_set() -> None:
    rows = _rows()
    manifest = build_fold_manifest(rows, n_splits=4)
    x = np.column_stack(
        [np.arange(len(rows), dtype=float), np.ones(len(rows), dtype=float)]
    )
    y = np.asarray([index % 2 for index in range(len(rows))])

    with pytest.raises(CrossfitError, match="estimator_class_set_mismatch"):
        crossfit_predict(
            x,
            y,
            manifest,
            MissingClassClassifier,
            target="CAT_OCUP",
            kind="probability",
            class_labels=(0, 1),
        )


def test_full_fit_score_is_separate_from_oof_and_preserves_score_row_ids() -> None:
    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.asarray([1.0, 2.0, 3.0, 4.0])
    x_score = np.asarray([[10.0], [11.0]])
    registry: list[tuple[set[float], set[float]]] = []

    artifact = fit_full_and_score(
        x_train,
        y_train,
        x_score,
        ("score-a", "score-b"),
        lambda: LeakageProbeRegressor(registry),
        target="P47T",
        run_id="full-score-test",
    )

    assert artifact.source == "score"
    assert artifact.fold_ids is None
    assert artifact.row_ids == ("score-a", "score-b")
    assert artifact.values.tolist() == [2.5, 2.5]
    assert registry == [({0.0, 1.0, 2.0, 3.0}, {10.0, 11.0})]


def test_crossfit_refuses_feature_manifest_length_mismatch() -> None:
    rows = _rows()
    manifest = build_fold_manifest(rows, n_splits=4)
    x = np.ones((len(rows) - 1, 2))
    y = np.ones(len(rows) - 1)
    factory: Callable[[], LeakageProbeRegressor] = lambda: LeakageProbeRegressor([])

    with pytest.raises(CrossfitError, match="feature_manifest_length_mismatch"):
        crossfit_predict(x, y, manifest, factory, target="P47T")
