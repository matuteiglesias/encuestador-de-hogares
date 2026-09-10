from __future__ import annotations

import numpy as np
import pytest

from encuestador.estimators import HGBClassifierAdapter, HGBRegressorAdapter
from encuestador.scientific_primitives import FoldManifest
from encuestador.terminal import (
    HurdleError,
    HurdleEstimator,
    classify_income_target,
    crossfit_hurdle,
    fit_hurdle_and_score,
)


def _manifest(n_splits: int = 4, rows_per_fold: int = 8) -> FoldManifest:
    row_ids = tuple(f"r{i}" for i in range(n_splits * rows_per_fold))
    household_ids = tuple(f"h{i}" for i in range(n_splits * rows_per_fold))
    fold_ids = tuple(i // rows_per_fold for i in range(n_splits * rows_per_fold))
    return FoldManifest(
        row_ids=row_ids,
        household_ids=household_ids,
        fold_ids=fold_ids,
        n_splits=n_splits,
    )


def _features(manifest: FoldManifest) -> np.ndarray:
    i = np.arange(len(manifest.row_ids), dtype=float)
    return np.column_stack([
        np.asarray(manifest.fold_ids, dtype=float),
        i % 3,
        20.0 + i,
    ])


def _hgb_hurdle(formulation: str) -> HurdleEstimator:
    params = {
        "max_iter": 12,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 3,
        "learning_rate": 0.08,
    }
    return HurdleEstimator(
        formulation=formulation,
        presence_factory=lambda: HGBClassifierAdapter(
            categorical_features=(1,), parameters=params
        ),
        amount_factory=lambda: HGBRegressorAdapter(
            loss="gamma" if formulation == "gamma" else "squared_error",
            categorical_features=(1,),
            parameters=params,
        ),
        retransformation="linear_identity_v1" if formulation == "gamma" else "duan_smearing_v1",
    )


def test_target_eligibility_preserves_zero_and_excludes_minus9_and_missing() -> None:
    result = classify_income_target([0, 120, -9, None, np.nan, "", 50])
    assert result.counts == {
        "rows": 7,
        "eligible": 3,
        "positive": 2,
        "zero": 1,
        "nonresponse": 1,
        "missing": 3,
    }
    assert result.numeric[0] == 0
    assert np.isnan(result.numeric[2])
    assert np.isnan(result.numeric[3])


def test_unexpected_negative_income_is_not_silently_reinterpreted() -> None:
    with pytest.raises(HurdleError, match="unexpected_negative_terminal_income"):
        classify_income_target([0, 10, -1])


@pytest.mark.parametrize("formulation", ["log", "gamma"])
def test_hurdle_oof_keeps_zero_state_and_emits_linear_scale_components(formulation: str) -> None:
    manifest = _manifest()
    x = _features(manifest)
    y: list[object] = []
    for i in range(len(x)):
        if i % 7 == 0:
            y.append(-9)
        elif i % 11 == 0:
            y.append(None)
        elif i % 3 == 0:
            y.append(0)
        else:
            y.append(100.0 + 7.0 * i)

    result = crossfit_hurdle(
        x,
        np.asarray(y, dtype=object),
        manifest,
        lambda: _hgb_hurdle(formulation),
        target="P47T",
        run_id=f"hurdle-{formulation}",
    )

    assert result.unconditional_expected_income.fold_ids == manifest.fold_ids
    assert result.p_positive.fold_ids == manifest.fold_ids
    assert result.positive_amount_prediction.fold_ids == manifest.fold_ids
    assert result.p_positive.class_labels == ("0", "1")
    p = result.p_positive.values[:, 1]
    amount = result.positive_amount_prediction.values
    unconditional = result.unconditional_expected_income.values
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(amount > 0)
    assert np.all(unconditional >= 0)
    assert np.allclose(unconditional, p * amount)
    assert result.target_eligibility["nonresponse"] > 0
    assert result.target_eligibility["missing"] > 0
    if formulation == "log":
        assert result.retransformation == "duan_smearing_v1"
    else:
        assert result.retransformation == "linear_identity_v1"


def test_gamma_amount_head_is_fit_only_on_strictly_positive_income() -> None:
    manifest = _manifest()
    x = _features(manifest)
    y = np.asarray([0.0 if i % 2 == 0 else 100.0 + i for i in range(len(x))], dtype=object)
    result = crossfit_hurdle(x, y, manifest, lambda: _hgb_hurdle("gamma"), target="P47T")
    assert np.all(result.positive_amount_prediction.values > 0)
    assert result.target_eligibility["zero"] == len(x) // 2


def test_empty_positive_population_fails_explicitly() -> None:
    manifest = _manifest()
    x = _features(manifest)
    with pytest.raises(HurdleError, match="positive_amount_training_empty"):
        crossfit_hurdle(x, np.zeros(len(x)), manifest, lambda: _hgb_hurdle("gamma"), target="P47T")


def test_fold_with_no_positive_training_supervision_fails_explicitly() -> None:
    manifest = _manifest(n_splits=3, rows_per_fold=5)
    x = _features(manifest)
    y = np.zeros(len(x), dtype=object)
    y[np.asarray(manifest.fold_ids) == 0] = 100.0
    with pytest.raises(HurdleError, match="positive_amount_training_empty"):
        crossfit_hurdle(x, y, manifest, lambda: _hgb_hurdle("log"), target="P47T")


class TrackingPresence:
    def __init__(self, registry: list[set[int]]) -> None:
        self.registry = registry
        self.classes_ = np.asarray([0, 1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> TrackingPresence:
        del y
        self.registry.append(set(x[:, 0].astype(int).tolist()))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(x), 0.4), np.full(len(x), 0.6)])


class TrackingAmount:
    def __init__(self, registry: list[set[int]]) -> None:
        self.registry = registry
        self.mean = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TrackingAmount:
        self.registry.append(set(x[:, 0].astype(int).tolist()))
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean)


def test_hurdle_outer_fold_never_fits_held_out_rows() -> None:
    manifest = _manifest(n_splits=4, rows_per_fold=6)
    x = _features(manifest)
    y = np.asarray([0.0 if i % 3 == 0 else 10.0 + i for i in range(len(x))], dtype=object)
    presence_registry: list[set[int]] = []
    amount_registry: list[set[int]] = []

    def factory() -> HurdleEstimator:
        return HurdleEstimator(
            formulation="gamma",
            presence_factory=lambda: TrackingPresence(presence_registry),
            amount_factory=lambda: TrackingAmount(amount_registry),
            retransformation="linear_identity_v1",
        )

    crossfit_hurdle(x, y, manifest, factory, target="P47T")
    all_folds = set(range(manifest.n_splits))
    assert len(presence_registry) == manifest.n_splits
    assert len(amount_registry) == manifest.n_splits
    for outer_fold, (presence_fitted, amount_fitted) in enumerate(
        zip(presence_registry, amount_registry, strict=True)
    ):
        assert outer_fold not in presence_fitted
        assert outer_fold not in amount_fitted
        assert presence_fitted == all_folds - {outer_fold}
        assert amount_fitted == all_folds - {outer_fold}


def test_full_fit_scoring_preserves_exact_external_row_ids() -> None:
    manifest = _manifest()
    x = _features(manifest)
    y = np.asarray([0.0 if i % 4 == 0 else 100.0 + i for i in range(len(x))], dtype=object)
    score = np.asarray([[0.0, 1.0, 33.0], [0.0, 2.0, 44.0]])
    result = fit_hurdle_and_score(
        x,
        y,
        score,
        ("sample-p1", "sample-p2"),
        lambda: _hgb_hurdle("gamma"),
        target="P47T",
    )
    assert result.unconditional_expected_income.row_ids == ("sample-p1", "sample-p2")
    assert result.unconditional_expected_income.source == "score"
    assert np.all(result.unconditional_expected_income.values >= 0)
