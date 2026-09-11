"""Nested household predictive distribution experimental primitive.

The empirical residual distribution is a deliberately narrow baseline, not a
general production uncertainty framework.  Calibration residuals must come
from inner household OOF predictions made solely inside an outer training set.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class PredictiveDistributionError(ValueError):
    """Raised when nesting or household identity invariants are violated."""


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def _ids(values: Sequence[Any], name: str) -> np.ndarray:
    ids = np.asarray([str(value) for value in values], object)
    if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids):
        raise PredictiveDistributionError(f"{name}_household_ids_invalid")
    return ids


def _matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, float)
    if array.ndim != 2 or not len(array) or np.isinf(array).any():
        raise PredictiveDistributionError(f"{name}_features_invalid")
    return array


@dataclass(frozen=True)
class EmpiricalResidualDistribution:
    """Nested OOF residual ECDF attached to an outer fitted point model."""

    estimator: Regressor
    residuals: np.ndarray
    calibration_household_ids: tuple[str, ...]
    outer_test_household_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def probability_below(self, x: Sequence[Sequence[float]], threshold: float | Sequence[float]) -> np.ndarray:
        features = _matrix(x, "scoring")
        point = np.asarray(self.estimator.predict(features), float)
        cut = np.asarray(threshold, float)
        if point.ndim != 1 or len(point) != len(features) or not np.isfinite(point).all():
            raise PredictiveDistributionError("outer_prediction_invalid")
        if cut.ndim == 0:
            cut = np.full(len(point), float(cut))
        if cut.ndim != 1 or len(cut) != len(point) or not np.isfinite(cut).all():
            raise PredictiveDistributionError("threshold_invalid")
        return np.mean(self.residuals[None, :] <= (cut - point)[:, None], axis=1)

    predict_cdf = probability_below


def fit_nested_empirical_residual_distribution(
    outer_train_x: Sequence[Sequence[float]], outer_train_y: Sequence[float],
    outer_train_household_ids: Sequence[Any], outer_test_household_ids: Sequence[Any],
    estimator_factory: Callable[[], Regressor], *, inner_folds: int = 5,
) -> EmpiricalResidualDistribution:
    """Fit inner household OOF residuals, then the outer-training point model."""
    x = _matrix(outer_train_x, "outer_train")
    y = np.asarray(outer_train_y, float)
    calibration_ids = _ids(outer_train_household_ids, "calibration")
    test_ids = _ids(outer_test_household_ids, "outer_test")
    if y.ndim != 1 or len(y) != len(x) or len(calibration_ids) != len(x) or not np.isfinite(y).all():
        raise PredictiveDistributionError("outer_training_shape_or_target_invalid")
    overlap = set(calibration_ids.tolist()) & set(test_ids.tolist())
    if overlap:
        raise PredictiveDistributionError("calibration_outer_test_household_overlap")
    if inner_folds < 2 or inner_folds > len(x):
        raise PredictiveDistributionError("inner_folds_invalid")
    # One row is one household in this primitive. Stable round-robin assignment
    # avoids Python hash randomization and makes the evidence reproducible.
    order = np.argsort(calibration_ids, kind="stable")
    fold = np.empty(len(x), int)
    fold[order] = np.arange(len(x)) % inner_folds
    oof = np.full(len(x), np.nan)
    for held_out in range(inner_folds):
        train, holdout = fold != held_out, fold == held_out
        if not train.any() or not holdout.any():
            raise PredictiveDistributionError("inner_fold_empty")
        model = estimator_factory()
        model.fit(x[train], y[train])
        prediction = np.asarray(model.predict(x[holdout]), float)
        if prediction.shape != (int(holdout.sum()),) or not np.isfinite(prediction).all():
            raise PredictiveDistributionError("inner_oof_prediction_invalid")
        oof[holdout] = prediction
    residuals = y - oof
    outer_model = estimator_factory()
    outer_model.fit(x, y)
    return EmpiricalResidualDistribution(
        estimator=outer_model, residuals=np.sort(residuals),
        calibration_household_ids=tuple(calibration_ids.tolist()),
        outer_test_household_ids=tuple(test_ids.tolist()),
        metadata={"method": "nested_household_oof_empirical_residual_v1", "experimental": True,
                  "production_framework": False, "survey_weights": "not_used",
                  "global_oof_residual_reuse_valid": False, "inner_folds": inner_folds},
    )


# Concise public alias for experiment code.
fit_predictive_distribution = fit_nested_empirical_residual_distribution
