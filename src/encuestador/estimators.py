"""Narrow sklearn estimator adapters for the active welfare experiment path.

HistGradientBoosting is the primary estimator family. These adapters make two
scientific choices explicit: categorical feature positions and the prohibition
on hidden estimator-level early-stopping splits. Outer validation remains owned
by the household-safe experiment runtime.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


class EstimatorConfigurationError(ValueError):
    """Raised when an estimator configuration violates the starter policy."""


def _categorical_positions(values: Sequence[int] | None) -> tuple[int, ...]:
    positions = tuple(int(value) for value in (values or ()))
    if any(value < 0 for value in positions):
        raise EstimatorConfigurationError("categorical_feature_index_negative")
    if len(set(positions)) != len(positions):
        raise EstimatorConfigurationError("categorical_feature_index_duplicate")
    return positions


def _base_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = dict(parameters or {})
    requested_early_stopping = resolved.pop("early_stopping", False)
    if requested_early_stopping is not False:
        raise EstimatorConfigurationError(
            "hidden_early_stopping_forbidden_without_household_safe_validation"
        )
    if "categorical_features" in resolved:
        raise EstimatorConfigurationError(
            "categorical_features_must_use_explicit_adapter_argument"
        )
    resolved["early_stopping"] = False
    resolved.setdefault("random_state", 0)
    return resolved


def _natural_classifier_target(value: np.ndarray) -> np.ndarray:
    """Canonicalize transport labels to a homogeneous discrete sklearn dtype."""
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1:
        return raw
    labels: list[str] = []
    for item in raw.tolist():
        if item is None:
            raise EstimatorConfigurationError("classifier_target_missing")
        if isinstance(item, (float, np.floating)) and np.isnan(float(item)):
            raise EstimatorConfigurationError("classifier_target_missing")
        label = str(item)
        if not label or label.lower() == "nan":
            raise EstimatorConfigurationError("classifier_target_missing")
        labels.append(label)
    return np.asarray(labels, dtype=str)


def _validate_hgb_features(features: np.ndarray, *, error: str) -> None:
    """HGB natively supports NaN feature values; infinities remain invalid."""
    if np.isinf(features).any():
        raise EstimatorConfigurationError(error)


class HGBClassifierAdapter:
    """HistGradientBoosting classifier with explicit probability semantics."""

    def __init__(
        self,
        *,
        categorical_features: Sequence[int] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.categorical_features = _categorical_positions(categorical_features)
        self.parameters = _base_parameters(parameters)
        self.model: HistGradientBoostingClassifier | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> HGBClassifierAdapter:
        features = np.asarray(x, dtype=float)
        target = _natural_classifier_target(y)
        if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
            raise EstimatorConfigurationError("classifier_training_shape_invalid")
        if not len(features):
            raise EstimatorConfigurationError("classifier_training_empty")
        _validate_hgb_features(features, error="classifier_features_infinite")
        if len(np.unique(target)) < 2:
            raise EstimatorConfigurationError("classifier_requires_two_classes")
        if self.categorical_features and max(self.categorical_features) >= features.shape[1]:
            raise EstimatorConfigurationError("categorical_feature_index_out_of_range")

        self.model = HistGradientBoostingClassifier(
            categorical_features=(
                list(self.categorical_features) if self.categorical_features else None
            ),
            **self.parameters,
        )
        self.model.fit(features, target)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def _fitted(self) -> HistGradientBoostingClassifier:
        if self.model is None or self.classes_ is None:
            raise EstimatorConfigurationError("classifier_not_fitted")
        return self.model

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        _validate_hgb_features(features, error="classifier_features_infinite")
        return np.asarray(self._fitted().predict(features))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        _validate_hgb_features(features, error="classifier_features_infinite")
        return np.asarray(self._fitted().predict_proba(features), dtype=float)


class HGBRegressorAdapter:
    """HistGradientBoosting regressor for squared-error or positive Gamma targets."""

    ALLOWED_LOSSES: ClassVar[frozenset[str]] = frozenset({"squared_error", "gamma"})

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        categorical_features: Sequence[int] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if loss not in self.ALLOWED_LOSSES:
            raise EstimatorConfigurationError(f"unsupported_hgb_regression_loss:{loss}")
        resolved = _base_parameters(parameters)
        if "loss" in resolved:
            raise EstimatorConfigurationError("loss_must_use_explicit_adapter_argument")
        self.loss = loss
        self.categorical_features = _categorical_positions(categorical_features)
        self.parameters = resolved
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> HGBRegressorAdapter:
        features = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
            raise EstimatorConfigurationError("regressor_training_shape_invalid")
        if not len(features):
            raise EstimatorConfigurationError("regressor_training_empty")
        _validate_hgb_features(features, error="regressor_features_infinite")
        if not np.isfinite(target).all():
            raise EstimatorConfigurationError("regressor_target_non_finite")
        if self.loss == "gamma" and np.any(target <= 0):
            raise EstimatorConfigurationError("gamma_target_must_be_strictly_positive")
        if self.categorical_features and max(self.categorical_features) >= features.shape[1]:
            raise EstimatorConfigurationError("categorical_feature_index_out_of_range")

        self.model = HistGradientBoostingRegressor(
            loss=self.loss,
            categorical_features=(
                list(self.categorical_features) if self.categorical_features else None
            ),
            **self.parameters,
        )
        self.model.fit(features, target)
        return self

    def _fitted(self) -> HistGradientBoostingRegressor:
        if self.model is None:
            raise EstimatorConfigurationError("regressor_not_fitted")
        return self.model

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        _validate_hgb_features(features, error="regressor_features_infinite")
        return np.asarray(self._fitted().predict(features), dtype=float)
