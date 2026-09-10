"""Generic household-safe cross-fitting independent of estimator family.

The functions here consume an explicit :class:`FoldManifest`; they never choose
or mutate outer splits themselves. Estimator-specific construction belongs in
adapter modules. This keeps household grouping as data/evidence rather than a
hidden side effect of a model API.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .scientific_primitives import (
    FoldManifest,
    PredictionArtifact,
    ScientificPrimitiveError,
)


class CrossfitError(ValueError):
    """Raised when cross-fitting cannot preserve the declared experiment contract."""


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Minimal estimator surface used by the generic runtime."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


EstimatorFactory = Callable[[], EstimatorProtocol]


def _matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise CrossfitError(f"{name}_must_be_2d")
    if not len(array):
        raise CrossfitError(f"{name}_must_not_be_empty")
    # Missing feature values are estimator policy. HGB supports NaN natively;
    # infinity is never a meaningful missing-value encoding.
    if np.isinf(array).any():
        raise CrossfitError(f"{name}_contains_infinite_value")
    return array


def _target(value: np.ndarray | Sequence[Any], expected_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise CrossfitError("target_must_be_1d")
    if len(array) != expected_rows:
        raise CrossfitError("target_length_mismatch")
    return array


def _validate_training_shape(
    x: np.ndarray,
    y: np.ndarray,
    manifest: FoldManifest,
) -> None:
    if len(x) != len(manifest.row_ids):
        raise CrossfitError("feature_manifest_length_mismatch")
    if len(y) != len(x):
        raise CrossfitError("target_feature_length_mismatch")


def _probability_columns(
    estimator: EstimatorProtocol,
    x: np.ndarray,
    class_labels: tuple[str, ...],
) -> np.ndarray:
    predict_proba = getattr(estimator, "predict_proba", None)
    classes = getattr(estimator, "classes_", None)
    if not callable(predict_proba) or classes is None:
        raise CrossfitError("probability_estimator_interface_missing")

    raw = np.asarray(predict_proba(x), dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(x):
        raise CrossfitError("probability_prediction_shape_invalid")

    estimator_labels = tuple(str(value) for value in np.asarray(classes).tolist())
    if len(estimator_labels) != raw.shape[1]:
        raise CrossfitError("estimator_class_metadata_mismatch")
    if len(set(estimator_labels)) != len(estimator_labels):
        raise CrossfitError("estimator_class_labels_not_unique")
    if set(estimator_labels) != set(class_labels):
        raise CrossfitError("estimator_class_set_mismatch")

    column_by_label = {label: index for index, label in enumerate(estimator_labels)}
    return raw[:, [column_by_label[label] for label in class_labels]]


def crossfit_predict(
    x: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Any],
    manifest: FoldManifest,
    estimator_factory: EstimatorFactory,
    *,
    target: str,
    kind: str = "regression",
    class_labels: Sequence[Any] | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PredictionArtifact:
    """Generate one honest OOF prediction for every row in ``manifest``.

    Each fold model is fitted only on rows whose manifest fold differs from the
    held-out fold. The function does not invoke sklearn CV or any estimator
    internal splitting surface.
    """
    features = _matrix(x, "features")
    target_values = _target(y, len(features))
    _validate_training_shape(features, target_values, manifest)

    if kind not in {"regression", "probability"}:
        raise CrossfitError(f"unknown_prediction_kind:{kind}")

    labels = tuple(str(value) for value in class_labels or ())
    if kind == "probability" and len(labels) < 2:
        raise CrossfitError("probability_class_labels_required")
    if kind == "regression" and labels:
        raise CrossfitError("regression_crossfit_has_class_labels")

    if kind == "regression":
        prediction = np.full(len(features), np.nan, dtype=float)
    else:
        prediction = np.full((len(features), len(labels)), np.nan, dtype=float)

    fold_ids = manifest.as_array()
    for fold in range(manifest.n_splits):
        train_mask = fold_ids != fold
        holdout_mask = fold_ids == fold
        if not np.any(train_mask) or not np.any(holdout_mask):
            raise CrossfitError(f"invalid_fold_partition:{fold}")

        estimator = estimator_factory()
        if not isinstance(estimator, EstimatorProtocol):
            raise CrossfitError("estimator_protocol_not_satisfied")
        estimator.fit(features[train_mask], target_values[train_mask])

        if kind == "regression":
            fold_prediction = np.asarray(
                estimator.predict(features[holdout_mask]),
                dtype=float,
            )
            if fold_prediction.ndim != 1 or len(fold_prediction) != int(
                holdout_mask.sum()
            ):
                raise CrossfitError("regression_prediction_shape_invalid")
            prediction[holdout_mask] = fold_prediction
        else:
            prediction[holdout_mask] = _probability_columns(
                estimator,
                features[holdout_mask],
                labels,
            )

    if not np.isfinite(prediction).all():
        raise CrossfitError("oof_prediction_incomplete_or_non_finite")

    try:
        return PredictionArtifact(
            target=target,
            kind=kind,
            row_ids=manifest.row_ids,
            values=prediction,
            source="oof",
            class_labels=labels,
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )
    except ScientificPrimitiveError as exc:
        raise CrossfitError(f"invalid_prediction_artifact:{exc}") from exc


def fit_full_and_score(
    x_train: np.ndarray | Sequence[Sequence[float]],
    y_train: np.ndarray | Sequence[Any],
    x_score: np.ndarray | Sequence[Sequence[float]],
    score_row_ids: Sequence[str],
    estimator_factory: EstimatorFactory,
    *,
    target: str,
    kind: str = "regression",
    class_labels: Sequence[Any] | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PredictionArtifact:
    """Fit once on all training rows and score a distinct declared row frame."""
    train_features = _matrix(x_train, "training_features")
    score_features = _matrix(x_score, "scoring_features")
    target_values = _target(y_train, len(train_features))
    if train_features.shape[1] != score_features.shape[1]:
        raise CrossfitError("training_scoring_feature_width_mismatch")
    if len(score_features) != len(score_row_ids):
        raise CrossfitError("scoring_row_id_length_mismatch")

    labels = tuple(str(value) for value in class_labels or ())
    if kind == "probability" and len(labels) < 2:
        raise CrossfitError("probability_class_labels_required")
    if kind == "regression" and labels:
        raise CrossfitError("regression_score_has_class_labels")
    if kind not in {"regression", "probability"}:
        raise CrossfitError(f"unknown_prediction_kind:{kind}")

    estimator = estimator_factory()
    if not isinstance(estimator, EstimatorProtocol):
        raise CrossfitError("estimator_protocol_not_satisfied")
    estimator.fit(train_features, target_values)

    if kind == "regression":
        prediction = np.asarray(estimator.predict(score_features), dtype=float)
        if prediction.ndim != 1 or len(prediction) != len(score_features):
            raise CrossfitError("regression_prediction_shape_invalid")
    else:
        prediction = _probability_columns(estimator, score_features, labels)

    try:
        return PredictionArtifact(
            target=target,
            kind=kind,
            row_ids=tuple(score_row_ids),
            values=prediction,
            source="score",
            class_labels=labels,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )
    except ScientificPrimitiveError as exc:
        raise CrossfitError(f"invalid_prediction_artifact:{exc}") from exc
