"""Reusable scientific primitives for household-safe welfare experiments.

This module is deliberately small. It owns identity, deterministic household
fold assignment, typed prediction artifacts, strict row alignment, household
aggregation, and basic regression metrics. It does not own estimators, DAG
execution, experiment resolution, or legacy compatibility.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HOUSEHOLD_COLUMNS = ("CODUSU", "NRO_HOGAR")
PERSON_COLUMNS = ("CODUSU", "NRO_HOGAR", "COMPONENTE")
FOLD_POLICY = "household_grouped_v1"


class ScientificPrimitiveError(ValueError):
    """Raised when a reusable scientific invariant is violated."""


def _identity_value(row: Mapping[str, Any], column: str) -> str:
    if column not in row:
        raise ScientificPrimitiveError(f"missing_identity_column:{column}")
    value = row[column]
    if value is None:
        raise ScientificPrimitiveError(f"null_identity_value:{column}")
    return str(value)


def _encode_identity(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    return "\x1f".join(_identity_value(row, column) for column in columns)


def household_id(
    row: Mapping[str, Any],
    columns: Sequence[str] = HOUSEHOLD_COLUMNS,
) -> str:
    """Return the canonical household identity used for grouped splitting."""
    return _encode_identity(row, columns)


def person_id(
    row: Mapping[str, Any],
    columns: Sequence[str] = PERSON_COLUMNS,
) -> str:
    """Return the canonical person/row identity used by prediction artifacts."""
    return _encode_identity(row, columns)


def _household_fold(encoded_household_id: str, n_splits: int) -> int:
    digest = hashlib.sha256(encoded_household_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_splits


@dataclass(frozen=True)
class FoldManifest:
    """Immutable row-level fold assignment with household grouping semantics."""

    row_ids: tuple[str, ...]
    household_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    n_splits: int
    policy: str = FOLD_POLICY

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ScientificPrimitiveError("n_splits_must_be_at_least_two")
        if not self.row_ids:
            raise ScientificPrimitiveError("fold_manifest_requires_rows")
        if not (
            len(self.row_ids) == len(self.household_ids) == len(self.fold_ids)
        ):
            raise ScientificPrimitiveError("fold_manifest_length_mismatch")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ScientificPrimitiveError("duplicate_person_identity")
        if any(fold < 0 or fold >= self.n_splits for fold in self.fold_ids):
            raise ScientificPrimitiveError("fold_id_out_of_range")

        household_folds: dict[str, set[int]] = defaultdict(set)
        for household, fold in zip(
            self.household_ids,
            self.fold_ids,
            strict=True,
        ):
            household_folds[household].add(fold)
        if any(len(folds) != 1 for folds in household_folds.values()):
            raise ScientificPrimitiveError("household_crosses_folds")

        used_folds = set(self.fold_ids)
        if used_folds != set(range(self.n_splits)):
            raise ScientificPrimitiveError("fold_assignment_has_empty_fold")

    def as_array(self) -> np.ndarray:
        """Return fold IDs as an integer NumPy array for estimator plumbing."""
        return np.asarray(self.fold_ids, dtype=int)

    def mapping(self) -> dict[str, int]:
        """Return a row-id to fold-id mapping for audit and comparison."""
        return dict(zip(self.row_ids, self.fold_ids, strict=True))


def build_fold_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
) -> FoldManifest:
    """Assign every person to a deterministic household-safe fold."""
    if n_splits < 2:
        raise ScientificPrimitiveError("n_splits_must_be_at_least_two")

    row_ids = tuple(person_id(row) for row in rows)
    household_ids = tuple(household_id(row) for row in rows)
    fold_ids = tuple(_household_fold(value, n_splits) for value in household_ids)
    return FoldManifest(
        row_ids=row_ids,
        household_ids=household_ids,
        fold_ids=fold_ids,
        n_splits=n_splits,
    )


@dataclass
class PredictionArtifact:
    """Typed person-level prediction artifact with explicit row semantics."""

    target: str
    kind: str
    row_ids: tuple[str, ...]
    values: np.ndarray
    source: str
    class_labels: tuple[str, ...] = ()
    fold_ids: tuple[int, ...] | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise ScientificPrimitiveError("prediction_target_required")
        if self.kind not in {"regression", "probability"}:
            raise ScientificPrimitiveError(f"unknown_prediction_kind:{self.kind}")
        if self.source not in {"oof", "score", "oracle_diagnostic"}:
            raise ScientificPrimitiveError(f"unknown_prediction_source:{self.source}")
        if not self.row_ids:
            raise ScientificPrimitiveError("prediction_artifact_requires_rows")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ScientificPrimitiveError("duplicate_prediction_row_id")

        values = np.asarray(self.values, dtype=float)
        if values.shape[0] != len(self.row_ids):
            raise ScientificPrimitiveError("prediction_length_mismatch")
        if not np.isfinite(values).all():
            raise ScientificPrimitiveError("non_finite_prediction")

        if self.kind == "regression":
            if values.ndim != 1:
                raise ScientificPrimitiveError("regression_prediction_must_be_1d")
            if self.class_labels:
                raise ScientificPrimitiveError("regression_prediction_has_class_labels")
        else:
            if values.ndim != 2:
                raise ScientificPrimitiveError("probability_prediction_must_be_2d")
            if values.shape[1] < 2:
                raise ScientificPrimitiveError("probability_prediction_needs_two_classes")
            if len(self.class_labels) != values.shape[1]:
                raise ScientificPrimitiveError("probability_class_metadata_mismatch")
            if len(set(self.class_labels)) != len(self.class_labels):
                raise ScientificPrimitiveError("duplicate_probability_class_label")
            tolerance = 1e-10
            if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
                raise ScientificPrimitiveError("probability_out_of_bounds")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
                raise ScientificPrimitiveError("probability_rows_must_sum_to_one")

        if self.fold_ids is not None:
            if len(self.fold_ids) != len(self.row_ids):
                raise ScientificPrimitiveError("prediction_fold_length_mismatch")
            if any(fold < 0 for fold in self.fold_ids):
                raise ScientificPrimitiveError("prediction_fold_id_negative")
            if self.source != "oof":
                raise ScientificPrimitiveError("fold_ids_only_valid_for_oof_predictions")

        self.values = values

    def validate_alignment(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Require exact row identity and order equality with the source frame."""
        expected = tuple(person_id(row) for row in rows)
        if self.row_ids == expected:
            return
        if set(self.row_ids) == set(expected):
            raise ScientificPrimitiveError("prediction_row_order_mismatch")
        raise ScientificPrimitiveError("prediction_row_identity_mismatch")


def aggregate_person_predictions(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, PredictionArtifact],
    *,
    truth_field: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate complete aligned person predictions to household linear sums.

    Incomplete or misaligned person membership fails explicitly before any
    household total is emitted; missing members are never silently dropped.
    """
    if not rows:
        raise ScientificPrimitiveError("household_aggregation_requires_rows")
    if not artifacts:
        raise ScientificPrimitiveError("household_aggregation_requires_predictions")

    for artifact in artifacts.values():
        artifact.validate_alignment(rows)
        if artifact.kind != "regression":
            raise ScientificPrimitiveError("household_aggregation_requires_regression")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[household_id(row)].append(index)

    output: list[dict[str, Any]] = []
    for encoded_household in sorted(grouped):
        indices = grouped[encoded_household]
        first = rows[indices[0]]
        record: dict[str, Any] = {
            "CODUSU": _identity_value(first, "CODUSU"),
            "NRO_HOGAR": _identity_value(first, "NRO_HOGAR"),
            "member_count": len(indices),
        }
        if truth_field is not None:
            try:
                truth_values = np.asarray(
                    [float(rows[index][truth_field]) for index in indices],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ScientificPrimitiveError(
                    f"invalid_household_truth:{truth_field}"
                ) from exc
            if not np.isfinite(truth_values).all():
                raise ScientificPrimitiveError(f"non_finite_household_truth:{truth_field}")
            record["truth_household_income"] = float(truth_values.sum())

        for artifact_id, artifact in artifacts.items():
            record[f"{artifact_id}_household_income"] = float(
                artifact.values[indices].sum()
            )
        record["status"] = "complete"
        output.append(record)
    return output


def regression_metrics(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return the minimal linear-scale regression metrics used by M1 plumbing."""
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if truth_array.ndim != 1 or prediction_array.ndim != 1:
        raise ScientificPrimitiveError("regression_metrics_require_1d")
    if len(truth_array) != len(prediction_array) or not len(truth_array):
        raise ScientificPrimitiveError("metric_length_error")
    if not np.isfinite(truth_array).all() or not np.isfinite(prediction_array).all():
        raise ScientificPrimitiveError("metrics_require_finite_values")

    residual = prediction_array - truth_array
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_error": float(np.mean(residual)),
    }
