"""Correct one-layer cascade execution with outer-fold-isolated latent fitting.

The first active lean architecture has one parallel categorical latent layer and
a terminal regression head. Final OOF evaluation must isolate each outer held-out
household set from every latent-label fit used to construct that outer fold's
terminal training design. A single global OOF latent matrix is insufficient for
that purpose because meta-training rows from other folds may have been predicted
by latent models that saw the outer held-out labels.

This module implements the smallest correct one-layer runtime. Deeper DAGs remain
a later extension rather than silently inheriting weaker leakage semantics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .crossfit import EstimatorFactory, crossfit_predict, fit_full_and_score
from .dag import prediction_feature_names
from .scientific_primitives import (
    FoldManifest,
    PredictionArtifact,
    ScientificPrimitiveError,
)


class CascadeError(ValueError):
    """Raised when a cascade cannot preserve its leakage/identity contract."""


@dataclass(frozen=True)
class CategoricalLatentSpec:
    """One categorical latent target and its stable class/estimator semantics."""

    target: str
    class_labels: tuple[str, ...]
    estimator_factory: EstimatorFactory

    def __post_init__(self) -> None:
        if not self.target:
            raise CascadeError("latent_target_required")
        if len(self.class_labels) < 2:
            raise CascadeError(f"latent_classes_required:{self.target}")
        if len(set(self.class_labels)) != len(self.class_labels):
            raise CascadeError(f"latent_classes_not_unique:{self.target}")


@dataclass(frozen=True)
class OneLayerOOFResult:
    """OOF latent diagnostics plus leakage-safe terminal OOF welfare prediction."""

    latent_artifacts: Mapping[str, PredictionArtifact]
    terminal_artifact: PredictionArtifact
    terminal_feature_names: tuple[str, ...]
    nesting_policy: str = "outer_fold_isolated_latent_crossfit_v1"


@dataclass(frozen=True)
class OneLayerScoreResult:
    """Final-fit training/scoring latent state and external terminal prediction."""

    training_latent_oof: Mapping[str, PredictionArtifact]
    scoring_latent: Mapping[str, PredictionArtifact]
    terminal_artifact: PredictionArtifact
    terminal_feature_names: tuple[str, ...]
    fitting_policy: str = "oof_meta_train_full_latent_external_score_v1"


def _matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not len(array):
        raise CascadeError(f"{name}_must_be_nonempty_2d")
    if not np.isfinite(array).all():
        raise CascadeError(f"{name}_must_be_finite")
    return array


def _target(value: np.ndarray | Sequence[Any], rows: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows:
        raise CascadeError(f"{name}_length_or_shape_invalid")
    return array


def _validate_manifest_rows(features: np.ndarray, manifest: FoldManifest) -> None:
    if len(features) != len(manifest.row_ids):
        raise CascadeError("feature_manifest_length_mismatch")
    if manifest.n_splits < 3:
        raise CascadeError("one_layer_nested_crossfit_requires_at_least_three_folds")


def _subset_manifest(manifest: FoldManifest, mask: np.ndarray) -> FoldManifest:
    """Restrict a manifest and compact its surviving fold labels deterministically."""
    if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(manifest.row_ids):
        raise CascadeError("subset_manifest_mask_invalid")
    source_folds = np.asarray(manifest.fold_ids, dtype=int)[mask]
    surviving = sorted(set(source_folds.tolist()))
    if len(surviving) < 2:
        raise CascadeError("nested_training_requires_two_inner_folds")
    remap = {fold: index for index, fold in enumerate(surviving)}
    compact = tuple(remap[int(fold)] for fold in source_folds)
    row_ids = tuple(
        row_id
        for row_id, keep in zip(manifest.row_ids, mask, strict=True)
        if keep
    )
    household_ids = tuple(
        household
        for household, keep in zip(manifest.household_ids, mask, strict=True)
        if keep
    )
    try:
        return FoldManifest(
            row_ids=row_ids,
            household_ids=household_ids,
            fold_ids=compact,
            n_splits=len(surviving),
            policy=manifest.policy,
        )
    except ScientificPrimitiveError as exc:
        raise CascadeError(f"invalid_nested_fold_manifest:{exc}") from exc


def _latent_feature_names(
    artifacts: Sequence[PredictionArtifact],
) -> tuple[str, ...]:
    output: list[str] = []
    for artifact in artifacts:
        output.extend(prediction_feature_names(artifact))
    if len(set(output)) != len(output):
        raise CascadeError("duplicate_latent_feature_name")
    return tuple(output)


def _validate_latent_targets(
    latent_targets: Mapping[str, np.ndarray | Sequence[Any]],
    latent_specs: Sequence[CategoricalLatentSpec],
    row_count: int,
) -> dict[str, np.ndarray]:
    if not latent_specs:
        raise CascadeError("lean_cascade_requires_latent_specs")
    if len({spec.target for spec in latent_specs}) != len(latent_specs):
        raise CascadeError("duplicate_latent_target_spec")
    expected = {spec.target for spec in latent_specs}
    if set(latent_targets) != expected:
        raise CascadeError("latent_target_mapping_mismatch")
    return {
        spec.target: _target(
            latent_targets[spec.target],
            row_count,
            f"latent_target:{spec.target}",
        )
        for spec in latent_specs
    }


def run_one_layer_oof(
    base_features: np.ndarray | Sequence[Sequence[float]],
    terminal_target: np.ndarray | Sequence[float],
    latent_targets: Mapping[str, np.ndarray | Sequence[Any]],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_estimator_factory: EstimatorFactory,
    *,
    terminal_target_name: str,
    base_feature_names: Sequence[str] | None = None,
    run_id: str | None = None,
) -> OneLayerOOFResult:
    """Evaluate a one-layer cascade without outer-fold latent-label leakage."""
    features = _matrix(base_features, "base_features")
    _validate_manifest_rows(features, manifest)
    y_terminal = _target(terminal_target, len(features), "terminal_target")
    y_latent = _validate_latent_targets(latent_targets, latent_specs, len(features))

    if base_feature_names is not None and len(base_feature_names) != features.shape[1]:
        raise CascadeError("base_feature_name_width_mismatch")
    names = tuple(base_feature_names or (f"base[{i}]" for i in range(features.shape[1])))

    terminal_values = np.full(len(features), np.nan, dtype=float)
    global_latent_values = {
        spec.target: np.full((len(features), len(spec.class_labels)), np.nan, dtype=float)
        for spec in latent_specs
    }
    fold_ids = manifest.as_array()
    latent_names: tuple[str, ...] | None = None

    for outer_fold in range(manifest.n_splits):
        train_mask = fold_ids != outer_fold
        holdout_mask = fold_ids == outer_fold
        if not np.any(train_mask) or not np.any(holdout_mask):
            raise CascadeError(f"invalid_outer_fold_partition:{outer_fold}")

        nested_manifest = _subset_manifest(manifest, train_mask)
        train_blocks: list[np.ndarray] = []
        holdout_blocks: list[np.ndarray] = []
        representative_artifacts: list[PredictionArtifact] = []

        for spec in latent_specs:
            nested_oof = crossfit_predict(
                features[train_mask],
                y_latent[spec.target][train_mask],
                nested_manifest,
                spec.estimator_factory,
                target=spec.target,
                kind="probability",
                class_labels=spec.class_labels,
                run_id=run_id,
                metadata={
                    "cascade_role": "nested_meta_training_latent",
                    "outer_fold": outer_fold,
                },
            )
            outer_score = fit_full_and_score(
                features[train_mask],
                y_latent[spec.target][train_mask],
                features[holdout_mask],
                tuple(
                    row_id
                    for row_id, held_out in zip(
                        manifest.row_ids,
                        holdout_mask,
                        strict=True,
                    )
                    if held_out
                ),
                spec.estimator_factory,
                target=spec.target,
                kind="probability",
                class_labels=spec.class_labels,
                run_id=run_id,
                metadata={
                    "cascade_role": "outer_holdout_latent",
                    "outer_fold": outer_fold,
                },
            )
            train_blocks.append(nested_oof.values)
            holdout_blocks.append(outer_score.values)
            global_latent_values[spec.target][holdout_mask] = outer_score.values
            representative_artifacts.append(outer_score)

        current_latent_names = _latent_feature_names(representative_artifacts)
        if latent_names is None:
            latent_names = current_latent_names
        elif latent_names != current_latent_names:
            raise CascadeError("latent_feature_name_drift_across_outer_folds")

        terminal_train_x = np.column_stack([features[train_mask], *train_blocks])
        terminal_holdout_x = np.column_stack([features[holdout_mask], *holdout_blocks])
        terminal_score = fit_full_and_score(
            terminal_train_x,
            y_terminal[train_mask],
            terminal_holdout_x,
            tuple(
                row_id
                for row_id, held_out in zip(
                    manifest.row_ids,
                    holdout_mask,
                    strict=True,
                )
                if held_out
            ),
            terminal_estimator_factory,
            target=terminal_target_name,
            kind="regression",
            run_id=run_id,
            metadata={
                "architecture": "one_layer_cascade",
                "outer_fold": outer_fold,
                "latent_training": "nested_oof_without_outer_holdout_rows",
            },
        )
        terminal_values[holdout_mask] = terminal_score.values

    if latent_names is None:
        raise CascadeError("latent_feature_names_missing")
    if not np.isfinite(terminal_values).all():
        raise CascadeError("terminal_oof_incomplete")

    latent_artifacts: dict[str, PredictionArtifact] = {}
    for spec in latent_specs:
        values = global_latent_values[spec.target]
        if not np.isfinite(values).all():
            raise CascadeError(f"latent_oof_incomplete:{spec.target}")
        latent_artifacts[spec.target] = PredictionArtifact(
            target=spec.target,
            kind="probability",
            row_ids=manifest.row_ids,
            values=values,
            source="oof",
            class_labels=spec.class_labels,
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "cascade_role": "outer_fold_latent_diagnostic",
                "nesting_policy": "fit_without_outer_holdout_rows",
            },
        )

    terminal_artifact = PredictionArtifact(
        target=terminal_target_name,
        kind="regression",
        row_ids=manifest.row_ids,
        values=terminal_values,
        source="oof",
        fold_ids=manifest.fold_ids,
        run_id=run_id,
        metadata={
            "architecture": "one_layer_cascade",
            "nesting_policy": "outer_fold_isolated_latent_crossfit_v1",
        },
    )
    return OneLayerOOFResult(
        latent_artifacts=latent_artifacts,
        terminal_artifact=terminal_artifact,
        terminal_feature_names=(*names, *latent_names),
    )


def fit_one_layer_and_score(
    base_features: np.ndarray | Sequence[Sequence[float]],
    terminal_target: np.ndarray | Sequence[float],
    latent_targets: Mapping[str, np.ndarray | Sequence[Any]],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_estimator_factory: EstimatorFactory,
    scoring_features: np.ndarray | Sequence[Sequence[float]],
    scoring_row_ids: Sequence[str],
    *,
    terminal_target_name: str,
    base_feature_names: Sequence[str] | None = None,
    run_id: str | None = None,
) -> OneLayerScoreResult:
    """Fit deployable one-layer state and score a distinct external person frame."""
    train_x = _matrix(base_features, "base_features")
    score_x = _matrix(scoring_features, "scoring_features")
    _validate_manifest_rows(train_x, manifest)
    if train_x.shape[1] != score_x.shape[1]:
        raise CascadeError("training_scoring_base_feature_width_mismatch")
    if len(score_x) != len(scoring_row_ids):
        raise CascadeError("scoring_row_id_length_mismatch")
    y_terminal = _target(terminal_target, len(train_x), "terminal_target")
    y_latent = _validate_latent_targets(latent_targets, latent_specs, len(train_x))
    if base_feature_names is not None and len(base_feature_names) != train_x.shape[1]:
        raise CascadeError("base_feature_name_width_mismatch")
    names = tuple(base_feature_names or (f"base[{i}]" for i in range(train_x.shape[1])))

    training_latent: dict[str, PredictionArtifact] = {}
    scoring_latent: dict[str, PredictionArtifact] = {}
    training_blocks: list[np.ndarray] = []
    scoring_blocks: list[np.ndarray] = []
    ordered_training_artifacts: list[PredictionArtifact] = []

    for spec in latent_specs:
        oof = crossfit_predict(
            train_x,
            y_latent[spec.target],
            manifest,
            spec.estimator_factory,
            target=spec.target,
            kind="probability",
            class_labels=spec.class_labels,
            run_id=run_id,
            metadata={"cascade_role": "full_meta_training_latent_oof"},
        )
        scored = fit_full_and_score(
            train_x,
            y_latent[spec.target],
            score_x,
            scoring_row_ids,
            spec.estimator_factory,
            target=spec.target,
            kind="probability",
            class_labels=spec.class_labels,
            run_id=run_id,
            metadata={"cascade_role": "full_latent_external_score"},
        )
        training_latent[spec.target] = oof
        scoring_latent[spec.target] = scored
        training_blocks.append(oof.values)
        scoring_blocks.append(scored.values)
        ordered_training_artifacts.append(oof)

    latent_names = _latent_feature_names(ordered_training_artifacts)
    terminal_train_x = np.column_stack([train_x, *training_blocks])
    terminal_score_x = np.column_stack([score_x, *scoring_blocks])
    terminal_artifact = fit_full_and_score(
        terminal_train_x,
        y_terminal,
        terminal_score_x,
        scoring_row_ids,
        terminal_estimator_factory,
        target=terminal_target_name,
        kind="regression",
        run_id=run_id,
        metadata={
            "architecture": "one_layer_cascade",
            "latent_training": "global_oof",
            "latent_scoring": "full_fit",
        },
    )
    return OneLayerScoreResult(
        training_latent_oof=training_latent,
        scoring_latent=scoring_latent,
        terminal_artifact=terminal_artifact,
        terminal_feature_names=(*names, *latent_names),
    )
