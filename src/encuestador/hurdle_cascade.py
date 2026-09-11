"""Leakage-safe one-layer lean cascade with an explicit hurdle terminal head.

This module deliberately reuses the established latent nesting semantics from
`cascade.py`: for every outer terminal fold, latent meta-training predictions are
rebuilt using only outer-training households, while latent predictions for the
outer holdout come from models fitted on that same outer-training population.
The only changed seam is the terminal head, which emits presence, positive amount,
and unconditional linear welfare rather than one opaque regression point.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .cascade import (
    CategoricalLatentSpec,
    _latent_feature_names,
    _matrix,
    _subset_manifest,
    _validate_latent_targets,
    _validate_manifest_rows,
)
from .crossfit import crossfit_predict, fit_full_and_score
from .scientific_primitives import FoldManifest, PredictionArtifact
from .terminal import (
    HurdleComponents,
    HurdleEstimator,
    HurdlePredictionBundle,
    classify_income_target,
)


class HurdleCascadeError(ValueError):
    """Raised when lean hurdle execution would violate the nesting contract."""


@dataclass(frozen=True)
class OneLayerHurdleOOFResult:
    latent_artifacts: Mapping[str, PredictionArtifact]
    hurdle: HurdlePredictionBundle
    terminal_feature_names: tuple[str, ...]
    nesting_policy: str = "outer_fold_isolated_latent_crossfit_v1"


@dataclass(frozen=True)
class OneLayerHurdleScoreResult:
    training_latent_oof: Mapping[str, PredictionArtifact]
    scoring_latent: Mapping[str, PredictionArtifact]
    hurdle: HurdlePredictionBundle
    terminal_feature_names: tuple[str, ...]
    fitting_policy: str = "oof_meta_train_full_latent_external_score_v1"


def _make_probability_artifact(
    *,
    target: str,
    row_ids: tuple[str, ...],
    values: np.ndarray,
    class_labels: tuple[str, ...],
    fold_ids: tuple[int, ...],
    run_id: str | None,
) -> PredictionArtifact:
    return PredictionArtifact(
        target=target,
        kind="probability",
        row_ids=row_ids,
        values=values,
        source="oof",
        class_labels=class_labels,
        fold_ids=fold_ids,
        run_id=run_id,
        metadata={
            "cascade_role": "outer_fold_latent_diagnostic",
            "nesting_policy": "fit_without_outer_holdout_rows",
        },
    )


def _make_hurdle_bundle(
    *,
    target: str,
    row_ids: tuple[str, ...],
    p_positive: np.ndarray,
    positive_amount: np.ndarray,
    unconditional: np.ndarray,
    source: str,
    formulation: str,
    retransformation: str,
    target_eligibility: Mapping[str, int],
    run_id: str | None,
    fold_ids: tuple[int, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> HurdlePredictionBundle:
    meta = {
        "terminal_formulation": f"hurdle_{formulation}",
        "retransformation": retransformation,
        **dict(metadata or {}),
    }
    probability = np.column_stack([1.0 - p_positive, p_positive])
    return HurdlePredictionBundle(
        p_positive=PredictionArtifact(
            target=f"{target}__positive",
            kind="probability",
            row_ids=row_ids,
            values=probability,
            source=source,
            class_labels=("0", "1"),
            fold_ids=fold_ids,
            run_id=run_id,
            metadata={**meta, "component": "p_positive"},
        ),
        positive_amount_prediction=PredictionArtifact(
            target=f"{target}__positive_amount",
            kind="regression",
            row_ids=row_ids,
            values=positive_amount,
            source=source,
            fold_ids=fold_ids,
            run_id=run_id,
            metadata={**meta, "component": "positive_amount_prediction"},
        ),
        unconditional_expected_income=PredictionArtifact(
            target=target,
            kind="regression",
            row_ids=row_ids,
            values=unconditional,
            source=source,
            fold_ids=fold_ids,
            run_id=run_id,
            metadata={**meta, "component": "unconditional_expected_income"},
        ),
        target_eligibility=dict(target_eligibility),
        formulation=formulation,
        retransformation=retransformation,
    )


def run_one_layer_hurdle_oof(
    base_features: np.ndarray | Sequence[Sequence[float]],
    terminal_target: np.ndarray | Sequence[object],
    latent_targets: Mapping[str, np.ndarray | Sequence[object]],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_estimator_factory: callable,
    *,
    terminal_target_name: str,
    base_feature_names: Sequence[str] | None = None,
    run_id: str | None = None,
) -> OneLayerHurdleOOFResult:
    """Evaluate lean latent probabilities feeding an outer-isolated hurdle head."""
    features = _matrix(base_features, "base_features")
    _validate_manifest_rows(features, manifest)
    raw_terminal = np.asarray(terminal_target, dtype=object)
    if raw_terminal.ndim != 1 or len(raw_terminal) != len(features):
        raise HurdleCascadeError("terminal_target_length_or_shape_invalid")
    target_eligibility = classify_income_target(raw_terminal)
    y_latent = _validate_latent_targets(latent_targets, latent_specs, len(features))

    if base_feature_names is not None and len(base_feature_names) != features.shape[1]:
        raise HurdleCascadeError("base_feature_name_width_mismatch")
    names = tuple(base_feature_names or (f"base[{i}]" for i in range(features.shape[1])))

    p_positive = np.full(len(features), np.nan, dtype=float)
    positive_amount = np.full(len(features), np.nan, dtype=float)
    unconditional = np.full(len(features), np.nan, dtype=float)
    global_latent_values = {
        spec.target: np.full((len(features), len(spec.class_labels)), np.nan, dtype=float)
        for spec in latent_specs
    }
    fold_ids = manifest.as_array()
    latent_names: tuple[str, ...] | None = None
    formulation: str | None = None
    retransformation: str | None = None

    for outer_fold in range(manifest.n_splits):
        train_mask = fold_ids != outer_fold
        holdout_mask = fold_ids == outer_fold
        if not np.any(train_mask) or not np.any(holdout_mask):
            raise HurdleCascadeError(f"invalid_outer_fold_partition:{outer_fold}")

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
            holdout_row_ids = tuple(
                row_id
                for row_id, held_out in zip(manifest.row_ids, holdout_mask, strict=True)
                if held_out
            )
            outer_score = fit_full_and_score(
                features[train_mask],
                y_latent[spec.target][train_mask],
                features[holdout_mask],
                holdout_row_ids,
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
            raise HurdleCascadeError("latent_feature_name_drift_across_outer_folds")

        terminal_train_x = np.column_stack([features[train_mask], *train_blocks])
        terminal_holdout_x = np.column_stack([features[holdout_mask], *holdout_blocks])
        estimator = terminal_estimator_factory()
        if not isinstance(estimator, HurdleEstimator):
            raise HurdleCascadeError("terminal_factory_must_return_hurdle_estimator")
        estimator.fit(terminal_train_x, raw_terminal[train_mask])
        components = estimator.predict_components(terminal_holdout_x)
        p_positive[holdout_mask] = components.p_positive
        positive_amount[holdout_mask] = components.positive_amount
        unconditional[holdout_mask] = components.unconditional_income
        current_retransform = (
            estimator.retransformation
            if estimator.formulation == "log"
            else "linear_identity_v1"
        )
        if formulation is None:
            formulation = estimator.formulation
            retransformation = current_retransform
        elif formulation != estimator.formulation or retransformation != current_retransform:
            raise HurdleCascadeError("terminal_formulation_drift_across_outer_folds")

    if latent_names is None or formulation is None or retransformation is None:
        raise HurdleCascadeError("lean_hurdle_execution_incomplete")
    if not all(
        np.isfinite(values).all()
        for values in (p_positive, positive_amount, unconditional)
    ):
        raise HurdleCascadeError("terminal_hurdle_oof_incomplete")

    latent_artifacts = {
        spec.target: _make_probability_artifact(
            target=spec.target,
            row_ids=manifest.row_ids,
            values=global_latent_values[spec.target],
            class_labels=spec.class_labels,
            fold_ids=manifest.fold_ids,
            run_id=run_id,
        )
        for spec in latent_specs
    }
    hurdle = _make_hurdle_bundle(
        target=terminal_target_name,
        row_ids=manifest.row_ids,
        p_positive=p_positive,
        positive_amount=positive_amount,
        unconditional=unconditional,
        source="oof",
        formulation=formulation,
        retransformation=retransformation,
        target_eligibility=target_eligibility.counts,
        run_id=run_id,
        fold_ids=manifest.fold_ids,
        metadata={
            "architecture": "one_layer_cascade",
            "nesting_policy": "outer_fold_isolated_latent_crossfit_v1",
        },
    )
    return OneLayerHurdleOOFResult(
        latent_artifacts=latent_artifacts,
        hurdle=hurdle,
        terminal_feature_names=(*names, *latent_names),
    )


def fit_one_layer_hurdle_and_score(
    base_features: np.ndarray | Sequence[Sequence[float]],
    terminal_target: np.ndarray | Sequence[object],
    latent_targets: Mapping[str, np.ndarray | Sequence[object]],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_estimator_factory: callable,
    scoring_features: np.ndarray | Sequence[Sequence[float]],
    scoring_row_ids: Sequence[str],
    *,
    terminal_target_name: str,
    base_feature_names: Sequence[str] | None = None,
    run_id: str | None = None,
) -> OneLayerHurdleScoreResult:
    """Fit deployable lean latent state and hurdle head, then score external rows."""
    train_x = _matrix(base_features, "base_features")
    score_x = _matrix(scoring_features, "scoring_features")
    _validate_manifest_rows(train_x, manifest)
    if train_x.shape[1] != score_x.shape[1]:
        raise HurdleCascadeError("training_scoring_base_feature_width_mismatch")
    if len(score_x) != len(scoring_row_ids):
        raise HurdleCascadeError("scoring_row_id_length_mismatch")
    raw_terminal = np.asarray(terminal_target, dtype=object)
    if raw_terminal.ndim != 1 or len(raw_terminal) != len(train_x):
        raise HurdleCascadeError("terminal_target_length_or_shape_invalid")
    target_eligibility = classify_income_target(raw_terminal)
    y_latent = _validate_latent_targets(latent_targets, latent_specs, len(train_x))
    if base_feature_names is not None and len(base_feature_names) != train_x.shape[1]:
        raise HurdleCascadeError("base_feature_name_width_mismatch")
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
    estimator = terminal_estimator_factory()
    if not isinstance(estimator, HurdleEstimator):
        raise HurdleCascadeError("terminal_factory_must_return_hurdle_estimator")
    estimator.fit(terminal_train_x, raw_terminal)
    components: HurdleComponents = estimator.predict_components(terminal_score_x)
    retransformation = (
        estimator.retransformation if estimator.formulation == "log" else "linear_identity_v1"
    )
    hurdle = _make_hurdle_bundle(
        target=terminal_target_name,
        row_ids=tuple(scoring_row_ids),
        p_positive=components.p_positive,
        positive_amount=components.positive_amount,
        unconditional=components.unconditional_income,
        source="score",
        formulation=estimator.formulation,
        retransformation=retransformation,
        target_eligibility=target_eligibility.counts,
        run_id=run_id,
        metadata={
            "architecture": "one_layer_cascade",
            "latent_training": "global_oof",
            "latent_scoring": "full_fit",
        },
    )
    return OneLayerHurdleScoreResult(
        training_latent_oof=training_latent,
        scoring_latent=scoring_latent,
        hurdle=hurdle,
        terminal_feature_names=(*names, *latent_names),
    )
