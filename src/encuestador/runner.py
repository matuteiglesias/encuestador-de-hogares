"""Execution bridge from resolved experiments to OOF evidence and sample scoring."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cascade import (
    CategoricalLatentSpec,
    fit_one_layer_and_score,
    run_one_layer_oof,
)
from .crossfit import EstimatorFactory, crossfit_predict, fit_full_and_score
from .estimators import HGBClassifierAdapter, HGBRegressorAdapter
from .evaluation import (
    distributional_regression_diagnostics,
    evaluate_cascade_triangle,
)
from .experiments import ResolvedExperiment
from .scientific_primitives import FoldManifest, PredictionArtifact


class ExperimentRunError(ValueError):
    """Raised when a resolved experiment cannot execute without changing its science."""


def _identity(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    values: list[str] = []
    for column in columns:
        if column not in row or row[column] is None or str(row[column]) == "":
            raise ExperimentRunError(f"missing_identity_value:{column}")
        values.append(str(row[column]))
    return "\x1f".join(values)


def _household_fold(group_id: str, n_splits: int) -> int:
    digest = hashlib.sha256(group_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_splits


def configured_fold_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    person_id_columns: Sequence[str],
    household_group_columns: Sequence[str],
    period_columns: Sequence[str] = (),
    n_splits: int,
) -> FoldManifest:
    """Build row-unique IDs while keeping repeated household waves in one split group."""
    row_columns = tuple(person_id_columns) + tuple(period_columns)
    if len(set(row_columns)) != len(row_columns):
        raise ExperimentRunError("row_identity_columns_not_unique")
    group_columns = tuple(household_group_columns)
    if not rows:
        raise ExperimentRunError("experiment_requires_training_rows")
    row_ids = tuple(_identity(row, row_columns) for row in rows)
    household_ids = tuple(_identity(row, group_columns) for row in rows)
    folds = tuple(_household_fold(group, n_splits) for group in household_ids)
    return FoldManifest(
        row_ids=row_ids,
        household_ids=household_ids,
        fold_ids=folds,
        n_splits=n_splits,
    )


def _feature_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    features = config["features"]
    columns = features["columns"]
    names = tuple(
        str(name)
        for name, spec in columns.items()
        if spec.get("allowed_as_external_input") is True
        and spec.get("dtype_role") in {"continuous", "categorical", "binary"}
    )
    if not names:
        raise ExperimentRunError("experiment_has_no_deployable_features")
    return names


def _matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    try:
        matrix = np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunError("feature_matrix_contains_missing_or_nonnumeric_value") from exc
    if matrix.ndim != 2 or matrix.shape != (len(rows), len(columns)):
        raise ExperimentRunError("feature_matrix_shape_invalid")
    if not np.isfinite(matrix).all():
        raise ExperimentRunError("feature_matrix_non_finite")
    return matrix


def _target(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunError(f"target_invalid:{name}") from exc
    if not np.isfinite(values).all():
        raise ExperimentRunError(f"target_non_finite:{name}")
    return values


def _raw_target(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    try:
        values = np.asarray([row[name] for row in rows])
    except KeyError as exc:
        raise ExperimentRunError(f"target_missing:{name}") from exc
    if values.ndim != 1:
        raise ExperimentRunError(f"target_shape_invalid:{name}")
    return values


def _categorical_positions(
    role: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[int, ...]:
    requested = tuple(str(name) for name in role.get("categorical_features", ()))
    missing = sorted(set(requested) - set(feature_names))
    if missing:
        raise ExperimentRunError(
            f"categorical_feature_not_in_design:{','.join(missing)}"
        )
    position = {name: index for index, name in enumerate(feature_names)}
    return tuple(position[name] for name in requested)


def _classifier_factory(
    role: Mapping[str, Any],
    feature_names: Sequence[str],
) -> EstimatorFactory:
    if role.get("family") != "hgb":
        raise ExperimentRunError("starter_runner_classifier_requires_hgb")
    categorical = _categorical_positions(role, feature_names)
    parameters = dict(role.get("params") or {})
    return lambda: HGBClassifierAdapter(
        categorical_features=categorical,
        parameters=parameters,
    )


def _regressor_factory(
    role: Mapping[str, Any],
    feature_names: Sequence[str],
) -> EstimatorFactory:
    if role.get("family") != "hgb":
        raise ExperimentRunError("starter_runner_regressor_requires_hgb")
    categorical = _categorical_positions(role, feature_names)
    parameters = dict(role.get("params") or {})
    loss = str(role.get("loss") or "squared_error")
    return lambda: HGBRegressorAdapter(
        loss=loss,
        categorical_features=categorical,
        parameters=parameters,
    )


def _latent_feature_names(specs: Sequence[CategoricalLatentSpec]) -> tuple[str, ...]:
    return tuple(
        f"latent.{spec.target}.p[{label}]"
        for spec in specs
        for label in spec.class_labels
    )


def _stable_classes(values: np.ndarray) -> tuple[str, ...]:
    labels = tuple(sorted({str(value) for value in values.tolist()}))
    if len(labels) < 2:
        raise ExperimentRunError("categorical_latent_requires_two_observed_classes")
    return labels


def _configured_household_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, PredictionArtifact],
    truth: np.ndarray,
    *,
    row_id_columns: Sequence[str],
    household_observation_columns: Sequence[str],
) -> dict[str, Any]:
    expected_ids = tuple(_identity(row, row_id_columns) for row in rows)
    for name, artifact in artifacts.items():
        if artifact.kind != "regression":
            raise ExperimentRunError(f"household_metric_requires_regression:{name}")
        if artifact.row_ids != expected_ids:
            raise ExperimentRunError(f"household_metric_row_identity_mismatch:{name}")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_identity(row, household_observation_columns)].append(index)
    truth_totals = np.asarray(
        [float(truth[indices].sum()) for _, indices in sorted(grouped.items())],
        dtype=float,
    )
    result: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        predicted = np.asarray(
            [float(artifact.values[indices].sum()) for _, indices in sorted(grouped.items())],
            dtype=float,
        )
        result[name] = distributional_regression_diagnostics(truth_totals, predicted)
    return {
        "household_observation_count": len(grouped),
        "person_observation_count": len(rows),
        "aggregation_identity_columns": list(household_observation_columns),
        "candidates": result,
    }


@dataclass(frozen=True)
class WelfareContext:
    welfare_period: str
    currency: str
    price_reference: str
    welfare_concept: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.welfare_period,
                self.currency,
                self.price_reference,
                self.welfare_concept,
            )
        ):
            raise ExperimentRunError("welfare_context_fields_required")


def materialize_household_welfare(
    scoring_rows: Sequence[Mapping[str, Any]],
    prediction: PredictionArtifact,
    *,
    context: WelfareContext,
    transport_model_release_id: str,
) -> list[dict[str, Any]]:
    """Resolve exact sample-person predictions into the downstream household handoff."""
    person_ids = tuple(_identity(row, ("sample_person_id",)) for row in scoring_rows)
    if prediction.row_ids != person_ids:
        raise ExperimentRunError("sample_scoring_prediction_identity_mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scoring_rows):
        grouped[_identity(row, ("sample_household_id",))].append(index)
    return [
        {
            "sample_household_id": sample_household_id,
            "welfare_period": context.welfare_period,
            "welfare_amount": float(prediction.values[indices].sum()),
            "currency": context.currency,
            "price_reference": context.price_reference,
            "welfare_concept": context.welfare_concept,
            "estimation_status": "complete",
            "transport_model_release_id": transport_model_release_id,
            "member_count": len(indices),
        }
        for sample_household_id, indices in sorted(grouped.items())
    ]


def derive_run_id(
    resolved: ResolvedExperiment,
    *,
    input_release_ids: Sequence[str] = (),
) -> str:
    payload = "\x1f".join((resolved.digest, *sorted(str(value) for value in input_release_ids)))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    experiment_id = str(resolved.config["experiment"]["id"])
    return f"{experiment_id}-{digest}"


@dataclass(frozen=True)
class ExperimentExecutionResult:
    run_id: str
    experiment_id: str
    config_digest: str
    architecture_id: str
    feature_names: tuple[str, ...]
    fold_manifest: FoldManifest
    oof_prediction: PredictionArtifact
    latent_predictions: Mapping[str, PredictionArtifact]
    metrics: Mapping[str, Any]
    score_prediction: PredictionArtifact | None
    household_welfare: tuple[dict[str, Any], ...]


def execute_experiment(
    resolved: ResolvedExperiment,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    scoring_rows: Sequence[Mapping[str, Any]] | None = None,
    welfare_context: WelfareContext | None = None,
    input_release_ids: Sequence[str] = (),
    run_id: str | None = None,
) -> ExperimentExecutionResult:
    """Execute the active direct or one-layer lean architecture from resolved config."""
    config = resolved.config
    data = config["data"]
    terminal = config["terminal"]
    estimators = config["estimators"]["roles"]
    splits = config["splits"]
    feature_names = _feature_names(config)
    person_columns = tuple(str(value) for value in data["person_id_columns"])
    household_columns = tuple(str(value) for value in data["household_id_columns"])
    period_columns = tuple(str(value) for value in data.get("period_columns", ()))
    row_columns = (*person_columns, *period_columns)
    household_observation_columns = (*household_columns, *period_columns)
    manifest = configured_fold_manifest(
        training_rows,
        person_id_columns=person_columns,
        household_group_columns=household_columns,
        period_columns=period_columns,
        n_splits=int(splits["n_splits"]),
    )
    x = _matrix(training_rows, feature_names)
    target_name = str(terminal["target"])
    y = _target(training_rows, target_name)
    effective_run_id = run_id or derive_run_id(
        resolved,
        input_release_ids=input_release_ids,
    )

    architecture = resolved.architecture
    score_prediction: PredictionArtifact | None = None
    household_welfare: tuple[dict[str, Any], ...] = ()

    if not architecture.nodes:
        if "terminal" not in estimators:
            raise ExperimentRunError("terminal_estimator_role_missing")
        terminal_factory = _regressor_factory(estimators["terminal"], feature_names)
        oof = crossfit_predict(
            x,
            y,
            manifest,
            terminal_factory,
            target=target_name,
            run_id=effective_run_id,
            metadata={"architecture": architecture.architecture_id},
        )
        metrics: dict[str, Any] = {
            "person": {"direct": distributional_regression_diagnostics(y, oof.values)},
            "household": _configured_household_diagnostics(
                training_rows,
                {"direct": oof},
                y,
                row_id_columns=row_columns,
                household_observation_columns=household_observation_columns,
            ),
        }
        latent_predictions: Mapping[str, PredictionArtifact] = {}
        if scoring_rows is not None:
            score_x = _matrix(scoring_rows, feature_names)
            score_ids = tuple(
                _identity(row, ("sample_person_id",))
                if "sample_person_id" in row
                else _identity(row, row_columns)
                for row in scoring_rows
            )
            score_prediction = fit_full_and_score(
                x,
                y,
                score_x,
                score_ids,
                terminal_factory,
                target=target_name,
                run_id=effective_run_id,
                metadata={"architecture": architecture.architecture_id},
            )
    else:
        order = architecture.topological_order()
        if len(order) != 1:
            raise ExperimentRunError("starter_runner_supports_direct_or_one_layer_only")
        node = architecture.nodes[order[0]]
        if node.upstream_nodes:
            raise ExperimentRunError("starter_runner_does_not_support_deep_upstreams")
        if tuple(node.base_inputs) != feature_names:
            raise ExperimentRunError("lean_node_must_use_declared_deployable_feature_plane_v1")
        latent_role = estimators.get(node.estimator_role)
        terminal_role = estimators.get("terminal")
        if not isinstance(latent_role, Mapping) or not isinstance(terminal_role, Mapping):
            raise ExperimentRunError("lean_estimator_roles_missing")
        latent_factory = _classifier_factory(latent_role, feature_names)
        latent_targets = {target: _raw_target(training_rows, target) for target in node.targets}
        latent_specs = tuple(
            CategoricalLatentSpec(
                target=target,
                class_labels=_stable_classes(latent_targets[target]),
                estimator_factory=latent_factory,
            )
            for target in node.targets
        )
        terminal_design_names = (*feature_names, *_latent_feature_names(latent_specs))
        terminal_factory = _regressor_factory(terminal_role, terminal_design_names)
        lean = run_one_layer_oof(
            x,
            y,
            latent_targets,
            manifest,
            latent_specs,
            terminal_factory,
            terminal_target_name=target_name,
            base_feature_names=feature_names,
            run_id=effective_run_id,
        )
        triangle = evaluate_cascade_triangle(
            x,
            y,
            latent_targets,
            manifest,
            latent_specs,
            terminal_factory,
            lean.terminal_artifact,
            lean.latent_artifacts,
            terminal_target_name=target_name,
            run_id=effective_run_id,
        )
        oof = lean.terminal_artifact
        latent_predictions = lean.latent_artifacts
        metrics = {
            "person": triangle.person_metrics,
            "cascade_gains": triangle.gains,
            "latents": triangle.latent_metrics,
            "household": _configured_household_diagnostics(
                training_rows,
                {
                    "direct": triangle.direct,
                    "oracle": triangle.oracle,
                    "deployable": triangle.deployable,
                },
                y,
                row_id_columns=row_columns,
                household_observation_columns=household_observation_columns,
            ),
        }
        if scoring_rows is not None:
            score_x = _matrix(scoring_rows, feature_names)
            score_ids = tuple(
                _identity(row, ("sample_person_id",))
                if "sample_person_id" in row
                else _identity(row, row_columns)
                for row in scoring_rows
            )
            scored = fit_one_layer_and_score(
                x,
                y,
                latent_targets,
                manifest,
                latent_specs,
                terminal_factory,
                score_x,
                score_ids,
                terminal_target_name=target_name,
                base_feature_names=feature_names,
                run_id=effective_run_id,
            )
            score_prediction = scored.terminal_artifact

    if score_prediction is not None and welfare_context is not None:
        if scoring_rows is None or not all(
            "sample_person_id" in row and "sample_household_id" in row
            for row in scoring_rows
        ):
            raise ExperimentRunError("household_welfare_requires_exact_sample_ids")
        household_welfare = tuple(
            materialize_household_welfare(
                scoring_rows,
                score_prediction,
                context=welfare_context,
                transport_model_release_id=f"run:{effective_run_id}",
            )
        )

    return ExperimentExecutionResult(
        run_id=effective_run_id,
        experiment_id=str(config["experiment"]["id"]),
        config_digest=resolved.digest,
        architecture_id=architecture.architecture_id,
        feature_names=feature_names,
        fold_manifest=manifest,
        oof_prediction=oof,
        latent_predictions=latent_predictions,
        metrics=metrics,
        score_prediction=score_prediction,
        household_welfare=household_welfare,
    )
