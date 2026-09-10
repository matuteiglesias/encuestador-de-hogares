"""Resolved experiment execution for direct/lean OOF evidence and sample welfare scoring."""
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
    classification_diagnostics,
    distributional_regression_diagnostics,
    evaluate_cascade_triangle,
)
from .experiments import ResolvedExperiment
from .hurdle_cascade import (
    fit_one_layer_hurdle_and_score,
    run_one_layer_hurdle_oof,
)
from .scientific_primitives import FoldManifest, PredictionArtifact
from .terminal import (
    HurdleEstimator,
    HurdlePredictionBundle,
    TargetEligibility,
    classify_income_target,
    crossfit_hurdle,
    fit_hurdle_and_score,
)


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
    columns = config["features"]["columns"]
    names = tuple(
        str(name)
        for name, spec in columns.items()
        if spec.get("allowed_as_external_input") is True
        and spec.get("dtype_role") in {"continuous", "categorical", "binary"}
    )
    if not names:
        raise ExperimentRunError("experiment_has_no_deployable_features")
    forbidden_weight_tokens = {
        "PONDERA",
        "PONDII",
        "PONDIH",
        "selection_probability",
        "design_inverse_probability_weight",
    }
    leaked = sorted(set(names) & forbidden_weight_tokens)
    if leaked:
        raise ExperimentRunError(f"forbidden_weight_or_design_feature:{','.join(leaked)}")
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
        values = np.asarray([row[name] for row in rows], dtype=object)
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


def _hurdle_factory(
    estimators: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    formulation: str,
) -> callable:
    presence_role = estimators.get("terminal_presence")
    amount_role = estimators.get("terminal_positive")
    if not isinstance(presence_role, Mapping) or not isinstance(amount_role, Mapping):
        raise ExperimentRunError("hurdle_terminal_estimator_roles_missing")
    presence_factory = _classifier_factory(presence_role, feature_names)
    amount_factory = _regressor_factory(amount_role, feature_names)
    expected_loss = "gamma" if formulation == "gamma" else "squared_error"
    actual_loss = str(amount_role.get("loss") or "squared_error")
    if actual_loss != expected_loss:
        raise ExperimentRunError(
            f"hurdle_positive_loss_mismatch:{actual_loss}!={expected_loss}"
        )
    retransformation = "duan_smearing_v1" if formulation == "log" else "linear_identity_v1"

    def factory() -> HurdleEstimator:
        return HurdleEstimator(
            formulation=formulation,
            presence_factory=presence_factory,
            amount_factory=amount_factory,
            retransformation=retransformation,
        )

    return factory


def _terminal_formulation(terminal: Mapping[str, Any]) -> str | None:
    value = str(terminal.get("formulation") or "")
    if value == "hurdle_log":
        return "log"
    if value == "hurdle_gamma":
        return "gamma"
    if value.startswith("hurdle"):
        raise ExperimentRunError(f"unsupported_hurdle_terminal_formulation:{value}")
    return None


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
    if any(label in {"", "None", "nan"} for label in labels):
        raise ExperimentRunError("categorical_latent_contains_unavailable_label")
    return labels


def _subset_probability(
    artifact: PredictionArtifact,
    mask: np.ndarray,
) -> PredictionArtifact:
    fold_ids = None
    if artifact.fold_ids is not None:
        fold_ids = tuple(
            fold for fold, keep in zip(artifact.fold_ids, mask, strict=True) if keep
        )
    return PredictionArtifact(
        target=artifact.target,
        kind="probability",
        row_ids=tuple(
            row_id for row_id, keep in zip(artifact.row_ids, mask, strict=True) if keep
        ),
        values=artifact.values[mask],
        source=artifact.source,
        class_labels=artifact.class_labels,
        fold_ids=fold_ids,
        run_id=artifact.run_id,
        metadata=dict(artifact.metadata),
    )


def _hurdle_person_metrics(
    eligibility: TargetEligibility,
    bundle: HurdlePredictionBundle,
) -> dict[str, Any]:
    valid = eligibility.valid
    positive = eligibility.positive
    counts = eligibility.counts
    if not np.any(valid):
        raise ExperimentRunError("terminal_evaluation_has_no_valid_rows")
    person = distributional_regression_diagnostics(
        eligibility.numeric[valid],
        bundle.unconditional_expected_income.values[valid],
    )
    presence = classification_diagnostics(
        eligibility.positive[valid].astype(int),
        _subset_probability(bundle.p_positive, valid),
    )
    positive_amount = None
    if np.any(positive):
        positive_amount = distributional_regression_diagnostics(
            eligibility.numeric[positive],
            bundle.positive_amount_prediction.values[positive],
        )
    return {
        "target_eligibility": {
            **counts,
            "zero_prevalence_among_eligible": counts["zero"] / counts["eligible"],
            "positive_prevalence_among_eligible": counts["positive"] / counts["eligible"],
        },
        "unconditional_income": person,
        "presence": presence,
        "positive_amount": positive_amount,
        "formulation": bundle.formulation,
        "retransformation": bundle.retransformation,
    }


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
        "complete_observed_income_household_count": len(grouped),
        "unavailable_observed_income_household_count": 0,
        "person_observation_count": len(rows),
        "aggregation_identity_columns": list(household_observation_columns),
        "candidates": result,
    }


def _hurdle_household_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, PredictionArtifact],
    eligibility: TargetEligibility,
    *,
    row_id_columns: Sequence[str],
    household_observation_columns: Sequence[str],
) -> dict[str, Any]:
    expected_ids = tuple(_identity(row, row_id_columns) for row in rows)
    for name, artifact in artifacts.items():
        if artifact.kind != "regression" or artifact.row_ids != expected_ids:
            raise ExperimentRunError(f"household_hurdle_prediction_alignment_invalid:{name}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_identity(row, household_observation_columns)].append(index)
    complete: list[tuple[str, list[int]]] = []
    unavailable: list[tuple[str, list[int]]] = []
    for key, indices in sorted(grouped.items()):
        if bool(np.all(eligibility.valid[indices])):
            complete.append((key, indices))
        else:
            unavailable.append((key, indices))
    if not complete:
        raise ExperimentRunError("no_complete_households_for_income_evaluation")
    truth = np.asarray(
        [float(eligibility.numeric[indices].sum()) for _, indices in complete], dtype=float
    )
    candidates: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        prediction = np.asarray(
            [float(artifact.values[indices].sum()) for _, indices in complete], dtype=float
        )
        candidates[name] = distributional_regression_diagnostics(truth, prediction)
    return {
        "household_observation_count": len(grouped),
        "complete_observed_income_household_count": len(complete),
        "unavailable_observed_income_household_count": len(unavailable),
        "person_observation_count": len(rows),
        "unavailable_member_income_count": int((~eligibility.valid).sum()),
        "aggregation_identity_columns": list(household_observation_columns),
        "evaluation_cohort": "all_members_have_valid_terminal_income_v1",
        "candidates": candidates,
    }


def _oracle_latent_design(
    latent_targets: Mapping[str, np.ndarray],
    specs: Sequence[CategoricalLatentSpec],
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for spec in specs:
        values = latent_targets[spec.target]
        labels = tuple(str(value) for value in values.tolist())
        lookup = {label: index for index, label in enumerate(spec.class_labels)}
        unknown = sorted(set(labels) - set(spec.class_labels))
        if unknown:
            raise ExperimentRunError(
                f"oracle_latent_class_missing:{spec.target}:{','.join(unknown)}"
            )
        index = np.asarray([lookup[label] for label in labels], dtype=int)
        blocks.append(np.eye(len(spec.class_labels), dtype=float)[index])
    return np.column_stack(blocks)


def _hurdle_gains(candidate_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("mae", "rmse"):
        direct = candidate_metrics["direct"]["unconditional_income"]["point"][metric]
        oracle = candidate_metrics["oracle"]["unconditional_income"]["point"][metric]
        deployable = candidate_metrics["deployable"]["unconditional_income"]["point"][metric]
        oracle_gain = direct - oracle
        deployable_gain = direct - deployable
        output[metric] = {
            "direct_risk": direct,
            "oracle_risk": oracle,
            "deployable_risk": deployable,
            "oracle_gain": oracle_gain,
            "deployable_gain": deployable_gain,
            "capture_ratio": deployable_gain / oracle_gain if oracle_gain != 0 else None,
        }
    return output


def _fold_counts(manifest: FoldManifest) -> dict[str, int]:
    values = np.asarray(manifest.fold_ids, dtype=int)
    return {str(fold): int((values == fold).sum()) for fold in range(manifest.n_splits)}


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
    hurdle: HurdlePredictionBundle | None = None
    score_hurdle: HurdlePredictionBundle | None = None


def execute_experiment(
    resolved: ResolvedExperiment,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    scoring_rows: Sequence[Mapping[str, Any]] | None = None,
    welfare_context: WelfareContext | None = None,
    input_release_ids: Sequence[str] = (),
    run_id: str | None = None,
) -> ExperimentExecutionResult:
    """Execute active direct or one-layer lean architecture from resolved config."""
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
    raw_y = _raw_target(training_rows, target_name)
    hurdle_formulation = _terminal_formulation(terminal)
    effective_run_id = run_id or derive_run_id(
        resolved,
        input_release_ids=input_release_ids,
    )
    architecture = resolved.architecture
    score_prediction: PredictionArtifact | None = None
    household_welfare: tuple[dict[str, Any], ...] = ()
    hurdle: HurdlePredictionBundle | None = None
    score_hurdle: HurdlePredictionBundle | None = None

    if not architecture.nodes:
        latent_predictions: Mapping[str, PredictionArtifact] = {}
        if hurdle_formulation is not None:
            eligibility = classify_income_target(raw_y)
            terminal_factory = _hurdle_factory(
                estimators, feature_names, hurdle_formulation
            )
            hurdle = crossfit_hurdle(
                x,
                raw_y,
                manifest,
                terminal_factory,
                target=target_name,
                run_id=effective_run_id,
                metadata={"architecture": architecture.architecture_id},
            )
            oof = hurdle.unconditional_expected_income
            metrics: dict[str, Any] = {
                "fold_counts": _fold_counts(manifest),
                "person": {"direct": _hurdle_person_metrics(eligibility, hurdle)},
                "household": _hurdle_household_diagnostics(
                    training_rows,
                    {"direct": oof},
                    eligibility,
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
                score_hurdle = fit_hurdle_and_score(
                    x,
                    raw_y,
                    score_x,
                    score_ids,
                    terminal_factory,
                    target=target_name,
                    run_id=effective_run_id,
                    metadata={"architecture": architecture.architecture_id},
                )
                score_prediction = score_hurdle.unconditional_expected_income
        else:
            y = _target(training_rows, target_name)
            terminal_role = estimators.get("terminal")
            if not isinstance(terminal_role, Mapping):
                raise ExperimentRunError("terminal_estimator_role_missing")
            terminal_factory = _regressor_factory(terminal_role, feature_names)
            oof = crossfit_predict(
                x,
                y,
                manifest,
                terminal_factory,
                target=target_name,
                run_id=effective_run_id,
                metadata={"architecture": architecture.architecture_id},
            )
            metrics = {
                "fold_counts": _fold_counts(manifest),
                "person": {
                    "direct": distributional_regression_diagnostics(y, oof.values)
                },
                "household": _configured_household_diagnostics(
                    training_rows,
                    {"direct": oof},
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
            raise ExperimentRunError(
                "lean_node_must_use_declared_deployable_feature_plane_v1"
            )
        latent_role = estimators.get(node.estimator_role)
        if not isinstance(latent_role, Mapping):
            raise ExperimentRunError("lean_latent_estimator_role_missing")
        latent_factory = _classifier_factory(latent_role, feature_names)
        latent_targets = {
            target: _raw_target(training_rows, target) for target in node.targets
        }
        latent_specs = tuple(
            CategoricalLatentSpec(
                target=target,
                class_labels=_stable_classes(latent_targets[target]),
                estimator_factory=latent_factory,
            )
            for target in node.targets
        )
        terminal_design_names = (*feature_names, *_latent_feature_names(latent_specs))

        if hurdle_formulation is not None:
            eligibility = classify_income_target(raw_y)
            terminal_factory = _hurdle_factory(
                estimators, terminal_design_names, hurdle_formulation
            )
            lean = run_one_layer_hurdle_oof(
                x,
                raw_y,
                latent_targets,
                manifest,
                latent_specs,
                terminal_factory,
                terminal_target_name=target_name,
                base_feature_names=feature_names,
                run_id=effective_run_id,
            )
            hurdle = lean.hurdle
            oof = hurdle.unconditional_expected_income
            latent_predictions = lean.latent_artifacts

            direct_factory = _hurdle_factory(
                estimators, feature_names, hurdle_formulation
            )
            direct = crossfit_hurdle(
                x,
                raw_y,
                manifest,
                direct_factory,
                target=target_name,
                run_id=effective_run_id,
                metadata={"evaluation_role": "direct_matched_baseline"},
            )
            oracle_x = np.column_stack(
                [x, _oracle_latent_design(latent_targets, latent_specs)]
            )
            oracle = crossfit_hurdle(
                oracle_x,
                raw_y,
                manifest,
                terminal_factory,
                target=target_name,
                run_id=effective_run_id,
                metadata={
                    "evaluation_role": "oracle_true_latents_diagnostic_only"
                },
            )
            candidate_metrics = {
                "direct": _hurdle_person_metrics(eligibility, direct),
                "oracle": _hurdle_person_metrics(eligibility, oracle),
                "deployable": _hurdle_person_metrics(eligibility, hurdle),
            }
            latent_metrics = {
                spec.target: classification_diagnostics(
                    latent_targets[spec.target], lean.latent_artifacts[spec.target]
                )
                for spec in latent_specs
            }
            metrics = {
                "fold_counts": _fold_counts(manifest),
                "person": candidate_metrics,
                "cascade_gains": _hurdle_gains(candidate_metrics),
                "latents": latent_metrics,
                "household": _hurdle_household_diagnostics(
                    training_rows,
                    {
                        "direct": direct.unconditional_expected_income,
                        "oracle": oracle.unconditional_expected_income,
                        "deployable": hurdle.unconditional_expected_income,
                    },
                    eligibility,
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
                scored = fit_one_layer_hurdle_and_score(
                    x,
                    raw_y,
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
                score_hurdle = scored.hurdle
                score_prediction = score_hurdle.unconditional_expected_income
        else:
            y = _target(training_rows, target_name)
            terminal_role = estimators.get("terminal")
            if not isinstance(terminal_role, Mapping):
                raise ExperimentRunError("terminal_estimator_role_missing")
            terminal_factory = _regressor_factory(
                terminal_role, terminal_design_names
            )
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
                "fold_counts": _fold_counts(manifest),
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
        hurdle=hurdle,
        score_hurdle=score_hurdle,
    )
