"""Scientific evaluation for latent probabilities, welfare distributions and cascades."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cascade import CategoricalLatentSpec
from .crossfit import EstimatorFactory, crossfit_predict
from .scientific_primitives import (
    FoldManifest,
    PredictionArtifact,
    aggregate_person_predictions,
    regression_metrics,
)


class EvaluationError(ValueError):
    """Raised when an evaluation would be incomplete or scientifically ambiguous."""


def _paired_arrays(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise EvaluationError("regression_evaluation_shape_invalid")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise EvaluationError("regression_evaluation_requires_finite_values")
    return y, p


def classification_diagnostics(
    truth: Sequence[Any] | np.ndarray,
    artifact: PredictionArtifact,
    *,
    reliability_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate class-labelled probability predictions without collapsing semantics."""
    if artifact.kind != "probability":
        raise EvaluationError("classification_requires_probability_artifact")
    if reliability_bins < 2:
        raise EvaluationError("reliability_bins_must_be_at_least_two")
    y = np.asarray(truth)
    if y.ndim != 1 or len(y) != len(artifact.row_ids):
        raise EvaluationError("classification_target_shape_invalid")

    labels = artifact.class_labels
    label_to_index = {label: index for index, label in enumerate(labels)}
    truth_labels = tuple(str(value) for value in y.tolist())
    unknown = sorted(set(truth_labels) - set(labels))
    if unknown:
        raise EvaluationError(f"classification_truth_class_missing:{','.join(unknown)}")
    target_index = np.asarray([label_to_index[label] for label in truth_labels], dtype=int)
    probabilities = np.asarray(artifact.values, dtype=float)
    predicted_index = probabilities.argmax(axis=1)

    eps = np.finfo(float).eps
    log_loss = float(
        -np.mean(np.log(np.clip(probabilities[np.arange(len(y)), target_index], eps, 1.0)))
    )
    one_hot = np.eye(len(labels), dtype=float)[target_index]
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for observed, predicted in zip(target_index, predicted_index, strict=True):
        confusion[observed, predicted] += 1

    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(labels):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())
        recall = tp / support if support else None
        precision = tp / predicted_count if predicted_count else None
        if recall is not None:
            recalls.append(recall)
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        elif support:
            f1 = 0.0
        else:
            f1 = None
        if f1 is not None:
            f1s.append(f1)
        per_class[label] = {
            "support": support,
            "prevalence": support / len(y),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    edges = np.linspace(0.0, 1.0, reliability_bins + 1)
    reliability: dict[str, list[dict[str, Any]]] = {}
    ece_numerator = 0.0
    for class_index, label in enumerate(labels):
        class_prob = probabilities[:, class_index]
        class_truth = (target_index == class_index).astype(float)
        records: list[dict[str, Any]] = []
        for bin_index in range(reliability_bins):
            lower = float(edges[bin_index])
            upper = float(edges[bin_index + 1])
            if bin_index == reliability_bins - 1:
                mask = (class_prob >= lower) & (class_prob <= upper)
            else:
                mask = (class_prob >= lower) & (class_prob < upper)
            count = int(mask.sum())
            mean_probability = float(class_prob[mask].mean()) if count else None
            observed_frequency = float(class_truth[mask].mean()) if count else None
            if count:
                ece_numerator += count * abs(mean_probability - observed_frequency)
            records.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                }
            )
        reliability[label] = records

    return {
        "row_count": len(y),
        "class_labels": list(labels),
        "class_count": len(labels),
        "accuracy": float(np.mean(predicted_index == target_index)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "log_loss": log_loss,
        "multiclass_brier": multiclass_brier,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "reliability": {
            "definition": "one_vs_rest_equal_width_v1",
            "bin_edges": edges.tolist(),
            "expected_calibration_error": ece_numerator / (len(y) * len(labels)),
            "by_class": reliability,
        },
    }


def _rank_deciles(
    reference: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    order = np.argsort(reference, kind="stable")
    output: list[dict[str, Any]] = []
    for decile, indices in enumerate(np.array_split(order, 10), start=1):
        if not len(indices):
            output.append({"decile": decile, "count": 0, "bias": None, "mae": None})
            continue
        residual = prediction[indices] - truth[indices]
        output.append(
            {
                "decile": decile,
                "count": len(indices),
                "reference_min": float(reference[indices].min()),
                "reference_max": float(reference[indices].max()),
                "bias": float(residual.mean()),
                "mae": float(np.abs(residual).mean()),
            }
        )
    return output


def distributional_regression_diagnostics(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    *,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, Any]:
    """Evaluate linear-scale fit and distributional compression explicitly."""
    y, p = _paired_arrays(truth, prediction)
    q = tuple(float(value) for value in quantiles)
    if not q or any(value <= 0.0 or value >= 1.0 for value in q):
        raise EvaluationError("quantile_grid_must_be_inside_unit_interval")
    if len(set(q)) != len(q):
        raise EvaluationError("quantile_grid_must_be_unique")

    point = regression_metrics(y, p)
    denominator = float(np.sum((y - y.mean()) ** 2))
    numerator = float(np.sum((y - p) ** 2))
    point["median_absolute_error"] = float(np.median(np.abs(p - y)))
    point["r2"] = 1.0 - numerator / denominator if denominator > 0 else None

    true_sd = float(np.std(y))
    pred_sd = float(np.std(p))
    quantile_records: dict[str, Any] = {}
    for value in q:
        observed = float(np.quantile(y, value))
        predicted = float(np.quantile(p, value))
        quantile_records[f"{value:g}"] = {
            "observed": observed,
            "predicted": predicted,
            "difference": predicted - observed,
        }

    p10 = float(np.quantile(y, 0.1))
    p90 = float(np.quantile(y, 0.9))
    low = y <= p10
    high = y >= p90
    residual = p - y

    return {
        "row_count": len(y),
        "point": point,
        "dispersion_ratio": pred_sd / true_sd if true_sd > 0 else None,
        "observed_sd": true_sd,
        "predicted_sd": pred_sd,
        "quantile_agreement": quantile_records,
        "deciles": {
            "definition": "stable_equal_count_rank_v1",
            "by_observed_income": _rank_deciles(y, y, p),
            "by_predicted_income": _rank_deciles(p, y, p),
        },
        "tails": {
            "observed_p10": p10,
            "observed_p90": p90,
            "low_tail_count": int(low.sum()),
            "low_tail_mean_bias": float(residual[low].mean()),
            "low_tail_median_bias": float(np.median(residual[low])),
            "high_tail_count": int(high.sum()),
            "high_tail_mean_bias": float(residual[high].mean()),
            "high_tail_median_bias": float(np.median(residual[high])),
        },
    }


def household_prediction_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, PredictionArtifact],
    *,
    truth_field: str,
) -> dict[str, Any]:
    """Aggregate complete households and evaluate each candidate on linear totals."""
    households = aggregate_person_predictions(rows, artifacts, truth_field=truth_field)
    if any(record.get("status") != "complete" for record in households):
        raise EvaluationError("household_metrics_require_complete_membership")
    truth = np.asarray([record["truth_household_income"] for record in households], dtype=float)

    candidates: dict[str, Any] = {}
    for artifact_id in artifacts:
        prediction_key = f"{artifact_id}_household_income"
        prediction = np.asarray([record[prediction_key] for record in households], dtype=float)
        by_size: dict[str, Any] = {}
        sizes = np.asarray([int(record["member_count"]) for record in households], dtype=int)
        for size in sorted(set(sizes.tolist())):
            mask = sizes == size
            by_size[str(size)] = {
                "household_count": int(mask.sum()),
                **regression_metrics(truth[mask], prediction[mask]),
            }
        candidates[artifact_id] = {
            **distributional_regression_diagnostics(truth, prediction),
            "by_household_size": by_size,
        }

    return {
        "household_count": len(households),
        "complete_household_count": len(households),
        "incomplete_household_count": 0,
        "person_count_accounted": int(sum(record["member_count"] for record in households)),
        "candidates": candidates,
    }


def _true_latent_design(
    latent_targets: Mapping[str, Sequence[Any] | np.ndarray],
    latent_specs: Sequence[CategoricalLatentSpec],
    row_count: int,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for spec in latent_specs:
        if spec.target not in latent_targets:
            raise EvaluationError(f"missing_oracle_latent_target:{spec.target}")
        target = np.asarray(latent_targets[spec.target])
        if target.ndim != 1 or len(target) != row_count:
            raise EvaluationError(f"oracle_latent_target_shape_invalid:{spec.target}")
        label_to_index = {label: index for index, label in enumerate(spec.class_labels)}
        labels = tuple(str(value) for value in target.tolist())
        unknown = sorted(set(labels) - set(spec.class_labels))
        if unknown:
            raise EvaluationError(
                f"oracle_latent_class_missing:{spec.target}:{','.join(unknown)}"
            )
        indices = np.asarray([label_to_index[label] for label in labels], dtype=int)
        blocks.append(np.eye(len(spec.class_labels), dtype=float)[indices])
    if not blocks:
        raise EvaluationError("oracle_requires_latent_targets")
    return np.column_stack(blocks)


@dataclass(frozen=True)
class CascadeTriangleResult:
    """Matched direct/oracle/deployable evidence under one outer fold manifest."""

    direct: PredictionArtifact
    oracle: PredictionArtifact
    deployable: PredictionArtifact
    person_metrics: Mapping[str, Any]
    gains: Mapping[str, Any]
    latent_metrics: Mapping[str, Any]
    household_metrics: Mapping[str, Any] | None


def evaluate_cascade_triangle(
    base_features: np.ndarray | Sequence[Sequence[float]],
    terminal_target: Sequence[float] | np.ndarray,
    latent_targets: Mapping[str, Sequence[Any] | np.ndarray],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_estimator_factory: EstimatorFactory,
    deployable_terminal: PredictionArtifact,
    deployable_latents: Mapping[str, PredictionArtifact],
    *,
    terminal_target_name: str,
    person_rows: Sequence[Mapping[str, Any]] | None = None,
    truth_field: str | None = None,
    run_id: str | None = None,
) -> CascadeTriangleResult:
    """Compare X, X+Z_true and leakage-safe X+Z_oof under matched outer folds."""
    x = np.asarray(base_features, dtype=float)
    y = np.asarray(terminal_target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) != len(manifest.row_ids):
        raise EvaluationError("cascade_triangle_shape_invalid")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise EvaluationError("cascade_triangle_requires_finite_values")
    if deployable_terminal.kind != "regression" or deployable_terminal.source != "oof":
        raise EvaluationError("deployable_terminal_must_be_oof_regression")
    if deployable_terminal.row_ids != manifest.row_ids or deployable_terminal.fold_ids != manifest.fold_ids:
        raise EvaluationError("deployable_terminal_manifest_mismatch")

    direct = crossfit_predict(
        x,
        y,
        manifest,
        terminal_estimator_factory,
        target=terminal_target_name,
        run_id=run_id,
        metadata={"evaluation_role": "direct_matched_baseline"},
    )
    oracle_latents = _true_latent_design(latent_targets, latent_specs, len(x))
    oracle = crossfit_predict(
        np.column_stack([x, oracle_latents]),
        y,
        manifest,
        terminal_estimator_factory,
        target=terminal_target_name,
        run_id=run_id,
        metadata={"evaluation_role": "oracle_true_latents_diagnostic_only"},
    )

    person_metrics = {
        "direct": distributional_regression_diagnostics(y, direct.values),
        "oracle": distributional_regression_diagnostics(y, oracle.values),
        "deployable": distributional_regression_diagnostics(y, deployable_terminal.values),
    }
    gains: dict[str, Any] = {}
    for metric in ("mae", "rmse"):
        direct_risk = person_metrics["direct"]["point"][metric]
        oracle_risk = person_metrics["oracle"]["point"][metric]
        deployable_risk = person_metrics["deployable"]["point"][metric]
        oracle_gain = direct_risk - oracle_risk
        deployable_gain = direct_risk - deployable_risk
        gains[metric] = {
            "direct_risk": direct_risk,
            "oracle_risk": oracle_risk,
            "deployable_risk": deployable_risk,
            "oracle_gain": oracle_gain,
            "deployable_gain": deployable_gain,
            "capture_ratio": deployable_gain / oracle_gain if oracle_gain > 0 else None,
        }

    latent_metrics: dict[str, Any] = {}
    expected_targets = {spec.target for spec in latent_specs}
    if set(deployable_latents) != expected_targets:
        raise EvaluationError("deployable_latent_mapping_mismatch")
    for spec in latent_specs:
        artifact = deployable_latents[spec.target]
        if artifact.row_ids != manifest.row_ids or artifact.fold_ids != manifest.fold_ids:
            raise EvaluationError(f"deployable_latent_manifest_mismatch:{spec.target}")
        latent_metrics[spec.target] = classification_diagnostics(
            latent_targets[spec.target],
            artifact,
        )

    household_metrics = None
    if person_rows is not None or truth_field is not None:
        if person_rows is None or truth_field is None:
            raise EvaluationError("household_triangle_requires_rows_and_truth_field")
        household_metrics = household_prediction_diagnostics(
            person_rows,
            {"direct": direct, "oracle": oracle, "deployable": deployable_terminal},
            truth_field=truth_field,
        )

    return CascadeTriangleResult(
        direct=direct,
        oracle=oracle,
        deployable=deployable_terminal,
        person_metrics=person_metrics,
        gains=gains,
        latent_metrics=latent_metrics,
        household_metrics=household_metrics,
    )
