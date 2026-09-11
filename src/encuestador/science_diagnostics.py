"""Explicit, unweighted diagnostics for person and household welfare predictions.

These helpers deliberately return plain dictionaries so that their output can be
stored in governed run bundles.  Oracle quantities are labelled as diagnostics;
they must never be presented as deployable model evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .scientific_primitives import regression_metrics


class ScienceDiagnosticsError(ValueError):
    """Raised when a diagnostic would have ambiguous scientific semantics."""


def _paired(truth: Sequence[float], prediction: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ScienceDiagnosticsError("diagnostic_shape_invalid")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ScienceDiagnosticsError("diagnostics_require_finite_values")
    return y, p


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, including deterministic handling of ties."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rank(truth: Sequence[float], prediction: Sequence[float]) -> float | None:
    """Return unweighted Spearman rank correlation (``None`` if undefined)."""
    y, p = _paired(truth, prediction)
    yr, pr = _rank(y), _rank(p)
    if np.std(yr) == 0 or np.std(pr) == 0:
        return None
    return float(np.corrcoef(yr, pr)[0, 1])


def positive_amount_head_metrics(
    truth: Sequence[float], prediction: Sequence[float]
) -> dict[str, Any]:
    """Evaluate amounts only for rows whose observed amount is positive."""
    y, p = _paired(truth, prediction)
    mask = y > 0
    return {
        "definition": "observed_positive_amount_head_v1",
        "count": int(mask.sum()),
        "metrics": regression_metrics(y[mask], p[mask]) if mask.any() else None,
    }


def transition_diagnostics(
    truth: Sequence[float], prediction: Sequence[float], *, bins: int
) -> dict[str, Any]:
    """Return an equal-count observed-rank to predicted-rank transition matrix."""
    if bins not in {5, 10}:
        raise ScienceDiagnosticsError("transition_bins_must_be_quintile_or_decile")
    y, p = _paired(truth, prediction)
    observed = np.empty(len(y), int)
    predicted = np.empty(len(y), int)
    for labels, values in ((observed, y), (predicted, p)):
        for index, members in enumerate(np.array_split(np.argsort(values, kind="stable"), bins)):
            labels[members] = index
    matrix = np.zeros((bins, bins), int)
    np.add.at(matrix, (observed, predicted), 1)
    return {
        "definition": "stable_equal_count_rank_transition_v1",
        "bins": bins,
        "matrix": matrix.tolist(),
        "same_bin_rate": float(np.mean(observed == predicted)),
        "mean_absolute_bin_move": float(np.mean(np.abs(observed - predicted))),
    }


def low_tail_cdf_bias(
    truth: Sequence[float], prediction: Sequence[float], *, probabilities: Sequence[float] = (.01, .05, .1)
) -> list[dict[str, float]]:
    """Compare predicted and observed empirical CDFs at observed low-tail cutoffs."""
    y, p = _paired(truth, prediction)
    output = []
    for probability in probabilities:
        q = float(probability)
        if not 0 < q < 0.5:
            raise ScienceDiagnosticsError("low_tail_probability_invalid")
        threshold = float(np.quantile(y, q))
        observed_cdf = float(np.mean(y <= threshold))
        predicted_cdf = float(np.mean(p <= threshold))
        output.append({"probability": q, "threshold": threshold, "observed_cdf": observed_cdf,
                       "predicted_cdf": predicted_cdf, "bias": predicted_cdf - observed_cdf})
    return output


def observed_decile_bias(truth: Sequence[float], prediction: Sequence[float]) -> list[dict[str, Any]]:
    """Compute prediction-minus-observation bias within observed rank deciles."""
    y, p = _paired(truth, prediction)
    output = []
    for decile, members in enumerate(np.array_split(np.argsort(y, kind="stable"), 10), 1):
        residual = p[members] - y[members]
        output.append({"decile": decile, "count": len(members),
                       "bias": float(residual.mean()) if len(members) else None})
    return output


def fold_stability(
    truth: Sequence[float], prediction: Sequence[float], fold_ids: Sequence[int]
) -> dict[str, Any]:
    """Report unweighted risk by fold and its between-fold range."""
    y, p = _paired(truth, prediction)
    folds = np.asarray(fold_ids)
    if folds.ndim != 1 or len(folds) != len(y):
        raise ScienceDiagnosticsError("fold_ids_shape_invalid")
    by_fold = {str(fold): {"count": int((folds == fold).sum()),
                           **regression_metrics(y[folds == fold], p[folds == fold])}
               for fold in sorted(set(folds.tolist()))}
    return {"definition": "unweighted_fold_stability_v1", "by_fold": by_fold,
            "mae_range": max(v["mae"] for v in by_fold.values()) - min(v["mae"] for v in by_fold.values()),
            "rmse_range": max(v["rmse"] for v in by_fold.values()) - min(v["rmse"] for v in by_fold.values())}


def select_household_evaluation(
    records: Sequence[Mapping[str, Any]], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    """Select complete households, or explicitly opt into incomplete evaluation."""
    complete = [row for row in records if row.get("status") == "complete"]
    incomplete = [row for row in records if row.get("status") != "complete"]
    selected = list(records) if allow_incomplete else complete
    return {"selection": "all_including_incomplete" if allow_incomplete else "complete_only",
            "complete_count": len(complete), "incomplete_count": len(incomplete), "records": selected}


def welfare_diagnostics(
    person_truth: Sequence[float], person_prediction: Sequence[float],
    household_truth: Sequence[float], household_prediction: Sequence[float],
    *, person_fold_ids: Sequence[int] | None = None,
    household_fold_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the baseline person/household diagnostic suite without weights."""
    py, pp = _paired(person_truth, person_prediction)
    hy, hp = _paired(household_truth, household_prediction)
    result: dict[str, Any] = {
        "weighting": "none", "person": regression_metrics(py, pp),
        "positive_amount_head": positive_amount_head_metrics(py, pp),
        "household": {**regression_metrics(hy, hp), "spearman_rank": spearman_rank(hy, hp)},
        "transitions": {"quintile": transition_diagnostics(hy, hp, bins=5),
                        "decile": transition_diagnostics(hy, hp, bins=10)},
        "low_tail_cdf_bias": low_tail_cdf_bias(hy, hp),
        "observed_decile_bias": observed_decile_bias(hy, hp),
    }
    if person_fold_ids is not None:
        result["person_fold_stability"] = fold_stability(py, pp, person_fold_ids)
    if household_fold_ids is not None:
        result["household_fold_stability"] = fold_stability(hy, hp, household_fold_ids)
    return result


def oracle_presence_diagnostic(truth: Sequence[float], oracle_prediction: Sequence[float]) -> dict[str, Any]:
    """Evaluate true-presence-conditioned output and label it non-deployable."""
    y, p = _paired(truth, oracle_prediction)
    return {"deployable": False, "role": "oracle_presence_diagnostic_only",
            "metrics": regression_metrics(y, p)}


def target_status_positive_upper_bound(
    truth: Sequence[float], prediction: Sequence[float], target_present: Sequence[bool]
) -> dict[str, Any]:
    """Positive-amount upper bound conditioned on true target status."""
    y, p = _paired(truth, prediction)
    present = np.asarray(target_present, bool)
    if present.ndim != 1 or len(present) != len(y):
        raise ScienceDiagnosticsError("target_status_shape_invalid")
    mask = present & (y > 0)
    return {"deployable": False, "role": "target_status_conditioned_positive_amount_upper_bound",
            "count": int(mask.sum()), "metrics": regression_metrics(y[mask], p[mask]) if mask.any() else None}
