"""Explicit KL/moment projection for governed aggregate anchors.

Anchors are interventions on predicted probability artifacts. They never mutate
source donor observations and are only valid when the caller has already proven
that the modeled rows and aggregate target share the same population universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class AnchorProjectionError(ValueError):
    """Raised when an aggregate moment cannot be applied truthfully."""


@dataclass(frozen=True)
class AnchorDiagnostics:
    method: str
    class_labels: tuple[str, ...]
    target_shares: tuple[float, ...]
    raw_shares: tuple[float, ...]
    anchored_shares: tuple[float, ...]
    lambdas: tuple[float, ...]
    weighted_mean_kl: float
    mean_probability_displacement: float
    median_probability_displacement: float
    max_probability_displacement: float
    iterations: int
    constraint_residual: float
    universe_contract: str
    anchor_release_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "class_labels": list(self.class_labels),
            "target_shares": dict(zip(self.class_labels, self.target_shares, strict=True)),
            "raw_shares": dict(zip(self.class_labels, self.raw_shares, strict=True)),
            "anchored_shares": dict(
                zip(self.class_labels, self.anchored_shares, strict=True)
            ),
            "lambda": dict(zip(self.class_labels, self.lambdas, strict=True)),
            "weighted_mean_kl": self.weighted_mean_kl,
            "mean_probability_displacement": self.mean_probability_displacement,
            "median_probability_displacement": self.median_probability_displacement,
            "max_probability_displacement": self.max_probability_displacement,
            "iterations": self.iterations,
            "constraint_residual": self.constraint_residual,
            "universe_contract": self.universe_contract,
            "anchor_release_id": self.anchor_release_id,
        }


def _weights(value: np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        output = np.ones(rows, dtype=float)
    else:
        output = np.asarray(value, dtype=float)
        if output.ndim != 1 or len(output) != rows:
            raise AnchorProjectionError("anchor_weight_shape_invalid")
        if not np.isfinite(output).all() or np.any(output <= 0):
            raise AnchorProjectionError("anchor_weights_must_be_positive_finite")
    return output / float(output.sum())


def _validate_probabilities(value: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(value, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise AnchorProjectionError("anchor_probabilities_must_be_nonempty_2d")
    if probabilities.shape[1] < 2:
        raise AnchorProjectionError("anchor_requires_at_least_two_classes")
    if not np.isfinite(probabilities).all():
        raise AnchorProjectionError("anchor_probabilities_non_finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise AnchorProjectionError("anchor_probabilities_out_of_bounds")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise AnchorProjectionError("anchor_probability_rows_must_sum_to_one")
    epsilon = 1e-12
    clipped = np.clip(probabilities, epsilon, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def multiclass_kl_moment_projection(
    probabilities: np.ndarray,
    *,
    class_labels: tuple[str, ...],
    target_shares: dict[str, float],
    universe_contract: str,
    anchor_release_id: str,
    weights: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iter: int = 500,
) -> tuple[np.ndarray, AnchorDiagnostics]:
    """Project probabilities to exact class moments by exponential tilting.

    The solution has q-star_ik proportional to q_ik * exp(lambda_k) and minimizes
    weighted row-wise categorical KL divergence subject to requested class shares.
    """
    raw = _validate_probabilities(probabilities)
    if len(class_labels) != raw.shape[1] or len(set(class_labels)) != len(class_labels):
        raise AnchorProjectionError("anchor_class_labels_invalid")
    if not universe_contract:
        raise AnchorProjectionError("anchor_universe_contract_required")
    if not anchor_release_id:
        raise AnchorProjectionError("anchor_release_id_required")
    if set(target_shares) != set(class_labels):
        raise AnchorProjectionError("anchor_target_class_set_mismatch")

    target = np.asarray([float(target_shares[label]) for label in class_labels])
    if (
        not np.isfinite(target).all()
        or np.any(target <= 0)
        or np.any(target >= 1)
        or not np.isclose(target.sum(), 1.0, atol=1e-10, rtol=0)
    ):
        raise AnchorProjectionError("anchor_target_shares_invalid")

    normalized_weights = _weights(weights, len(raw))
    raw_shares = normalized_weights @ raw
    log_raw = np.log(raw)
    lambdas = np.zeros(raw.shape[1], dtype=float)
    anchored = raw.copy()
    residual = float("inf")

    for iteration in range(1, max_iter + 1):
        logits = log_raw + lambdas
        logits -= logits.max(axis=1, keepdims=True)
        anchored = np.exp(logits)
        anchored /= anchored.sum(axis=1, keepdims=True)
        current = normalized_weights @ anchored
        residual = float(np.max(np.abs(current - target)))
        if residual <= tolerance:
            break
        if np.any(current <= 0):
            raise AnchorProjectionError("anchor_projection_degenerate_moment")
        lambdas += np.log(target / current)
        lambdas -= lambdas[-1]
    else:
        raise AnchorProjectionError(
            f"anchor_projection_did_not_converge:{residual:.6g}"
        )

    displacement = np.abs(anchored - raw)
    row_kl = np.sum(anchored * (np.log(anchored) - log_raw), axis=1)
    diagnostics = AnchorDiagnostics(
        method="categorical_kl_exponential_tilt_v1",
        class_labels=class_labels,
        target_shares=tuple(float(value) for value in target),
        raw_shares=tuple(float(value) for value in raw_shares),
        anchored_shares=tuple(float(value) for value in normalized_weights @ anchored),
        lambdas=tuple(float(value) for value in lambdas),
        weighted_mean_kl=float(normalized_weights @ row_kl),
        mean_probability_displacement=float(displacement.mean()),
        median_probability_displacement=float(np.median(displacement)),
        max_probability_displacement=float(displacement.max()),
        iterations=iteration,
        constraint_residual=residual,
        universe_contract=universe_contract,
        anchor_release_id=anchor_release_id,
    )
    return anchored, diagnostics


def binary_kl_logit_shift(
    positive_probability: np.ndarray,
    *,
    target_positive_share: float,
    universe_contract: str,
    anchor_release_id: str,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, AnchorDiagnostics]:
    """Binary convenience wrapper around the governed categorical projector."""
    positive = np.asarray(positive_probability, dtype=float)
    if positive.ndim != 1:
        raise AnchorProjectionError("binary_anchor_probability_must_be_1d")
    raw = np.column_stack([1.0 - positive, positive])
    anchored, diagnostics = multiclass_kl_moment_projection(
        raw,
        class_labels=("0", "1"),
        target_shares={
            "0": 1.0 - float(target_positive_share),
            "1": float(target_positive_share),
        },
        universe_contract=universe_contract,
        anchor_release_id=anchor_release_id,
        weights=weights,
    )
    return anchored[:, 1], diagnostics
