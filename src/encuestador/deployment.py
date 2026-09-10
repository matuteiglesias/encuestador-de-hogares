"""Serializable full-fit model states for deployable direct and lean hurdle scoring."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cascade import CategoricalLatentSpec
from .crossfit import EstimatorProtocol, crossfit_predict
from .scientific_primitives import FoldManifest
from .terminal import HurdleComponents, HurdleEstimator


class DeploymentError(ValueError):
    """Raised when a deployable fitted state cannot preserve training semantics."""


def _matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise DeploymentError(f"{name}_must_be_nonempty_finite_2d")
    return array


def _ordered_probabilities(
    model: EstimatorProtocol,
    x: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if not callable(predict_proba) or classes is None:
        raise DeploymentError("latent_probability_interface_missing")
    raw = np.asarray(predict_proba(x), dtype=float)
    model_labels = tuple(str(value) for value in np.asarray(classes).tolist())
    if set(model_labels) != set(labels) or raw.shape != (len(x), len(model_labels)):
        raise DeploymentError("latent_probability_class_contract_changed")
    lookup = {label: index for index, label in enumerate(model_labels)}
    values = raw[:, [lookup[label] for label in labels]]
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0):
        raise DeploymentError("latent_probability_values_invalid")
    return values


@dataclass(frozen=True)
class FittedHurdleState:
    """Prediction-only hurdle state with no unpickleable estimator factories."""

    formulation: str
    retransformation: str
    smearing_factor: float
    constant_positive_probability: float | None
    presence_model: EstimatorProtocol | None
    amount_model: EstimatorProtocol
    training_eligibility: Mapping[str, int]

    @classmethod
    def from_estimator(cls, estimator: HurdleEstimator) -> FittedHurdleState:
        if estimator.amount_model is None or estimator.training_eligibility is None:
            raise DeploymentError("hurdle_estimator_not_fitted")
        retransformation = (
            estimator.retransformation
            if estimator.formulation == "log"
            else "linear_identity_v1"
        )
        return cls(
            formulation=estimator.formulation,
            retransformation=retransformation,
            smearing_factor=float(estimator.smearing_factor),
            constant_positive_probability=estimator.constant_positive_probability,
            presence_model=estimator.presence_model,
            amount_model=estimator.amount_model,
            training_eligibility=dict(estimator.training_eligibility),
        )

    def predict_components(
        self, x: np.ndarray | Sequence[Sequence[float]]
    ) -> HurdleComponents:
        features = _matrix(x, "scoring_features")
        if self.constant_positive_probability is not None:
            p_positive = np.full(
                len(features), self.constant_positive_probability, dtype=float
            )
        else:
            if self.presence_model is None:
                raise DeploymentError("fitted_presence_state_missing")
            classes = tuple(
                str(value)
                for value in np.asarray(getattr(self.presence_model, "classes_", [])).tolist()
            )
            if set(classes) != {"0", "1"}:
                raise DeploymentError("fitted_presence_class_contract_changed")
            probabilities = np.asarray(
                self.presence_model.predict_proba(features), dtype=float
            )
            p_positive = probabilities[:, classes.index("1")]
        raw_amount = np.asarray(self.amount_model.predict(features), dtype=float)
        if raw_amount.ndim != 1 or len(raw_amount) != len(features):
            raise DeploymentError("fitted_amount_prediction_shape_invalid")
        if self.formulation == "log":
            amount = np.exp(raw_amount) * self.smearing_factor
        elif self.formulation == "gamma":
            amount = raw_amount
        else:
            raise DeploymentError(f"fitted_hurdle_formulation_unknown:{self.formulation}")
        if not np.isfinite(amount).all() or np.any(amount <= 0):
            raise DeploymentError("fitted_positive_amount_invalid")
        if not np.isfinite(p_positive).all() or np.any(p_positive < 0) or np.any(p_positive > 1):
            raise DeploymentError("fitted_positive_probability_invalid")
        unconditional = p_positive * amount
        return HurdleComponents(
            p_positive=p_positive,
            positive_amount=amount,
            unconditional_income=unconditional,
        )

    def predict(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        return self.predict_components(x).unconditional_income


@dataclass(frozen=True)
class FittedDirectHurdleModel:
    feature_names: tuple[str, ...]
    terminal: FittedHurdleState
    architecture_id: str = "direct_v1"

    def predict_components(
        self, x: np.ndarray | Sequence[Sequence[float]]
    ) -> HurdleComponents:
        features = _matrix(x, "direct_scoring_features")
        if features.shape[1] != len(self.feature_names):
            raise DeploymentError("direct_scoring_feature_width_mismatch")
        return self.terminal.predict_components(features)


@dataclass(frozen=True)
class FittedLatentState:
    target: str
    class_labels: tuple[str, ...]
    model: EstimatorProtocol


@dataclass(frozen=True)
class FittedLeanHurdleModel:
    feature_names: tuple[str, ...]
    latent_states: tuple[FittedLatentState, ...]
    terminal: FittedHurdleState
    architecture_id: str = "lean_labor_state_v1"

    def predict_components(
        self, x: np.ndarray | Sequence[Sequence[float]]
    ) -> HurdleComponents:
        features = _matrix(x, "lean_scoring_features")
        if features.shape[1] != len(self.feature_names):
            raise DeploymentError("lean_scoring_feature_width_mismatch")
        blocks = [
            _ordered_probabilities(state.model, features, state.class_labels)
            for state in self.latent_states
        ]
        design = np.column_stack([features, *blocks])
        return self.terminal.predict_components(design)

    def latent_probabilities(
        self, x: np.ndarray | Sequence[Sequence[float]]
    ) -> dict[str, np.ndarray]:
        features = _matrix(x, "lean_scoring_features")
        return {
            state.target: _ordered_probabilities(
                state.model, features, state.class_labels
            )
            for state in self.latent_states
        }


def fit_direct_hurdle_model(
    x: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[Any] | np.ndarray,
    terminal_factory: callable,
    *,
    feature_names: Sequence[str],
) -> FittedDirectHurdleModel:
    features = _matrix(x, "direct_training_features")
    estimator = terminal_factory()
    if not isinstance(estimator, HurdleEstimator):
        raise DeploymentError("direct_terminal_factory_must_return_hurdle")
    estimator.fit(features, np.asarray(y, dtype=object))
    return FittedDirectHurdleModel(
        feature_names=tuple(feature_names),
        terminal=FittedHurdleState.from_estimator(estimator),
    )


def fit_lean_hurdle_model(
    x: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[Any] | np.ndarray,
    latent_targets: Mapping[str, np.ndarray | Sequence[Any]],
    manifest: FoldManifest,
    latent_specs: Sequence[CategoricalLatentSpec],
    terminal_factory: callable,
    *,
    feature_names: Sequence[str],
) -> FittedLeanHurdleModel:
    """Fit deployable full latent models and terminal on OOF latent meta-features."""
    features = _matrix(x, "lean_training_features")
    if len(features) != len(manifest.row_ids):
        raise DeploymentError("lean_training_manifest_length_mismatch")
    training_blocks: list[np.ndarray] = []
    fitted_latents: list[FittedLatentState] = []
    for spec in latent_specs:
        values = np.asarray(latent_targets[spec.target])
        if values.ndim != 1 or len(values) != len(features):
            raise DeploymentError(f"latent_target_shape_invalid:{spec.target}")
        oof = crossfit_predict(
            features,
            values,
            manifest,
            spec.estimator_factory,
            target=spec.target,
            kind="probability",
            class_labels=spec.class_labels,
            metadata={"deployment_role": "terminal_meta_training_oof"},
        )
        full_model = spec.estimator_factory()
        full_model.fit(features, values)
        _ordered_probabilities(full_model, features[:1], spec.class_labels)
        training_blocks.append(oof.values)
        fitted_latents.append(
            FittedLatentState(
                target=spec.target,
                class_labels=spec.class_labels,
                model=full_model,
            )
        )
    design = np.column_stack([features, *training_blocks])
    terminal = terminal_factory()
    if not isinstance(terminal, HurdleEstimator):
        raise DeploymentError("lean_terminal_factory_must_return_hurdle")
    terminal.fit(design, np.asarray(y, dtype=object))
    return FittedLeanHurdleModel(
        feature_names=tuple(feature_names),
        latent_states=tuple(fitted_latents),
        terminal=FittedHurdleState.from_estimator(terminal),
    )
