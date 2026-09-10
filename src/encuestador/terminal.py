"""Terminal hurdle welfare model with explicit target eligibility and linear-scale outputs."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .crossfit import EstimatorProtocol
from .scientific_primitives import FoldManifest, PredictionArtifact


class HurdleError(ValueError):
    """Raised when terminal welfare supervision or fitting is scientifically invalid."""


EstimatorFactory = Callable[[], EstimatorProtocol]
NONRESPONSE_CODE = -9.0


@dataclass(frozen=True)
class TargetEligibility:
    numeric: np.ndarray
    valid: np.ndarray
    positive: np.ndarray
    zero: np.ndarray
    nonresponse: np.ndarray
    missing: np.ndarray

    @property
    def counts(self) -> dict[str, int]:
        return {
            "rows": len(self.numeric),
            "eligible": int(self.valid.sum()),
            "positive": int(self.positive.sum()),
            "zero": int(self.zero.sum()),
            "nonresponse": int(self.nonresponse.sum()),
            "missing": int(self.missing.sum()),
        }


def classify_income_target(values: Sequence[Any] | np.ndarray) -> TargetEligibility:
    """Classify zero/positive supervision while keeping -9 and missing unavailable."""
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise HurdleError("terminal_target_must_be_nonempty_1d")
    numeric = np.full(len(raw), np.nan, dtype=float)
    missing = np.zeros(len(raw), dtype=bool)
    nonresponse = np.zeros(len(raw), dtype=bool)
    for index, value in enumerate(raw.tolist()):
        if value is None or (isinstance(value, str) and not value.strip()):
            missing[index] = True
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise HurdleError(f"terminal_target_not_numeric:{index}") from exc
        if not np.isfinite(number):
            missing[index] = True
            continue
        if number == NONRESPONSE_CODE:
            nonresponse[index] = True
            continue
        if number < 0:
            raise HurdleError(f"unexpected_negative_terminal_income:{number:g}")
        numeric[index] = number
    valid = ~(missing | nonresponse)
    positive = valid & (numeric > 0)
    zero = valid & (numeric == 0)
    if not np.array_equal(valid, positive | zero):
        raise HurdleError("terminal_target_eligibility_internal_error")
    return TargetEligibility(
        numeric=numeric,
        valid=valid,
        positive=positive,
        zero=zero,
        nonresponse=nonresponse,
        missing=missing,
    )


@dataclass(frozen=True)
class HurdleComponents:
    p_positive: np.ndarray
    positive_amount: np.ndarray
    unconditional_income: np.ndarray


class HurdleEstimator:
    """Estimator-protocol terminal head: presence × conditional positive amount."""

    def __init__(
        self,
        *,
        formulation: str,
        presence_factory: EstimatorFactory,
        amount_factory: EstimatorFactory,
        retransformation: str = "duan_smearing_v1",
    ) -> None:
        if formulation not in {"log", "gamma"}:
            raise HurdleError(f"unsupported_hurdle_formulation:{formulation}")
        if formulation == "log" and retransformation != "duan_smearing_v1":
            raise HurdleError(f"unsupported_log_retransformation:{retransformation}")
        if formulation == "gamma" and retransformation not in {"identity", "linear_identity_v1"}:
            raise HurdleError(f"unsupported_gamma_retransformation:{retransformation}")
        self.formulation = formulation
        self.presence_factory = presence_factory
        self.amount_factory = amount_factory
        self.retransformation = retransformation
        self.presence_model: EstimatorProtocol | None = None
        self.amount_model: EstimatorProtocol | None = None
        self.constant_positive_probability: float | None = None
        self.smearing_factor = 1.0
        self.training_eligibility: dict[str, int] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> HurdleEstimator:
        features = np.asarray(x, dtype=float)
        if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
            raise HurdleError("hurdle_training_features_invalid")
        eligibility = classify_income_target(y)
        if len(eligibility.numeric) != len(features):
            raise HurdleError("hurdle_target_feature_length_mismatch")
        if not np.any(eligibility.valid):
            raise HurdleError("terminal_income_supervision_empty")
        if not np.any(eligibility.positive):
            raise HurdleError("positive_amount_training_empty")

        presence_y = eligibility.positive[eligibility.valid].astype(int)
        unique_presence = np.unique(presence_y)
        if len(unique_presence) == 1:
            self.constant_positive_probability = float(unique_presence[0])
            self.presence_model = None
        else:
            model = self.presence_factory()
            model.fit(features[eligibility.valid], presence_y)
            classes = getattr(model, "classes_", None)
            predict_proba = getattr(model, "predict_proba", None)
            if classes is None or not callable(predict_proba):
                raise HurdleError("presence_estimator_probability_interface_missing")
            if {str(value) for value in np.asarray(classes).tolist()} != {"0", "1"}:
                raise HurdleError("presence_estimator_class_set_invalid")
            self.presence_model = model
            self.constant_positive_probability = None

        positive_y = eligibility.numeric[eligibility.positive]
        amount_model = self.amount_factory()
        if self.formulation == "log":
            transformed = np.log(positive_y)
            amount_model.fit(features[eligibility.positive], transformed)
            fitted_log = np.asarray(
                amount_model.predict(features[eligibility.positive]), dtype=float
            )
            if fitted_log.ndim != 1 or len(fitted_log) != len(transformed):
                raise HurdleError("log_amount_training_prediction_shape_invalid")
            factor = float(np.mean(np.exp(transformed - fitted_log)))
            if not np.isfinite(factor) or factor <= 0:
                raise HurdleError("log_smearing_factor_invalid")
            self.smearing_factor = factor
        else:
            amount_model.fit(features[eligibility.positive], positive_y)
            self.smearing_factor = 1.0
        self.amount_model = amount_model
        self.training_eligibility = eligibility.counts
        return self

    def _positive_probability(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise HurdleError("hurdle_scoring_features_invalid")
        if self.constant_positive_probability is not None:
            return np.full(len(features), self.constant_positive_probability, dtype=float)
        if self.presence_model is None:
            raise HurdleError("hurdle_presence_model_not_fitted")
        classes = tuple(str(value) for value in np.asarray(self.presence_model.classes_).tolist())
        positive_index = classes.index("1")
        probabilities = np.asarray(self.presence_model.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape != (len(features), len(classes)):
            raise HurdleError("presence_probability_shape_invalid")
        output = probabilities[:, positive_index]
        if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1):
            raise HurdleError("presence_probability_invalid")
        return output

    def predict_components(self, x: np.ndarray) -> HurdleComponents:
        features = np.asarray(x, dtype=float)
        if self.amount_model is None:
            raise HurdleError("hurdle_amount_model_not_fitted")
        p_positive = self._positive_probability(features)
        raw_amount = np.asarray(self.amount_model.predict(features), dtype=float)
        if raw_amount.ndim != 1 or len(raw_amount) != len(features):
            raise HurdleError("positive_amount_prediction_shape_invalid")
        if self.formulation == "log":
            positive_amount = np.exp(raw_amount) * self.smearing_factor
        else:
            positive_amount = raw_amount
        if not np.isfinite(positive_amount).all() or np.any(positive_amount <= 0):
            raise HurdleError("positive_amount_prediction_must_be_positive_finite")
        unconditional = p_positive * positive_amount
        if not np.isfinite(unconditional).all() or np.any(unconditional < 0):
            raise HurdleError("unconditional_income_prediction_invalid")
        return HurdleComponents(
            p_positive=p_positive,
            positive_amount=positive_amount,
            unconditional_income=unconditional,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_components(x).unconditional_income

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "terminal_formulation": f"hurdle_{self.formulation}",
            "retransformation": (
                self.retransformation if self.formulation == "log" else "linear_identity_v1"
            ),
            "smearing_factor": self.smearing_factor if self.formulation == "log" else None,
            "training_eligibility": self.training_eligibility,
        }


@dataclass(frozen=True)
class HurdlePredictionBundle:
    p_positive: PredictionArtifact
    positive_amount_prediction: PredictionArtifact
    unconditional_expected_income: PredictionArtifact
    target_eligibility: Mapping[str, int]
    formulation: str
    retransformation: str


def _bundle(
    *,
    target: str,
    row_ids: Sequence[str],
    components: HurdleComponents,
    source: str,
    formulation: str,
    retransformation: str,
    target_eligibility: Mapping[str, int],
    fold_ids: Sequence[int] | None = None,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HurdlePredictionBundle:
    base = {
        "terminal_formulation": f"hurdle_{formulation}",
        "retransformation": retransformation,
        **dict(metadata or {}),
    }
    ids = tuple(row_ids)
    folds = tuple(fold_ids) if fold_ids is not None else None
    presence_values = np.column_stack([1.0 - components.p_positive, components.p_positive])
    return HurdlePredictionBundle(
        p_positive=PredictionArtifact(
            target=f"{target}__positive",
            kind="probability",
            row_ids=ids,
            values=presence_values,
            source=source,
            class_labels=("0", "1"),
            fold_ids=folds,
            run_id=run_id,
            metadata={**base, "component": "p_positive"},
        ),
        positive_amount_prediction=PredictionArtifact(
            target=f"{target}__positive_amount",
            kind="regression",
            row_ids=ids,
            values=components.positive_amount,
            source=source,
            fold_ids=folds,
            run_id=run_id,
            metadata={**base, "component": "positive_amount_prediction"},
        ),
        unconditional_expected_income=PredictionArtifact(
            target=target,
            kind="regression",
            row_ids=ids,
            values=components.unconditional_income,
            source=source,
            fold_ids=folds,
            run_id=run_id,
            metadata={**base, "component": "unconditional_expected_income"},
        ),
        target_eligibility=dict(target_eligibility),
        formulation=formulation,
        retransformation=retransformation,
    )


def crossfit_hurdle(
    x: np.ndarray,
    y: Sequence[Any] | np.ndarray,
    manifest: FoldManifest,
    estimator_factory: Callable[[], HurdleEstimator],
    *,
    target: str,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HurdlePredictionBundle:
    """Generate hurdle components under the supplied household-safe outer folds."""
    features = np.asarray(x, dtype=float)
    raw_target = np.asarray(y, dtype=object)
    if features.ndim != 2 or len(features) != len(raw_target):
        raise HurdleError("hurdle_crossfit_shape_invalid")
    if len(features) != len(manifest.row_ids) or not np.isfinite(features).all():
        raise HurdleError("hurdle_crossfit_manifest_or_features_invalid")
    eligibility = classify_income_target(raw_target)
    p_positive = np.full(len(features), np.nan, dtype=float)
    positive_amount = np.full(len(features), np.nan, dtype=float)
    unconditional = np.full(len(features), np.nan, dtype=float)
    fold_array = manifest.as_array()
    formulation: str | None = None
    retransformation: str | None = None
    for fold in range(manifest.n_splits):
        train = fold_array != fold
        holdout = fold_array == fold
        if not np.any(train) or not np.any(holdout):
            raise HurdleError(f"invalid_hurdle_fold_partition:{fold}")
        estimator = estimator_factory()
        estimator.fit(features[train], raw_target[train])
        components = estimator.predict_components(features[holdout])
        p_positive[holdout] = components.p_positive
        positive_amount[holdout] = components.positive_amount
        unconditional[holdout] = components.unconditional_income
        formulation = estimator.formulation
        retransformation = (
            estimator.retransformation if estimator.formulation == "log" else "linear_identity_v1"
        )
    if not all(np.isfinite(value).all() for value in (p_positive, positive_amount, unconditional)):
        raise HurdleError("hurdle_oof_prediction_incomplete")
    assert formulation is not None and retransformation is not None
    return _bundle(
        target=target,
        row_ids=manifest.row_ids,
        components=HurdleComponents(p_positive, positive_amount, unconditional),
        source="oof",
        formulation=formulation,
        retransformation=retransformation,
        target_eligibility=eligibility.counts,
        fold_ids=manifest.fold_ids,
        run_id=run_id,
        metadata=metadata,
    )


def fit_hurdle_and_score(
    x_train: np.ndarray,
    y_train: Sequence[Any] | np.ndarray,
    x_score: np.ndarray,
    score_row_ids: Sequence[str],
    estimator_factory: Callable[[], HurdleEstimator],
    *,
    target: str,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HurdlePredictionBundle:
    """Fit deployable hurdle state once and score a distinct external frame."""
    train = np.asarray(x_train, dtype=float)
    score = np.asarray(x_score, dtype=float)
    if train.ndim != 2 or score.ndim != 2 or train.shape[1] != score.shape[1]:
        raise HurdleError("hurdle_full_score_feature_shape_invalid")
    if len(score) != len(score_row_ids):
        raise HurdleError("hurdle_score_row_id_length_mismatch")
    eligibility = classify_income_target(y_train)
    estimator = estimator_factory()
    estimator.fit(train, np.asarray(y_train, dtype=object))
    components = estimator.predict_components(score)
    retransformation = (
        estimator.retransformation if estimator.formulation == "log" else "linear_identity_v1"
    )
    return _bundle(
        target=target,
        row_ids=score_row_ids,
        components=components,
        source="score",
        formulation=estimator.formulation,
        retransformation=retransformation,
        target_eligibility=eligibility.counts,
        run_id=run_id,
        metadata=metadata,
    )
