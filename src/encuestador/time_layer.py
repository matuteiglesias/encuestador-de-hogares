"""Leakage-safe explicit year/quarter/exception time correction for welfare hurdles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .terminal import classify_income_target

EXCEPTIONAL_PERIODS = {
    "2020-Q2": "pandemic_fieldwork_regime",
    "2024-Q1": "2024_h1_macroeconomic_shock",
    "2024-Q2": "2024_h1_macroeconomic_shock",
}


class TimeLayerError(ValueError):
    """Raised when an explicit measurement-time correction is not identifiable."""


def parse_period(period: str) -> tuple[int, int]:
    try:
        year_text, quarter_text = str(period).split("-Q", 1)
        year, quarter = int(year_text), int(quarter_text)
    except (TypeError, ValueError) as exc:
        raise TimeLayerError(f"invalid_period:{period}") from exc
    if quarter not in {1, 2, 3, 4}:
        raise TimeLayerError(f"invalid_period:{period}")
    return year, quarter


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _solve_logit_shift(probability: np.ndarray, target_mean: float) -> float:
    if not 0 <= target_mean <= 1:
        raise TimeLayerError("presence_target_mean_out_of_range")
    target = float(np.clip(target_mean, 1e-6, 1.0 - 1e-6))
    logits = _logit(probability)
    low, high = -30.0, 30.0
    for _ in range(120):
        midpoint = (low + high) / 2.0
        mean = float(_sigmoid(logits + midpoint).mean())
        if mean < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _mean_log_ratio(observed: np.ndarray, predicted: np.ndarray) -> float:
    if not len(observed):
        raise TimeLayerError("amount_effect_has_no_positive_rows")
    if (
        not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or np.any(observed <= 0)
        or np.any(predicted <= 0)
    ):
        raise TimeLayerError("amount_effect_requires_positive_finite_values")
    ratio = float(observed.mean() / predicted.mean())
    if not np.isfinite(ratio) or ratio <= 0:
        raise TimeLayerError("amount_effect_ratio_invalid")
    return float(np.log(ratio))


@dataclass(frozen=True)
class TimeLayerFit:
    presence_year: dict[int, float]
    presence_quarter: dict[int, float]
    presence_exception: dict[str, float]
    amount_year: dict[int, float]
    amount_quarter: dict[int, float]
    amount_exception: dict[str, float]
    year_support: dict[int, dict[str, Any]]
    exception_policy: dict[str, str]
    fit_row_count: int
    ordinary_fit_row_count: int

    def apply(
        self,
        periods: Sequence[str],
        base_positive_probability: np.ndarray,
        base_positive_amount: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        period_values = tuple(str(value) for value in periods)
        probability = np.asarray(base_positive_probability, dtype=float)
        amount = np.asarray(base_positive_amount, dtype=float)
        if (
            probability.ndim != 1
            or amount.ndim != 1
            or len(probability) != len(period_values)
            or len(amount) != len(period_values)
        ):
            raise TimeLayerError("time_layer_apply_shape_invalid")
        if (
            not np.isfinite(probability).all()
            or not np.isfinite(amount).all()
            or np.any(probability < 0)
            or np.any(probability > 1)
            or np.any(amount <= 0)
        ):
            raise TimeLayerError("time_layer_apply_base_predictions_invalid")

        presence_effect = np.zeros(len(period_values), dtype=float)
        amount_effect = np.zeros(len(period_values), dtype=float)
        for index, period in enumerate(period_values):
            year, quarter = parse_period(period)
            presence_effect[index] = (
                self.presence_year.get(year, 0.0)
                + self.presence_quarter.get(quarter, 0.0)
                + self.presence_exception.get(period, 0.0)
            )
            amount_effect[index] = (
                self.amount_year.get(year, 0.0)
                + self.amount_quarter.get(quarter, 0.0)
                + self.amount_exception.get(period, 0.0)
            )
        final_probability = _sigmoid(_logit(probability) + presence_effect)
        final_amount = amount * np.exp(amount_effect)
        return final_probability, final_amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "presence": {
                "year_effect": {str(key): value for key, value in self.presence_year.items()},
                "quarter_effect": {
                    str(key): value for key, value in self.presence_quarter.items()
                },
                "exceptional_period_effect": self.presence_exception,
            },
            "positive_amount": {
                "year_effect": {str(key): value for key, value in self.amount_year.items()},
                "quarter_effect": {
                    str(key): value for key, value in self.amount_quarter.items()
                },
                "exceptional_period_effect": self.amount_exception,
            },
            "year_support": {
                str(key): value for key, value in sorted(self.year_support.items())
            },
            "exception_policy": dict(self.exception_policy),
            "fit_row_count": self.fit_row_count,
            "ordinary_fit_row_count": self.ordinary_fit_row_count,
            "semantics": {
                "ordinary_year_excludes_exceptional_periods": True,
                "ordinary_quarter_excludes_exceptional_periods": True,
                "exceptional_rows_remain_measurable": True,
                "forecasting_authorized": False,
            },
        }


def fit_time_layer(
    periods: Sequence[str],
    base_positive_probability: np.ndarray,
    base_positive_amount: np.ndarray,
    target_income: Sequence[Any] | np.ndarray,
    *,
    exception_policy: dict[str, str] | None = None,
) -> TimeLayerFit:
    """Fit explicit time effects using training-fold evidence only.

    Ordinary year and quarter effects exclude exceptional periods. Dedicated
    exceptional-period corrections are then fitted on the retained exceptional
    observations after the ordinary effects have been established.
    """
    exception_policy = dict(exception_policy or EXCEPTIONAL_PERIODS)
    period_values = np.asarray([str(value) for value in periods], dtype=object)
    probability = np.asarray(base_positive_probability, dtype=float)
    amount = np.asarray(base_positive_amount, dtype=float)
    if (
        period_values.ndim != 1
        or probability.ndim != 1
        or amount.ndim != 1
        or not (len(period_values) == len(probability) == len(amount))
        or not len(period_values)
    ):
        raise TimeLayerError("time_layer_fit_shape_invalid")
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(amount).all()
        or np.any(probability < 0)
        or np.any(probability > 1)
        or np.any(amount <= 0)
    ):
        raise TimeLayerError("time_layer_fit_base_predictions_invalid")

    eligibility = classify_income_target(target_income)
    if len(eligibility.numeric) != len(period_values):
        raise TimeLayerError("time_layer_target_length_mismatch")
    years = np.asarray([parse_period(value)[0] for value in period_values], dtype=int)
    quarters = np.asarray([parse_period(value)[1] for value in period_values], dtype=int)
    exceptional = np.asarray(
        [value in exception_policy for value in period_values], dtype=bool
    )
    ordinary_valid = (~exceptional) & eligibility.valid

    presence_year: dict[int, float] = {}
    presence_quarter: dict[int, float] = {}
    amount_year: dict[int, float] = {}
    amount_quarter: dict[int, float] = {}

    presence_adjusted = probability.copy()
    amount_adjusted = amount.copy()

    for year in sorted(set(years.tolist())):
        mask = ordinary_valid & (years == year)
        if np.any(mask):
            shift = _solve_logit_shift(
                presence_adjusted[mask],
                float(eligibility.positive[mask].mean()),
            )
            presence_year[year] = shift
            presence_adjusted[years == year] = _sigmoid(
                _logit(presence_adjusted[years == year]) + shift
            )

        positive_mask = (~exceptional) & eligibility.positive & (years == year)
        if np.any(positive_mask):
            effect = _mean_log_ratio(
                eligibility.numeric[positive_mask],
                amount_adjusted[positive_mask],
            )
            amount_year[year] = effect
            amount_adjusted[years == year] *= np.exp(effect)

    for quarter in (1, 2, 3, 4):
        mask = ordinary_valid & (quarters == quarter)
        if np.any(mask):
            shift = _solve_logit_shift(
                presence_adjusted[mask],
                float(eligibility.positive[mask].mean()),
            )
            presence_quarter[quarter] = shift
            presence_adjusted[quarters == quarter] = _sigmoid(
                _logit(presence_adjusted[quarters == quarter]) + shift
            )

        positive_mask = (~exceptional) & eligibility.positive & (quarters == quarter)
        if np.any(positive_mask):
            effect = _mean_log_ratio(
                eligibility.numeric[positive_mask],
                amount_adjusted[positive_mask],
            )
            amount_quarter[quarter] = effect
            amount_adjusted[quarters == quarter] *= np.exp(effect)

    presence_exception: dict[str, float] = {}
    amount_exception: dict[str, float] = {}
    for period in sorted(exception_policy):
        period_mask = period_values == period
        valid_mask = period_mask & eligibility.valid
        if np.any(valid_mask):
            presence_exception[period] = _solve_logit_shift(
                presence_adjusted[valid_mask],
                float(eligibility.positive[valid_mask].mean()),
            )
        positive_mask = period_mask & eligibility.positive
        if np.any(positive_mask):
            amount_exception[period] = _mean_log_ratio(
                eligibility.numeric[positive_mask],
                amount_adjusted[positive_mask],
            )

    year_support: dict[int, dict[str, Any]] = {}
    for year in sorted(set(years.tolist())):
        observed = sorted(
            {
                str(period)
                for period, row_year in zip(period_values, years, strict=True)
                if row_year == year
            }
        )
        ordinary = [period for period in observed if period not in exception_policy]
        year_support[year] = {
            "quarters_observed_for_year": len(observed),
            "periods_observed": observed,
            "ordinary_quarters_used_for_year_effect": len(ordinary),
            "ordinary_periods_used_for_year_effect": ordinary,
            "partial_year": len(observed) < 4,
        }

    return TimeLayerFit(
        presence_year=presence_year,
        presence_quarter=presence_quarter,
        presence_exception=presence_exception,
        amount_year=amount_year,
        amount_quarter=amount_quarter,
        amount_exception=amount_exception,
        year_support=year_support,
        exception_policy=exception_policy,
        fit_row_count=len(period_values),
        ordinary_fit_row_count=int(ordinary_valid.sum()),
    )
