"""Fixture-first longitudinal welfare runtime for L10/L11/L12 measurement arms."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .anchors import multiclass_kl_moment_projection
from .cascade import _subset_manifest
from .crossfit import crossfit_predict, fit_full_and_score
from .estimators import HGBClassifierAdapter, HGBRegressorAdapter
from .evaluation import (
    classification_diagnostics,
    distributional_regression_diagnostics,
)
from .scientific_primitives import FoldManifest, PredictionArtifact
from .terminal import (
    HurdleEstimator,
    HurdlePredictionBundle,
    classify_income_target,
)
from .time_layer import (
    EXCEPTIONAL_PERIODS,
    TimeLayerFit,
    fit_time_layer,
    parse_period,
)

LABOR_INDICATORS = ("activity_rate", "unemployment_rate", "subemployment_rate")
LABOR_CONTEXT_FIELDS = (
    "labor_national_activity_rate",
    "labor_national_unemployment_rate",
    "labor_national_subemployment_rate",
    "labor_regional_activity_deviation",
    "labor_regional_unemployment_deviation",
    "labor_regional_subemployment_deviation",
)
ARM_IDS = {"L10", "L11", "L12"}
FORBIDDEN_COMPOSITION_FIELDS = {
    "ESTADO",
    "CONDACT",
    "donor_condact",
    "donor_condact_vintage",
    "stale_labor_state",
    "target_current_labor_state",
}


class LongitudinalRuntimeError(ValueError):
    """Raised when a longitudinal arm would violate its clock or fold contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _period_index(period: str) -> int:
    year, quarter = parse_period(period)
    return year * 4 + quarter - 1


@dataclass(frozen=True)
class LongitudinalConfig:
    arm: str
    composition_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_field: str
    period_field: str
    region_field: str
    row_id_field: str
    group_field: str
    household_observation_field: str
    panel_person_field: str
    observed_labor_fields: tuple[str, ...]
    stale_labor_field: str
    current_labor_field: str
    elapsed_quarters_field: str
    allowed_panel_gaps: tuple[int, ...]
    n_splits: int
    presence_params: Mapping[str, Any]
    amount_params: Mapping[str, Any]
    transition_params: Mapping[str, Any]
    anchor_enabled: bool
    raw: Mapping[str, Any]
    digest: str

    @property
    def base_feature_names(self) -> tuple[str, ...]:
        return (*self.composition_features, *LABOR_CONTEXT_FIELDS)


def load_longitudinal_config(path: Path) -> LongitudinalConfig:
    path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LongitudinalRuntimeError("longitudinal_config_invalid") from exc
    if not isinstance(value, Mapping):
        raise LongitudinalRuntimeError("longitudinal_config_mapping_required")
    if value.get("schema") != "research.encuestador-longitudinal-config/v1":
        raise LongitudinalRuntimeError("unexpected_longitudinal_config_schema")
    arm = str(value.get("arm") or "")
    if arm not in ARM_IDS:
        raise LongitudinalRuntimeError(f"unknown_longitudinal_arm:{arm}")

    features = value.get("features")
    identity = value.get("identity")
    panel = value.get("panel")
    estimators = value.get("estimators")
    splits = value.get("splits")
    anchor = value.get("anchor") or {}
    if not all(isinstance(item, Mapping) for item in (features, identity, panel, estimators, splits, anchor)):
        raise LongitudinalRuntimeError("longitudinal_config_sections_missing")

    composition = tuple(str(name) for name in features.get("composition", ()))
    categorical = tuple(str(name) for name in features.get("categorical", ()))
    if not composition:
        raise LongitudinalRuntimeError("longitudinal_composition_features_required")
    forbidden = sorted(set(composition) & FORBIDDEN_COMPOSITION_FIELDS)
    if forbidden:
        raise LongitudinalRuntimeError(
            "current_or_donor_labor_forbidden_in_composition:" + ",".join(forbidden)
        )
    if not set(categorical).issubset(composition):
        raise LongitudinalRuntimeError("longitudinal_categorical_not_in_composition")

    allowed_gaps = tuple(int(value) for value in panel.get("allowed_elapsed_quarters", (1, 3)))
    if not allowed_gaps or any(gap <= 0 or gap > 5 for gap in allowed_gaps):
        raise LongitudinalRuntimeError("panel_gap_support_must_be_short_positive")
    n_splits = int(splits.get("n_splits", 5))
    if n_splits < 3:
        raise LongitudinalRuntimeError("longitudinal_runtime_requires_at_least_three_folds")
    if bool(anchor.get("enabled", False)) and arm != "L12":
        raise LongitudinalRuntimeError("aggregate_anchor_only_allowed_in_L12")

    raw = json.loads(json.dumps(value))
    digest = hashlib.sha256(_canonical_json(raw).encode()).hexdigest()
    return LongitudinalConfig(
        arm=arm,
        composition_features=composition,
        categorical_features=categorical,
        target_field=str(value.get("target_field") or "P47T_real"),
        period_field=str(identity.get("period_field") or "period"),
        region_field=str(identity.get("region_field") or "region_id"),
        row_id_field=str(identity.get("row_id_field") or ("row_id" if arm == "L10" else "pair_id")),
        group_field=str(identity.get("group_field") or "panel_household_id"),
        household_observation_field=str(
            identity.get("household_observation_field") or "household_observation_id"
        ),
        panel_person_field=str(panel.get("person_candidate_field") or "person_linkage_candidate_id"),
        observed_labor_fields=tuple(
            str(value)
            for value in panel.get("observed_labor_fields", ("ESTADO", "CONDACT"))
        ),
        stale_labor_field=str(panel.get("stale_labor_field") or "stale_labor_state"),
        current_labor_field=str(panel.get("current_labor_field") or "target_current_labor_state"),
        elapsed_quarters_field=str(panel.get("elapsed_quarters_field") or "elapsed_quarters"),
        allowed_panel_gaps=allowed_gaps,
        n_splits=n_splits,
        presence_params=dict(estimators.get("presence") or {}),
        amount_params=dict(estimators.get("positive_amount") or {}),
        transition_params=dict(estimators.get("transition") or {}),
        anchor_enabled=bool(anchor.get("enabled", False)),
        raw=raw,
        digest=digest,
    )


def _rate(value: str, key: tuple[str, str, str]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LongitudinalRuntimeError(f"labor_context_value_invalid:{key}") from exc
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise LongitudinalRuntimeError(f"labor_context_value_out_of_range:{key}")
    return number


def labor_context_index(
    observations: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index exact official cells without interpolation or person-state interpretation."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in observations:
        indicator = str(row.get("indicator_id") or "")
        geography = str(row.get("geography_id") or "")
        if indicator not in LABOR_INDICATORS:
            continue
        if geography != "total_31_agglomerates" and str(row.get("geography_level")) != "region":
            continue
        period = str(row.get("period") or "")
        key = (period, geography, indicator)
        if key in index:
            raise LongitudinalRuntimeError(f"duplicate_labor_context_cell:{key}")
        if row.get("value_status") != "observed":
            raise LongitudinalRuntimeError(f"labor_context_cell_not_observed:{key}")
        index[key] = {
            "value": _rate(str(row.get("value") or ""), key),
            "source_id": str(row.get("source_id") or ""),
            "source_snapshot_sha256": str(
                row.get("source_snapshot_sha256") or ""
            ),
            "source_cell_identity": str(row.get("source_cell_identity") or ""),
        }
    if not index:
        raise LongitudinalRuntimeError("labor_context_index_empty")
    return index


def attach_labor_context(
    rows: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    period_field: str = "period",
    region_field: str = "region_id",
) -> list[dict[str, Any]]:
    """Attach national levels plus regional deviations for three official rates."""
    index = labor_context_index(observations)
    output: list[dict[str, Any]] = []
    field_by_indicator = {
        "activity_rate": (
            "labor_national_activity_rate",
            "labor_regional_activity_deviation",
        ),
        "unemployment_rate": (
            "labor_national_unemployment_rate",
            "labor_regional_unemployment_deviation",
        ),
        "subemployment_rate": (
            "labor_national_subemployment_rate",
            "labor_regional_subemployment_deviation",
        ),
    }
    for source in rows:
        row = dict(source)
        period = str(row.get(period_field) or "")
        region = str(row.get(region_field) or "")
        if not period or not region:
            raise LongitudinalRuntimeError("labor_context_join_keys_missing")
        source_cells: list[dict[str, str]] = []
        for indicator in LABOR_INDICATORS:
            national_key = (period, "total_31_agglomerates", indicator)
            regional_key = (period, region, indicator)
            if national_key not in index or regional_key not in index:
                raise LongitudinalRuntimeError(
                    f"labor_context_join_incomplete:{period}:{region}:{indicator}"
                )
            national_record = index[national_key]
            regional_record = index[regional_key]
            national = float(national_record["value"])
            regional = float(regional_record["value"])
            national_field, deviation_field = field_by_indicator[indicator]
            row[national_field] = national
            row[deviation_field] = regional - national
            for role, key, record in (
                ("national", national_key, national_record),
                ("regional", regional_key, regional_record),
            ):
                source_cells.append(
                    {
                        "role": role,
                        "period": key[0],
                        "geography_id": key[1],
                        "indicator_id": key[2],
                        "source_id": str(record["source_id"]),
                        "source_snapshot_sha256": str(
                            record["source_snapshot_sha256"]
                        ),
                        "source_cell_identity": str(
                            record["source_cell_identity"]
                        ),
                    }
                )
        row["labor_context_semantics"] = (
            "aggregate_context_not_individual_probability"
        )
        row["labor_context_source_cells"] = tuple(source_cells)
        output.append(row)
    return output


def _labor_state(value: Any, field: str) -> str:
    text = str(value).strip()
    text = text.removesuffix(".0")
    if text not in {"1", "2", "3"}:
        raise LongitudinalRuntimeError(
            f"labor_state_outside_reviewed_classes:{field}:{text}"
        )
    return text


def _observed_labor_state(
    row: Mapping[str, Any],
    config: LongitudinalConfig,
) -> str | None:
    valid: list[tuple[str, str]] = []
    special: list[tuple[str, str]] = []
    for field in config.observed_labor_fields:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text in {"0", "4"}:
            special.append((field, text))
            continue
        valid.append((field, _labor_state(text, field)))
    if not valid:
        if special:
            return None
        raise LongitudinalRuntimeError(
            "observed_labor_state_missing:" + "|".join(config.observed_labor_fields)
        )
    distinct = {value for _, value in valid}
    if len(distinct) != 1:
        detail = ",".join(f"{field}={value}" for field, value in valid)
        raise LongitudinalRuntimeError(f"observed_labor_state_disagreement:{detail}")
    return valid[0][1]


def build_panel_pairs(
    rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
) -> list[dict[str, Any]]:
    """Create honest earlier-state -> later-welfare pairs on supported EPH gaps only."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = str(row.get(config.panel_person_field) or "")
        if not candidate:
            raise LongitudinalRuntimeError("panel_person_candidate_missing")
        grouped[candidate].append(row)

    output: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    for candidate, observations in grouped.items():
        ordered = sorted(
            observations,
            key=lambda row: _period_index(str(row.get(config.period_field) or "")),
        )
        period_ids = [str(row.get(config.period_field) or "") for row in ordered]
        if len(period_ids) != len(set(period_ids)):
            raise LongitudinalRuntimeError(
                f"panel_candidate_duplicate_period:{candidate}"
            )
        for earlier, later in pairwise(ordered):
            earlier_period = str(earlier[config.period_field])
            later_period = str(later[config.period_field])
            gap = _period_index(later_period) - _period_index(earlier_period)
            if gap not in config.allowed_panel_gaps:
                continue
            earlier_id = str(earlier.get("row_id") or "")
            later_id = str(later.get("row_id") or "")
            if not earlier_id or not later_id or earlier_id == later_id or gap <= 0:
                raise LongitudinalRuntimeError("panel_pair_clock_identity_invalid")
            if later_id in seen_target_ids:
                raise LongitudinalRuntimeError(
                    f"panel_target_observation_reused:{later_id}"
                )
            earlier_household = str(earlier.get(config.group_field) or "")
            later_household = str(later.get(config.group_field) or "")
            if not earlier_household or earlier_household != later_household:
                raise LongitudinalRuntimeError("panel_pair_household_identity_changed")

            stale_state = _observed_labor_state(earlier, config)
            current_state = _observed_labor_state(later, config)
            if stale_state is None or current_state is None:
                continue
            paired = dict(later)
            paired["pair_id"] = f"{earlier_id}->{later_id}"
            paired["stale_observation_row_id"] = earlier_id
            paired["target_observation_row_id"] = later_id
            paired["stale_period"] = earlier_period
            paired["target_period"] = later_period
            paired[config.elapsed_quarters_field] = gap
            paired[config.stale_labor_field] = stale_state
            paired[config.current_labor_field] = current_state
            if paired["stale_period"] == paired["target_period"]:
                raise LongitudinalRuntimeError("stale_state_not_earlier_than_target")
            output.append(paired)
            seen_target_ids.add(later_id)
    if not output:
        raise LongitudinalRuntimeError("no_supported_panel_pairs")
    return output


def stale_proxy_diagnostics(
    pair_rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
) -> dict[str, Any]:
    by_gap: dict[int, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in pair_rows:
        gap = int(row[config.elapsed_quarters_field])
        stale = _labor_state(row[config.stale_labor_field], config.stale_labor_field)
        current = _labor_state(row[config.current_labor_field], config.current_labor_field)
        by_gap[gap][(stale, current)] += 1
    return {
        "pair_count": len(pair_rows),
        "supported_elapsed_quarters": sorted(by_gap),
        "transition_counts": {
            str(gap): {
                f"{source}->{target}": count
                for (source, target), count in sorted(counts.items())
            }
            for gap, counts in sorted(by_gap.items())
        },
        "identification_boundary": (
            "observed_short_gap_eph_only_no_extrapolation_to_cpv2010_long_horizon"
        ),
    }


def _encoded_id(row: Mapping[str, Any], field: str) -> str:
    value = str(row.get(field) or "")
    if not value:
        raise LongitudinalRuntimeError(f"longitudinal_identity_missing:{field}")
    return value


def build_longitudinal_fold_manifest(
    rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
) -> FoldManifest:
    row_ids = tuple(_encoded_id(row, config.row_id_field) for row in rows)
    groups = tuple(_encoded_id(row, config.group_field) for row in rows)
    folds = []
    for group in groups:
        digest = hashlib.sha256(group.encode()).digest()
        folds.append(int.from_bytes(digest[:8], "big") % config.n_splits)
    return FoldManifest(
        row_ids=row_ids,
        household_ids=groups,
        fold_ids=tuple(folds),
        n_splits=config.n_splits,
        policy="panel_household_grouped_longitudinal_v1",
    )


def _float_feature(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LongitudinalRuntimeError(f"feature_not_numeric:{field}") from exc
    if math.isinf(number):
        raise LongitudinalRuntimeError(f"feature_infinite:{field}")
    return number


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[_float_feature(row, field) for field in fields] for row in rows],
        dtype=float,
    )


def _categorical_positions(
    feature_names: Sequence[str],
    config: LongitudinalConfig,
    *,
    include_stale: bool,
) -> tuple[int, ...]:
    categories = set(config.categorical_features)
    if include_stale:
        categories.add(config.stale_labor_field)
    return tuple(
        index for index, name in enumerate(feature_names) if name in categories
    )


def _hurdle_estimator(
    config: LongitudinalConfig,
    feature_names: Sequence[str],
    *,
    include_stale: bool = False,
) -> HurdleEstimator:
    categorical = _categorical_positions(
        feature_names, config, include_stale=include_stale
    )
    presence = lambda: HGBClassifierAdapter(
        categorical_features=categorical,
        parameters=config.presence_params,
    )
    amount = lambda: HGBRegressorAdapter(
        loss="gamma",
        categorical_features=categorical,
        parameters=config.amount_params,
    )
    return HurdleEstimator(
        formulation="gamma",
        presence_factory=presence,
        amount_factory=amount,
        retransformation="linear_identity_v1",
    )


def _transition_factory(
    config: LongitudinalConfig,
    feature_names: Sequence[str],
):
    categorical = _categorical_positions(
        feature_names, config, include_stale=True
    )
    return lambda: HGBClassifierAdapter(
        categorical_features=categorical,
        parameters=config.transition_params,
    )


@dataclass(frozen=True)
class LaborMomentAnchor:
    period: str
    region_id: str
    class_shares: Mapping[str, float]
    release_id: str
    universe_contract: str


def _anchor_probabilities(
    probabilities: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
    anchors: Sequence[LaborMomentAnchor],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not config.anchor_enabled:
        if anchors:
            raise LongitudinalRuntimeError("anchor_targets_supplied_to_nonanchored_config")
        return probabilities, []
    target_by_key = {(item.period, item.region_id): item for item in anchors}
    keys = [
        (str(row[config.period_field]), str(row[config.region_field]))
        for row in rows
    ]
    missing = sorted(set(keys) - set(target_by_key))
    if missing:
        raise LongitudinalRuntimeError(
            "l12_anchor_target_missing:" + ",".join(f"{p}:{r}" for p, r in missing)
        )
    output = probabilities.copy()
    diagnostics: list[dict[str, Any]] = []
    for key in sorted(set(keys)):
        mask = np.asarray([value == key for value in keys], dtype=bool)
        anchor = target_by_key[key]
        projected, report = multiclass_kl_moment_projection(
            output[mask],
            class_labels=("1", "2", "3"),
            target_shares=dict(anchor.class_shares),
            universe_contract=anchor.universe_contract,
            anchor_release_id=anchor.release_id,
        )
        output[mask] = projected
        diagnostics.append(
            {
                "period": key[0],
                "region_id": key[1],
                **report.as_dict(),
            }
        )
    return output, diagnostics


def labor_class_shares_from_official_rates(
    *,
    activity_rate: float,
    unemployment_rate: float,
) -> dict[str, float]:
    """Translate compatible activity/unemployment margins to E/U/I class shares.

    This arithmetic is only valid after the caller has established a common
    age/population universe for the anchor.
    """
    activity = float(activity_rate) / 100.0
    unemployment = float(unemployment_rate) / 100.0
    if not 0 < activity < 1 or not 0 < unemployment < 1:
        raise LongitudinalRuntimeError("official_labor_anchor_rates_invalid")
    return {
        "1": activity * (1.0 - unemployment),
        "2": activity * unemployment,
        "3": 1.0 - activity,
    }


def _subset_probability_artifact(
    artifact: PredictionArtifact,
    mask: np.ndarray,
) -> PredictionArtifact:
    return PredictionArtifact(
        target=artifact.target,
        kind="probability",
        row_ids=tuple(
            row_id
            for row_id, keep in zip(artifact.row_ids, mask, strict=True)
            if keep
        ),
        values=artifact.values[mask],
        source="oof",
        class_labels=artifact.class_labels,
        fold_ids=tuple(
            fold
            for fold, keep in zip(artifact.fold_ids or (), mask, strict=True)
            if keep
        ),
        run_id=artifact.run_id,
        metadata=dict(artifact.metadata),
    )


def _household_metrics(
    rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
    eligibility,
    prediction: np.ndarray,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = _encoded_id(row, config.household_observation_field)
        grouped[key].append(index)
    truth: list[float] = []
    predicted: list[float] = []
    unavailable = 0
    for indices in grouped.values():
        if not bool(np.all(eligibility.valid[indices])):
            unavailable += 1
            continue
        truth.append(float(eligibility.numeric[indices].sum()))
        predicted.append(float(prediction[indices].sum()))
    if not truth:
        raise LongitudinalRuntimeError("longitudinal_household_evaluation_empty")
    return {
        "complete_households": len(truth),
        "unavailable_households": unavailable,
        "point": distributional_regression_diagnostics(
            np.asarray(truth), np.asarray(predicted)
        ),
    }


@dataclass(frozen=True)
class LongitudinalRunResult:
    arm: str
    run_id: str
    config_digest: str
    feature_names: tuple[str, ...]
    fold_manifest: FoldManifest
    hurdle: HurdlePredictionBundle
    metrics: Mapping[str, Any]
    time_layers: tuple[Mapping[str, Any], ...]
    parent_metadata: Mapping[str, Any]
    panel_diagnostics: Mapping[str, Any] | None = None
    transition_raw: PredictionArtifact | None = None
    transition_anchored: PredictionArtifact | None = None
    anchor_diagnostics: tuple[Mapping[str, Any], ...] = ()
    matched_l10_oof: PredictionArtifact | None = None


def _run_id(
    config: LongitudinalConfig,
    parent_metadata: Mapping[str, Any],
    row_ids: Sequence[str],
) -> str:
    payload = {
        "config_digest": config.digest,
        "parents": parent_metadata,
        "rows_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]
    return f"longitudinal-{config.arm.lower()}-{digest}"


def execute_longitudinal_arm(
    rows: Sequence[Mapping[str, Any]],
    config: LongitudinalConfig,
    *,
    parent_metadata: Mapping[str, Any],
    fold_manifest: FoldManifest | None = None,
    anchors: Sequence[LaborMomentAnchor] = (),
) -> LongitudinalRunResult:
    """Run L10, L11, or L12 under one household/panel-safe OOF engine."""
    if not rows:
        raise LongitudinalRuntimeError("longitudinal_run_requires_rows")
    if config.arm in {"L11", "L12"}:
        for row in rows:
            stale_period = str(row.get("stale_period") or "")
            target_period = str(row.get("target_period") or "")
            if not stale_period or _period_index(stale_period) >= _period_index(target_period):
                raise LongitudinalRuntimeError("stale_labor_clock_not_strictly_earlier")
            _labor_state(row.get(config.stale_labor_field), config.stale_labor_field)
            _labor_state(row.get(config.current_labor_field), config.current_labor_field)
    elif anchors:
        raise LongitudinalRuntimeError("aggregate_anchor_only_allowed_in_L12")

    manifest = fold_manifest or build_longitudinal_fold_manifest(rows, config)
    expected_ids = tuple(_encoded_id(row, config.row_id_field) for row in rows)
    if manifest.row_ids != expected_ids:
        raise LongitudinalRuntimeError("longitudinal_fold_manifest_row_mismatch")

    base_feature_names = config.base_feature_names
    base_x = _matrix(rows, base_feature_names)
    target = np.asarray([row.get(config.target_field) for row in rows], dtype=object)
    eligibility = classify_income_target(target)
    periods = np.asarray([str(row[config.period_field]) for row in rows], dtype=object)

    if config.arm == "L11":
        terminal_feature_names = (*base_feature_names, config.stale_labor_field)
        terminal_base_x = np.column_stack(
            [base_x, _matrix(rows, (config.stale_labor_field,))]
        )
    else:
        terminal_feature_names = base_feature_names
        terminal_base_x = base_x

    p_positive = np.full(len(rows), np.nan, dtype=float)
    positive_amount = np.full(len(rows), np.nan, dtype=float)
    matched_l10_probability = (
        np.full(len(rows), np.nan, dtype=float)
        if config.arm in {"L11", "L12"}
        else None
    )
    matched_l10_amount = (
        np.full(len(rows), np.nan, dtype=float)
        if config.arm in {"L11", "L12"}
        else None
    )
    transition_raw_values = (
        np.full((len(rows), 3), np.nan, dtype=float) if config.arm == "L12" else None
    )
    transition_anchored_values = (
        np.full((len(rows), 3), np.nan, dtype=float) if config.arm == "L12" else None
    )
    anchor_reports: list[dict[str, Any]] = []
    time_reports: list[dict[str, Any]] = []
    fold_array = manifest.as_array()

    for outer_fold in range(manifest.n_splits):
        train = fold_array != outer_fold
        holdout = fold_array == outer_fold
        if not np.any(train) or not np.any(holdout):
            raise LongitudinalRuntimeError(f"longitudinal_fold_partition_invalid:{outer_fold}")

        if config.arm == "L12":
            transition_feature_names = (
                *base_feature_names,
                config.stale_labor_field,
                config.elapsed_quarters_field,
            )
            transition_x = np.column_stack(
                [
                    base_x,
                    _matrix(
                        rows,
                        (config.stale_labor_field, config.elapsed_quarters_field),
                    ),
                ]
            )
            transition_y = np.asarray(
                [
                    _labor_state(row[config.current_labor_field], config.current_labor_field)
                    for row in rows
                ],
                dtype=str,
            )
            if set(transition_y.tolist()) != {"1", "2", "3"}:
                raise LongitudinalRuntimeError(
                    "l12_transition_requires_all_three_labor_classes"
                )
            inner_manifest = _subset_manifest(manifest, train)
            factory = _transition_factory(config, transition_feature_names)
            nested = crossfit_predict(
                transition_x[train],
                transition_y[train],
                inner_manifest,
                factory,
                target=config.current_labor_field,
                kind="probability",
                class_labels=("1", "2", "3"),
                metadata={
                    "arm": "L12",
                    "role": "nested_transition_meta_training",
                    "outer_fold": outer_fold,
                },
            )
            holdout_ids = tuple(
                row_id
                for row_id, held_out in zip(manifest.row_ids, holdout, strict=True)
                if held_out
            )
            scored = fit_full_and_score(
                transition_x[train],
                transition_y[train],
                transition_x[holdout],
                holdout_ids,
                factory,
                target=config.current_labor_field,
                kind="probability",
                class_labels=("1", "2", "3"),
                metadata={
                    "arm": "L12",
                    "role": "outer_holdout_transition",
                    "outer_fold": outer_fold,
                },
            )
            raw_train = nested.values
            raw_holdout = scored.values
            if config.anchor_enabled:
                train_rows = [row for row, keep in zip(rows, train, strict=True) if keep]
                holdout_rows = [
                    row for row, keep in zip(rows, holdout, strict=True) if keep
                ]
                latent_train, train_reports = _anchor_probabilities(
                    raw_train, train_rows, config, anchors
                )
                latent_holdout, holdout_reports = _anchor_probabilities(
                    raw_holdout, holdout_rows, config, anchors
                )
                for report in (*train_reports, *holdout_reports):
                    anchor_reports.append({"outer_fold": outer_fold, **report})
            else:
                latent_train, latent_holdout = raw_train, raw_holdout

            transition_raw_values[holdout] = raw_holdout
            transition_anchored_values[holdout] = latent_holdout
            terminal_train_x = np.column_stack([base_x[train], latent_train])
            terminal_holdout_x = np.column_stack([base_x[holdout], latent_holdout])
            terminal_feature_names = (
                *base_feature_names,
                "latent.current_labor.p[1]",
                "latent.current_labor.p[2]",
                "latent.current_labor.p[3]",
            )
            estimator = _hurdle_estimator(
                config, terminal_feature_names, include_stale=False
            )
        else:
            terminal_train_x = terminal_base_x[train]
            terminal_holdout_x = terminal_base_x[holdout]
            estimator = _hurdle_estimator(
                config,
                terminal_feature_names,
                include_stale=config.arm == "L11",
            )

        estimator.fit(terminal_train_x, target[train])
        train_components = estimator.predict_components(terminal_train_x)
        holdout_components = estimator.predict_components(terminal_holdout_x)
        time_fit: TimeLayerFit = fit_time_layer(
            periods[train],
            train_components.p_positive,
            train_components.positive_amount,
            target[train],
            exception_policy=EXCEPTIONAL_PERIODS,
        )
        corrected_p, corrected_amount = time_fit.apply(
            periods[holdout],
            holdout_components.p_positive,
            holdout_components.positive_amount,
        )
        p_positive[holdout] = corrected_p
        positive_amount[holdout] = corrected_amount
        time_reports.append(
            {
                "outer_fold": outer_fold,
                "model_role": config.arm,
                **time_fit.as_dict(),
            }
        )

        if matched_l10_probability is not None and matched_l10_amount is not None:
            baseline_estimator = _hurdle_estimator(
                config,
                base_feature_names,
                include_stale=False,
            )
            baseline_estimator.fit(base_x[train], target[train])
            baseline_train = baseline_estimator.predict_components(base_x[train])
            baseline_holdout = baseline_estimator.predict_components(base_x[holdout])
            baseline_time = fit_time_layer(
                periods[train],
                baseline_train.p_positive,
                baseline_train.positive_amount,
                target[train],
                exception_policy=EXCEPTIONAL_PERIODS,
            )
            baseline_p, baseline_amount = baseline_time.apply(
                periods[holdout],
                baseline_holdout.p_positive,
                baseline_holdout.positive_amount,
            )
            matched_l10_probability[holdout] = baseline_p
            matched_l10_amount[holdout] = baseline_amount
            time_reports.append(
                {
                    "outer_fold": outer_fold,
                    "model_role": "matched_L10_baseline",
                    **baseline_time.as_dict(),
                }
            )

    if not np.isfinite(p_positive).all() or not np.isfinite(positive_amount).all():
        raise LongitudinalRuntimeError("longitudinal_oof_prediction_incomplete")

    run_id = _run_id(config, parent_metadata, manifest.row_ids)
    unconditional = p_positive * positive_amount
    hurdle = HurdlePredictionBundle(
        p_positive=PredictionArtifact(
            target=f"{config.target_field}__positive",
            kind="probability",
            row_ids=manifest.row_ids,
            values=np.column_stack([1.0 - p_positive, p_positive]),
            source="oof",
            class_labels=("0", "1"),
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "arm": config.arm,
                "time_layer": "explicit_year_quarter_exception_v1",
            },
        ),
        positive_amount_prediction=PredictionArtifact(
            target=f"{config.target_field}__positive_amount",
            kind="regression",
            row_ids=manifest.row_ids,
            values=positive_amount,
            source="oof",
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "arm": config.arm,
                "time_layer": "explicit_year_quarter_exception_v1",
            },
        ),
        unconditional_expected_income=PredictionArtifact(
            target=config.target_field,
            kind="regression",
            row_ids=manifest.row_ids,
            values=unconditional,
            source="oof",
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "arm": config.arm,
                "measurement_mode": True,
                "forecasting_authorized": False,
            },
        ),
        target_eligibility=eligibility.counts,
        formulation="gamma",
        retransformation="linear_identity_v1",
    )

    valid = eligibility.valid
    positive = eligibility.positive
    presence_artifact = _subset_probability_artifact(hurdle.p_positive, valid)
    metrics: dict[str, Any] = {
        "person": {
            "unconditional": distributional_regression_diagnostics(
                eligibility.numeric[valid], unconditional[valid]
            ),
            "presence": classification_diagnostics(
                eligibility.positive[valid].astype(int), presence_artifact
            ),
            "positive_amount": (
                distributional_regression_diagnostics(
                    eligibility.numeric[positive], positive_amount[positive]
                )
                if np.any(positive)
                else None
            ),
        },
        "household": _household_metrics(
            rows, config, eligibility, unconditional
        ),
        "fold_counts": {
            str(fold): int((fold_array == fold).sum())
            for fold in range(manifest.n_splits)
        },
        "measurement_mode": True,
        "forecasting_authorized": False,
        "exceptional_periods": dict(EXCEPTIONAL_PERIODS),
    }

    matched_l10_artifact = None
    if matched_l10_probability is not None and matched_l10_amount is not None:
        if (
            not np.isfinite(matched_l10_probability).all()
            or not np.isfinite(matched_l10_amount).all()
        ):
            raise LongitudinalRuntimeError("matched_l10_oof_prediction_incomplete")
        matched_values = matched_l10_probability * matched_l10_amount
        matched_l10_artifact = PredictionArtifact(
            target=config.target_field,
            kind="regression",
            row_ids=manifest.row_ids,
            values=matched_values,
            source="oof",
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "arm": "L10",
                "evaluation_role": f"matched_baseline_for_{config.arm}",
                "same_rows_and_folds": True,
            },
        )
        matched_person = distributional_regression_diagnostics(
            eligibility.numeric[valid], matched_values[valid]
        )
        matched_household = _household_metrics(
            rows, config, eligibility, matched_values
        )
        metrics["matched_l10_baseline"] = {
            "person": matched_person,
            "household": matched_household,
            "delta": {
                "person_mae_gain": (
                    matched_person["point"]["mae"]
                    - metrics["person"]["unconditional"]["point"]["mae"]
                ),
                "person_rmse_gain": (
                    matched_person["point"]["rmse"]
                    - metrics["person"]["unconditional"]["point"]["rmse"]
                ),
                "household_mae_gain": (
                    matched_household["point"]["point"]["mae"]
                    - metrics["household"]["point"]["point"]["mae"]
                ),
                "household_rmse_gain": (
                    matched_household["point"]["point"]["rmse"]
                    - metrics["household"]["point"]["point"]["rmse"]
                ),
            },
            "promotion_authorized": False,
        }

    transition_raw_artifact = None
    transition_anchored_artifact = None
    if config.arm == "L12":
        assert transition_raw_values is not None
        assert transition_anchored_values is not None
        transition_y = np.asarray(
            [str(row[config.current_labor_field]) for row in rows], dtype=str
        )
        transition_raw_artifact = PredictionArtifact(
            target=config.current_labor_field,
            kind="probability",
            row_ids=manifest.row_ids,
            values=transition_raw_values,
            source="oof",
            class_labels=("1", "2", "3"),
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={"arm": "L12", "anchored": False},
        )
        transition_anchored_artifact = PredictionArtifact(
            target=config.current_labor_field,
            kind="probability",
            row_ids=manifest.row_ids,
            values=transition_anchored_values,
            source="oof",
            class_labels=("1", "2", "3"),
            fold_ids=manifest.fold_ids,
            run_id=run_id,
            metadata={
                "arm": "L12",
                "anchored": config.anchor_enabled,
                "anchor_policy": "external_moment_only",
            },
        )
        metrics["transition_raw"] = classification_diagnostics(
            transition_y, transition_raw_artifact
        )
        metrics["transition_terminal_input"] = classification_diagnostics(
            transition_y, transition_anchored_artifact
        )

    panel_diagnostics = (
        stale_proxy_diagnostics(rows, config) if config.arm in {"L11", "L12"} else None
    )
    return LongitudinalRunResult(
        arm=config.arm,
        run_id=run_id,
        config_digest=config.digest,
        feature_names=tuple(terminal_feature_names),
        fold_manifest=manifest,
        hurdle=hurdle,
        metrics=metrics,
        time_layers=tuple(time_reports),
        parent_metadata=dict(parent_metadata),
        panel_diagnostics=panel_diagnostics,
        transition_raw=transition_raw_artifact,
        transition_anchored=transition_anchored_artifact,
        anchor_diagnostics=tuple(anchor_reports),
        matched_l10_oof=matched_l10_artifact,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_longitudinal_run_bundle(
    output_root: Path,
    result: LongitudinalRunResult,
    config: LongitudinalConfig,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically package one immutable C4 measurement-mode OOF run."""
    output_root = Path(output_root).expanduser().resolve()
    destination = output_root / result.run_id
    if destination.exists():
        raise LongitudinalRuntimeError(f"immutable_longitudinal_run_exists:{destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{result.run_id}.", dir=output_root))
    try:
        (staging / "resolved_config.json").write_text(
            _canonical_json(config.raw), encoding="utf-8"
        )
        (staging / "parents.json").write_text(
            _canonical_json(result.parent_metadata), encoding="utf-8"
        )
        (staging / "metrics.json").write_text(
            _canonical_json(result.metrics), encoding="utf-8"
        )
        (staging / "time_layer.json").write_text(
            _canonical_json(list(result.time_layers)), encoding="utf-8"
        )
        (staging / "panel_diagnostics.json").write_text(
            _canonical_json(result.panel_diagnostics), encoding="utf-8"
        )
        (staging / "anchor_diagnostics.json").write_text(
            _canonical_json(list(result.anchor_diagnostics)), encoding="utf-8"
        )
        fold_rows = [
            {
                "row_id": row_id,
                "panel_household_group_id": group,
                "fold_id": fold,
            }
            for row_id, group, fold in zip(
                result.fold_manifest.row_ids,
                result.fold_manifest.household_ids,
                result.fold_manifest.fold_ids,
                strict=True,
            )
        ]
        (staging / "fold_manifest.json").write_text(
            _canonical_json(
                {
                    "policy": result.fold_manifest.policy,
                    "n_splits": result.fold_manifest.n_splits,
                    "rows": fold_rows,
                }
            ),
            encoding="utf-8",
        )
        with (staging / "person_oof.jsonl").open("w", encoding="utf-8") as stream:
            eligibility = classify_income_target(
                [row.get(config.target_field) for row in rows]
            )
            for index, row_id in enumerate(result.fold_manifest.row_ids):
                stream.write(
                    _canonical_json(
                        {
                            "row_id": row_id,
                            "fold_id": result.fold_manifest.fold_ids[index],
                            "period": rows[index][config.period_field],
                            "region_id": rows[index][config.region_field],
                            "observed_income": (
                                float(eligibility.numeric[index])
                                if eligibility.valid[index]
                                else None
                            ),
                            "p_positive": float(
                                result.hurdle.p_positive.values[index, 1]
                            ),
                            "positive_amount_prediction": float(
                                result.hurdle.positive_amount_prediction.values[index]
                            ),
                            "unconditional_expected_income": float(
                                result.hurdle.unconditional_expected_income.values[index]
                            ),
                            "matched_l10_expected_income": (
                                float(result.matched_l10_oof.values[index])
                                if result.matched_l10_oof is not None
                                else None
                            ),
                        }
                    )
                )
        if result.transition_raw is not None:
            with (staging / "transition_oof.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                assert result.transition_anchored is not None
                for index, row_id in enumerate(result.transition_raw.row_ids):
                    stream.write(
                        _canonical_json(
                            {
                                "row_id": row_id,
                                "raw_probabilities": {
                                    label: float(result.transition_raw.values[index, column])
                                    for column, label in enumerate(
                                        result.transition_raw.class_labels
                                    )
                                },
                                "terminal_input_probabilities": {
                                    label: float(
                                        result.transition_anchored.values[index, column]
                                    )
                                    for column, label in enumerate(
                                        result.transition_anchored.class_labels
                                    )
                                },
                            }
                        )
                    )

        artifacts = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "contract": "research.encuestador-longitudinal-run/v1",
            "run_id": result.run_id,
            "arm": result.arm,
            "config_digest": result.config_digest,
            "parents": result.parent_metadata,
            "feature_names": list(result.feature_names),
            "target": config.target_field,
            "monetary_target_semantics": "linear_real_common_reference_from_parent",
            "time_layer": "explicit_year_quarter_exception_v1",
            "exceptional_periods": dict(EXCEPTIONAL_PERIODS),
            "panel_grouping": result.fold_manifest.policy,
            "anchor_enabled": config.anchor_enabled,
            "measurement_mode": True,
            "forecasting_authorized": False,
            "artifacts": artifacts,
        }
        (staging / "run_manifest.json").write_text(
            _canonical_json(manifest), encoding="utf-8"
        )
        os.replace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compare_longitudinal_runs(roots: Sequence[Path]) -> dict[str, Any]:
    if len(roots) < 2:
        raise LongitudinalRuntimeError("longitudinal_compare_requires_two_runs")
    records = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        if manifest.get("contract") != "research.encuestador-longitudinal-run/v1":
            raise LongitudinalRuntimeError("longitudinal_compare_contract_invalid")
        records.append((manifest, metrics))
    parent_signatures = {
        _canonical_json(record[0].get("parents")) for record in records
    }
    if len(parent_signatures) != 1:
        raise LongitudinalRuntimeError("longitudinal_compare_parent_mismatch")
    output = []
    for manifest, metrics in records:
        point = metrics["person"]["unconditional"]["point"]
        household = metrics["household"]["point"]["point"]
        output.append(
            {
                "run_id": manifest["run_id"],
                "arm": manifest["arm"],
                "person_mae": point["mae"],
                "person_rmse": point["rmse"],
                "household_mae": household["mae"],
                "household_rmse": household["rmse"],
            }
        )
    return {
        "contract": "research.encuestador-longitudinal-comparison/v1",
        "status": "descriptive_no_fixture_promotion",
        "runs": output,
        "promotion_authorized": False,
    }
