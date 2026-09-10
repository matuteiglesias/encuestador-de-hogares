"""Minimal declarative experiment loading, resolution, validation and hashing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dag import ArchitectureSpec, DAGSpecError


class ExperimentResolutionError(ValueError):
    """Raised before fitting when a declarative experiment is invalid."""


_COMPONENT_KEYS = (
    "data",
    "features",
    "architecture",
    "terminal",
    "estimators",
    "splits",
    "calibration",
    "anchors",
    "uncertainty",
    "evaluation",
)
_REQUIRED_COMPONENTS = set(_COMPONENT_KEYS) - {"anchors"}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentResolutionError(f"invalid_yaml:{path}") from exc
    if not isinstance(value, dict):
        raise ExperimentResolutionError(f"yaml_root_must_be_mapping:{path}")
    return value


def _safe_component_path(root: Path, experiment_path: Path, reference: str) -> Path:
    candidate = (experiment_path.parent / reference).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ExperimentResolutionError(
            f"component_reference_escapes_config_root:{reference}"
        ) from exc
    return candidate


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


def _resolve_component(
    key: str,
    value: Any,
    *,
    config_root: Path,
    experiment_path: Path,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        raise ExperimentResolutionError(f"invalid_component_reference:{key}")

    path = _safe_component_path(config_root, experiment_path, value)
    fragment = _load_yaml_mapping(path)
    if key in fragment and len(fragment) == 1:
        resolved = fragment[key]
    else:
        resolved = fragment
    if not isinstance(resolved, Mapping):
        raise ExperimentResolutionError(f"component_must_resolve_to_mapping:{key}")
    return dict(resolved)


def _validate_data(data: Mapping[str, Any]) -> None:
    required = {
        "source_release",
        "training_periods",
        "row_unit",
        "household_id_columns",
        "person_id_columns",
        "weight_policy",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ExperimentResolutionError(
            f"data_component_missing:{','.join(missing)}"
        )
    if data.get("row_unit") != "person":
        raise ExperimentResolutionError("starter_data_row_unit_must_be_person")
    if tuple(data.get("household_id_columns", ())) != ("CODUSU", "NRO_HOGAR"):
        raise ExperimentResolutionError("starter_household_identity_changed")
    if tuple(data.get("person_id_columns", ())) != (
        "CODUSU",
        "NRO_HOGAR",
        "COMPONENTE",
    ):
        raise ExperimentResolutionError("starter_person_identity_changed")
    weights = data.get("weight_policy")
    if not isinstance(weights, Mapping):
        raise ExperimentResolutionError("data_weight_policy_missing")
    if any(weights.get(key) is not None for key in ("fit", "calibration", "evaluation")):
        raise ExperimentResolutionError("starter_weight_policy_must_be_none")


def _validate_features(features: Mapping[str, Any]) -> set[str]:
    columns = features.get("columns")
    if not isinstance(columns, Mapping) or not columns:
        raise ExperimentResolutionError("feature_columns_required")
    forbidden: set[str] = set()
    for name, spec in columns.items():
        if not isinstance(spec, Mapping):
            raise ExperimentResolutionError(f"feature_spec_must_be_mapping:{name}")
        dtype_role = spec.get("dtype_role")
        if dtype_role not in {
            "continuous",
            "categorical",
            "binary",
            "identifier",
            "audit",
            "forbidden",
        }:
            raise ExperimentResolutionError(f"invalid_feature_dtype_role:{name}")
        if dtype_role == "forbidden" or spec.get("allowed_as_external_input") is False:
            forbidden.add(str(name))
        if not spec.get("temporal_role"):
            raise ExperimentResolutionError(f"feature_temporal_role_required:{name}")
    return forbidden


def _validate_splits(splits: Mapping[str, Any]) -> None:
    if splits.get("strategy") != "household_grouped_v1":
        raise ExperimentResolutionError("starter_split_strategy_must_be_household_grouped")
    if tuple(splits.get("group_columns", ())) != ("CODUSU", "NRO_HOGAR"):
        raise ExperimentResolutionError("split_household_identity_changed")
    n_splits = splits.get("n_splits")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ExperimentResolutionError("split_n_splits_invalid")


def _validate_terminal(terminal: Mapping[str, Any]) -> None:
    if not terminal.get("id"):
        raise ExperimentResolutionError("terminal_id_required")
    if terminal.get("prediction_unit") != "person":
        raise ExperimentResolutionError("starter_terminal_prediction_unit_must_be_person")
    if not terminal.get("target"):
        raise ExperimentResolutionError("terminal_target_required")


def _validate_estimators(estimators: Mapping[str, Any]) -> None:
    roles = estimators.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ExperimentResolutionError("estimator_roles_required")
    for role, spec in roles.items():
        if not isinstance(spec, Mapping):
            raise ExperimentResolutionError(f"estimator_role_invalid:{role}")
        if spec.get("family") not in {"hgb", "random_forest"}:
            raise ExperimentResolutionError(f"unsupported_estimator_family:{role}")


def _validate_named_component(
    component: Mapping[str, Any],
    key: str,
) -> None:
    if not component.get("id"):
        raise ExperimentResolutionError(f"{key}_id_required")


@dataclass(frozen=True)
class ResolvedExperiment:
    """Immutable effective experiment config and stable digest."""

    config: Mapping[str, Any]
    digest: str
    source_path: str
    architecture: ArchitectureSpec

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.config)


def resolve_experiment(
    path: Path,
    *,
    config_root: Path | None = None,
) -> ResolvedExperiment:
    """Resolve one experiment into its full effective pre-fit authority."""
    experiment_path = Path(path).resolve()
    root = Path(config_root).resolve() if config_root is not None else experiment_path.parent
    try:
        experiment_path.relative_to(root)
    except ValueError as exc:
        raise ExperimentResolutionError("experiment_path_outside_config_root") from exc

    raw = _load_yaml_mapping(experiment_path)
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping) or not experiment.get("id"):
        raise ExperimentResolutionError("experiment_id_required")

    missing = sorted(key for key in _REQUIRED_COMPONENTS if key not in raw)
    if missing:
        raise ExperimentResolutionError(
            f"experiment_components_missing:{','.join(missing)}"
        )

    resolved: dict[str, Any] = {
        "schema_version": 1,
        "experiment": dict(experiment),
    }
    for key in _COMPONENT_KEYS:
        resolved[key] = _resolve_component(
            key,
            raw.get(key),
            config_root=root,
            experiment_path=experiment_path,
        )

    overrides = raw.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ExperimentResolutionError("experiment_overrides_must_be_mapping")
    unknown_overrides = sorted(set(overrides) - set(_COMPONENT_KEYS))
    if unknown_overrides:
        raise ExperimentResolutionError(
            f"unknown_experiment_override:{','.join(unknown_overrides)}"
        )
    for key, override in overrides.items():
        if resolved[key] is None:
            raise ExperimentResolutionError(f"override_of_null_component:{key}")
        resolved[key] = _deep_merge(resolved[key], override)

    for key in _REQUIRED_COMPONENTS:
        if not isinstance(resolved[key], Mapping):
            raise ExperimentResolutionError(f"resolved_component_missing:{key}")

    data = resolved["data"]
    features = resolved["features"]
    terminal = resolved["terminal"]
    estimators = resolved["estimators"]
    splits = resolved["splits"]
    calibration = resolved["calibration"]
    uncertainty = resolved["uncertainty"]
    evaluation = resolved["evaluation"]
    assert isinstance(data, Mapping)
    assert isinstance(features, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(estimators, Mapping)
    assert isinstance(splits, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(uncertainty, Mapping)
    assert isinstance(evaluation, Mapping)

    _validate_data(data)
    forbidden_features = _validate_features(features)
    _validate_terminal(terminal)
    _validate_estimators(estimators)
    _validate_splits(splits)
    _validate_named_component(calibration, "calibration")
    _validate_named_component(uncertainty, "uncertainty")
    _validate_named_component(evaluation, "evaluation")
    if resolved["anchors"] is not None:
        anchors = resolved["anchors"]
        assert isinstance(anchors, Mapping)
        _validate_named_component(anchors, "anchors")

    architecture_mapping = resolved["architecture"]
    assert isinstance(architecture_mapping, Mapping)
    try:
        architecture = ArchitectureSpec.from_mapping(
            architecture_mapping,
            forbidden_external_features=forbidden_features,
        )
    except DAGSpecError as exc:
        raise ExperimentResolutionError(f"invalid_architecture:{exc}") from exc

    digest = hashlib.sha256(_canonical_json(resolved)).hexdigest()
    return ResolvedExperiment(
        config=resolved,
        digest=digest,
        source_path=str(experiment_path),
        architecture=architecture,
    )
