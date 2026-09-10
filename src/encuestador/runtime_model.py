"""High-level full-fit deployment state from the same resolved experiment authority."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cascade import CategoricalLatentSpec
from .deployment import (
    FittedDirectHurdleModel,
    FittedLeanHurdleModel,
    fit_direct_hurdle_model,
    fit_lean_hurdle_model,
)
from .experiments import ResolvedExperiment
from .runner import (
    ExperimentRunError,
    _classifier_factory,
    _feature_names,
    _hurdle_factory,
    _latent_feature_names,
    _matrix,
    _raw_target,
    _stable_classes,
    _terminal_formulation,
    configured_fold_manifest,
)

FittedTransportModel = FittedDirectHurdleModel | FittedLeanHurdleModel


def fit_deployable_hurdle_model(
    resolved: ResolvedExperiment,
    training_rows: Sequence[Mapping[str, Any]],
) -> FittedTransportModel:
    """Fit the deployable state implied by a direct or one-layer lean hurdle config."""
    config = resolved.config
    data = config["data"]
    terminal = config["terminal"]
    estimators = config["estimators"]["roles"]
    splits = config["splits"]
    feature_names = _feature_names(config)
    formulation = _terminal_formulation(terminal)
    if formulation is None:
        raise ExperimentRunError("deployable_model_requires_hurdle_terminal")
    x = _matrix(training_rows, feature_names)
    raw_y = _raw_target(training_rows, str(terminal["target"]))
    manifest = configured_fold_manifest(
        training_rows,
        person_id_columns=tuple(str(value) for value in data["person_id_columns"]),
        household_group_columns=tuple(
            str(value) for value in data["household_id_columns"]
        ),
        period_columns=tuple(str(value) for value in data.get("period_columns", ())),
        n_splits=int(splits["n_splits"]),
    )
    architecture = resolved.architecture
    if not architecture.nodes:
        terminal_factory = _hurdle_factory(estimators, feature_names, formulation)
        return fit_direct_hurdle_model(
            x,
            raw_y,
            terminal_factory,
            feature_names=feature_names,
        )

    order = architecture.topological_order()
    if len(order) != 1:
        raise ExperimentRunError("deployable_model_supports_direct_or_one_layer_only")
    node = architecture.nodes[order[0]]
    if node.upstream_nodes or tuple(node.base_inputs) != feature_names:
        raise ExperimentRunError("deployable_lean_architecture_not_supported_v1")
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
    terminal_names = (*feature_names, *_latent_feature_names(latent_specs))
    terminal_factory = _hurdle_factory(estimators, terminal_names, formulation)
    return fit_lean_hurdle_model(
        x,
        raw_y,
        latent_targets,
        manifest,
        latent_specs,
        terminal_factory,
        feature_names=feature_names,
    )
