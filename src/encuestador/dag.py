"""Small validated DAG specification and prediction representation helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any

from .scientific_primitives import PredictionArtifact


class DAGSpecError(ValueError):
    """Raised when an architecture graph is scientifically or structurally invalid."""


_ALLOWED_TASKS = {"binary", "multiclass", "regression"}
_ALLOWED_REPRESENTATIONS = {"probabilities", "points"}


@dataclass(frozen=True)
class NodeSpec:
    """Estimator-independent declaration of one learned DAG node."""

    node_id: str
    targets: tuple[str, ...]
    task_by_target: Mapping[str, str]
    base_inputs: tuple[str, ...]
    upstream_nodes: tuple[str, ...]
    estimator_role: str
    output_representation: str

    @classmethod
    def from_mapping(cls, node_id: str, value: Mapping[str, Any]) -> NodeSpec:
        targets = tuple(str(item) for item in value.get("targets", ()))
        task_by_target = {
            str(key): str(task)
            for key, task in dict(value.get("task_by_target", {})).items()
        }
        base_inputs = tuple(str(item) for item in value.get("base_inputs", ()))
        upstream_nodes = tuple(str(item) for item in value.get("upstream_nodes", ()))
        estimator_role = str(value.get("estimator_role", ""))
        output_representation = str(value.get("output_representation", ""))
        spec = cls(
            node_id=str(node_id),
            targets=targets,
            task_by_target=task_by_target,
            base_inputs=base_inputs,
            upstream_nodes=upstream_nodes,
            estimator_role=estimator_role,
            output_representation=output_representation,
        )
        spec.validate_local()
        return spec

    def validate_local(self) -> None:
        if not self.node_id:
            raise DAGSpecError("node_id_required")
        if not self.targets:
            raise DAGSpecError(f"node_targets_required:{self.node_id}")
        if len(set(self.targets)) != len(self.targets):
            raise DAGSpecError(f"duplicate_target_within_node:{self.node_id}")
        if set(self.task_by_target) != set(self.targets):
            raise DAGSpecError(f"task_map_must_cover_targets:{self.node_id}")
        invalid_tasks = sorted(set(self.task_by_target.values()) - _ALLOWED_TASKS)
        if invalid_tasks:
            raise DAGSpecError(
                f"unsupported_task:{self.node_id}:{','.join(invalid_tasks)}"
            )
        if not self.base_inputs and not self.upstream_nodes:
            raise DAGSpecError(f"node_has_no_usable_inputs:{self.node_id}")
        if len(set(self.base_inputs)) != len(self.base_inputs):
            raise DAGSpecError(f"duplicate_base_input:{self.node_id}")
        if len(set(self.upstream_nodes)) != len(self.upstream_nodes):
            raise DAGSpecError(f"duplicate_upstream_node:{self.node_id}")
        if not self.estimator_role:
            raise DAGSpecError(f"estimator_role_required:{self.node_id}")
        if self.output_representation not in _ALLOWED_REPRESENTATIONS:
            raise DAGSpecError(
                f"unsupported_output_representation:{self.node_id}:"
                f"{self.output_representation}"
            )
        categorical_tasks = {
            target
            for target, task in self.task_by_target.items()
            if task in {"binary", "multiclass"}
        }
        regression_tasks = {
            target
            for target, task in self.task_by_target.items()
            if task == "regression"
        }
        if categorical_tasks and self.output_representation != "probabilities":
            raise DAGSpecError(
                f"categorical_node_must_emit_probabilities:{self.node_id}"
            )
        if regression_tasks and categorical_tasks:
            raise DAGSpecError(f"mixed_task_node_not_supported_v1:{self.node_id}")
        if regression_tasks and self.output_representation != "points":
            raise DAGSpecError(f"regression_node_must_emit_points:{self.node_id}")


@dataclass(frozen=True)
class ArchitectureSpec:
    """Validated learned dependency graph independent of estimator parameters."""

    architecture_id: str
    nodes: Mapping[str, NodeSpec]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        forbidden_external_features: Sequence[str] = (),
    ) -> ArchitectureSpec:
        architecture_id = str(value.get("id", ""))
        if not architecture_id:
            raise DAGSpecError("architecture_id_required")
        raw_nodes = value.get("nodes", {})
        if raw_nodes is None:
            raw_nodes = {}
        if not isinstance(raw_nodes, Mapping):
            raise DAGSpecError("architecture_nodes_must_be_mapping")
        nodes = {
            str(node_id): NodeSpec.from_mapping(str(node_id), node)
            for node_id, node in raw_nodes.items()
        }
        spec = cls(architecture_id=architecture_id, nodes=nodes)
        spec.validate(forbidden_external_features=forbidden_external_features)
        return spec

    def validate(self, *, forbidden_external_features: Sequence[str] = ()) -> None:
        node_ids = set(self.nodes)
        emitted_targets: dict[str, str] = {}
        all_targets = {
            target
            for node in self.nodes.values()
            for target in node.targets
        }
        forbidden = set(forbidden_external_features)

        for node_id, node in self.nodes.items():
            missing = sorted(set(node.upstream_nodes) - node_ids)
            if missing:
                raise DAGSpecError(
                    f"missing_upstream_node:{node_id}:{','.join(missing)}"
                )
            if node_id in node.upstream_nodes:
                raise DAGSpecError(f"self_dependency:{node_id}")
            for target in node.targets:
                previous = emitted_targets.get(target)
                if previous is not None:
                    raise DAGSpecError(
                        f"duplicate_output_target:{target}:{previous}:{node_id}"
                    )
                emitted_targets[target] = node_id
            observed_injection = sorted(set(node.base_inputs) & all_targets)
            if observed_injection:
                raise DAGSpecError(
                    f"observed_stage_target_in_base_inputs:{node_id}:"
                    f"{','.join(observed_injection)}"
                )
            forbidden_inputs = sorted(set(node.base_inputs) & forbidden)
            if forbidden_inputs:
                raise DAGSpecError(
                    f"forbidden_external_feature:{node_id}:"
                    f"{','.join(forbidden_inputs)}"
                )

        sorter = TopologicalSorter(
            {
                node_id: set(node.upstream_nodes)
                for node_id, node in self.nodes.items()
            }
        )
        try:
            tuple(sorter.static_order())
        except CycleError as exc:
            raise DAGSpecError("architecture_cycle_detected") from exc

    def topological_order(self) -> tuple[str, ...]:
        sorter = TopologicalSorter(
            {
                node_id: set(node.upstream_nodes)
                for node_id, node in self.nodes.items()
            }
        )
        try:
            return tuple(sorter.static_order())
        except CycleError as exc:
            raise DAGSpecError("architecture_cycle_detected") from exc


def prediction_feature_names(artifact: PredictionArtifact) -> tuple[str, ...]:
    """Expand a typed prediction artifact into stable downstream feature names."""
    if artifact.kind == "probability":
        return tuple(
            f"latent.{artifact.target}.p[{label}]"
            for label in artifact.class_labels
        )
    if artifact.kind == "regression":
        return (f"latent.{artifact.target}.point",)
    raise DAGSpecError(f"unsupported_prediction_kind:{artifact.kind}")
