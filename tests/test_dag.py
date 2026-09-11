from __future__ import annotations

import numpy as np
import pytest

from encuestador.dag import (
    ArchitectureSpec,
    DAGSpecError,
    prediction_feature_names,
)
from encuestador.scientific_primitives import PredictionArtifact


def test_direct_architecture_has_empty_topological_order() -> None:
    architecture = ArchitectureSpec.from_mapping({"id": "direct_v1", "nodes": {}})
    assert architecture.topological_order() == ()


def test_lean_architecture_orders_latent_nodes_before_downstream_nodes() -> None:
    architecture = ArchitectureSpec.from_mapping(
        {
            "id": "lean_v1",
            "nodes": {
                "labor_state": {
                    "targets": ["CAT_OCUP", "CAT_INAC"],
                    "task_by_target": {
                        "CAT_OCUP": "multiclass",
                        "CAT_INAC": "multiclass",
                    },
                    "base_inputs": ["P02", "P03"],
                    "upstream_nodes": [],
                    "estimator_role": "latent_categorical",
                    "output_representation": "probabilities",
                },
                "income_state": {
                    "targets": ["positive_income"],
                    "task_by_target": {"positive_income": "binary"},
                    "base_inputs": ["P02", "P03"],
                    "upstream_nodes": ["labor_state"],
                    "estimator_role": "latent_binary",
                    "output_representation": "probabilities",
                },
            },
        }
    )
    order = architecture.topological_order()
    assert order.index("labor_state") < order.index("income_state")


def test_dag_rejects_cycles_missing_nodes_duplicate_targets_and_forbidden_features() -> None:
    with pytest.raises(DAGSpecError, match="architecture_cycle_detected"):
        ArchitectureSpec.from_mapping(
            {
                "id": "cycle",
                "nodes": {
                    "a": {
                        "targets": ["A"],
                        "task_by_target": {"A": "binary"},
                        "base_inputs": ["P02"],
                        "upstream_nodes": ["b"],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                    "b": {
                        "targets": ["B"],
                        "task_by_target": {"B": "binary"},
                        "base_inputs": ["P03"],
                        "upstream_nodes": ["a"],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                },
            }
        )

    with pytest.raises(DAGSpecError, match="missing_upstream_node"):
        ArchitectureSpec.from_mapping(
            {
                "id": "missing",
                "nodes": {
                    "a": {
                        "targets": ["A"],
                        "task_by_target": {"A": "binary"},
                        "base_inputs": ["P02"],
                        "upstream_nodes": ["does_not_exist"],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    }
                },
            }
        )

    with pytest.raises(DAGSpecError, match="duplicate_output_target"):
        ArchitectureSpec.from_mapping(
            {
                "id": "duplicate",
                "nodes": {
                    "a": {
                        "targets": ["STATE"],
                        "task_by_target": {"STATE": "binary"},
                        "base_inputs": ["P02"],
                        "upstream_nodes": [],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                    "b": {
                        "targets": ["STATE"],
                        "task_by_target": {"STATE": "binary"},
                        "base_inputs": ["P03"],
                        "upstream_nodes": [],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                },
            }
        )

    with pytest.raises(DAGSpecError, match="forbidden_external_feature"):
        ArchitectureSpec.from_mapping(
            {
                "id": "forbidden",
                "nodes": {
                    "a": {
                        "targets": ["STATE"],
                        "task_by_target": {"STATE": "binary"},
                        "base_inputs": ["AGLO_rk"],
                        "upstream_nodes": [],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    }
                },
            },
            forbidden_external_features=("AGLO_rk",),
        )


def test_dag_rejects_observed_stage_target_injection() -> None:
    with pytest.raises(DAGSpecError, match="observed_stage_target_in_base_inputs"):
        ArchitectureSpec.from_mapping(
            {
                "id": "observed-injection",
                "nodes": {
                    "labor": {
                        "targets": ["CAT_OCUP"],
                        "task_by_target": {"CAT_OCUP": "multiclass"},
                        "base_inputs": ["P02"],
                        "upstream_nodes": [],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                    "income": {
                        "targets": ["positive_income"],
                        "task_by_target": {"positive_income": "binary"},
                        "base_inputs": ["P03", "CAT_OCUP"],
                        "upstream_nodes": ["labor"],
                        "estimator_role": "latent",
                        "output_representation": "probabilities",
                    },
                },
            }
        )


def test_probability_artifact_expands_to_stable_named_latent_columns() -> None:
    artifact = PredictionArtifact(
        target="CAT_OCUP",
        kind="probability",
        row_ids=("a", "b"),
        values=np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
        source="oof",
        class_labels=("0", "1", "2"),
        fold_ids=(0, 1),
    )
    assert prediction_feature_names(artifact) == (
        "latent.CAT_OCUP.p[0]",
        "latent.CAT_OCUP.p[1]",
        "latent.CAT_OCUP.p[2]",
    )
