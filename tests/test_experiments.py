from __future__ import annotations

from pathlib import Path

import pytest

from encuestador.experiments import ExperimentResolutionError, resolve_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
DIRECT = CONFIG_ROOT / "experiments" / "synthetic_direct_hgb_v1.yaml"


def test_committed_direct_experiment_resolves_to_stable_complete_authority() -> None:
    first = resolve_experiment(DIRECT, config_root=CONFIG_ROOT)
    second = resolve_experiment(DIRECT, config_root=CONFIG_ROOT)

    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()
    assert len(first.digest) == 64
    assert first.config["experiment"]["id"] == "synthetic_direct_hgb_v1"
    assert first.config["architecture"]["id"] == "direct_v1"
    assert first.architecture.topological_order() == ()
    assert first.config["data"]["weight_policy"] == {
        "fit": None,
        "calibration": None,
        "evaluation": None,
        "claim_boundary": "sample_conditional",
    }
    assert first.config["anchors"] is None
    assert first.config["splits"]["strategy"] == "household_grouped_v1"
    assert first.config["splits"]["group_columns"] == ["CODUSU", "NRO_HOGAR"]
    assert first.config["features"]["columns"]["AGLO_rk"]["allowed_as_external_input"] is False
    assert first.config["features"]["columns"]["Reg_rk"]["allowed_as_external_input"] is False


def test_experiment_override_is_explicit_and_changes_digest(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        DIRECT.read_text(encoding="utf-8")
        + "\noverrides:\n  estimators:\n    roles:\n      terminal:\n        params:\n          max_iter: 17\n",
        encoding="utf-8",
    )

    resolved = resolve_experiment(base, config_root=tmp_path)
    canonical = resolve_experiment(DIRECT, config_root=CONFIG_ROOT)
    assert resolved.config["estimators"]["roles"]["terminal"]["params"]["max_iter"] == 17
    assert resolved.digest != canonical.digest


def test_invalid_science_fails_before_fitting(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    text = DIRECT.read_text(encoding="utf-8")
    invalid.write_text(
        text.replace("fit: null", "fit: PONDERA", 1),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentResolutionError,
        match="starter_weight_policy_must_be_none",
    ):
        resolve_experiment(invalid, config_root=tmp_path)


def test_forbidden_feature_cannot_enter_a_deployable_dag_node(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-feature.yaml"
    text = DIRECT.read_text(encoding="utf-8")
    text = text.replace(
        "architecture:\n  id: direct_v1\n  nodes: {}",
        """architecture:
  id: invalid_v1
  nodes:
    labor_state:
      targets: [CAT_OCUP]
      task_by_target: {CAT_OCUP: multiclass}
      base_inputs: [P02, AGLO_rk]
      upstream_nodes: []
      estimator_role: latent_categorical
      output_representation: probabilities""",
    )
    invalid.write_text(text, encoding="utf-8")

    with pytest.raises(
        ExperimentResolutionError,
        match="forbidden_external_feature",
    ):
        resolve_experiment(invalid, config_root=tmp_path)


def test_component_reference_cannot_escape_config_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.yaml"
    outside.write_text("id: outside\n", encoding="utf-8")
    experiment = tmp_path / "escape.yaml"
    experiment.write_text(
        """experiment: {id: escape}
data: ../outside.yaml
features: {id: f, columns: {X: {dtype_role: continuous, temporal_role: observed, allowed_as_external_input: true}}}
architecture: {id: direct, nodes: {}}
terminal: {id: t, prediction_unit: person, target: Y}
estimators: {id: e, roles: {terminal: {family: hgb}}}
splits: {id: s, strategy: household_grouped_v1, group_columns: [CODUSU, NRO_HOGAR], n_splits: 2}
calibration: {id: raw}
anchors: null
uncertainty: {id: none}
evaluation: {id: core}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentResolutionError,
        match="component_reference_escapes_config_root",
    ):
        resolve_experiment(experiment, config_root=tmp_path)
