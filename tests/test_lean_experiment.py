from __future__ import annotations

from pathlib import Path

from encuestador.experiments import resolve_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
LEAN = CONFIG_ROOT / "experiments" / "synthetic_lean_hgb_v1.yaml"


def test_committed_lean_experiment_resolves_to_one_probability_latent_layer() -> None:
    resolved = resolve_experiment(LEAN, config_root=CONFIG_ROOT)

    assert resolved.config["experiment"]["id"] == "synthetic_lean_hgb_v1"
    assert resolved.architecture.topological_order() == ("labor_state",)
    node = resolved.architecture.nodes["labor_state"]
    assert node.targets == ("CAT_OCUP", "CAT_INAC", "CH07")
    assert node.output_representation == "probabilities"
    assert node.estimator_role == "latent_categorical"
    assert resolved.config["splits"]["cascade_nesting"] == "outer_fold_isolated_latent_crossfit_v1"
    assert set(resolved.config["estimators"]["roles"]) == {"latent_categorical", "terminal"}
    assert resolved.config["anchors"] is None
