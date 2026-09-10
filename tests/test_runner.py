from __future__ import annotations

import json
from pathlib import Path

from encuestador.experiments import resolve_experiment
from encuestador.runner import (
    WelfareContext,
    configured_fold_manifest,
    execute_experiment,
)
from encuestador.transport_proof import (
    DEFAULT_SEMANTIC_PLANE,
    DEFAULT_SPEC,
    make_synthetic_people,
    validate_semantic_plane,
    validate_spec,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
LEAN = CONFIG_ROOT / "experiments" / "synthetic_lean_hgb_v1.yaml"


def test_period_qualified_rows_keep_repeated_household_waves_in_one_split_group() -> None:
    rows = [
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 2},
        {"CODUSU": "B", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "C", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "D", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "E", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "F", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "G", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
        {"CODUSU": "H", "NRO_HOGAR": "1", "COMPONENTE": 1, "ANO4": 2024, "TRIMESTRE": 1},
    ]
    manifest = configured_fold_manifest(
        rows,
        person_id_columns=("CODUSU", "NRO_HOGAR", "COMPONENTE"),
        household_group_columns=("CODUSU", "NRO_HOGAR"),
        period_columns=("ANO4", "TRIMESTRE"),
        n_splits=3,
    )

    assert manifest.row_ids[0] != manifest.row_ids[1]
    assert manifest.household_ids[0] == manifest.household_ids[1]
    assert manifest.fold_ids[0] == manifest.fold_ids[1]


def _approved_rows() -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    plane = json.loads(DEFAULT_SEMANTIC_PLANE.read_text(encoding="utf-8"))
    validate_spec(spec)
    approved = validate_semantic_plane(spec, plane)
    training = make_synthetic_people(
        approved,
        household_count=28,
        codusu_prefix="EPH-RUNNER",
        include_census_design=False,
    )
    scoring = make_synthetic_people(
        approved,
        household_count=9,
        codusu_prefix="CPV-RUNNER",
        include_census_design=True,
    )
    for row in scoring:
        row["sample_person_id"] = (
            f"sample-person:{row['CODUSU']}:{row['NRO_HOGAR']}:{row['COMPONENTE']}"
        )
        row["sample_household_id"] = f"sample-household:{row['CODUSU']}:{row['NRO_HOGAR']}"
    return approved, training, scoring


def test_resolved_lean_experiment_runs_to_exact_sample_household_welfare(tmp_path: Path) -> None:
    config = tmp_path / "lean.yaml"
    config.write_text(
        LEAN.read_text(encoding="utf-8")
        + """
overrides:
  splits:
    n_splits: 4
  estimators:
    roles:
      latent_categorical:
        params:
          early_stopping: false
          random_state: 42
          max_iter: 6
          max_leaf_nodes: 7
          min_samples_leaf: 5
      terminal:
        params:
          early_stopping: false
          random_state: 42
          max_iter: 8
          max_leaf_nodes: 9
          min_samples_leaf: 5
""",
        encoding="utf-8",
    )
    resolved = resolve_experiment(config, config_root=tmp_path)
    _approved, training, scoring = _approved_rows()
    context = WelfareContext(
        welfare_period="synthetic-2025Q1",
        currency="SYN",
        price_reference="synthetic_linear_units",
        welfare_concept="household_total_person_income",
    )

    result = execute_experiment(
        resolved,
        training,
        scoring_rows=scoring,
        welfare_context=context,
        input_release_ids=("synthetic-eph", "synthetic-census-sample"),
    )

    assert result.architecture_id == "lean_labor_v1"
    assert result.oof_prediction.fold_ids == result.fold_manifest.fold_ids
    assert set(result.latent_predictions) == {"CAT_OCUP", "CAT_INAC", "CH07"}
    assert result.score_prediction is not None
    assert len(result.household_welfare) == 9
    assert sum(row["member_count"] for row in result.household_welfare) == len(scoring)
    assert all(row["estimation_status"] == "complete" for row in result.household_welfare)
    assert all(row["transport_model_release_id"] == f"run:{result.run_id}" for row in result.household_welfare)
    assert set(result.metrics["person"]) == {"direct", "oracle", "deployable"}
    assert result.metrics["household"]["person_observation_count"] == len(training)
