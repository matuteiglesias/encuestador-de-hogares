from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from encuestador.transport_proof import (
    DEFAULT_SEMANTIC_PLANE,
    DEFAULT_SPEC,
    aggregate_households,
    run_synthetic_proof,
    validate_semantic_plane,
    validate_spec,
)


def test_historical_staged_v1_freezes_current_default_and_starter_policies() -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    validate_spec(spec)

    assert spec["prediction_unit"] == "person"
    assert spec["terminal_person_income_target"] == "P47T"
    assert spec["fold_policy"]["strategy"] == "household_grouped_v1"
    assert spec["fold_policy"]["learned_intermediates_for_training"] == (
        "out_of_fold_predictions"
    )
    assert spec["eph_design_policy"]["fit_weight"] is None
    assert spec["eph_design_policy"]["calibration_weight"] is None
    assert spec["eph_design_policy"]["evaluation_weight"] is None
    assert spec["temporal_policy"]["target_period_calibration"] == "none"
    assert spec["household_aggregation"]["operation"] == "sum_linear_person_income"
    assert spec["stages"][3]["targets"] == [
        "P21",
        "P47T",
        "PP08D1",
        "TOT_P12",
        "T_VI",
        "V12_M",
        "V2_M",
        "V3_M",
        "V5_M",
    ]


def test_synthetic_semantic_plane_blocks_target_derived_rank_inputs() -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    plane = json.loads(DEFAULT_SEMANTIC_PLANE.read_text(encoding="utf-8"))
    approved = validate_semantic_plane(spec, plane)

    assert "AGLO_rk" not in approved
    assert "Reg_rk" not in approved
    assert plane["features"]["AGLO_rk"]["semantic_class"] == "research_only"
    assert plane["features"]["Reg_rk"]["semantic_class"] == "research_only"
    assert plane["real_vintage_approval"] is False
    assert all(plane["features"][field]["transport_time_role"] for field in approved)


def test_synthetic_proof_is_household_grouped_unweighted_and_uncalibrated() -> None:
    summary = run_synthetic_proof()

    assert summary["status"] == "synthetic_proof_only"
    assert summary["fold_policy"]["households_crossing_folds"] == 0
    assert summary["fold_policy"]["learned_intermediate_training_source"] == (
        "out_of_fold_predictions"
    )
    assert summary["weight_policy"] == {
        "fit": None,
        "calibration": None,
        "evaluation": None,
        "claim_boundary": "sample_conditional",
        "census_selection_probability_used_as_model_weight": False,
    }
    assert summary["temporal_policy"] == {
        "target_period_calibration": "none",
        "aggregate_anchor": None,
        "donor_field_mutation": False,
    }
    assert all(stage["observed_intermediate_inputs_used"] is False for stage in summary["stages"])
    assert all(stage["weight_used"] is None for stage in summary["stages"])
    assert all("AGLO_rk" not in stage["features"] for stage in summary["stages"])
    assert all("Reg_rk" not in stage["features"] for stage in summary["stages"])


def test_direct_and_staged_are_compared_at_person_and_household_terminal_income() -> None:
    summary = run_synthetic_proof()

    for surface in (
        "eph_oof_person",
        "eph_oof_household",
        "synthetic_census_person",
        "synthetic_census_household",
    ):
        assert set(summary["metrics"][surface]) == {"direct", "staged"}
        for model in ("direct", "staged"):
            assert summary["metrics"][surface][model]["mae"] >= 0
            assert summary["metrics"][surface][model]["rmse"] >= 0
            assert np.isfinite(summary["metrics"][surface][model]["mean_error"])


def test_household_aggregation_refuses_to_hide_an_invalid_member() -> None:
    rows = [
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 1, "P47T": 100.0},
        {"CODUSU": "A", "NRO_HOGAR": "1", "COMPONENTE": 2, "P47T": 200.0},
    ]
    records = aggregate_households(
        rows,
        {
            "direct": np.asarray([90.0, 210.0]),
            "staged": np.asarray([95.0, np.nan]),
        },
    )

    assert records[0]["truth_household_income"] == 300.0
    assert records[0]["direct_household_income"] == 300.0
    assert records[0]["staged_household_income"] is None
    assert records[0]["staged_missing_members"] == [2]
    assert records[0]["status"] == "incomplete"
    assert records[0]["incomplete_models"] == ["staged"]


def test_proof_is_deterministic_and_emits_person_level_audit_files(tmp_path: Path) -> None:
    first = run_synthetic_proof(output_dir=tmp_path / "one")
    second = run_synthetic_proof(output_dir=tmp_path / "two")
    assert first == second

    for filename in (
        "eph_oof_person_predictions.csv",
        "eph_oof_household_predictions.csv",
        "census_person_predictions.csv",
        "census_household_predictions.csv",
        "summary.json",
    ):
        assert (tmp_path / "one" / filename).read_bytes() == (
            tmp_path / "two" / filename
        ).read_bytes()

    with (tmp_path / "one" / "census_person_predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert "selection_probability" in rows[0]
    assert "design_inverse_probability_weight" in rows[0]
    assert "direct_person_income" in rows[0]
    assert "staged_person_income" in rows[0]
