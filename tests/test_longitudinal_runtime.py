from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from encuestador.anchors import multiclass_kl_moment_projection
from encuestador.longitudinal_intake import (
    exact_parent_metadata,
    load_donor_labor_release,
    load_labor_context_release,
    load_longitudinal_eph_release,
)
from encuestador.longitudinal_runtime import (
    LaborMomentAnchor,
    LongitudinalRuntimeError,
    attach_labor_context,
    build_longitudinal_fold_manifest,
    build_panel_pairs,
    execute_longitudinal_arm,
    load_longitudinal_config,
)
from encuestador.time_layer import fit_time_layer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "longitudinal"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups_by_fold(n_per_fold: int = 9, n_splits: int = 5) -> dict[int, list[str]]:
    output = {fold: [] for fold in range(n_splits)}
    candidate = 0
    while any(len(values) < n_per_fold for values in output.values()):
        group = f"panel-hh-{candidate:04d}"
        digest = hashlib.sha256(group.encode()).digest()
        fold = int.from_bytes(digest[:8], "big") % n_splits
        if len(output[fold]) < n_per_fold:
            output[fold].append(group)
        candidate += 1
    return output


def _base_fields(index: int, group: str, *, period: str, current_state: int) -> dict:
    return {
        "row_id": f"row-{period}-{group}",
        "household_observation_id": f"hhobs-{period}-{group}",
        "panel_household_id": group,
        "person_linkage_candidate_id": f"person-{group}",
        "period": period,
        "region_id": "pampeana",
        "CH04": 1 + index % 2,
        "CH06": 20 + index % 45,
        "CH09": 1 + index % 2,
        "CH10": 1 + index % 3,
        "CH12": 1 + index % 8,
        "CH13": 1 + index % 2,
        "CH15": 1 + index % 5,
        "IX_TOT": 1 + index % 5,
        "ESTADO": current_state,
        "labor_national_activity_rate": 48.0,
        "labor_national_unemployment_rate": 7.0,
        "labor_national_subemployment_rate": 11.0,
        "labor_regional_activity_deviation": 1.0,
        "labor_regional_unemployment_deviation": -0.5,
        "labor_regional_subemployment_deviation": 0.75,
    }


def _l10_rows() -> list[dict]:
    periods = [
        "2023-Q1",
        "2023-Q2",
        "2023-Q3",
        "2023-Q4",
        "2024-Q1",
        "2024-Q2",
        "2024-Q3",
        "2024-Q4",
        "2025-Q1",
    ]
    rows: list[dict] = []
    index = 0
    for fold, groups in _groups_by_fold(18).items():
        for within, group in enumerate(groups):
            period = periods[(fold * 3 + within) % len(periods)]
            state = 1 + (within % 3)
            row = _base_fields(index, group, period=period, current_state=state)
            row["P47T_real"] = 0.0 if within % 3 == 0 else 1000.0 + 25 * index
            rows.append(row)
            index += 1
    return rows


def _panel_source_rows() -> list[dict]:
    rows: list[dict] = []
    index = 0
    for groups in _groups_by_fold(9).values():
        for within, group in enumerate(groups):
            stale = 1 + (within % 3)
            current = 1 + ((within + 1) % 3)
            earlier = _base_fields(
                index, group, period="2024-Q2", current_state=stale
            )
            later = _base_fields(
                index + 1000, group, period="2024-Q3", current_state=current
            )
            earlier["row_id"] = f"earlier-{group}"
            later["row_id"] = f"later-{group}"
            earlier["P47T_real"] = 500.0 + index
            later["P47T_real"] = 0.0 if within % 4 == 0 else 1500.0 + 20 * index
            rows.extend([earlier, later])
            index += 1
    return rows


def test_kl_projection_hits_exact_moments_without_hard_flips() -> None:
    raw = np.asarray(
        [[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.20, 0.20, 0.60], [0.40, 0.30, 0.30]]
    )
    anchored, diagnostics = multiclass_kl_moment_projection(
        raw,
        class_labels=("1", "2", "3"),
        target_shares={"1": 0.40, "2": 0.35, "3": 0.25},
        universe_contract="fixture_age14plus_population",
        anchor_release_id="labor-anchor-fixture",
    )
    assert np.allclose(anchored.mean(axis=0), [0.40, 0.35, 0.25], atol=1e-9)
    assert np.allclose(anchored.sum(axis=1), 1.0)
    assert diagnostics.constraint_residual < 1e-9
    assert diagnostics.weighted_mean_kl > 0
    assert diagnostics.max_probability_displacement > 0
    assert np.array_equal(raw.argmax(axis=1), anchored.argmax(axis=1))


def test_labor_context_join_is_exact_period_region_context_not_person_probability() -> None:
    observations = []
    for geography_level, geography_id, activity, unemployment, subemployment in (
        ("total_31", "total_31_agglomerates", 48.0, 7.0, 11.0),
        ("region", "pampeana", 50.0, 6.5, 12.0),
    ):
        for indicator, value in (
            ("activity_rate", activity),
            ("unemployment_rate", unemployment),
            ("subemployment_rate", subemployment),
        ):
            observations.append(
                {
                    "period": "2024-Q3",
                    "geography_level": geography_level,
                    "geography_id": geography_id,
                    "indicator_id": indicator,
                    "value": str(value),
                    "value_status": "observed",
                    "source_id": "indec-fixture",
                    "source_snapshot_sha256": "a" * 64,
                    "source_cell_identity": f"{geography_id}:{indicator}:2024-Q3",
                }
            )
    joined = attach_labor_context(
        [{"period": "2024-Q3", "region_id": "pampeana", "row_id": "x"}],
        observations,
    )
    assert joined[0]["labor_national_unemployment_rate"] == 7.0
    assert joined[0]["labor_regional_unemployment_deviation"] == -0.5
    assert (
        joined[0]["labor_context_semantics"]
        == "aggregate_context_not_individual_probability"
    )
    assert len(joined[0]["labor_context_source_cells"]) == 6
    assert all(
        cell["source_id"] == "indec-fixture"
        for cell in joined[0]["labor_context_source_cells"]
    )
    broken = [row for row in observations if row["indicator_id"] != "subemployment_rate"]
    with pytest.raises(LongitudinalRuntimeError, match="labor_context_join_incomplete"):
        attach_labor_context(
            [{"period": "2024-Q3", "region_id": "pampeana"}], broken
        )


def test_exceptional_periods_do_not_estimate_ordinary_2024_level() -> None:
    periods = np.asarray(
        [
            "2024-Q1",
            "2024-Q1",
            "2024-Q2",
            "2024-Q2",
            "2024-Q3",
            "2024-Q3",
            "2024-Q4",
            "2024-Q4",
            "2026-Q1",
            "2026-Q1",
        ],
        dtype=object,
    )
    probability = np.full(len(periods), 0.5)
    amount = np.full(len(periods), 100.0)
    target_a = np.asarray([0, 20, 0, 30, 0, 100, 0, 120, 0, 140], dtype=object)
    target_b = np.asarray([0, 2, 0, 3, 0, 100, 0, 120, 0, 140], dtype=object)
    fit_a = fit_time_layer(periods, probability, amount, target_a)
    fit_b = fit_time_layer(periods, probability, amount, target_b)

    assert fit_a.amount_year[2024] == pytest.approx(fit_b.amount_year[2024])
    assert fit_a.amount_quarter[3] == pytest.approx(fit_b.amount_quarter[3])
    assert fit_a.amount_quarter[4] == pytest.approx(fit_b.amount_quarter[4])
    assert fit_a.amount_exception["2024-Q1"] != pytest.approx(
        fit_b.amount_exception["2024-Q1"]
    )
    assert fit_a.year_support[2024]["ordinary_periods_used_for_year_effect"] == [
        "2024-Q3",
        "2024-Q4",
    ]
    assert fit_a.year_support[2026]["quarters_observed_for_year"] == 1
    assert fit_a.year_support[2026]["partial_year"] is True


def test_panel_pairs_use_actual_earlier_state_and_panel_households_never_cross_folds() -> None:
    config = load_longitudinal_config(CONFIG / "l11.yaml")
    pairs = build_panel_pairs(_panel_source_rows(), config)
    assert pairs
    for pair in pairs:
        assert pair["stale_period"] == "2024-Q2"
        assert pair["target_period"] == "2024-Q3"
        assert pair["elapsed_quarters"] == 1
        assert pair["stale_observation_row_id"].startswith("earlier-")
        assert pair["target_observation_row_id"].startswith("later-")
        assert pair["stale_labor_state"] != pair["target_current_labor_state"]

    manifest = build_longitudinal_fold_manifest(pairs, config)
    observed: dict[str, set[int]] = {}
    for group, fold in zip(manifest.household_ids, manifest.fold_ids, strict=True):
        observed.setdefault(group, set()).add(fold)
    assert all(len(folds) == 1 for folds in observed.values())


def test_l10_holdout_predictions_do_not_depend_on_holdout_targets() -> None:
    config = load_longitudinal_config(CONFIG / "l10.yaml")
    rows = _l10_rows()
    manifest = build_longitudinal_fold_manifest(rows, config)
    parents = {
        "longitudinal_eph": {"release_id": "eph-longitudinal-fixture"},
        "labor_context": {"release_id": "labor-context-fixture"},
        "monetary_conversion": {
            "release_id": "money-fixture",
            "reference_period": "2025-11-01",
        },
    }
    first = execute_longitudinal_arm(
        rows, config, parent_metadata=parents, fold_manifest=manifest
    )
    changed = [dict(row) for row in rows]
    for index, fold in enumerate(manifest.fold_ids):
        if fold == 0:
            value = float(changed[index]["P47T_real"])
            changed[index]["P47T_real"] = value * 25.0 if value > 0 else 0.0
    second = execute_longitudinal_arm(
        changed, config, parent_metadata=parents, fold_manifest=manifest
    )
    mask = np.asarray(manifest.fold_ids) == 0
    assert np.allclose(
        first.hurdle.unconditional_expected_income.values[mask],
        second.hurdle.unconditional_expected_income.values[mask],
    )


def test_l11_and_l12_execute_without_current_state_copy_and_anchor_is_l12_only() -> None:
    source = _panel_source_rows()
    l11 = load_longitudinal_config(CONFIG / "l11.yaml")
    pairs = build_panel_pairs(source, l11)
    parents = {
        "longitudinal_eph": {"release_id": "eph-longitudinal-fixture"},
        "labor_context": {"release_id": "labor-context-fixture"},
        "monetary_conversion": {
            "release_id": "money-fixture",
            "reference_period": "2025-11-01",
        },
        "donor_labor": {
            "semantic_plane_release_id": "semantic-fixture",
            "donor_vintage": 2010,
        },
    }
    shared_manifest = build_longitudinal_fold_manifest(pairs, l11)
    result_l11 = execute_longitudinal_arm(
        pairs, l11, parent_metadata=parents, fold_manifest=shared_manifest
    )
    assert "stale_labor_state" in result_l11.feature_names
    assert result_l11.panel_diagnostics["supported_elapsed_quarters"] == [1]

    l12 = load_longitudinal_config(CONFIG / "l12.yaml")
    result_l12 = execute_longitudinal_arm(
        pairs, l12, parent_metadata=parents, fold_manifest=shared_manifest
    )
    assert "stale_labor_state" not in result_l12.feature_names
    assert result_l12.transition_raw is not None
    assert result_l12.transition_raw.source == "oof"

    anchored = load_longitudinal_config(CONFIG / "l12_anchored.yaml")
    anchor = LaborMomentAnchor(
        period="2024-Q3",
        region_id="pampeana",
        class_shares={"1": 0.40, "2": 0.35, "3": 0.25},
        release_id="official-labor-anchor-fixture",
        universe_contract="fixture_age14plus_same_population",
    )
    result_anchor = execute_longitudinal_arm(
        pairs,
        anchored,
        parent_metadata=parents,
        fold_manifest=shared_manifest,
        anchors=(anchor,),
    )
    assert result_anchor.transition_anchored is not None
    assert np.allclose(
        result_anchor.transition_anchored.values.mean(axis=0),
        [0.40, 0.35, 0.25],
        atol=1e-9,
    )
    assert result_anchor.anchor_diagnostics

    with pytest.raises(LongitudinalRuntimeError, match="aggregate_anchor_only_allowed_in_L12"):
        execute_longitudinal_arm(
            pairs,
            l11,
            parent_metadata=parents,
            fold_manifest=shared_manifest,
            anchors=(anchor,),
        )


def test_artifact_consumers_bind_exact_c1_c2_c3_parents(tmp_path: Path) -> None:
    eph_id = "eph-longitudinal-fixture"
    eph_root = tmp_path / eph_id
    eph_root.mkdir()
    for name, text in (
        ("persons.csv", "row_id,period\na,2017-Q1\n"),
        ("coverage.csv", "period,status\n2017-Q1,present\n"),
        ("monetary_lineage.csv", "period,factor\n2017-Q1,1\n"),
        ("qa.json", "{}\n"),
    ):
        (eph_root / name).write_text(text, encoding="utf-8")
    eph_artifacts = {
        name: {"sha256": _sha(eph_root / name), "size_bytes": (eph_root / name).stat().st_size}
        for name in ("persons.csv", "coverage.csv", "monetary_lineage.csv", "qa.json")
    }
    (eph_root / "manifest.json").write_text(
        json.dumps(
            {
                "contract": "research.eph-longitudinal-analysis-frame/v1",
                "release_id": eph_id,
                "coverage": {
                    "period_start": "2017-Q1",
                    "period_end": "2026-Q1",
                    "period_count": 37,
                    "all_expected_periods_present": True,
                },
                "artifacts": eph_artifacts,
                "parents": {
                    "monetary_conversion": {"release_id": "money-release-fixture"}
                },
                "monetary_lineage": {
                    "common_reference_period": "2025-11-01",
                    "source_field": "P47T",
                    "real_output": "P47T_real",
                },
                "field_policy": {"zero_income_rows_retained": True},
            }
        ),
        encoding="utf-8",
    )

    labor_id = "labor-context-fixture"
    labor_root = tmp_path / labor_id
    labor_root.mkdir()
    (labor_root / "labor_state.csv").write_text(
        "period,geography_level,geography_id,indicator_id,value,value_status\n"
        "2017-Q1,total_31,total_31_agglomerates,activity_rate,48,observed\n",
        encoding="utf-8",
    )
    (labor_root / "coverage.csv").write_text(
        "period,coverage_status\n2017-Q1,present\n", encoding="utf-8"
    )
    (labor_root / "geographies.json").write_text("[]\n", encoding="utf-8")
    (labor_root / "indicators.json").write_text("[]\n", encoding="utf-8")
    labor_files = {
        name: _sha(labor_root / name)
        for name in (
            "labor_state.csv",
            "coverage.csv",
            "geographies.json",
            "indicators.json",
        )
    }
    (labor_root / "manifest.json").write_text(
        json.dumps(
            {
                "contract_id": "publicdata.indec-eph-labor-state/v1",
                "release_id": labor_id,
                "period_min": "2017-Q1",
                "period_max": "2026-Q2",
                "required_coverage_complete": True,
                "files": labor_files,
            }
        ),
        encoding="utf-8",
    )

    donor_root = tmp_path / "semantic-plane"
    donor_root.mkdir()
    (donor_root / "donor.parquet").write_bytes(b"fixture-not-read")
    (donor_root / "donor_labor_qa.json").write_text(
        json.dumps(
            {
                "identity_row_count_preserved": True,
                "clock_separation": {"same_clock": False},
            }
        ),
        encoding="utf-8",
    )
    (donor_root / "feature_plane_manifest.json").write_text(
        json.dumps(
            {
                "schema": "research.eph-census-semantic-feature-plane/v1",
                "release_id": "semantic-fixture-v2",
                "donor_labor_handoff": {
                    "schema": "research.eph-census-donor-labor-handoff/v1",
                    "path": "donor.parquet",
                    "sha256": _sha(donor_root / "donor.parquet"),
                    "qa_path": "donor_labor_qa.json",
                    "qa_sha256": _sha(donor_root / "donor_labor_qa.json"),
                    "donor_vintage": 2010,
                    "target_period_current_state_claimed": False,
                    "eph_training_analogue_materialized": False,
                },
            }
        ),
        encoding="utf-8",
    )

    eph = load_longitudinal_eph_release(eph_root)
    labor = load_labor_context_release(labor_root)
    donor = load_donor_labor_release(donor_root)
    parents = exact_parent_metadata(eph, labor, donor=donor)
    assert parents["longitudinal_eph"]["release_id"] == eph_id
    assert parents["labor_context"]["release_id"] == labor_id
    assert parents["monetary_conversion"]["release_id"] == "money-release-fixture"
    assert parents["donor_labor"]["donor_vintage"] == 2010
