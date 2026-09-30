from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from encuestador.longitudinal_c6 import (
    C6_RUN_CONTRACT,
    MODEL_PLANE_CONTRACT,
    compare_c6_runs,
    load_longitudinal_model_plane,
    materialize_longitudinal_model_plane,
    resolve_config_for_model_plane,
    run_resource_safe_l10,
)
from encuestador.longitudinal_intake import (
    CompositionPlaneProfile,
    LaborContextRelease,
    LongitudinalEPHRelease,
    canonical_composition_parent_metadata,
    exact_parent_metadata,
    load_composition_plane_profile,
)
from encuestador.longitudinal_runtime import (
    attach_labor_context,
    build_longitudinal_fold_manifest,
    execute_longitudinal_arm,
    load_longitudinal_config,
    resolve_composition_profile,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fold(group: str, n_splits: int) -> int:
    digest = hashlib.sha256(group.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_splits


def _groups_for_folds(n_splits: int, per_fold: int) -> list[str]:
    output: list[str] = []
    for fold in range(n_splits):
        found = 0
        candidate = 0
        while found < per_fold:
            group = f"panel-f{fold}-{candidate}"
            candidate += 1
            if _fold(group, n_splits) == fold:
                output.append(group)
                found += 1
    return output


def _fixture(tmp_path: Path):
    n_splits = 3
    periods = (
        "2017-Q1",
        "2018-Q2",
        "2019-Q4",
        "2020-Q1",
        "2024-Q3",
        "2025-Q4",
        "2026-Q1",
    )
    groups = _groups_for_folds(n_splits, 10)
    eph_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        period = periods[index % len(periods)]
        row_id = f"row-{index:04d}"
        eph_rows.append(
            {
                "row_id": row_id,
                "period": period,
                "region_id": "pampeana",
                "panel_household_id": group,
                "household_observation_id": f"obs-{index:04d}",
                "P47T_real": 0.0 if index % 4 == 0 else 800.0 + 17.0 * index,
            }
        )
        composition_rows.append(
            {
                "row_id": row_id,
                "P02": 1 + index % 2,
                "P03": 18 + index % 55,
            }
        )

    eph_root = tmp_path / "eph-fixture"
    eph_root.mkdir()
    _write_csv(eph_root / "persons.csv", eph_rows)
    (eph_root / "manifest.json").write_text(
        json.dumps(
            {
                "contract": "research.eph-longitudinal-analysis-frame/v1",
                "release_id": "eph-fixture",
                "artifacts": {"persons.csv": {"rows": len(eph_rows)}},
            }
        ),
        encoding="utf-8",
    )
    eph = LongitudinalEPHRelease(
        root=eph_root,
        release_id="eph-fixture",
        persons_path=eph_root / "persons.csv",
        coverage_path=eph_root / "coverage.csv",
        manifest=json.loads((eph_root / "manifest.json").read_text(encoding="utf-8")),
        monetary_release_id="money-fixture",
        monetary_reference_period="2026-01-01",
    )

    profile_id = "PTEST_LONG"
    composition_release_id = "composition-fixture"
    composition_root = tmp_path / composition_release_id
    composition_root.mkdir()
    _write_csv(composition_root / "composition_plane.csv", composition_rows)
    profile_record = {
        "artifact": "composition_plane.csv",
        "features": ["P02", "P03"],
        "categorical_features": ["P02"],
    }
    composition_manifest = {
        "contract": "research.eph-longitudinal-composition-plane/v1",
        "release_id": composition_release_id,
        "row_id_field": "row_id",
        "profiles": {profile_id: profile_record},
        "artifacts": {
            "composition_plane.csv": {
                "sha256": _sha(composition_root / "composition_plane.csv"),
                "rows": len(composition_rows),
            }
        },
    }
    (composition_root / "manifest.json").write_text(
        json.dumps(composition_manifest), encoding="utf-8"
    )
    profile = load_composition_plane_profile(composition_root, profile_id)

    labor_rows: list[dict[str, object]] = []
    for period_index, period in enumerate(periods):
        for geography_level, geography_id, activity, unemployment, subemployment in (
            ("total_31", "total_31_agglomerates", 46.0, 7.0, 11.0),
            (
                "region",
                "pampeana",
                47.0 + 0.1 * period_index,
                6.5 + 0.05 * period_index,
                12.0,
            ),
        ):
            for indicator, value in (
                ("activity_rate", activity),
                ("unemployment_rate", unemployment),
                ("subemployment_rate", subemployment),
            ):
                labor_rows.append(
                    {
                        "period": period,
                        "geography_level": geography_level,
                        "geography_id": geography_id,
                        "indicator_id": indicator,
                        "value": value,
                        "value_status": "observed",
                        "source_id": "labor-fixture",
                        "source_snapshot_sha256": "a" * 64,
                        "source_cell_identity": f"{period}:{geography_id}:{indicator}",
                    }
                )
    labor_root = tmp_path / "labor-fixture"
    labor_root.mkdir()
    _write_csv(labor_root / "labor_state.csv", labor_rows)
    (labor_root / "manifest.json").write_text(
        json.dumps(
            {
                "contract_id": "publicdata.indec-eph-labor-state/v1",
                "release_id": "labor-fixture",
            }
        ),
        encoding="utf-8",
    )
    labor = LaborContextRelease(
        root=labor_root,
        release_id="labor-fixture",
        observations_path=labor_root / "labor_state.csv",
        manifest=json.loads((labor_root / "manifest.json").read_text(encoding="utf-8")),
    )

    config_path = tmp_path / "l10_c6.yaml"
    config_path.write_text(
        f"""
arm: L10
schema: research.encuestador-longitudinal-config/v1
composition_source: canonical_parent
composition_parent_contract: research.eph-longitudinal-composition-plane/v1
feature_profile_id: {profile_id}
features:
  composition: []
  categorical: []
target_field: P47T_real
identity:
  period_field: period
  region_field: region_id
  group_field: panel_household_id
  row_id_field: row_id
  household_observation_field: household_observation_id
panel:
  person_candidate_field: person_linkage_candidate_id
  observed_labor_fields: [ESTADO, CONDACT]
  stale_labor_field: stale_labor_state
  current_labor_field: target_current_labor_state
  elapsed_quarters_field: elapsed_quarters
  allowed_elapsed_quarters: [1, 3]
splits:
  n_splits: {n_splits}
estimators:
  presence: {{early_stopping: false, random_state: 42, max_iter: 4, max_leaf_nodes: 7, min_samples_leaf: 2, learning_rate: 0.08}}
  positive_amount: {{early_stopping: false, random_state: 42, max_iter: 4, max_leaf_nodes: 7, min_samples_leaf: 2, learning_rate: 0.08}}
  transition: {{early_stopping: false, random_state: 42, max_iter: 4, max_leaf_nodes: 7, min_samples_leaf: 2, learning_rate: 0.08}}
anchor: {{enabled: false}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = resolve_composition_profile(
        load_longitudinal_config(config_path),
        profile,
    )
    return eph_rows, composition_rows, eph, labor, profile, config


def test_c6_model_plane_streams_exact_identity_and_persists_folds(tmp_path: Path) -> None:
    eph_rows, _, eph, labor, profile, config = _fixture(tmp_path)
    root = materialize_longitudinal_model_plane(
        eph,
        labor,
        profile,
        config,
        tmp_path / "planes",
    )
    plane = load_longitudinal_model_plane(root)
    assert plane.manifest["contract"] == MODEL_PLANE_CONTRACT
    assert plane.row_count == len(eph_rows)
    assert plane.feature_names == (
        "P02",
        "P03",
        "labor_national_activity_rate",
        "labor_national_unemployment_rate",
        "labor_national_subemployment_rate",
        "labor_regional_activity_deviation",
        "labor_regional_unemployment_deviation",
        "labor_regional_subemployment_deviation",
    )
    assert np.load(plane.features_path, mmap_mode="r").shape == (
        len(eph_rows),
        len(plane.feature_names),
    )
    row_ids = plane.row_ids_path.read_text(encoding="utf-8").splitlines()
    assert row_ids == [str(row["row_id"]) for row in eph_rows]
    folds = np.load(plane.fold_ids_path, mmap_mode="r")
    assert set(folds.tolist()) == {0, 1, 2}
    for row, fold in zip(eph_rows, folds.tolist(), strict=True):
        assert fold == _fold(str(row["panel_household_id"]), config.n_splits)
    assert plane.manifest["execution_semantics"][
        "million_row_python_dict_collection"
    ] is False


def test_c6_l10_matches_legacy_c4b_semantics_and_resumes_fold(tmp_path: Path) -> None:
    eph_rows, composition_rows, eph, labor, profile, config = _fixture(tmp_path)
    root = materialize_longitudinal_model_plane(
        eph,
        labor,
        profile,
        config,
        tmp_path / "planes",
    )
    plane = load_longitudinal_model_plane(root)
    c6_config = resolve_config_for_model_plane(
        load_longitudinal_config(tmp_path / "l10_c6.yaml"),
        plane,
    )

    work = run_resource_safe_l10(
        plane,
        c6_config,
        tmp_path / "runs",
        labor_mode="national_regional",
        outer_fold=0,
    )
    assert work.name.endswith(".work")
    assert (work / "checkpoints" / "fold_0" / "checkpoint.json").is_file()

    c6_root = run_resource_safe_l10(
        plane,
        c6_config,
        tmp_path / "runs",
        labor_mode="national_regional",
    )
    c6_manifest = json.loads(
        (c6_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert c6_manifest["contract"] == C6_RUN_CONTRACT
    assert len(c6_manifest["checkpoints"]) == config.n_splits

    composition_by_id = {
        str(row["row_id"]): row for row in composition_rows
    }
    legacy_rows = []
    for source in eph_rows:
        merged = dict(source)
        comp = composition_by_id[str(source["row_id"])]
        merged["P02"] = comp["P02"]
        merged["P03"] = comp["P03"]
        legacy_rows.append(merged)
    legacy_rows = attach_labor_context(
        legacy_rows,
        labor.read_observations(),
        period_field=config.period_field,
        region_field=config.region_field,
    )
    parents = exact_parent_metadata(
        eph,
        labor,
        composition_metadata=canonical_composition_parent_metadata(profile),
    )
    manifest = build_longitudinal_fold_manifest(legacy_rows, config)
    legacy = execute_longitudinal_arm(
        legacy_rows,
        config,
        parent_metadata=parents,
        fold_manifest=manifest,
    )

    c6_p = np.load(c6_root / "p_positive.npy")
    c6_amount = np.load(c6_root / "positive_amount.npy")
    c6_income = np.load(c6_root / "expected_income.npy")
    assert np.allclose(
        c6_p,
        legacy.hurdle.p_positive.values[:, 1],
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        c6_amount,
        legacy.hurdle.positive_amount_prediction.values,
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        c6_income,
        legacy.hurdle.unconditional_expected_income.values,
        rtol=1e-10,
        atol=1e-10,
    )


def test_c6_compare_requires_matched_identity_and_folds(tmp_path: Path) -> None:
    _, _, eph, labor, profile, config = _fixture(tmp_path)
    plane_root = materialize_longitudinal_model_plane(
        eph,
        labor,
        profile,
        config,
        tmp_path / "planes",
    )
    plane = load_longitudinal_model_plane(plane_root)
    c6_config = resolve_config_for_model_plane(
        load_longitudinal_config(tmp_path / "l10_c6.yaml"),
        plane,
    )
    run_none = run_resource_safe_l10(
        plane,
        c6_config,
        tmp_path / "runs",
        labor_mode="none",
    )
    run_full = run_resource_safe_l10(
        plane,
        c6_config,
        tmp_path / "runs",
        labor_mode="national_regional",
    )
    comparison = compare_c6_runs([run_none, run_full])
    assert comparison["matched_row_identity_sequence"] is True
    assert comparison["matched_fold_assignments"] is True
    assert [row["labor_mode"] for row in comparison["runs"]] == [
        "none",
        "national_regional",
    ]
    assert comparison["promotion_authorized"] is False
