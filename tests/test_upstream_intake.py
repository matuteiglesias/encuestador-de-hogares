from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from encuestador.upstream_intake import (
    UpstreamIntakeError,
    load_census_sample_v2_release,
    load_eph_annual_release,
    read_eph_rows,
    sha256_file,
)


def _write_eph_fixture(tmp_path: Path, columns: list[str]) -> Path:
    data = tmp_path / "data" / "annual.csv"
    data.parent.mkdir(parents=True)
    with data.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow({name: str(index + 1) for index, name in enumerate(columns)})
        writer.writerow({name: str(index + 11) for index, name in enumerate(columns)})

    manifest = {
        "manifest_schema": "research-artifact-manifest/v1",
        "manifest_schema_version": "1.0",
        "method_contract_version": "research.eph-annual-preprocessed/v1",
        "artifact_type": "research.eph-annual-preprocessed",
        "release_id": "artifact:research.eph-annual-preprocessed@1+2025",
        "release_status": "candidate",
        "data_vintage": 2025,
        "coverage": {"years": [2025], "quarters": [1, 2, 3, 4]},
        "rows": 2,
        "column_inventory": columns,
        "artifact": {
            "path": "data/annual.csv",
            "role": "annual_preprocessed_input",
            "sha256": sha256_file(data),
        },
        "inputs": [
            {
                "release_id": "artifact:publicdata.eph-microdata@1+2025",
                "sha256": "a" * 64,
            }
        ],
        "limitations": [],
    }
    path = tmp_path / "manifests" / "annual.manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_eph_annual_release_requires_real_household_person_identity_and_reads_declared_columns(
    tmp_path: Path,
) -> None:
    columns = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "P02", "P47T"]
    manifest = _write_eph_fixture(tmp_path, columns)
    release = load_eph_annual_release(
        manifest,
        artifact_root=tmp_path,
        required_columns=("P02", "P47T"),
    )
    rows = read_eph_rows(
        release,
        columns=("CODUSU", "NRO_HOGAR", "COMPONENTE", "P02", "P47T"),
    )

    assert release.data_vintage == 2025
    assert release.rows == 2
    assert len(rows) == 2
    assert set(rows[0]) == {"CODUSU", "NRO_HOGAR", "COMPONENTE", "P02", "P47T"}


def test_eph_release_refuses_historical_nonunique_identity_shape(tmp_path: Path) -> None:
    manifest = _write_eph_fixture(tmp_path, ["CODUSU", "ANO4", "TRIMESTRE", "P47T"])
    with pytest.raises(
        UpstreamIntakeError,
        match="eph_release_missing_required_columns:COMPONENTE,NRO_HOGAR",
    ):
        load_eph_annual_release(manifest, artifact_root=tmp_path)


def _write_census_fixture(tmp_path: Path, *, analysis_weight: object = None) -> Path:
    root = tmp_path / "census-sample-2025-fixture"
    root.mkdir()
    (root / "selection.parquet").write_bytes(b"selection fixture")
    (root / "person_membership.parquet").write_bytes(b"membership fixture")
    qa = {
        "frame_vintage": "2010",
        "sampling_target_year": 2025,
        "selection_unit": "household",
        "target_mass_unit": "person",
        "complete_household_membership": True,
        "selected_counts": {"households": 2, "persons": 5},
        "materialization": "selection-only",
    }
    (root / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    artifacts = {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in ("selection.parquet", "person_membership.parquet", "qa.json")
    }
    manifest = {
        "contract": "research.census-target-year-sample/v2",
        "release_id": root.name,
        "frame": {
            "frame_release_id": "census-frame-2010-fixture",
            "census_vintage": "2010",
            "manifest_sha256": "b" * 64,
        },
        "target_population_parent": {"target_year": 2025, "sha256": "c" * 64},
        "selection": {
            "algorithm": "fixture",
            "unit": "household",
            "target_mass_unit": "person",
            "fraction": 0.01,
            "seed": 42,
        },
        "weight_semantics": {
            "selection_probability": "household inclusion probability",
            "design_inverse_probability_weight": "1 / selection_probability",
            "analysis_weight": analysis_weight,
            "generic_sample_weight": None,
        },
        "materialization": "selection-only",
        "artifacts": artifacts,
        "qa": qa,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_sampler_v2_manifest_intake_preserves_exact_design_boundary(tmp_path: Path) -> None:
    root = _write_census_fixture(tmp_path)
    release = load_census_sample_v2_release(root)

    assert release.release_id == "census-sample-2025-fixture"
    assert release.target_year == 2025
    assert release.selected_households == 2
    assert release.selected_persons == 5
    assert release.materialization == "selection-only"
    assert release.selection_path.name == "selection.parquet"
    assert release.person_membership_path.name == "person_membership.parquet"


def test_sampler_v2_intake_rejects_invented_analysis_weight(tmp_path: Path) -> None:
    root = _write_census_fixture(tmp_path, analysis_weight="selection_probability")
    with pytest.raises(
        UpstreamIntakeError,
        match="census_sample_analysis_weight_must_be_unset",
    ):
        load_census_sample_v2_release(root)


def test_sampler_v2_intake_verifies_artifact_hashes(tmp_path: Path) -> None:
    root = _write_census_fixture(tmp_path)
    (root / "selection.parquet").write_bytes(b"mutated")
    with pytest.raises(
        UpstreamIntakeError,
        match="census_sample_artifact_hash_mismatch:selection.parquet",
    ):
        load_census_sample_v2_release(root)
