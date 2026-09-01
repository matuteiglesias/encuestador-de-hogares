import csv
import hashlib
import json
from pathlib import Path

import pytest

from encuestador.census_sample_intake import (
    CensusSampleIntakeError,
    validate_census_sample_release,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _release(tmp_path: Path, *, target_year: int = 2024) -> Path:
    root = tmp_path / "sample"
    root.mkdir()
    household_fields = [
        "sample_household_id",
        "department_2010_id",
        "selection_probability",
        "design_inverse_probability_weight",
    ]
    person_fields = [
        "sample_person_id",
        "sample_household_id",
        "department_2010_id",
        "radio_2010_id",
        "sex_code",
        "age_years",
        "selection_probability",
        "design_inverse_probability_weight",
    ]
    households = [
        {
            "sample_household_id": "cpv2010:household:H1",
            "department_2010_id": "06028",
            "selection_probability": 0.01,
            "design_inverse_probability_weight": 100.0,
        },
        {
            "sample_household_id": "cpv2010:household:H2",
            "department_2010_id": "06035",
            "selection_probability": 0.02,
            "design_inverse_probability_weight": 50.0,
        },
    ]
    persons = [
        {
            "sample_person_id": "cpv2010:person:P1",
            "sample_household_id": "cpv2010:household:H1",
            "department_2010_id": "06028",
            "radio_2010_id": "R1",
            "sex_code": 1,
            "age_years": 44,
            "selection_probability": 0.01,
            "design_inverse_probability_weight": 100.0,
        },
        {
            "sample_person_id": "cpv2010:person:P2",
            "sample_household_id": "cpv2010:household:H1",
            "department_2010_id": "06028",
            "radio_2010_id": "R1",
            "sex_code": 2,
            "age_years": 40,
            "selection_probability": 0.01,
            "design_inverse_probability_weight": 100.0,
        },
        {
            "sample_person_id": "cpv2010:person:P3",
            "sample_household_id": "cpv2010:household:H2",
            "department_2010_id": "06035",
            "radio_2010_id": "R2",
            "sex_code": 1,
            "age_years": 9,
            "selection_probability": 0.02,
            "design_inverse_probability_weight": 50.0,
        },
    ]
    _write_csv(root / "households.csv", household_fields, households)
    _write_csv(root / "persons.csv", person_fields, persons)
    qa = {
        "selection_unit": "household",
        "target_mass_unit": "person",
        "frame_vintage": 2010,
        "sampling_target_year": target_year,
        "selected_counts": {"households": 2, "persons": 3},
        "complete_household_membership": True,
    }
    (root / "qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "contract": "research.census-target-year-sample/v1",
        "release_id": f"fixture-census-target-{target_year}",
        "frame": {"vintage": 2010},
        "target_population_parent": {
            "source_id": "fixture-population-parent",
            "target_year": target_year,
            "sha256": "fixture-target-sha",
        },
        "selection": {
            "unit": "household",
            "target_mass_unit": "person",
            "common_score_across_target_years": True,
        },
        "weight_semantics": {
            "selection_probability": "household inclusion probability",
            "design_inverse_probability_weight": "1 / selection_probability",
            "analysis_weight": None,
            "generic_sample_weight": None,
        },
        "artifacts": {
            filename: {
                "sha256": _sha256(root / filename),
                "size_bytes": (root / filename).stat().st_size,
            }
            for filename in ("households.csv", "persons.csv", "qa.json")
        },
        "qa": qa,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _refresh_person_hash(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["persons.csv"]["sha256"] = _sha256(root / "persons.csv")
    manifest["artifacts"]["persons.csv"]["size_bytes"] = (root / "persons.csv").stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_valid_sampler_release_is_accepted_for_audit_not_scoring(tmp_path: Path) -> None:
    root = _release(tmp_path)
    report = validate_census_sample_release(root)

    assert report["status"] == "accepted_for_audit_semantic_alignment_required"
    assert report["sampling_target_year"] == 2024
    assert report["households"] == 2
    assert report["persons"] == 3
    assert report["household_membership_complete"] is True
    assert report["design_audit_fields"] == [
        "design_inverse_probability_weight",
        "selection_probability",
    ]
    assert report["model_forbidden_design_fields"] == report["design_audit_fields"]
    assert report["generic_weight_fields_present"] == []
    assert report["model_scoring_authorized"] is False
    assert {"sex_code", "age_years", "department_2010_id", "radio_2010_id"}.issubset(
        report["donor_observation_columns"]
    )


def test_generic_sample_weight_column_fails_closed(tmp_path: Path) -> None:
    root = _release(tmp_path)
    rows = list(csv.DictReader((root / "persons.csv").open(encoding="utf-8")))
    fields = list(rows[0]) + ["sample_weight"]
    for row in rows:
        row["sample_weight"] = "1"
    _write_csv(root / "persons.csv", fields, rows)
    _refresh_person_hash(root)

    with pytest.raises(CensusSampleIntakeError, match="forbidden_generic_weight_fields"):
        validate_census_sample_release(root)


def test_person_probability_must_match_selected_household(tmp_path: Path) -> None:
    root = _release(tmp_path)
    rows = list(csv.DictReader((root / "persons.csv").open(encoding="utf-8")))
    rows[0]["selection_probability"] = "0.02"
    rows[0]["design_inverse_probability_weight"] = "50"
    _write_csv(root / "persons.csv", list(rows[0]), rows)
    _refresh_person_hash(root)

    with pytest.raises(
        CensusSampleIntakeError,
        match="census_person_household_probability_mismatch",
    ):
        validate_census_sample_release(root)


def test_payload_hash_drift_fails_before_semantic_use(tmp_path: Path) -> None:
    root = _release(tmp_path)
    with (root / "persons.csv").open("a", encoding="utf-8") as stream:
        stream.write("corruption\n")

    with pytest.raises(CensusSampleIntakeError, match="sample_payload_hash_mismatch:persons.csv"):
        validate_census_sample_release(root)


def test_only_governed_2024_2025_target_years_are_accepted(tmp_path: Path) -> None:
    root = _release(tmp_path, target_year=2026)

    with pytest.raises(CensusSampleIntakeError, match="census_sampling_target_year_not_approved"):
        validate_census_sample_release(root)
