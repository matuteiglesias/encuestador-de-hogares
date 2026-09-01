"""Validate samplerCensoARG target-year releases at the surveyor boundary.

This module owns a deliberately narrow consumer gate. It validates sampler
identity, payload hashes, household membership and weight semantics, then stops
before semantic feature alignment. Passing this intake does not authorize model
scoring and does not rename Census fields into historical EPH-shaped features.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SAMPLE_CONTRACT = "research.census-target-year-sample/v1"
TARGET_YEARS = {2024, 2025}
HOUSEHOLD_REQUIRED = {
    "sample_household_id",
    "department_2010_id",
    "selection_probability",
    "design_inverse_probability_weight",
}
PERSON_REQUIRED = {
    "sample_person_id",
    "sample_household_id",
    "department_2010_id",
    "radio_2010_id",
    "selection_probability",
    "design_inverse_probability_weight",
}
DESIGN_AUDIT_FIELDS = {
    "selection_probability",
    "design_inverse_probability_weight",
}
FORBIDDEN_GENERIC_WEIGHT_FIELDS = {
    "weight",
    "sample_weight",
    "analysis_weight",
}
IDENTITY_FIELDS = {
    "sample_person_id",
    "sample_household_id",
}


class CensusSampleIntakeError(ValueError):
    """Raised when a target-year Census sample cannot cross the intake gate."""


def _load_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CensusSampleIntakeError(reason) from exc
    if not isinstance(value, dict):
        raise CensusSampleIntakeError(reason)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [
                {key: (value or "").strip() for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    except OSError as exc:
        raise CensusSampleIntakeError(f"sample_payload_unreadable:{path.name}") from exc
    if not rows:
        raise CensusSampleIntakeError(f"sample_payload_empty:{path.name}")
    return rows


def _artifact_hash(manifest: dict[str, Any], filename: str) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CensusSampleIntakeError("sample_manifest_artifacts_missing")
    record = artifacts.get(filename)
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise CensusSampleIntakeError(f"sample_manifest_hash_missing:{filename}")
    return record["sha256"]


def _require_columns(
    rows: list[dict[str, str]], required: set[str], table: str
) -> set[str]:
    columns = set(rows[0])
    missing = sorted(required - columns)
    if missing:
        raise CensusSampleIntakeError(
            f"{table}:missing_required_columns:{','.join(missing)}"
        )
    forbidden = sorted(columns & FORBIDDEN_GENERIC_WEIGHT_FIELDS)
    if forbidden:
        raise CensusSampleIntakeError(
            f"{table}:forbidden_generic_weight_fields:{','.join(forbidden)}"
        )
    return columns


def _assert_unique(rows: list[dict[str, str]], key: str, table: str) -> None:
    counts = Counter(row[key] for row in rows)
    invalid = sorted(value for value, count in counts.items() if not value or count != 1)
    if invalid:
        raise CensusSampleIntakeError(
            f"{table}:duplicate_or_empty_identity:{key}:{invalid[:5]}"
        )


def _probability(row: dict[str, str], identity: str) -> tuple[float, float]:
    try:
        probability = float(row["selection_probability"])
        inverse = float(row["design_inverse_probability_weight"])
    except (TypeError, ValueError) as exc:
        raise CensusSampleIntakeError(f"invalid_design_value:{identity}") from exc
    if not math.isfinite(probability) or not (0 < probability <= 1):
        raise CensusSampleIntakeError(f"invalid_selection_probability:{identity}")
    if not math.isfinite(inverse) or inverse <= 0:
        raise CensusSampleIntakeError(f"invalid_design_inverse_weight:{identity}")
    if not math.isclose(inverse, 1.0 / probability, rel_tol=1e-12, abs_tol=1e-12):
        raise CensusSampleIntakeError(f"design_inverse_probability_mismatch:{identity}")
    return probability, inverse


def validate_census_sample_release(root: Path) -> dict[str, Any]:
    """Validate one target-year sample without authorizing semantic scoring."""
    root = Path(root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json", "sample_manifest_missing_or_invalid")
    qa = _load_json(root / "qa.json", "sample_qa_missing_or_invalid")

    if manifest.get("contract") != SAMPLE_CONTRACT:
        raise CensusSampleIntakeError("unexpected_census_sample_contract")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise CensusSampleIntakeError("census_sample_release_id_missing")
    frame = manifest.get("frame") or {}
    if frame.get("vintage") != 2010:
        raise CensusSampleIntakeError("census_donor_vintage_must_be_2010")
    target_parent = manifest.get("target_population_parent") or {}
    target_year = target_parent.get("target_year")
    if target_year not in TARGET_YEARS:
        raise CensusSampleIntakeError("census_sampling_target_year_not_approved")

    selection = manifest.get("selection") or {}
    if selection.get("unit") != "household":
        raise CensusSampleIntakeError("census_sample_unit_must_be_household")
    if selection.get("target_mass_unit") != "person":
        raise CensusSampleIntakeError("census_target_mass_unit_must_be_person")
    if selection.get("common_score_across_target_years") is not True:
        raise CensusSampleIntakeError("census_common_household_score_required")

    semantics = manifest.get("weight_semantics") or {}
    if semantics.get("analysis_weight") is not None:
        raise CensusSampleIntakeError("census_analysis_weight_must_be_unset")
    if semantics.get("generic_sample_weight") is not None:
        raise CensusSampleIntakeError("census_generic_sample_weight_must_be_unset")
    if "selection_probability" not in semantics:
        raise CensusSampleIntakeError("selection_probability_semantics_missing")
    if "design_inverse_probability_weight" not in semantics:
        raise CensusSampleIntakeError("design_inverse_probability_semantics_missing")

    for filename in ("households.csv", "persons.csv", "qa.json"):
        path = root / filename
        if not path.is_file():
            raise CensusSampleIntakeError(f"sample_payload_missing:{filename}")
        if _sha256(path) != _artifact_hash(manifest, filename):
            raise CensusSampleIntakeError(f"sample_payload_hash_mismatch:{filename}")

    households = _read_csv(root / "households.csv")
    persons = _read_csv(root / "persons.csv")
    household_columns = _require_columns(households, HOUSEHOLD_REQUIRED, "households")
    person_columns = _require_columns(persons, PERSON_REQUIRED, "persons")
    _assert_unique(households, "sample_household_id", "households")
    _assert_unique(persons, "sample_person_id", "persons")

    household_design: dict[str, tuple[float, float, str]] = {}
    for row in households:
        household_id = row["sample_household_id"]
        probability, inverse = _probability(row, household_id)
        household_design[household_id] = (
            probability,
            inverse,
            row["department_2010_id"],
        )

    member_counts: Counter[str] = Counter()
    for row in persons:
        person_id = row["sample_person_id"]
        household_id = row["sample_household_id"]
        if household_id not in household_design:
            raise CensusSampleIntakeError(
                f"census_person_orphan_household:{person_id}:{household_id}"
            )
        probability, inverse = _probability(row, person_id)
        household_probability, household_inverse, household_department = household_design[
            household_id
        ]
        if row["department_2010_id"] != household_department:
            raise CensusSampleIntakeError(
                f"census_person_household_department_mismatch:{person_id}"
            )
        if not math.isclose(probability, household_probability, rel_tol=0, abs_tol=1e-15):
            raise CensusSampleIntakeError(
                f"census_person_household_probability_mismatch:{person_id}"
            )
        if not math.isclose(inverse, household_inverse, rel_tol=1e-12, abs_tol=1e-12):
            raise CensusSampleIntakeError(
                f"census_person_household_inverse_weight_mismatch:{person_id}"
            )
        member_counts[household_id] += 1

    empty_households = sorted(set(household_design) - set(member_counts))
    if empty_households:
        raise CensusSampleIntakeError(
            f"selected_households_without_members:{empty_households[:5]}"
        )
    if qa.get("complete_household_membership") is not True:
        raise CensusSampleIntakeError("sampler_did_not_assert_complete_household_membership")
    selected_counts = qa.get("selected_counts") or {}
    if selected_counts.get("households") != len(households):
        raise CensusSampleIntakeError("sampler_household_count_mismatch")
    if selected_counts.get("persons") != len(persons):
        raise CensusSampleIntakeError("sampler_person_count_mismatch")

    donor_observation_columns = sorted(
        person_columns - IDENTITY_FIELDS - DESIGN_AUDIT_FIELDS
    )
    return {
        "contract": "research.encuestador-census-sample-intake/v1",
        "status": "accepted_for_audit_semantic_alignment_required",
        "sample_release_id": release_id,
        "frame_vintage": 2010,
        "sampling_target_year": target_year,
        "selection_unit": "household",
        "target_mass_unit": "person",
        "households": len(households),
        "persons": len(persons),
        "household_membership_complete": True,
        "design_audit_fields": sorted(DESIGN_AUDIT_FIELDS),
        "model_forbidden_design_fields": sorted(DESIGN_AUDIT_FIELDS),
        "generic_weight_fields_present": [],
        "donor_observation_columns": donor_observation_columns,
        "household_columns": sorted(household_columns),
        "person_columns": sorted(person_columns),
        "model_scoring_authorized": False,
        "model_scoring_blocker": (
            "Exact eph-censo semantic-plane mapping and required feature availability "
            "must be approved separately."
        ),
        "claim_boundary": (
            "This gate validates sampler design/custody only; selection probabilities "
            "remain audit metadata and are not fitting, calibration, or evaluation weights."
        ),
    }
