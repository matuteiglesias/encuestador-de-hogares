"""Fail-closed intake for upstream EPH and Census producer releases.

This module validates producer contracts and lineage only. It does not reproduce
EPH preprocessing, Census sampling, semantic alignment, or analysis weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPH_MANIFEST_SCHEMA = "research-artifact-manifest/v1"
EPH_METHOD_CONTRACT = "research.eph-annual-preprocessed/v1"
EPH_ARTIFACT_TYPE = "research.eph-annual-preprocessed"
CENSUS_SAMPLE_CONTRACT = "research.census-target-year-sample/v2"
REQUIRED_EPH_IDENTITY = ("CODUSU", "NRO_HOGAR", "COMPONENTE")


class UpstreamIntakeError(ValueError):
    """Raised when upstream evidence is absent, stale, ambiguous, or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamIntakeError(error) from exc
    if not isinstance(value, dict):
        raise UpstreamIntakeError(error)
    return value


def _safe_relative(root: Path, relative: str, error: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise UpstreamIntakeError(error) from exc
    return candidate


@dataclass(frozen=True)
class EPHAnnualRelease:
    release_id: str
    manifest_path: Path
    data_path: Path
    data_vintage: int
    rows: int
    columns: tuple[str, ...]
    artifact_sha256: str
    upstream_inputs: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]

    def require_columns(self, columns: Sequence[str]) -> None:
        missing = sorted(set(columns) - set(self.columns))
        if missing:
            raise UpstreamIntakeError(
                f"eph_release_missing_required_columns:{','.join(missing)}"
            )

    def require_household_safe_identity(self) -> None:
        self.require_columns(REQUIRED_EPH_IDENTITY)


def load_eph_annual_release(
    manifest_path: Path,
    *,
    artifact_root: Path,
    required_columns: Sequence[str] = (),
    verify_hash: bool = True,
) -> EPHAnnualRelease:
    """Validate one `research.eph-annual-preprocessed/v1` release manifest."""
    manifest_path = Path(manifest_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    manifest = _json(manifest_path, "eph_manifest_invalid")

    if manifest.get("manifest_schema") != EPH_MANIFEST_SCHEMA:
        raise UpstreamIntakeError("unexpected_eph_manifest_schema")
    if manifest.get("method_contract_version") != EPH_METHOD_CONTRACT:
        raise UpstreamIntakeError("unexpected_eph_method_contract")
    if manifest.get("artifact_type") != EPH_ARTIFACT_TYPE:
        raise UpstreamIntakeError("unexpected_eph_artifact_type")
    release_id = str(manifest.get("release_id") or "")
    if not release_id.startswith("artifact:research.eph-annual-preprocessed@1+"):
        raise UpstreamIntakeError("unexpected_eph_release_id")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("role") != "annual_preprocessed_input":
        raise UpstreamIntakeError("eph_artifact_record_invalid")
    relative_path = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise UpstreamIntakeError("eph_artifact_path_missing")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise UpstreamIntakeError("eph_artifact_hash_missing")
    data_path = _safe_relative(
        artifact_root,
        relative_path,
        "eph_artifact_path_escapes_root",
    )
    if not data_path.is_file():
        raise UpstreamIntakeError("eph_artifact_missing")
    if verify_hash and sha256_file(data_path) != expected_hash:
        raise UpstreamIntakeError("eph_artifact_hash_mismatch")

    inventory = manifest.get("column_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise UpstreamIntakeError("eph_column_inventory_missing")
    columns = tuple(str(column) for column in inventory)
    if len(set(columns)) != len(columns):
        raise UpstreamIntakeError("eph_column_inventory_duplicate")
    rows = manifest.get("rows")
    vintage = manifest.get("data_vintage")
    if not isinstance(rows, int) or rows <= 0:
        raise UpstreamIntakeError("eph_row_count_invalid")
    if not isinstance(vintage, int):
        raise UpstreamIntakeError("eph_data_vintage_invalid")

    coverage = manifest.get("coverage") or {}
    years = coverage.get("years") if isinstance(coverage, dict) else None
    if years != [vintage]:
        raise UpstreamIntakeError("eph_coverage_vintage_mismatch")

    release = EPHAnnualRelease(
        release_id=release_id,
        manifest_path=manifest_path,
        data_path=data_path,
        data_vintage=vintage,
        rows=rows,
        columns=columns,
        artifact_sha256=expected_hash,
        upstream_inputs=tuple(
            dict(value)
            for value in manifest.get("inputs", [])
            if isinstance(value, dict)
        ),
        limitations=tuple(str(value) for value in manifest.get("limitations", [])),
    )
    release.require_household_safe_identity()
    release.require_columns(required_columns)
    return release


def read_eph_rows(
    release: EPHAnnualRelease,
    *,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    """Read only declared columns from one validated annual CSV artifact."""
    requested = tuple(str(column) for column in columns)
    if not requested:
        raise UpstreamIntakeError("eph_read_columns_required")
    release.require_columns(requested)
    output: list[dict[str, str]] = []
    try:
        with release.data_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = sorted(set(requested) - set(fieldnames))
            if missing:
                raise UpstreamIntakeError(
                    f"eph_csv_missing_manifest_columns:{','.join(missing)}"
                )
            for row in reader:
                output.append({column: row[column] for column in requested})
    except UnicodeDecodeError as exc:
        raise UpstreamIntakeError("eph_csv_not_utf8") from exc
    if len(output) != release.rows:
        raise UpstreamIntakeError(
            f"eph_csv_row_count_mismatch:{len(output)}!={release.rows}"
        )
    return output


@dataclass(frozen=True)
class CensusSampleV2Release:
    release_id: str
    root: Path
    frame_release_id: str
    frame_vintage: str
    target_year: int
    materialization: str
    selected_households: int
    selected_persons: int
    artifacts: dict[str, Path]

    @property
    def selection_path(self) -> Path:
        return self.artifacts["selection.parquet"]

    @property
    def person_membership_path(self) -> Path:
        return self.artifacts["person_membership.parquet"]


def load_census_sample_v2_release(
    root: Path,
    *,
    verify_hashes: bool = True,
) -> CensusSampleV2Release:
    """Validate sampler-v2 manifest, lineage, weight semantics and artifact hashes."""
    root = Path(root).resolve()
    manifest = _json(root / "manifest.json", "census_sample_v2_manifest_invalid")
    qa = _json(root / "qa.json", "census_sample_v2_qa_invalid")
    if manifest.get("contract") != CENSUS_SAMPLE_CONTRACT:
        raise UpstreamIntakeError("unexpected_census_sample_contract")
    release_id = str(manifest.get("release_id") or "")
    if not release_id or root.name != release_id:
        raise UpstreamIntakeError("census_sample_release_directory_mismatch")

    materialization = str(manifest.get("materialization") or "")
    if materialization not in {"selection-only", "full-payload"}:
        raise UpstreamIntakeError("census_sample_materialization_invalid")
    selection = manifest.get("selection") or {}
    if not isinstance(selection, dict) or selection.get("unit") != "household":
        raise UpstreamIntakeError("census_sample_selection_unit_changed")
    if selection.get("target_mass_unit") != "person":
        raise UpstreamIntakeError("census_sample_target_mass_unit_changed")

    semantics = manifest.get("weight_semantics") or {}
    if not isinstance(semantics, dict):
        raise UpstreamIntakeError("census_sample_weight_semantics_missing")
    if semantics.get("analysis_weight") is not None:
        raise UpstreamIntakeError("census_sample_analysis_weight_must_be_unset")
    if semantics.get("generic_sample_weight") is not None:
        raise UpstreamIntakeError("census_sample_generic_weight_must_be_unset")

    if qa.get("complete_household_membership") is not True:
        raise UpstreamIntakeError("census_sample_membership_not_complete")
    counts = qa.get("selected_counts") or {}
    if not isinstance(counts, dict):
        raise UpstreamIntakeError("census_sample_selected_counts_missing")
    households = counts.get("households")
    persons = counts.get("persons")
    if not isinstance(households, int) or households <= 0:
        raise UpstreamIntakeError("census_sample_household_count_invalid")
    if not isinstance(persons, int) or persons < households:
        raise UpstreamIntakeError("census_sample_person_count_invalid")

    frame = manifest.get("frame") or {}
    target_parent = manifest.get("target_population_parent") or {}
    if not isinstance(frame, dict) or not frame.get("frame_release_id"):
        raise UpstreamIntakeError("census_sample_frame_identity_missing")
    target_year = target_parent.get("target_year") if isinstance(target_parent, dict) else None
    if not isinstance(target_year, int):
        raise UpstreamIntakeError("census_sample_target_year_missing")

    records = manifest.get("artifacts")
    if not isinstance(records, dict):
        raise UpstreamIntakeError("census_sample_artifact_manifest_missing")
    required = {"selection.parquet", "person_membership.parquet", "qa.json"}
    if materialization == "full-payload":
        required |= {"vivienda.parquet", "hogar.parquet", "persona.parquet"}
    missing = sorted(required - set(records))
    if missing:
        raise UpstreamIntakeError(
            f"census_sample_required_artifact_missing:{','.join(missing)}"
        )

    artifacts: dict[str, Path] = {}
    for name, record in records.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise UpstreamIntakeError("census_sample_artifact_name_unsafe")
        if not isinstance(record, dict):
            raise UpstreamIntakeError(f"census_sample_artifact_record_invalid:{name}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise UpstreamIntakeError(f"census_sample_artifact_hash_missing:{name}")
        path = root / name
        if not path.is_file():
            raise UpstreamIntakeError(f"census_sample_artifact_file_missing:{name}")
        if verify_hashes and sha256_file(path) != expected_hash:
            raise UpstreamIntakeError(f"census_sample_artifact_hash_mismatch:{name}")
        artifacts[name] = path

    return CensusSampleV2Release(
        release_id=release_id,
        root=root,
        frame_release_id=str(frame["frame_release_id"]),
        frame_vintage=str(frame.get("census_vintage") or ""),
        target_year=target_year,
        materialization=materialization,
        selected_households=households,
        selected_persons=persons,
        artifacts=artifacts,
    )
