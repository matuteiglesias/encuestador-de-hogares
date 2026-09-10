"""Thin manifest-driven adapter for exact `microdatos-EPH-INDEC` releases.

This adapter performs only source-faithful CSV reading and the mechanical
individual LEFT JOIN household on CODUSU + NRO_HOGAR. It does not own income
harmonization, semantic EPH/Census alignment, monetary transformations, weights,
or model feature choices.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .upstream_intake import UpstreamIntakeError, sha256_file

EPH_EXTRACTION_CONTRACT = "eph-zip-v2"
HOUSEHOLD_KEY = ("CODUSU", "NRO_HOGAR")
PERSON_KEY = ("CODUSU", "NRO_HOGAR", "COMPONENTE")


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise UpstreamIntakeError("eph_microdata_artifact_path_escapes_root") from exc
    return path


@dataclass(frozen=True)
class EPHMicrodataTable:
    role: str
    path: Path
    rows: int
    columns: int
    sha256: str
    delimiter: str
    encoding: str


@dataclass(frozen=True)
class EPHMicrodataRelease:
    release_id: str
    root: Path
    year: int
    quarter: str
    period: str
    source_archive_sha256: str
    source_manifest_sha256: str
    individual: EPHMicrodataTable
    household: EPHMicrodataTable
    warnings: tuple[str, ...]


def _read_header(table: EPHMicrodataTable) -> tuple[str, ...]:
    try:
        with table.path.open("r", encoding=table.encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=table.delimiter)
            return tuple(next(reader))
    except (OSError, UnicodeDecodeError, StopIteration) as exc:
        raise UpstreamIntakeError(f"eph_microdata_header_invalid:{table.role}") from exc


def load_eph_microdata_release(
    root: Path,
    *,
    expected_release_id: str | None = None,
    verify_hashes: bool = True,
) -> EPHMicrodataRelease:
    """Validate one immutable source-faithful producer release directory."""
    root = Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "output-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamIntakeError("eph_microdata_output_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise UpstreamIntakeError("eph_microdata_output_manifest_invalid")
    release_id = str(manifest.get("release_id") or "")
    if not release_id or release_id != root.name:
        raise UpstreamIntakeError("eph_microdata_release_directory_mismatch")
    if expected_release_id is not None and release_id != expected_release_id:
        raise UpstreamIntakeError(
            f"eph_microdata_release_id_mismatch:{release_id}!={expected_release_id}"
        )
    if manifest.get("extraction_contract_version") != EPH_EXTRACTION_CONTRACT:
        raise UpstreamIntakeError("unexpected_eph_microdata_extraction_contract")
    year = manifest.get("requested_year")
    quarter = str(manifest.get("requested_quarter") or "").upper()
    if not isinstance(year, int) or quarter not in {"Q1", "Q2", "Q3", "Q4"}:
        raise UpstreamIntakeError("eph_microdata_period_invalid")

    records = manifest.get("files")
    if not isinstance(records, list):
        raise UpstreamIntakeError("eph_microdata_file_inventory_missing")
    tables: dict[str, EPHMicrodataTable] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        role = str(record.get("role") or "")
        if role not in {"individual", "household"}:
            continue
        if role in tables:
            raise UpstreamIntakeError(f"duplicate_eph_microdata_role:{role}")
        relative = record.get("file")
        expected_hash = record.get("sha256")
        rows = record.get("rows")
        columns = record.get("columns")
        delimiter = str(record.get("delimiter") or "")
        encoding = str(record.get("encoding") or "")
        if not isinstance(relative, str) or not relative:
            raise UpstreamIntakeError(f"eph_microdata_file_path_missing:{role}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise UpstreamIntakeError(f"eph_microdata_file_hash_missing:{role}")
        if not isinstance(rows, int) or rows <= 0 or not isinstance(columns, int) or columns <= 0:
            raise UpstreamIntakeError(f"eph_microdata_file_shape_invalid:{role}")
        if delimiter != ";" or encoding not in {"utf-8", "latin-1"}:
            raise UpstreamIntakeError(f"eph_microdata_text_contract_invalid:{role}")
        path = _safe_path(root, relative)
        if not path.is_file():
            raise UpstreamIntakeError(f"eph_microdata_file_missing:{role}")
        if verify_hashes and sha256_file(path) != expected_hash:
            raise UpstreamIntakeError(f"eph_microdata_file_hash_mismatch:{role}")
        tables[role] = EPHMicrodataTable(
            role=role,
            path=path,
            rows=rows,
            columns=columns,
            sha256=expected_hash,
            delimiter=delimiter,
            encoding=encoding,
        )
    if set(tables) != {"individual", "household"}:
        raise UpstreamIntakeError("eph_microdata_individual_household_tables_required")

    individual_header = _read_header(tables["individual"])
    household_header = _read_header(tables["household"])
    missing_person = sorted(set(PERSON_KEY) - set(individual_header))
    missing_household = sorted(set(HOUSEHOLD_KEY) - set(household_header))
    if missing_person:
        raise UpstreamIntakeError(
            f"eph_microdata_person_identity_missing:{','.join(missing_person)}"
        )
    if missing_household:
        raise UpstreamIntakeError(
            f"eph_microdata_household_identity_missing:{','.join(missing_household)}"
        )

    return EPHMicrodataRelease(
        release_id=release_id,
        root=root,
        year=year,
        quarter=quarter,
        period=f"{year}-{quarter}",
        source_archive_sha256=str(manifest.get("source_archive_sha256") or ""),
        source_manifest_sha256=str(manifest.get("source_manifest_sha256") or ""),
        individual=tables["individual"],
        household=tables["household"],
        warnings=tuple(str(value) for value in manifest.get("warnings", [])),
    )


def _read_selected(
    table: EPHMicrodataTable,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    requested = tuple(dict.fromkeys(str(value) for value in columns))
    if not requested:
        raise UpstreamIntakeError(f"eph_microdata_read_columns_required:{table.role}")
    output: list[dict[str, str]] = []
    try:
        with table.path.open("r", encoding=table.encoding, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=table.delimiter)
            fieldnames = tuple(reader.fieldnames or ())
            missing = sorted(set(requested) - set(fieldnames))
            if missing:
                raise UpstreamIntakeError(
                    f"eph_microdata_columns_missing:{table.role}:{','.join(missing)}"
                )
            for row in reader:
                output.append({name: row[name] for name in requested})
    except UnicodeDecodeError as exc:
        raise UpstreamIntakeError(f"eph_microdata_decode_error:{table.role}") from exc
    if len(output) != table.rows:
        raise UpstreamIntakeError(
            f"eph_microdata_row_count_mismatch:{table.role}:{len(output)}!={table.rows}"
        )
    return output


def read_eph_person_observation_frame(
    release: EPHMicrodataRelease,
    *,
    individual_columns: Sequence[str],
    household_columns: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Mechanically join exact source rows without inventing analytical transforms."""
    individual_requested = tuple(
        dict.fromkeys((*PERSON_KEY, "ANO4", "TRIMESTRE", *individual_columns))
    )
    household_requested = tuple(dict.fromkeys((*HOUSEHOLD_KEY, *household_columns)))
    collision = sorted(
        (set(individual_requested) & set(household_requested)) - set(HOUSEHOLD_KEY)
    )
    if collision:
        raise UpstreamIntakeError(
            f"eph_microdata_join_column_collision:{','.join(collision)}"
        )
    households = _read_selected(release.household, household_requested)
    household_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in households:
        key = (row["CODUSU"], row["NRO_HOGAR"])
        if key in household_by_key:
            raise UpstreamIntakeError("duplicate_eph_microdata_household_key")
        household_by_key[key] = row

    persons = _read_selected(release.individual, individual_requested)
    seen_persons: set[tuple[str, str, str]] = set()
    output: list[dict[str, str]] = []
    missing_household = 0
    for person in persons:
        person_key = (
            person["CODUSU"],
            person["NRO_HOGAR"],
            person["COMPONENTE"],
        )
        if person_key in seen_persons:
            raise UpstreamIntakeError("duplicate_eph_microdata_person_key")
        seen_persons.add(person_key)
        hh_key = (person["CODUSU"], person["NRO_HOGAR"])
        household = household_by_key.get(hh_key)
        if household is None:
            missing_household += 1
            continue
        merged = dict(person)
        for name in household_columns:
            merged[str(name)] = household[str(name)]
        output.append(merged)
    if missing_household:
        raise UpstreamIntakeError(
            f"eph_microdata_person_households_missing:{missing_household}"
        )
    return output
