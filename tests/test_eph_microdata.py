from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from encuestador.eph_microdata import (
    load_eph_microdata_release,
    read_eph_person_observation_frame,
)
from encuestador.upstream_intake import UpstreamIntakeError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_table(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "eph-2024-q3-testrelease"
    individual = root / "individual" / "usu_individual_t324.txt"
    household = root / "household" / "usu_hogar_t324.txt"
    individual_rows = [
        {
            "CODUSU": "A",
            "NRO_HOGAR": "1",
            "COMPONENTE": "1",
            "ANO4": "2024",
            "TRIMESTRE": "3",
            "CH04": "1",
            "CH06": "40",
            "CAT_OCUP": "1",
            "P47T": "100000",
        },
        {
            "CODUSU": "A",
            "NRO_HOGAR": "1",
            "COMPONENTE": "2",
            "ANO4": "2024",
            "TRIMESTRE": "3",
            "CH04": "2",
            "CH06": "38",
            "CAT_OCUP": "3",
            "P47T": "-9",
        },
        {
            "CODUSU": "B",
            "NRO_HOGAR": "1",
            "COMPONENTE": "1",
            "ANO4": "2024",
            "TRIMESTRE": "3",
            "CH04": "1",
            "CH06": "21",
            "CAT_OCUP": "2",
            "P47T": "0",
        },
    ]
    household_rows = [
        {"CODUSU": "A", "NRO_HOGAR": "1", "IX_TOT": "2", "IV1": "1"},
        {"CODUSU": "B", "NRO_HOGAR": "1", "IX_TOT": "1", "IV1": "2"},
    ]
    _write_table(
        individual,
        ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "CH04", "CH06", "CAT_OCUP", "P47T"],
        individual_rows,
    )
    _write_table(household, ["CODUSU", "NRO_HOGAR", "IX_TOT", "IV1"], household_rows)
    files = []
    for role, path, rows in (
        ("individual", individual, len(individual_rows)),
        ("household", household, len(household_rows)),
    ):
        with path.open("r", encoding="utf-8", newline="") as stream:
            columns = len(next(csv.reader(stream, delimiter=";")))
        files.append(
            {
                "role": role,
                "period": "2024-Q3",
                "schema_hash": "0" * 64,
                "original_name": path.name,
                "normalized_name": path.name,
                "file": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "delimiter": ";",
                "encoding": "utf-8",
                "rows": rows,
                "columns": columns,
            }
        )
    manifest = {
        "schema_version": 1,
        "release_id": root.name,
        "extraction_contract_version": "eph-zip-v2",
        "source_archive_sha256": "1" * 64,
        "requested_year": 2024,
        "requested_quarter": "Q3",
        "source_manifest_sha256": "2" * 64,
        "files": files,
        "warnings": [],
        "producing_command": "fixture",
        "software_version": "test",
    }
    (root / "output-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_exact_microdata_release_builds_person_frame_by_mechanical_household_join(tmp_path: Path) -> None:
    root = _release(tmp_path)
    release = load_eph_microdata_release(
        root, expected_release_id="eph-2024-q3-testrelease"
    )
    rows = read_eph_person_observation_frame(
        release,
        individual_columns=("CH04", "CH06", "CAT_OCUP", "P47T"),
        household_columns=("IX_TOT", "IV1"),
    )
    assert release.period == "2024-Q3"
    assert len(rows) == 3
    assert rows[0]["IX_TOT"] == "2"
    assert rows[1]["P47T"] == "-9"
    assert rows[2]["P47T"] == "0"
    assert {tuple(row[name] for name in ("CODUSU", "NRO_HOGAR", "COMPONENTE")) for row in rows} == {
        ("A", "1", "1"),
        ("A", "1", "2"),
        ("B", "1", "1"),
    }


def test_microdata_release_rejects_hash_drift(tmp_path: Path) -> None:
    root = _release(tmp_path)
    individual = root / "individual" / "usu_individual_t324.txt"
    individual.write_text(individual.read_text(encoding="utf-8") + "corruption\n", encoding="utf-8")
    with pytest.raises(UpstreamIntakeError, match="eph_microdata_file_hash_mismatch:individual"):
        load_eph_microdata_release(root)


def test_microdata_join_rejects_missing_household_record(tmp_path: Path) -> None:
    root = _release(tmp_path)
    household = root / "household" / "usu_hogar_t324.txt"
    lines = household.read_text(encoding="utf-8").splitlines()
    household.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    manifest_path = root / "output-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(value for value in manifest["files"] if value["role"] == "household")
    record["sha256"] = _sha(household)
    record["rows"] = 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release = load_eph_microdata_release(root)
    with pytest.raises(UpstreamIntakeError, match="eph_microdata_person_households_missing:1"):
        read_eph_person_observation_frame(
            release,
            individual_columns=("P47T",),
            household_columns=("IX_TOT",),
        )
