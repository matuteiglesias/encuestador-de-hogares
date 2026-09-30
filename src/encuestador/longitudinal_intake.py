"""Fail-closed consumers for longitudinal EPH, official labor context, and donor labor."""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LONGITUDINAL_EPH_CONTRACT = "research.eph-longitudinal-analysis-frame/v1"
LABOR_CONTEXT_CONTRACT = "publicdata.indec-eph-labor-state/v1"
SEMANTIC_PLANE_CONTRACT = "research.eph-census-semantic-feature-plane/v1"
COMPOSITION_PLANE_CONTRACT = "research.eph-longitudinal-composition-plane/v1"
FIXTURE_COMPOSITION_CONTRACT = "fixture.raw-c2-composition-inline/v1"
FIXTURE_COMPOSITION_PROFILE_ID = "RAW_C2_8VAR_TEST_FIXTURE_V1"


class LongitudinalIntakeError(ValueError):
    """Raised when a parent artifact cannot be consumed without weakening lineage."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LongitudinalIntakeError(reason) from exc
    if not isinstance(value, dict):
        raise LongitudinalIntakeError(reason)
    return value


def _iter_csv(path: Path):
    """Yield normalized CSV rows without materializing the whole artifact."""
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise LongitudinalIntakeError(f"csv_header_missing:{path.name}")
            seen = False
            for row in reader:
                seen = True
                yield {key: (value or "").strip() for key, value in row.items()}
            if not seen:
                raise LongitudinalIntakeError(f"csv_empty:{path.name}")
    except OSError as exc:
        raise LongitudinalIntakeError(f"csv_unreadable:{path.name}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(_iter_csv(path))


def _csv_header_and_count(path: Path) -> tuple[tuple[str, ...], int]:
    """Validate one CSV structurally while keeping peak memory independent of rows."""
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = tuple(reader.fieldnames or ())
            if not header:
                raise LongitudinalIntakeError(f"csv_header_missing:{path.name}")
            count = sum(1 for _ in reader)
    except OSError as exc:
        raise LongitudinalIntakeError(f"csv_unreadable:{path.name}") from exc
    if count == 0:
        raise LongitudinalIntakeError(f"csv_empty:{path.name}")
    return header, count


def _verify_artifact(
    root: Path,
    record: dict[str, Any],
    filename: str,
    *,
    hash_key: str = "sha256",
) -> Path:
    path = (Path(root) / filename).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise LongitudinalIntakeError(f"artifact_path_escapes_root:{filename}") from exc
    expected = record.get(hash_key)
    if not path.is_file():
        raise LongitudinalIntakeError(f"artifact_missing:{filename}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise LongitudinalIntakeError(f"artifact_hash_missing:{filename}")
    if sha256_file(path) != expected:
        raise LongitudinalIntakeError(f"artifact_hash_mismatch:{filename}")
    return path


@dataclass(frozen=True)
class LongitudinalEPHRelease:
    root: Path
    release_id: str
    persons_path: Path
    coverage_path: Path
    manifest: dict[str, Any]
    monetary_release_id: str
    monetary_reference_period: str

    def iter_persons(self):
        return _iter_csv(self.persons_path)

    def read_persons(self) -> list[dict[str, str]]:
        """Compatibility helper for bounded fixtures; real C6 execution streams."""
        return list(self.iter_persons())


def load_longitudinal_eph_release(root: Path) -> LongitudinalEPHRelease:
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "manifest.json", "longitudinal_eph_manifest_invalid")
    if manifest.get("contract") != LONGITUDINAL_EPH_CONTRACT:
        raise LongitudinalIntakeError("unexpected_longitudinal_eph_contract")
    release_id = str(manifest.get("release_id") or "")
    if not release_id or root.name != release_id:
        raise LongitudinalIntakeError("longitudinal_eph_release_directory_mismatch")

    coverage = manifest.get("coverage") or {}
    if (
        coverage.get("period_start") != "2017-Q1"
        or coverage.get("period_end") != "2026-Q1"
        or coverage.get("period_count") != 37
        or coverage.get("all_expected_periods_present") is not True
    ):
        raise LongitudinalIntakeError("longitudinal_eph_coverage_contract_invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LongitudinalIntakeError("longitudinal_eph_artifacts_missing")
    for filename in ("persons.csv", "coverage.csv", "monetary_lineage.csv", "qa.json"):
        if not isinstance(artifacts.get(filename), dict):
            raise LongitudinalIntakeError(
                f"longitudinal_eph_artifact_record_missing:{filename}"
            )
    persons_path = _verify_artifact(root, artifacts["persons.csv"], "persons.csv")
    coverage_path = _verify_artifact(root, artifacts["coverage.csv"], "coverage.csv")
    _verify_artifact(root, artifacts["monetary_lineage.csv"], "monetary_lineage.csv")
    _verify_artifact(root, artifacts["qa.json"], "qa.json")

    parents = manifest.get("parents") or {}
    monetary = parents.get("monetary_conversion") or {}
    release = str(monetary.get("release_id") or "")
    lineage = manifest.get("monetary_lineage") or {}
    reference = str(lineage.get("common_reference_period") or "")
    if not release or not reference:
        raise LongitudinalIntakeError("longitudinal_eph_monetary_parent_missing")
    if lineage.get("source_field") != "P47T" or lineage.get("real_output") != "P47T_real":
        raise LongitudinalIntakeError("longitudinal_eph_monetary_semantics_changed")
    if manifest.get("field_policy", {}).get("zero_income_rows_retained") is not True:
        raise LongitudinalIntakeError("longitudinal_eph_zero_income_contract_changed")

    return LongitudinalEPHRelease(
        root=root,
        release_id=release_id,
        persons_path=persons_path,
        coverage_path=coverage_path,
        manifest=manifest,
        monetary_release_id=release,
        monetary_reference_period=reference,
    )


@dataclass(frozen=True)
class CompositionPlaneProfile:
    root: Path
    release_id: str
    profile_id: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    rows_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    profile_sha256: str

    def iter_rows(self):
        return _iter_csv(self.rows_path)

    def read_rows(self) -> list[dict[str, str]]:
        """Compatibility helper for bounded fixtures; real C6 execution streams."""
        return list(self.iter_rows())


def load_composition_plane_profile(
    root: Path,
    profile_id: str,
) -> CompositionPlaneProfile:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path, "composition_plane_manifest_invalid")
    if manifest.get("contract") != COMPOSITION_PLANE_CONTRACT:
        raise LongitudinalIntakeError("unexpected_composition_plane_contract")
    release_id = str(manifest.get("release_id") or "")
    if not release_id or root.name != release_id:
        raise LongitudinalIntakeError("composition_plane_release_directory_mismatch")
    if manifest.get("row_id_field") != "row_id":
        raise LongitudinalIntakeError("composition_plane_row_identity_not_canonical")

    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise LongitudinalIntakeError(
            f"composition_profile_not_declared:{profile_id}"
        )
    profile = profiles[profile_id]
    if not isinstance(profile, dict):
        raise LongitudinalIntakeError("composition_profile_record_invalid")
    features = tuple(str(value) for value in profile.get("features", ()))
    categorical = tuple(
        str(value) for value in profile.get("categorical_features", ())
    )
    if not features or len(set(features)) != len(features):
        raise LongitudinalIntakeError("composition_profile_features_invalid")
    if not set(categorical).issubset(features):
        raise LongitudinalIntakeError(
            "composition_profile_categorical_not_subset"
        )
    forbidden = {
        "ESTADO",
        "CONDACT",
        "donor_condact",
        "donor_condact_vintage",
        "stale_labor_state",
        "target_current_labor_state",
    }
    overlap = sorted(set(features) & forbidden)
    if overlap:
        raise LongitudinalIntakeError(
            "composition_profile_contains_labor_state:" + ",".join(overlap)
        )

    artifact_name = str(profile.get("artifact") or "")
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise LongitudinalIntakeError("composition_profile_artifact_name_invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LongitudinalIntakeError("composition_plane_artifacts_missing")
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, dict):
        raise LongitudinalIntakeError("composition_profile_artifact_record_missing")
    rows_path = _verify_artifact(root, artifact, artifact_name)
    header, row_count = _csv_header_and_count(rows_path)
    declared_rows = artifact.get("rows")
    if declared_rows is not None and int(declared_rows) != row_count:
        raise LongitudinalIntakeError("composition_profile_row_count_mismatch")
    required_columns = {"row_id", *features}
    missing_columns = sorted(required_columns - set(header))
    if missing_columns:
        raise LongitudinalIntakeError(
            "composition_profile_columns_missing:" + ",".join(missing_columns)
        )

    profile_sha256 = hashlib.sha256(
        (
            json.dumps(
                profile,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return CompositionPlaneProfile(
        root=root,
        release_id=release_id,
        profile_id=profile_id,
        features=features,
        categorical_features=categorical,
        rows_path=rows_path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        profile_sha256=profile_sha256,
    )


def join_composition_profile(
    rows: list[dict[str, Any]],
    profile: CompositionPlaneProfile,
) -> list[dict[str, Any]]:
    """Join an exact canonical composition profile to C2 observations by row_id."""
    base_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if not row_id:
            raise LongitudinalIntakeError("composition_join_base_row_id_missing")
        if row_id in base_by_id:
            raise LongitudinalIntakeError(
                f"composition_join_duplicate_base_row_id:{row_id}"
            )
        base_by_id[row_id] = row

    profile_by_id: dict[str, dict[str, str]] = {}
    for row in profile.read_rows():
        row_id = str(row.get("row_id") or "")
        if not row_id:
            raise LongitudinalIntakeError("composition_join_parent_row_id_missing")
        if row_id in profile_by_id:
            raise LongitudinalIntakeError(
                f"composition_join_duplicate_parent_row_id:{row_id}"
            )
        profile_by_id[row_id] = row

    base_ids = set(base_by_id)
    profile_ids = set(profile_by_id)
    if base_ids != profile_ids:
        missing = sorted(base_ids - profile_ids)
        extra = sorted(profile_ids - base_ids)
        raise LongitudinalIntakeError(
            "composition_join_identity_mismatch:"
            f"missing={missing[:5]}:extra={extra[:5]}"
        )

    output: list[dict[str, Any]] = []
    for source in rows:
        row_id = str(source["row_id"])
        governed = profile_by_id[row_id]
        merged = dict(source)
        for feature in profile.features:
            if feature not in governed:
                raise LongitudinalIntakeError(
                    f"composition_join_feature_missing:{feature}"
                )
            merged[feature] = governed[feature]
        merged["composition_profile_id"] = profile.profile_id
        merged["composition_release_id"] = profile.release_id
        output.append(merged)
    return output


def canonical_composition_parent_metadata(
    profile: CompositionPlaneProfile,
) -> dict[str, Any]:
    return {
        "source": "canonical_parent",
        "contract": COMPOSITION_PLANE_CONTRACT,
        "release_id": profile.release_id,
        "manifest_sha256": profile.manifest_sha256,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "features": list(profile.features),
        "categorical_features": list(profile.categorical_features),
        "fixture_only": False,
    }


def fixture_composition_parent_metadata() -> dict[str, Any]:
    return {
        "source": "raw_c2_fixture",
        "contract": FIXTURE_COMPOSITION_CONTRACT,
        "release_id": None,
        "manifest_sha256": None,
        "profile_id": FIXTURE_COMPOSITION_PROFILE_ID,
        "fixture_only": True,
    }


@dataclass(frozen=True)
class LaborContextRelease:
    root: Path
    release_id: str
    observations_path: Path
    manifest: dict[str, Any]
    completion: LaborContextCompletion | None = None

    def read_observations(self) -> list[dict[str, str]]:
        rows = _read_csv(self.observations_path)
        if self.completion is None:
            return rows
        # Synthetic rows deliberately carry no official source/quality
        # metadata. Their provenance is the overlay's source identity.
        for row in self.completion.rows:
            rows.append({
                "period": row["period"],
                "geography_level": row["geography_level"],
                "geography_id": row["geography_id"],
                "indicator_id": row["indicator_id"],
                "value": row["value"],
                "value_status": row["value_status"],
                "source_id": "",
                "source_snapshot_sha256": "",
                "source_cell_identity": "derived_from:" + row["source_observation_identity"],
                "completion_release_id": self.completion.release_id,
                "completion_manifest_sha256": self.completion.manifest_sha256,
                "fill_method": row["fill_method"],
                "fill_source_period": row["fill_source_period"],
                "fill_distance_quarters": row["fill_distance_quarters"],
            })
        return rows


@dataclass(frozen=True)
class LaborContextCompletion:
    root: Path
    release_id: str
    manifest_sha256: str
    rows: list[dict[str, str]]


def _completion_period_offset(period: str, delta: int) -> str:
    year, quarter = int(period[:4]), int(period[-1])
    index = year * 4 + quarter - 1 + delta
    return f"{index // 4:04d}-Q{index % 4 + 1}"


def load_labor_completion_overlay(
    official_root: Path,
    overlay_root: Path,
    official_manifest: dict[str, Any],
) -> LaborContextCompletion:
    overlay_root = Path(overlay_root).expanduser().resolve()
    manifest_path = overlay_root / "manifest.json"
    manifest = _json(manifest_path, "labor_completion_manifest_invalid")
    if manifest.get("contract") != "research.indec-eph-labor-context-completion/v1":
        raise LongitudinalIntakeError("unexpected_labor_completion_contract")
    if manifest.get("official_parent_release_id") != official_manifest.get("release_id"):
        raise LongitudinalIntakeError("labor_completion_parent_release_mismatch")
    parent_hash = sha256_file(Path(official_root) / "manifest.json")
    if manifest.get("official_parent_manifest_sha256") != parent_hash:
        raise LongitudinalIntakeError("labor_completion_parent_hash_mismatch")
    completion_path = overlay_root / "completion.csv"
    expected = (manifest.get("files") or {}).get("completion.csv")
    if not completion_path.is_file() or sha256_file(completion_path) != expected:
        raise LongitudinalIntakeError("labor_completion_artifact_hash_mismatch")
    rows = _read_csv(completion_path)
    official_rows = _read_csv(Path(official_root) / "labor_state.csv")
    official = {(r["period"], r["geography_level"], r["geography_id"], r["indicator_id"]): r for r in official_rows}
    coverage = _read_csv(Path(official_root) / "coverage.csv")
    missing = {(r["period"], r["geography_level"], r["geography_id"], r["indicator_id"])
               for r in coverage if r["coverage_status"] != "present"}
    seen = set()
    for row in rows:
        key = (row.get("period", ""), row.get("geography_level", ""), row.get("geography_id", ""), row.get("indicator_id", ""))
        if key in seen or key in official or key not in missing:
            raise LongitudinalIntakeError("labor_completion_not_missing_or_duplicate")
        seen.add(key)
        status = row.get("value_status")
        delta = 1 if status == "derived_bfill" else -1 if status == "derived_ffill" else 0
        if delta == 0 or row.get("fill_distance_quarters") != "1":
            raise LongitudinalIntakeError("labor_completion_bounds_invalid")
        if row.get("fill_source_period") != _completion_period_offset(row["period"], delta):
            raise LongitudinalIntakeError("labor_completion_source_period_invalid")
        source = official.get((row["fill_source_period"], row["geography_level"], row["geography_id"], row["indicator_id"]))
        if source is None or source.get("value_status") != "observed" or source.get("value") != row.get("value"):
            raise LongitudinalIntakeError("labor_completion_source_not_observed")
        if not row.get("source_observation_identity"):
            raise LongitudinalIntakeError("labor_completion_source_identity_missing")
        if any(field in row for field in ("cv", "ci90_low", "ci90_high", "quality_status")):
            raise LongitudinalIntakeError("labor_completion_quality_metadata_forbidden")
    if seen != missing:
        raise LongitudinalIntakeError("labor_completion_required_grid_incomplete")
    return LaborContextCompletion(
        root=overlay_root, release_id=str(manifest["release_id"]),
        manifest_sha256=sha256_file(manifest_path), rows=rows,
    )


def load_labor_context_release(root: Path, completion_root: Path | None = None) -> LaborContextRelease:
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "manifest.json", "labor_context_manifest_invalid")
    if manifest.get("contract_id") != LABOR_CONTEXT_CONTRACT:
        raise LongitudinalIntakeError("unexpected_labor_context_contract")
    release_id = str(manifest.get("release_id") or "")
    if not release_id or root.name != release_id:
        raise LongitudinalIntakeError("labor_context_release_directory_mismatch")
    completion = None
    if manifest.get("required_coverage_complete") is not True and completion_root is None:
        raise LongitudinalIntakeError("labor_context_required_coverage_incomplete")
    if str(manifest.get("period_min")) > "2017-Q1" or str(manifest.get("period_max")) < "2026-Q1":
        raise LongitudinalIntakeError("labor_context_period_coverage_insufficient")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise LongitudinalIntakeError("labor_context_files_missing")
    required_files = {
        "labor_state.csv",
        "coverage.csv",
        "geographies.json",
        "indicators.json",
    }
    missing = sorted(required_files - set(files))
    if missing:
        raise LongitudinalIntakeError(
            "labor_context_required_files_missing:" + ",".join(missing)
        )
    for filename, expected in files.items():
        if Path(filename).name != filename:
            raise LongitudinalIntakeError("labor_context_artifact_name_unsafe")
        path = root / filename
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise LongitudinalIntakeError(
                f"labor_context_artifact_hash_mismatch:{filename}"
            )
    observations_path = root / "labor_state.csv"
    if completion_root is not None:
        if manifest.get("required_coverage_complete") is True:
            raise LongitudinalIntakeError("labor_completion_unnecessary_for_complete_parent")
        completion = load_labor_completion_overlay(root, completion_root, manifest)
    return LaborContextRelease(
        root=root,
        release_id=release_id,
        observations_path=observations_path,
        manifest=manifest,
        completion=completion,
    )


@dataclass(frozen=True)
class DonorLaborRelease:
    root: Path
    release_id: str
    donor_vintage: int
    handoff_path: Path
    qa_path: Path
    manifest: dict[str, Any]

    def read_rows(self) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise LongitudinalIntakeError(
                "donor_labor_parquet_requires_pandas_pyarrow"
            ) from exc
        frame = pd.read_parquet(self.handoff_path)
        return frame.to_dict(orient="records")


def load_donor_labor_release(root: Path) -> DonorLaborRelease:
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "feature_plane_manifest.json", "semantic_plane_manifest_invalid")
    if manifest.get("schema") != SEMANTIC_PLANE_CONTRACT:
        raise LongitudinalIntakeError("unexpected_semantic_plane_contract")
    release_id = str(manifest.get("release_id") or "")
    if not release_id:
        raise LongitudinalIntakeError("semantic_plane_release_id_missing")
    handoff = manifest.get("donor_labor_handoff")
    if not isinstance(handoff, dict):
        raise LongitudinalIntakeError("semantic_plane_donor_labor_handoff_missing")
    if handoff.get("schema") != "research.eph-census-donor-labor-handoff/v1":
        raise LongitudinalIntakeError("unexpected_donor_labor_handoff_contract")
    if handoff.get("target_period_current_state_claimed") is not False:
        raise LongitudinalIntakeError("donor_labor_mislabeled_as_current")
    if handoff.get("eph_training_analogue_materialized") is not False:
        raise LongitudinalIntakeError("donor_labor_fake_eph_analogue_present")
    donor_vintage = handoff.get("donor_vintage")
    if not isinstance(donor_vintage, int):
        raise LongitudinalIntakeError("donor_labor_vintage_missing")
    handoff_name = str(handoff.get("path") or "")
    qa_name = str(handoff.get("qa_path") or "")
    if Path(handoff_name).name != handoff_name or Path(qa_name).name != qa_name:
        raise LongitudinalIntakeError("donor_labor_artifact_name_unsafe")
    handoff_path = root / handoff_name
    qa_path = root / qa_name
    if (
        not handoff_path.is_file()
        or sha256_file(handoff_path) != handoff.get("sha256")
        or not qa_path.is_file()
        or sha256_file(qa_path) != handoff.get("qa_sha256")
    ):
        raise LongitudinalIntakeError("donor_labor_artifact_hash_mismatch")
    qa = _json(qa_path, "donor_labor_qa_invalid")
    if qa.get("identity_row_count_preserved") is not True:
        raise LongitudinalIntakeError("donor_labor_identity_not_preserved")
    if qa.get("clock_separation", {}).get("same_clock") is not False:
        raise LongitudinalIntakeError("donor_labor_clock_separation_missing")
    return DonorLaborRelease(
        root=root,
        release_id=release_id,
        donor_vintage=donor_vintage,
        handoff_path=handoff_path,
        qa_path=qa_path,
        manifest=manifest,
    )


def exact_parent_metadata(
    eph: LongitudinalEPHRelease,
    labor: LaborContextRelease,
    *,
    composition_metadata: Mapping[str, Any],
    donor: DonorLaborRelease | None = None,
    anchor_release_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "longitudinal_eph": {
            "contract": LONGITUDINAL_EPH_CONTRACT,
            "release_id": eph.release_id,
            "manifest_sha256": sha256_file(eph.root / "manifest.json"),
        },
        "labor_context": {
            "contract": LABOR_CONTEXT_CONTRACT,
            "release_id": labor.release_id,
            "manifest_sha256": sha256_file(labor.root / "manifest.json"),
        },
        "monetary_conversion": {
            "release_id": eph.monetary_release_id,
            "reference_period": eph.monetary_reference_period,
        },
        "composition": dict(composition_metadata),
        "anchors": list(anchor_release_ids),
    }
    if labor.completion is not None:
        output["labor_context_completion"] = {
            "contract": "research.indec-eph-labor-context-completion/v1",
            "release_id": labor.completion.release_id,
            "manifest_sha256": labor.completion.manifest_sha256,
        }
    if donor is not None:
        output["donor_labor"] = {
            "semantic_plane_release_id": donor.release_id,
            "donor_vintage": donor.donor_vintage,
            "contract": "research.eph-census-donor-labor-handoff/v1",
            "manifest_sha256": sha256_file(
                donor.root / "feature_plane_manifest.json"
            ),
        }
    return output
