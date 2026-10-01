"""Resource-safe longitudinal execution plane for real L10 commissioning.

C6 changes storage and execution mechanics only.  Scientific semantics remain owned by
C4/C4B: canonical C5 composition, official aggregate labor context, deterministic
panel-household folds, nested household-safe OOF base predictions, and the explicit
year/quarter/exception time layer.

The real-scale path deliberately avoids million-row Python dict collections.  C2 and C5
are streamed in lockstep into typed NumPy .npy arrays, and outer folds are executed one
at a time with atomic checkpoints.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import distributional_regression_diagnostics
from .longitudinal_intake import (
    CompositionPlaneProfile,
    LaborContextRelease,
    LongitudinalEPHRelease,
    canonical_composition_parent_metadata,
    exact_parent_metadata,
    sha256_file,
)
from .longitudinal_runtime import (
    EXCEPTIONAL_PERIODS,
    LABOR_CONTEXT_FIELDS,
    LABOR_INDICATORS,
    LongitudinalConfig,
    _hurdle_estimator,
    labor_context_index,
)
from .terminal import classify_income_target
from .time_layer import fit_time_layer

MODEL_PLANE_CONTRACT = "research.encuestador-longitudinal-model-plane/v1"
C6_RUN_CONTRACT = "research.encuestador-longitudinal-c6-run/v1"
FOLD_POLICY = "panel_household_grouped_longitudinal_v1"
LABOR_MODES = ("none", "national", "national_regional")
PERIODS = tuple(
    f"{year}-Q{quarter}"
    for year in range(2017, 2027)
    for quarter in range(1, 5)
    if (year, quarter) <= (2026, 1)
)
PERIOD_TO_CODE = {period: index for index, period in enumerate(PERIODS)}


class C6ExecutionError(ValueError):
    """Raised when resource-safe execution would weaken an existing contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_gib() -> float:
    # Linux ru_maxrss is KiB.  C6 real commissioning currently targets Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def _current_rss_gib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kib = float(line.split()[1])
                return kib / (1024.0**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _current_swap_gib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmSwap:"):
                kib = float(line.split()[1])
                return kib / (1024.0**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _count_csv_rows(path: Path) -> int:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                next(reader)
            except StopIteration as exc:
                raise C6ExecutionError(f"csv_empty:{path.name}") from exc
            return sum(1 for _ in reader)
    except OSError as exc:
        raise C6ExecutionError(f"csv_unreadable:{path.name}") from exc


def _declared_rows(record: Mapping[str, Any] | None, path: Path) -> int:
    value = (record or {}).get("rows")
    if value is not None:
        count = int(value)
        if count <= 0:
            raise C6ExecutionError(f"artifact_declared_row_count_invalid:{path.name}")
        return count
    return _count_csv_rows(path)


def _feature_value(value: Any, field: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise C6ExecutionError(f"feature_not_numeric:{field}:{value}") from exc
    if math.isinf(number):
        raise C6ExecutionError(f"feature_infinite:{field}")
    return number


def _target_value(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return float("nan")
    try:
        number = float(text)
    except ValueError as exc:
        raise C6ExecutionError(f"target_not_numeric:{text}") from exc
    if math.isinf(number):
        raise C6ExecutionError("target_infinite")
    return number


def _household_fold(group: str, n_splits: int) -> int:
    digest = hashlib.sha256(group.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_splits


def _labor_values(
    index: Mapping[tuple[str, str, str], Mapping[str, Any]],
    period: str,
    region: str,
) -> tuple[float, ...]:
    values: list[float] = []
    for indicator in LABOR_INDICATORS:
        national_key = (period, "total_31_agglomerates", indicator)
        regional_key = (period, region, indicator)
        if national_key not in index or regional_key not in index:
            raise C6ExecutionError(
                f"labor_context_join_incomplete:{period}:{region}:{indicator}"
            )
        national = float(index[national_key]["value"])
        regional = float(index[regional_key]["value"])
        values.append(national)
    for indicator in LABOR_INDICATORS:
        national = float(index[(period, "total_31_agglomerates", indicator)]["value"])
        regional = float(index[(period, region, indicator)]["value"])
        values.append(regional - national)
    return tuple(values)


def _derived_labor_context(
    index: Mapping[tuple[str, str, str], Mapping[str, Any]],
    period: str,
    region: str,
) -> bool:
    for indicator in LABOR_INDICATORS:
        for geography in ("total_31_agglomerates", region):
            if str(index[(period, geography, indicator)].get("value_status")) != "observed":
                return True
    return False


@dataclass(frozen=True)
class LongitudinalModelPlane:
    root: Path
    release_id: str
    manifest: dict[str, Any]
    manifest_sha256: str
    row_count: int
    feature_names: tuple[str, ...]
    composition_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    n_splits: int

    @property
    def features_path(self) -> Path:
        return self.root / "features.npy"

    @property
    def target_path(self) -> Path:
        return self.root / "target.npy"

    @property
    def period_codes_path(self) -> Path:
        return self.root / "period_codes.npy"

    @property
    def fold_ids_path(self) -> Path:
        return self.root / "fold_ids.npy"

    @property
    def household_codes_path(self) -> Path:
        return self.root / "household_codes.npy"

    @property
    def row_ids_path(self) -> Path:
        return self.root / "row_ids.txt"

    def feature_indices(self, labor_mode: str) -> tuple[int, ...]:
        if labor_mode not in LABOR_MODES:
            raise C6ExecutionError(f"unknown_labor_mode:{labor_mode}")
        composition_count = len(self.composition_features)
        if labor_mode == "none":
            return tuple(range(composition_count))
        if labor_mode == "national":
            return tuple(range(composition_count + 3))
        return tuple(range(len(self.feature_names)))

    def selected_feature_names(self, labor_mode: str) -> tuple[str, ...]:
        return tuple(self.feature_names[index] for index in self.feature_indices(labor_mode))


def resolve_config_for_model_plane(
    config: LongitudinalConfig,
    plane: LongitudinalModelPlane,
) -> LongitudinalConfig:
    """Bind a canonical config to the exact feature profile stored in a C6 plane."""
    if config.arm != "L10":
        raise C6ExecutionError("c6_first_executor_supports_l10_only")
    if config.composition_source != "canonical_parent":
        raise C6ExecutionError("c6_requires_canonical_composition_parent")
    if config.feature_profile_id != plane.manifest.get("profile_id"):
        raise C6ExecutionError("c6_run_profile_mismatch")
    if config.n_splits != plane.n_splits:
        raise C6ExecutionError("c6_run_fold_count_mismatch")
    return replace(
        config,
        composition_features=plane.composition_features,
        categorical_features=plane.categorical_features,
    )


def _artifact_rows_from_profile(profile: CompositionPlaneProfile) -> int:
    profile_record = (profile.manifest.get("profiles") or {}).get(profile.profile_id) or {}
    artifact_name = str(profile_record.get("artifact") or "")
    artifact = (profile.manifest.get("artifacts") or {}).get(artifact_name) or {}
    return _declared_rows(artifact, profile.rows_path)


def materialize_longitudinal_model_plane(
    eph: LongitudinalEPHRelease,
    labor: LaborContextRelease,
    profile: CompositionPlaneProfile,
    config: LongitudinalConfig,
    output_root: Path,
) -> Path:
    """Stream C2+C5+L1 into one typed L10 model plane with persistent folds."""
    if config.arm != "L10":
        raise C6ExecutionError("c6_model_plane_first_release_is_l10_only")
    if config.composition_source != "canonical_parent":
        raise C6ExecutionError("c6_requires_canonical_composition_parent")
    if config.feature_profile_id != profile.profile_id:
        raise C6ExecutionError("c6_profile_config_mismatch")
    if tuple(config.composition_features) != tuple(profile.features):
        raise C6ExecutionError("c6_config_profile_not_resolved")
    if tuple(config.categorical_features) != tuple(profile.categorical_features):
        raise C6ExecutionError("c6_categorical_profile_mismatch")

    eph_artifact = (eph.manifest.get("artifacts") or {}).get("persons.csv") or {}
    eph_rows = _declared_rows(eph_artifact, eph.persons_path)
    composition_rows = _artifact_rows_from_profile(profile)
    if eph_rows != composition_rows:
        raise C6ExecutionError(
            f"c6_parent_row_count_mismatch:{eph_rows}:{composition_rows}"
        )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".longitudinal-model-plane.", dir=output_root))

    feature_names = (*profile.features, *LABOR_CONTEXT_FIELDS)
    feature_matrix = np.lib.format.open_memmap(
        staging / "features.npy",
        mode="w+",
        dtype=np.float64,
        shape=(eph_rows, len(feature_names)),
    )
    target = np.lib.format.open_memmap(
        staging / "target.npy", mode="w+", dtype=np.float64, shape=(eph_rows,)
    )
    period_codes = np.lib.format.open_memmap(
        staging / "period_codes.npy", mode="w+", dtype=np.int16, shape=(eph_rows,)
    )
    fold_ids = np.lib.format.open_memmap(
        staging / "fold_ids.npy", mode="w+", dtype=np.uint8, shape=(eph_rows,)
    )
    household_codes = np.lib.format.open_memmap(
        staging / "household_codes.npy", mode="w+", dtype=np.int32, shape=(eph_rows,)
    )

    labor_index = labor_context_index(labor.read_observations())
    row_sequence = hashlib.sha256()
    household_code_by_id: dict[str, int] = {}
    derived_context_rows = 0

    try:
        with (staging / "row_ids.txt").open("w", encoding="utf-8") as row_stream:
            paired = zip_longest(eph.iter_persons(), profile.iter_rows())
            observed_rows = 0
            for index, pair in enumerate(paired):
                eph_row, composition_row = pair
                if eph_row is None or composition_row is None:
                    raise C6ExecutionError("c6_parent_stream_length_mismatch")
                if index >= eph_rows:
                    raise C6ExecutionError("c6_parent_stream_exceeds_declared_rows")
                row_id = str(eph_row.get(config.row_id_field) or "")
                composition_id = str(composition_row.get("row_id") or "")
                if not row_id or row_id != composition_id:
                    raise C6ExecutionError(
                        f"c6_lockstep_row_identity_mismatch:{index}:{row_id}:{composition_id}"
                    )
                period = str(eph_row.get(config.period_field) or "")
                region = str(eph_row.get(config.region_field) or "")
                panel_group = str(eph_row.get(config.group_field) or "")
                household = str(eph_row.get(config.household_observation_field) or "")
                if period not in PERIOD_TO_CODE:
                    raise C6ExecutionError(f"c6_period_outside_window:{period}")
                if not region or not panel_group or not household:
                    raise C6ExecutionError(f"c6_identity_field_missing:{index}")

                for column, feature in enumerate(profile.features):
                    feature_matrix[index, column] = _feature_value(
                        composition_row.get(feature), feature
                    )
                labor_values = _labor_values(labor_index, period, region)
                feature_matrix[index, len(profile.features) :] = np.asarray(
                    labor_values, dtype=np.float64
                )
                target[index] = _target_value(eph_row.get(config.target_field))
                period_codes[index] = PERIOD_TO_CODE[period]
                fold_ids[index] = _household_fold(panel_group, config.n_splits)

                code = household_code_by_id.get(household)
                if code is None:
                    code = len(household_code_by_id)
                    household_code_by_id[household] = code
                household_codes[index] = code

                if _derived_labor_context(labor_index, period, region):
                    derived_context_rows += 1
                row_stream.write(row_id + "\n")
                row_sequence.update((row_id + "\n").encode())
                observed_rows = index + 1

        if observed_rows != eph_rows:
            raise C6ExecutionError(
                f"c6_parent_stream_row_count_mismatch:{observed_rows}:{eph_rows}"
            )
        used_folds = set(np.asarray(fold_ids).tolist())
        if used_folds != set(range(config.n_splits)):
            raise C6ExecutionError("c6_fold_assignment_has_empty_fold")

        feature_matrix.flush()
        target.flush()
        period_codes.flush()
        fold_ids.flush()
        household_codes.flush()
        del feature_matrix, target, period_codes, fold_ids, household_codes
        gc.collect()

        composition_metadata = canonical_composition_parent_metadata(profile)
        parents = exact_parent_metadata(
            eph,
            labor,
            composition_metadata=composition_metadata,
        )
        row_sequence_sha256 = row_sequence.hexdigest()
        fold_ids_sha256 = _sha256(staging / "fold_ids.npy")
        payload = {
            "contract": MODEL_PLANE_CONTRACT,
            "config_digest": config.digest,
            "parents": parents,
            "profile_id": profile.profile_id,
            "feature_names": list(feature_names),
            "row_count": eph_rows,
            "row_identity_sequence_sha256": row_sequence_sha256,
            "fold_ids_sha256": fold_ids_sha256,
        }
        release_digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        release_id = (
            "longitudinal-model-plane-"
            + profile.profile_id.lower()
            + "-"
            + release_digest[:16]
        )
        destination = output_root / release_id
        if destination.exists():
            raise C6ExecutionError(f"immutable_c6_model_plane_exists:{destination}")

        artifacts: dict[str, Any] = {}
        for name in (
            "features.npy",
            "target.npy",
            "period_codes.npy",
            "fold_ids.npy",
            "household_codes.npy",
            "row_ids.txt",
        ):
            artifact_path = staging / name
            artifacts[name] = {
                "sha256": _sha256(artifact_path),
                "bytes": artifact_path.stat().st_size,
            }
        manifest = {
            **payload,
            "release_id": release_id,
            "composition_features": list(profile.features),
            "categorical_features": list(profile.categorical_features),
            "labor_context_fields": list(LABOR_CONTEXT_FIELDS),
            "target_field": config.target_field,
            "period_field": config.period_field,
            "region_field": config.region_field,
            "row_id_field": config.row_id_field,
            "group_field": config.group_field,
            "household_observation_field": config.household_observation_field,
            "fold_policy": FOLD_POLICY,
            "n_splits": config.n_splits,
            "period_labels": list(PERIODS),
            "household_group_count": len(household_code_by_id),
            "derived_labor_context_person_rows": derived_context_rows,
            "storage": {
                "features": "float64_c_order_npy",
                "target": "float64_npy",
                "period_codes": "int16_npy",
                "fold_ids": "uint8_npy",
                "household_codes": "int32_npy",
                "row_ids": "utf8_newline_exact_sequence",
            },
            "execution_semantics": {
                "one_lockstep_c2_c5_join": True,
                "million_row_python_dict_collection": False,
                "measurement_mode": True,
                "forecasting_authorized": False,
            },
            "build_resource": {
                "peak_rss_gib": _peak_rss_gib(),
                "current_rss_gib": _current_rss_gib(),
                "current_process_swap_gib": _current_swap_gib(),
            },
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            _canonical_json(manifest), encoding="utf-8"
        )
        os.replace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_longitudinal_model_plane(
    root: Path,
    *,
    verify_hashes: bool = True,
) -> LongitudinalModelPlane:
    root = Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C6ExecutionError("c6_model_plane_manifest_invalid") from exc
    if manifest.get("contract") != MODEL_PLANE_CONTRACT:
        raise C6ExecutionError("unexpected_c6_model_plane_contract")
    if manifest.get("release_id") != root.name:
        raise C6ExecutionError("c6_model_plane_directory_mismatch")
    artifacts = manifest.get("artifacts") or {}
    required = (
        "features.npy",
        "target.npy",
        "period_codes.npy",
        "fold_ids.npy",
        "household_codes.npy",
        "row_ids.txt",
    )
    for name in required:
        record = artifacts.get(name)
        path = root / name
        if not isinstance(record, Mapping) or not path.is_file():
            raise C6ExecutionError(f"c6_model_plane_artifact_missing:{name}")
        if verify_hashes and _sha256(path) != record.get("sha256"):
            raise C6ExecutionError(f"c6_model_plane_artifact_hash_mismatch:{name}")

    row_count = int(manifest.get("row_count") or 0)
    features = tuple(str(value) for value in manifest.get("feature_names", ()))
    composition = tuple(str(value) for value in manifest.get("composition_features", ()))
    categorical = tuple(str(value) for value in manifest.get("categorical_features", ()))
    n_splits = int(manifest.get("n_splits") or 0)
    if row_count <= 0 or not features or not composition or n_splits < 3:
        raise C6ExecutionError("c6_model_plane_manifest_semantics_invalid")

    feature_array = np.load(root / "features.npy", mmap_mode="r")
    if feature_array.shape != (row_count, len(features)):
        raise C6ExecutionError("c6_model_plane_feature_shape_mismatch")
    for name in ("target.npy", "period_codes.npy", "fold_ids.npy", "household_codes.npy"):
        array = np.load(root / name, mmap_mode="r")
        if array.shape != (row_count,):
            raise C6ExecutionError(f"c6_model_plane_vector_shape_mismatch:{name}")

    return LongitudinalModelPlane(
        root=root,
        release_id=root.name,
        manifest=manifest,
        manifest_sha256=sha256_file(root / "manifest.json"),
        row_count=row_count,
        feature_names=features,
        composition_features=composition,
        categorical_features=categorical,
        n_splits=n_splits,
    )


def _dense_rows(
    source: np.ndarray,
    indices: np.ndarray,
    columns: Sequence[int],
) -> np.ndarray:
    output = np.empty((len(indices), len(columns)), dtype=np.float64)
    for output_column, source_column in enumerate(columns):
        output[:, output_column] = source[indices, source_column]
    return output


def _nested_hurdle_oof(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fold_train: np.ndarray,
    config: LongitudinalConfig,
    feature_names: Sequence[str],
    *,
    include_stale: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    surviving = sorted({int(value) for value in fold_train.tolist()})
    if len(surviving) < 2:
        raise C6ExecutionError("c6_nested_training_requires_two_inner_folds")
    probability = np.full(len(x_train), np.nan, dtype=float)
    amount = np.full(len(x_train), np.nan, dtype=float)
    reports: list[dict[str, Any]] = []
    for inner_label in surviving:
        holdout = fold_train == inner_label
        fit = ~holdout
        if not np.any(fit) or not np.any(holdout):
            raise C6ExecutionError(f"c6_inner_fold_partition_invalid:{inner_label}")
        estimator = _hurdle_estimator(
            config,
            feature_names,
            include_stale=include_stale,
        )
        estimator.fit(x_train[fit], y_train[fit])
        components = estimator.predict_components(x_train[holdout])
        probability[holdout] = components.p_positive
        amount[holdout] = components.positive_amount
        reports.append(
            {
                "source_fold_label": inner_label,
                "fit_rows": int(fit.sum()),
                "holdout_rows": int(holdout.sum()),
                "group_overlap": 0,
            }
        )
        del estimator, components
        gc.collect()
    if not np.isfinite(probability).all() or not np.isfinite(amount).all():
        raise C6ExecutionError("c6_inner_oof_prediction_incomplete")
    return probability, amount, {
        "source": "inner_household_safe_oof",
        "policy": FOLD_POLICY,
        "n_splits": len(surviving),
        "row_count": len(x_train),
        "in_sample_base_predictions_used": False,
        "outer_holdout_rows_used": 0,
        "folds": reports,
    }


def _run_id(
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    labor_mode: str,
) -> str:
    payload = {
        "plane_release_id": plane.release_id,
        "plane_manifest_sha256": plane.manifest_sha256,
        "config_digest": config.digest,
        "labor_mode": labor_mode,
        "selected_features": list(plane.selected_feature_names(labor_mode)),
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]
    return f"longitudinal-l10-c6-{labor_mode.replace('_', '-')}-{digest}"


def _checkpoint_identity(
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    labor_mode: str,
    outer_fold: int,
) -> dict[str, Any]:
    return {
        "schema": "research.encuestador-longitudinal-c6-fold-checkpoint/v1",
        "plane_release_id": plane.release_id,
        "plane_manifest_sha256": plane.manifest_sha256,
        "config_digest": config.digest,
        "labor_mode": labor_mode,
        "outer_fold": outer_fold,
        "fold_policy": FOLD_POLICY,
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
    }


def _checkpoint_valid(
    checkpoint_root: Path,
    identity: Mapping[str, Any],
) -> bool:
    path = checkpoint_root / "checkpoint.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in identity.items():
        if record.get(key) != value:
            return False
    files = record.get("artifacts") or {}
    for name in ("indices.npy", "p_positive.npy", "positive_amount.npy"):
        artifact = files.get(name)
        path_value = checkpoint_root / name
        if (
            not isinstance(artifact, Mapping)
            or not path_value.is_file()
            or _sha256(path_value) != artifact.get("sha256")
        ):
            return False
    return True


def _write_fold_checkpoint(
    work_root: Path,
    identity: Mapping[str, Any],
    holdout_indices: np.ndarray,
    p_positive: np.ndarray,
    positive_amount: np.ndarray,
    time_report: Mapping[str, Any],
) -> Path:
    fold = int(identity["outer_fold"])
    destination = work_root / "checkpoints" / f"fold_{fold}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _checkpoint_valid(destination, identity):
            return destination
        raise C6ExecutionError(f"c6_checkpoint_identity_conflict:{fold}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".fold_{fold}.", dir=destination.parent)
    )
    try:
        np.save(staging / "indices.npy", np.asarray(holdout_indices, dtype=np.int64))
        np.save(staging / "p_positive.npy", np.asarray(p_positive, dtype=np.float64))
        np.save(
            staging / "positive_amount.npy",
            np.asarray(positive_amount, dtype=np.float64),
        )
        artifacts = {
            name: {
                "sha256": _sha256(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
            for name in ("indices.npy", "p_positive.npy", "positive_amount.npy")
        }
        record = {
            **dict(identity),
            "holdout_rows": len(holdout_indices),
            "time_layer": dict(time_report),
            "resource": {
                "peak_rss_gib": _peak_rss_gib(),
                "current_rss_gib": _current_rss_gib(),
            },
            "artifacts": artifacts,
        }
        (staging / "checkpoint.json").write_text(
            _canonical_json(record), encoding="utf-8"
        )
        os.replace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _binary_presence_diagnostics(
    truth_positive: np.ndarray,
    p_positive: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(truth_positive, dtype=int)
    probability = np.asarray(p_positive, dtype=float)
    if truth.ndim != 1 or probability.ndim != 1 or len(truth) != len(probability):
        raise C6ExecutionError("c6_presence_metric_shape_invalid")
    eps = np.finfo(float).eps
    predicted = probability >= 0.5
    tp = int(np.sum((truth == 1) & predicted))
    tn = int(np.sum((truth == 0) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    log_loss = float(
        -np.mean(
            truth * np.log(np.clip(probability, eps, 1.0))
            + (1 - truth) * np.log(np.clip(1.0 - probability, eps, 1.0))
        )
    )
    brier = float(np.mean((probability - truth) ** 2))
    return {
        "row_count": len(truth),
        "accuracy": float(np.mean(predicted == truth)),
        "log_loss": log_loss,
        "binary_brier": brier,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "prevalence_positive": float(np.mean(truth)),
    }


def _household_metrics(
    household_codes: np.ndarray,
    eligibility,
    prediction: np.ndarray,
    household_group_count: int,
) -> dict[str, Any]:
    codes = np.asarray(household_codes, dtype=np.int64)
    counts = np.bincount(codes, minlength=household_group_count)
    valid_counts = np.bincount(
        codes,
        weights=eligibility.valid.astype(np.int64),
        minlength=household_group_count,
    )
    truth_sum = np.bincount(
        codes,
        weights=np.where(eligibility.valid, eligibility.numeric, 0.0),
        minlength=household_group_count,
    )
    prediction_sum = np.bincount(
        codes,
        weights=np.where(eligibility.valid, prediction, 0.0),
        minlength=household_group_count,
    )
    complete = valid_counts == counts
    if not np.any(complete):
        raise C6ExecutionError("c6_household_evaluation_empty")
    return {
        "group_count": int(complete.sum()),
        "groups_excluded_for_invalid_target": int((~complete).sum()),
        "aggregation_scope": "complete_observed_household",
        "full_household_welfare_claim": True,
        "point": distributional_regression_diagnostics(
            truth_sum[complete], prediction_sum[complete]
        ),
    }


def _prepare_work_root(
    output_root: Path,
    run_id: str,
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    labor_mode: str,
) -> tuple[Path, Path]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        return destination, destination
    work = output_root / f".{run_id}.work"
    identity = {
        "schema": "research.encuestador-longitudinal-c6-work/v1",
        "run_id": run_id,
        "plane_release_id": plane.release_id,
        "plane_manifest_sha256": plane.manifest_sha256,
        "config_digest": config.digest,
        "labor_mode": labor_mode,
    }
    if work.exists():
        try:
            current = json.loads((work / "work_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise C6ExecutionError("c6_work_manifest_invalid") from exc
        if current != identity:
            raise C6ExecutionError("c6_work_identity_conflict")
    else:
        work.mkdir()
        (work / "work_manifest.json").write_text(
            _canonical_json(identity), encoding="utf-8"
        )
    return work, destination


def _execute_outer_fold(
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    labor_mode: str,
    outer_fold: int,
    work_root: Path,
) -> Path:
    identity = _checkpoint_identity(plane, config, labor_mode, outer_fold)
    existing = work_root / "checkpoints" / f"fold_{outer_fold}"
    if _checkpoint_valid(existing, identity):
        return existing

    features = np.load(plane.features_path, mmap_mode="r")
    target = np.load(plane.target_path, mmap_mode="r")
    period_codes = np.load(plane.period_codes_path, mmap_mode="r")
    fold_ids = np.load(plane.fold_ids_path, mmap_mode="r")

    train_indices = np.flatnonzero(fold_ids != outer_fold)
    holdout_indices = np.flatnonzero(fold_ids == outer_fold)
    if not len(train_indices) or not len(holdout_indices):
        raise C6ExecutionError(f"c6_outer_fold_partition_invalid:{outer_fold}")

    selected = plane.feature_indices(labor_mode)
    feature_names = plane.selected_feature_names(labor_mode)
    x_train = _dense_rows(features, train_indices, selected)
    x_holdout = _dense_rows(features, holdout_indices, selected)
    y_train = np.asarray(target[train_indices], dtype=float)
    fold_train = np.asarray(fold_ids[train_indices], dtype=int)
    period_lookup = np.asarray(plane.manifest["period_labels"], dtype="<U7")
    periods_train = period_lookup[np.asarray(period_codes[train_indices], dtype=int)]
    periods_holdout = period_lookup[np.asarray(period_codes[holdout_indices], dtype=int)]

    time_oof_p, time_oof_amount, time_evidence = _nested_hurdle_oof(
        x_train,
        y_train,
        fold_train,
        config,
        feature_names,
    )
    time_fit = fit_time_layer(
        periods_train,
        time_oof_p,
        time_oof_amount,
        y_train,
        exception_policy=EXCEPTIONAL_PERIODS,
    )

    estimator = _hurdle_estimator(config, feature_names, include_stale=False)
    estimator.fit(x_train, y_train)
    components = estimator.predict_components(x_holdout)
    corrected_p, corrected_amount = time_fit.apply(
        periods_holdout,
        components.p_positive,
        components.positive_amount,
    )
    report = {
        "outer_fold": outer_fold,
        "model_role": "L10",
        "labor_mode": labor_mode,
        "selected_features": list(feature_names),
        "base_prediction_evidence": time_evidence,
        **time_fit.as_dict(),
    }
    checkpoint = _write_fold_checkpoint(
        work_root,
        identity,
        holdout_indices,
        corrected_p,
        corrected_amount,
        report,
    )
    del (
        features,
        target,
        period_codes,
        fold_ids,
        train_indices,
        holdout_indices,
        x_train,
        x_holdout,
        y_train,
        fold_train,
        periods_train,
        periods_holdout,
        time_oof_p,
        time_oof_amount,
        estimator,
        components,
        corrected_p,
        corrected_amount,
    )
    gc.collect()
    return checkpoint


def _finalize_l10(
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    labor_mode: str,
    work_root: Path,
    destination: Path,
) -> Path:
    if destination.exists():
        return destination
    row_count = plane.row_count
    p_positive = np.lib.format.open_memmap(
        work_root / "p_positive.npy", mode="w+", dtype=np.float64, shape=(row_count,)
    )
    positive_amount = np.lib.format.open_memmap(
        work_root / "positive_amount.npy",
        mode="w+",
        dtype=np.float64,
        shape=(row_count,),
    )
    p_positive[:] = np.nan
    positive_amount[:] = np.nan
    time_reports: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    for outer_fold in range(plane.n_splits):
        checkpoint = work_root / "checkpoints" / f"fold_{outer_fold}"
        identity = _checkpoint_identity(plane, config, labor_mode, outer_fold)
        if not _checkpoint_valid(checkpoint, identity):
            raise C6ExecutionError(f"c6_checkpoint_missing_or_invalid:{outer_fold}")
        indices = np.load(checkpoint / "indices.npy", mmap_mode="r")
        p_positive[indices] = np.load(checkpoint / "p_positive.npy", mmap_mode="r")
        positive_amount[indices] = np.load(
            checkpoint / "positive_amount.npy", mmap_mode="r"
        )
        record = json.loads(
            (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
        )
        time_reports.append(record["time_layer"])
        checkpoint_records.append(
            {
                "outer_fold": outer_fold,
                "checkpoint_manifest_sha256": _sha256(checkpoint / "checkpoint.json"),
                "holdout_rows": record["holdout_rows"],
                "peak_rss_gib_after_fold": record["resource"]["peak_rss_gib"],
            }
        )

    p_positive.flush()
    positive_amount.flush()
    if not np.isfinite(p_positive).all() or not np.isfinite(positive_amount).all():
        raise C6ExecutionError("c6_oof_prediction_incomplete")

    expected = np.lib.format.open_memmap(
        work_root / "expected_income.npy",
        mode="w+",
        dtype=np.float64,
        shape=(row_count,),
    )
    expected[:] = p_positive * positive_amount
    expected.flush()

    target = np.load(plane.target_path, mmap_mode="r")
    household_codes = np.load(plane.household_codes_path, mmap_mode="r")
    fold_ids = np.load(plane.fold_ids_path, mmap_mode="r")
    eligibility = classify_income_target(np.asarray(target))
    valid = eligibility.valid
    positive = eligibility.positive
    metrics = {
        "person": {
            "unconditional": distributional_regression_diagnostics(
                eligibility.numeric[valid], np.asarray(expected)[valid]
            ),
            "presence": _binary_presence_diagnostics(
                eligibility.positive[valid].astype(int),
                np.asarray(p_positive)[valid],
            ),
            "positive_amount": (
                distributional_regression_diagnostics(
                    eligibility.numeric[positive],
                    np.asarray(positive_amount)[positive],
                )
                if np.any(positive)
                else None
            ),
        },
        "household": _household_metrics(
            household_codes,
            eligibility,
            np.asarray(expected),
            int(plane.manifest["household_group_count"]),
        ),
        "fold_counts": {
            str(fold): int(np.sum(fold_ids == fold))
            for fold in range(plane.n_splits)
        },
        "measurement_mode": True,
        "forecasting_authorized": False,
        "exceptional_periods": dict(EXCEPTIONAL_PERIODS),
        "execution_plane": "c6_resource_safe_v1",
        "labor_mode": labor_mode,
    }
    (work_root / "metrics.json").write_text(
        _canonical_json(metrics), encoding="utf-8"
    )
    (work_root / "time_layer.json").write_text(
        _canonical_json(time_reports), encoding="utf-8"
    )
    (work_root / "resolved_config.json").write_text(
        _canonical_json(config.raw), encoding="utf-8"
    )
    (work_root / "parents.json").write_text(
        _canonical_json(plane.manifest["parents"]), encoding="utf-8"
    )
    resource_record = {
        "peak_rss_gib": _peak_rss_gib(),
        "current_rss_gib": _current_rss_gib(),
        "current_process_swap_gib": _current_swap_gib(),
        "checkpoint_resources": checkpoint_records,
        "acceptance_target_peak_rss_gib": 10.0,
        "acceptance_target_no_swap_thrashing": True,
    }
    (work_root / "resource.json").write_text(
        _canonical_json(resource_record), encoding="utf-8"
    )

    artifacts: dict[str, Any] = {}
    for name in (
        "p_positive.npy",
        "positive_amount.npy",
        "expected_income.npy",
        "metrics.json",
        "time_layer.json",
        "resolved_config.json",
        "parents.json",
        "resource.json",
    ):
        path = work_root / name
        artifacts[name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "contract": C6_RUN_CONTRACT,
        "run_id": _run_id(plane, config, labor_mode),
        "arm": "L10",
        "execution_plane": "c6_resource_safe_v1",
        "model_plane_release_id": plane.release_id,
        "model_plane_manifest_sha256": plane.manifest_sha256,
        "parents": plane.manifest["parents"],
        "config_digest": config.digest,
        "feature_profile_id": config.feature_profile_id,
        "feature_names": list(plane.selected_feature_names(labor_mode)),
        "labor_mode": labor_mode,
        "target": config.target_field,
        "time_layer": "explicit_year_quarter_exception_v1",
        "nested_base_prediction_policy": "inner_household_safe_oof",
        "fold_policy": FOLD_POLICY,
        "n_splits": plane.n_splits,
        "row_count": row_count,
        "row_identity_sequence_sha256": plane.manifest[
            "row_identity_sequence_sha256"
        ],
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
        "measurement_mode": True,
        "forecasting_authorized": False,
        "restartable_outer_fold_checkpoints": True,
        "checkpoints": checkpoint_records,
        "artifacts": artifacts,
    }
    (work_root / "run_manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    work_manifest = work_root / "work_manifest.json"
    if work_manifest.exists():
        work_manifest.unlink()
    os.replace(work_root, destination)
    return destination


def run_resource_safe_l10(
    plane: LongitudinalModelPlane,
    config: LongitudinalConfig,
    output_root: Path,
    *,
    labor_mode: str = "national_regional",
    outer_fold: int | None = None,
) -> Path:
    """Run missing L10 outer folds sequentially and finalize only when complete."""
    if config.arm != "L10":
        raise C6ExecutionError("c6_first_executor_supports_l10_only")
    if config.feature_profile_id != plane.manifest.get("profile_id"):
        raise C6ExecutionError("c6_run_profile_mismatch")
    if config.n_splits != plane.n_splits:
        raise C6ExecutionError("c6_run_fold_count_mismatch")
    if labor_mode not in LABOR_MODES:
        raise C6ExecutionError(f"unknown_labor_mode:{labor_mode}")
    run_id = _run_id(plane, config, labor_mode)
    work_root, destination = _prepare_work_root(
        output_root, run_id, plane, config, labor_mode
    )
    if work_root == destination:
        return destination

    folds = (
        (outer_fold,)
        if outer_fold is not None
        else tuple(range(plane.n_splits))
    )
    for fold in folds:
        if fold < 0 or fold >= plane.n_splits:
            raise C6ExecutionError(f"c6_outer_fold_out_of_range:{fold}")
        _execute_outer_fold(plane, config, labor_mode, fold, work_root)

    if outer_fold is not None:
        return work_root
    return _finalize_l10(
        plane,
        config,
        labor_mode,
        work_root,
        destination,
    )


def compare_c6_runs(roots: Sequence[Path]) -> dict[str, Any]:
    if len(roots) < 2:
        raise C6ExecutionError("c6_compare_requires_two_runs")
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for value in roots:
        root = Path(value).expanduser().resolve()
        try:
            manifest = json.loads(
                (root / "run_manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise C6ExecutionError(f"c6_run_unreadable:{root}") from exc
        if manifest.get("contract") != C6_RUN_CONTRACT:
            raise C6ExecutionError("c6_compare_contract_invalid")
        records.append((manifest, metrics))

    cohorts = {record[0]["row_identity_sequence_sha256"] for record in records}
    folds = {record[0]["fold_ids_sha256"] for record in records}
    if len(cohorts) != 1:
        raise C6ExecutionError("c6_compare_cohort_mismatch")
    if len(folds) != 1:
        raise C6ExecutionError("c6_compare_fold_mismatch")

    runs = []
    for manifest, metrics in records:
        person = metrics["person"]["unconditional"]["point"]
        household = metrics["household"]["point"]["point"]
        runs.append(
            {
                "run_id": manifest["run_id"],
                "feature_profile_id": manifest["feature_profile_id"],
                "labor_mode": manifest["labor_mode"],
                "person_mae": person["mae"],
                "person_rmse": person["rmse"],
                "household_mae": household["mae"],
                "household_rmse": household["rmse"],
            }
        )
    return {
        "contract": "research.encuestador-longitudinal-c6-comparison/v1",
        "matched_row_identity_sequence": True,
        "matched_fold_assignments": True,
        "runs": runs,
        "promotion_authorized": False,
    }
