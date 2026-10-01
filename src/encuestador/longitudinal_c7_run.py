"""C7A: restartable matched C7-0 (historical-X) versus C7-1 (L11).

Both arms receive the same donor-clock composition, later aggregate context,
elapsed gap, target and household-group folds.  The single increment in C7-1
is observed earlier labor.  The C6 fitting/time primitives are reused.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import distributional_regression_diagnostics
from .longitudinal_c6 import (
    FOLD_POLICY,
    _binary_presence_diagnostics,
    _canonical_json,
    _current_rss_gib,
    _current_swap_gib,
    _dense_rows,
    _hurdle_estimator,
    _nested_hurdle_oof,
    _peak_rss_gib,
    _sha256,
)
from .longitudinal_c7 import C7_RUN_CONTRACT, C7PanelPlane
from .longitudinal_c7_contract import C7_COMPOSITION_PROFILE, C7ContractError
from .longitudinal_runtime import EXCEPTIONAL_PERIODS, LongitudinalConfig
from .time_layer import fit_time_layer

ARMS = ("C7-0", "C7-1")
FILE_PREFIXES = {"C7-0": "c7_0", "C7-1": "c7_1"}


def resolve_c7_config(config: LongitudinalConfig, plane: C7PanelPlane) -> LongitudinalConfig:
    if config.arm != "L11" or config.composition_source != "canonical_parent":
        raise C7ContractError("c7_requires_l11_canonical_config")
    if (config.feature_profile_id != C7_COMPOSITION_PROFILE
            or config.feature_profile_id != plane.manifest["feature_profile_id"]):
        raise C7ContractError("c7_composition_profile_drift")
    if config.n_splits != plane.n_splits:
        raise C7ContractError("c7_config_fold_count_mismatch")
    if config.target_field != "P47T_real" or config.anchor_enabled:
        raise C7ContractError("c7_target_or_anchor_policy_invalid")
    if tuple(config.allowed_panel_gaps) != (1, 3):
        raise C7ContractError("c7_allowed_gaps_changed")
    if (
        config.stale_labor_field != "stale_labor_state"
        or config.current_labor_field != "target_current_labor_state"
        or config.elapsed_quarters_field != "elapsed_quarters"
    ):
        raise C7ContractError("c7_labour_column_semantics_changed")
    return replace(
        config,
        composition_features=tuple(plane.manifest["composition_features"]),
        categorical_features=tuple(plane.manifest["categorical_features"]),
    )


def _identity(plane: C7PanelPlane, config: LongitudinalConfig) -> dict[str, Any]:
    return {
        "contract": C7_RUN_CONTRACT,
        "plane_release_id": plane.root.name,
        "plane_manifest_sha256": plane.manifest_sha256,
        "config_digest": config.digest,
        "row_identity_sequence_sha256": plane.manifest["row_identity_sequence_sha256"],
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
        "estimation_policy": "c7_pooled_1q3q_matched_nested_time_v1",
    }


def _run_id(identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(identity)).encode()).hexdigest()[:16]
    return "longitudinal-c7-matched-l11-" + digest


def _checkpoint_identity(
    identity: Mapping[str, Any], fold: int
) -> dict[str, Any]:
    return {
        "schema": "research.encuestador-c7-fold-checkpoint/v1",
        "run_id": _run_id(identity),
        "plane_manifest_sha256": identity["plane_manifest_sha256"],
        "config_digest": identity["config_digest"],
        "fold_ids_sha256": identity["fold_ids_sha256"],
        "outer_fold": fold,
        "arms": list(ARMS),
    }


def _checkpoint_valid(directory: Path, expected: Mapping[str, Any]) -> bool:
    file = directory / "checkpoint.json"
    if not file.is_file():
        return False
    try:
        record = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if any(record.get(key) != value for key, value in expected.items()):
        return False
    for name in (
        "indices.npy",
        "c7_0_p_positive.npy", "c7_0_positive_amount.npy",
        "c7_1_p_positive.npy", "c7_1_positive_amount.npy",
    ):
        path = directory / name
        if not path.is_file() or _sha256(path) != (
            (record.get("artifacts") or {}).get(name) or {}
        ).get("sha256"):
            return False
    return True


def _write_fold(
    work: Path, identity: Mapping[str, Any], fold: int,
    indices: np.ndarray,
    outputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    time_reports: list[dict[str, Any]],
) -> Path:
    destination = work / "checkpoints" / f"fold_{fold}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _checkpoint_identity(identity, fold)
    if destination.exists():
        if _checkpoint_valid(destination, expected):
            return destination
        raise C7ContractError(f"c7_fold_checkpoint_conflict:{fold}")
    staging = Path(tempfile.mkdtemp(prefix=f".fold_{fold}.", dir=destination.parent))
    try:
        np.save(staging / "indices.npy", np.asarray(indices, dtype=np.int64))
        for arm in ARMS:
            prefix = FILE_PREFIXES[arm]
            probability, amount = outputs[arm]
            np.save(staging / f"{prefix}_p_positive.npy", probability)
            np.save(staging / f"{prefix}_positive_amount.npy", amount)
        names = (
            "indices.npy",
            "c7_0_p_positive.npy", "c7_0_positive_amount.npy",
            "c7_1_p_positive.npy", "c7_1_positive_amount.npy",
        )
        record = {
            **expected,
            "holdout_rows": len(indices),
            "time_layers": time_reports,
            "resource": {
                "peak_rss_gib": _peak_rss_gib(),
                "current_rss_gib": _current_rss_gib(),
                "current_process_swap_gib": _current_swap_gib(),
            },
            "artifacts": {
                name: {"sha256": _sha256(staging / name)}
                for name in names
            },
        }
        (staging / "checkpoint.json").write_text(
            _canonical_json(record), encoding="utf-8"
        )
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _execute_fold(
    plane: C7PanelPlane,
    config: LongitudinalConfig,
    identity: Mapping[str, Any],
    work: Path,
    fold: int,
) -> Path:
    directory = work / "checkpoints" / f"fold_{fold}"
    expected = _checkpoint_identity(identity, fold)
    if _checkpoint_valid(directory, expected):
        return directory
    if directory.exists():
        raise C7ContractError(f"c7_fold_checkpoint_conflict:{fold}")
    x = np.load(plane.root / "features.npy", mmap_mode="r")
    y = np.load(plane.root / "target.npy", mmap_mode="r")
    folds = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    times = np.load(plane.root / "period_codes.npy", mmap_mode="r")
    train = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    if not len(train) or not len(test):
        raise C7ContractError(f"c7_empty_outer_fold:{fold}")
    nbase = len(plane.baseline_features)
    labels = np.asarray(plane.manifest["period_labels"], dtype="<U7")
    y_train = np.asarray(y[train], dtype=float)
    if not np.isfinite(y_train).all() or (y_train < 0).any():
        raise C7ContractError("c7_invalid_model_target")
    train_folds = np.asarray(folds[train], dtype=int)
    train_times = labels[np.asarray(times[train], dtype=int)]
    test_times = labels[np.asarray(times[test], dtype=int)]
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    time_reports: list[dict[str, Any]] = []
    for arm in ARMS:
        is_l11 = arm == "C7-1"
        columns = tuple(range(nbase + int(is_l11)))
        feature_names = plane.feature_names if is_l11 else plane.baseline_features
        x_train = _dense_rows(x, train, columns)
        x_test = _dense_rows(x, test, columns)
        # The C6 nested helper keeps the same categorical positions for the
        # original composition columns, adding stale labor only in C7-1.
        p_oof, amount_oof, oof_receipt = _nested_hurdle_oof(
            x_train, y_train, train_folds, config, feature_names,
            include_stale=is_l11,
        )
        time_fit = fit_time_layer(
            train_times, p_oof, amount_oof, y_train,
            exception_policy=EXCEPTIONAL_PERIODS,
        )
        estimator = _hurdle_estimator(
            config, feature_names, include_stale=is_l11,
        )
        estimator.fit(x_train, y_train)
        predicted = estimator.predict_components(x_test)
        p, amount = time_fit.apply(
            test_times, predicted.p_positive, predicted.positive_amount,
        )
        if (
            len(p) != len(test) or len(amount) != len(test)
            or not np.isfinite(p).all() or not np.isfinite(amount).all()
        ):
            raise C7ContractError(f"c7_nonfinite_fold_predictions:{fold}:{arm}")
        outputs[arm] = (np.asarray(p, dtype=float), np.asarray(amount, dtype=float))
        time_reports.append({
            "arm": arm, "outer_fold": fold, "feature_names": list(feature_names),
            "base_prediction_evidence": oof_receipt,
            **time_fit.as_dict(),
        })
        del x_train, x_test, p_oof, amount_oof, estimator, predicted, time_fit
        gc.collect()
    result = _write_fold(work, identity, fold, test, outputs, time_reports)
    del x, y, folds, times
    gc.collect()
    return result


def _point_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(y) < 2:
        return {"n": len(y), "mae": float(np.mean(np.abs(y - predicted)))}
    return {
        "n": len(y),
        **distributional_regression_diagnostics(y, predicted),
    }


def _selected_household_metrics(
    group: np.ndarray, y: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    # Target-household codes group only SELECTED Gate-B-linked members, not
    # all surveyed residents. Never claim complete household income.
    _, inverse = np.unique(group, return_inverse=True)
    truth = np.bincount(inverse, weights=y)
    fitted = np.bincount(inverse, weights=predicted)
    return {
        "aggregation_scope": "eligible_panel_pair_members_within_target_household",
        "full_household_welfare_claim": False,
        "selected_household_group_count": len(truth),
        "point": _point_metrics(truth, fitted),
    }


def _finish(
    plane: C7PanelPlane,
    config: LongitudinalConfig,
    identity: Mapping[str, Any],
    work: Path,
    destination: Path,
) -> Path:
    if destination.exists():
        return destination
    n = plane.row_count
    arrays = {
        f"{prefix}_{quantity}": np.lib.format.open_memmap(
            work / f"{prefix}_{quantity}.npy",
            mode="w+", dtype=np.float64, shape=(n,),
        )
        for prefix in ("c7_0", "c7_1")
        for quantity in ("p_positive", "positive_amount", "expected_income")
    }
    for item in arrays.values():
        item[:] = np.nan
    reports: list[dict[str, Any]] = []
    checkpoints = []
    for fold in range(plane.n_splits):
        root = work / "checkpoints" / f"fold_{fold}"
        expected = _checkpoint_identity(identity, fold)
        if not _checkpoint_valid(root, expected):
            raise C7ContractError(f"c7_checkpoint_missing_or_invalid:{fold}")
        indices = np.load(root / "indices.npy", mmap_mode="r")
        for arm, prefix in FILE_PREFIXES.items():
            for quantity in ("p_positive", "positive_amount"):
                arrays[f"{prefix}_{quantity}"][indices] = np.load(
                    root / f"{prefix}_{quantity}.npy", mmap_mode="r"
                )
        record = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        reports.extend(record["time_layers"])
        checkpoints.append({
            "outer_fold": fold, "holdout_rows": len(indices),
            "checkpoint_sha256": _sha256(root / "checkpoint.json"),
            "resource": record["resource"],
        })
    y = np.load(plane.root / "target.npy", mmap_mode="r")
    folds = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    gap = np.load(plane.root / "gap.npy", mmap_mode="r")
    stale = np.load(plane.root / "stale_labor.npy", mmap_mode="r")
    group = np.load(plane.root / "target_household_codes.npy", mmap_mode="r")
    metrics: dict[str, Any] = {
        "target": "later_P47T_real",
        "row_count": n, "fold_policy": FOLD_POLICY,
        "household_scope": "selected_panel_pair_members_only",
        "arms": {},
    }
    estimates = {}
    for arm, prefix in FILE_PREFIXES.items():
        probability = arrays[f"{prefix}_p_positive"]
        amount = arrays[f"{prefix}_positive_amount"]
        if not np.isfinite(probability).all() or not np.isfinite(amount).all():
            raise C7ContractError(f"c7_prediction_incomplete:{arm}")
        arrays[f"{prefix}_expected_income"][:] = probability * amount
        if not np.isfinite(arrays[f"{prefix}_expected_income"]).all():
            raise C7ContractError(f"c7_expected_income_invalid:{arm}")
        predicted = np.asarray(arrays[f"{prefix}_expected_income"])
        estimates[arm] = predicted
        positive = y > 0
        metrics["arms"][arm] = {
            "person_unconditional": _point_metrics(y, predicted),
            "presence": _binary_presence_diagnostics(
                positive.astype(int), np.asarray(probability)
            ),
            "positive_amount": _point_metrics(y[positive], np.asarray(amount)[positive]),
            "selected_household_groups": _selected_household_metrics(
                group, y, predicted
            ),
            "feature_names": list(
                plane.feature_names if arm == "C7-1" else plane.baseline_features
            ),
        }
    base, l11 = estimates["C7-0"], estimates["C7-1"]
    error_gain = np.abs(y - base) - np.abs(y - l11)
    metrics["paired"] = {
        "person_mae_gain_positive_favors_L11": float(np.mean(error_gain)),
        "person_rmse_gain_positive_favors_L11": float(
            np.sqrt(np.mean((y - base)**2)) - np.sqrt(np.mean((y - l11)**2))
        ),
        "foldwise": [],
        "gap_by_stale_labor": [],
        "gap_only": [],
        "stale_labor_only": [],
    }
    for fold in range(plane.n_splits):
        selection = np.asarray(folds == fold)
        metrics["paired"]["foldwise"].append({
            "outer_fold": fold, "n": int(selection.sum()),
            "mae_gain": float(error_gain[selection].mean()),
        })
    for values, axis in ((gap, "gap_only"), (stale, "stale_labor_only")):
        for value in sorted(np.unique(values)):
            selection = np.asarray(values == value)
            metrics["paired"][axis].append({
                "value": int(value), "n": int(selection.sum()),
                "baseline_mae": float(np.abs(y[selection] - base[selection]).mean()),
                "l11_mae": float(np.abs(y[selection] - l11[selection]).mean()),
                "mae_gain": float(error_gain[selection].mean()),
            })
    for g in (1, 3):
        for state in (1, 2, 3):
            selection = np.asarray((gap == g) & (stale == state))
            if selection.any():
                metrics["paired"]["gap_by_stale_labor"].append({
                    "gap": g, "stale_labor_state": state, "n": int(selection.sum()),
                    "baseline_mae": float(np.abs(y[selection] - base[selection]).mean()),
                    "l11_mae": float(np.abs(y[selection] - l11[selection]).mean()),
                    "mae_gain": float(error_gain[selection].mean()),
                })
    for array in arrays.values():
        array.flush()
    (work / "metrics.json").write_text(_canonical_json(metrics), encoding="utf-8")
    (work / "time_layer.json").write_text(_canonical_json(reports), encoding="utf-8")
    (work / "resolved_config.json").write_text(
        _canonical_json(config.raw), encoding="utf-8"
    )
    resource = {
        "peak_rss_gib": _peak_rss_gib(),
        "current_rss_gib": _current_rss_gib(),
        "current_process_swap_gib": _current_swap_gib(),
        "acceptance_target_peak_rss_gib": 10.0,
        "checkpoint_resources": checkpoints,
    }
    (work / "resource.json").write_text(_canonical_json(resource), encoding="utf-8")
    names = (
        *(f"{prefix}_{qty}.npy" for prefix in ("c7_0", "c7_1")
          for qty in ("p_positive", "positive_amount", "expected_income")),
        "metrics.json", "time_layer.json", "resolved_config.json", "resource.json",
    )
    artifacts = {
        name: {"sha256": _sha256(work / name), "bytes": (work / name).stat().st_size}
        for name in names
    }
    manifest = {
        **dict(identity),
        "run_id": destination.name,
        "arm": "matched_C7_0_C7_1",
        "panel_release_id": plane.root.name,
        "panel_manifest_sha256": plane.manifest_sha256,
        "source_parents": {
            key: plane.manifest[key] for key in (
                "l2_release_id", "l2_manifest_sha256",
                "gate_b_release_id", "gate_b_receipt_sha256",
                "c6_release_id", "c6_manifest_sha256",
            )
        },
        "row_count": n, "n_splits": plane.n_splits,
        "features_baseline": list(plane.baseline_features),
        "features_l11": list(plane.feature_names),
        "matched_rows_folds_target": True,
        "time_layer": "explicit_year_quarter_exception_v1",
        "nested_base_prediction_policy": "inner_household_safe_oof",
        "selection": "valid_later_income_after_exact_gate_b",
        "household_scope": "eligible_panel_pair_members_within_target_household",
        "full_household_welfare_claim": False,
        "measurement_mode": True, "forecasting_authorized": False,
        "long_horizon_census_transport_authorized": False,
        "scientific_promotion_authorized": False,
        "restartable_outer_fold_checkpoints": True,
        "checkpoints": checkpoints,
        "artifacts": artifacts,
    }
    (work / "run_manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    (work / "work_manifest.json").unlink()
    del arrays
    gc.collect()
    os.replace(work, destination)
    return destination


def run_resource_safe_c7(
    plane: C7PanelPlane,
    config: LongitudinalConfig,
    output_root: Path,
    *,
    outer_fold: int | None = None,
) -> Path:
    """Sequential, resumable matched comparison; no external Census deployment."""
    config = resolve_c7_config(config, plane)
    identity = _identity(plane, config)
    run_id = _run_id(identity)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        existing = json.loads(
            (destination / "run_manifest.json").read_text(encoding="utf-8")
        )
        if any(existing.get(k) != v for k, v in identity.items()):
            raise C7ContractError("c7_existing_run_identity_conflict")
        return destination
    work = output_root / f".{run_id}.work"
    work_identity = {"schema": "research.encuestador-c7-work/v1", **identity}
    if work.exists():
        try:
            current = json.loads(
                (work / "work_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise C7ContractError("c7_work_manifest_invalid") from exc
        if current != work_identity:
            raise C7ContractError("c7_work_identity_conflict")
    else:
        work.mkdir()
        (work / "work_manifest.json").write_text(
            _canonical_json(work_identity), encoding="utf-8"
        )
    folds = (outer_fold,) if outer_fold is not None else tuple(range(plane.n_splits))
    for fold in folds:
        if fold < 0 or fold >= plane.n_splits:
            raise C7ContractError(f"c7_outer_fold_out_of_range:{fold}")
        _execute_fold(plane, config, identity, work, fold)
    if outer_fold is not None:
        return work
    return _finish(plane, config, identity, work, destination)
