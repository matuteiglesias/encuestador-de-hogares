"""C8 stage 2: nested raw-L12 probability-feature welfare against accepted C7.

A completed/verified C8 transition release AND completed/verified C7 matched
L11 run are required.  C8-1 adds three E/U/I *predicted* probabilities to
donor-time base composition + target aggregate context + gap; observed current
labor is never a terminal predictor.  All intermediate labor labels are
cross-fitted inside the relevant training set, including the time layer.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .longitudinal_c6 import (
    FOLD_POLICY,
    _binary_presence_diagnostics,
    _canonical_json,
    _current_rss_gib,
    _current_swap_gib,
    _hurdle_estimator,
    _peak_rss_gib,
    _sha256,
)
from .longitudinal_c7 import C7_RUN_CONTRACT, C7PanelPlane
from .longitudinal_c7_contract import C7ContractError
from .longitudinal_c7_run import _point_metrics, _selected_household_metrics
from .longitudinal_c8 import (
    C8_WELFARE_CONTRACT,
    PROBABILITY_FEATURES,
    CLASS_ORDER,
    _normalize,
    crossfit_transitions,
    load_c8_transition,
    resolve_c8_config,
    t1_conditional,
)
from .longitudinal_runtime import EXCEPTIONAL_PERIODS, LongitudinalConfig
from .time_layer import fit_time_layer


def _c7_comparator(
    root: Path, plane: C7PanelPlane, config: LongitudinalConfig,
) -> dict[str, Any]:
    """Exact C7 parent/metric identity, not an assumed available local run."""
    root = Path(root).expanduser().resolve()
    try:
        receipt = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        c7_config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise C7ContractError("c8_complete_c7_comparator_required") from exc
    if (
        receipt.get("contract") != C7_RUN_CONTRACT
        or receipt.get("run_id") != root.name
        or receipt.get("panel_release_id") != plane.root.name
        or receipt.get("panel_manifest_sha256") != plane.manifest_sha256
        or receipt.get("row_count") != plane.row_count
        or receipt.get("n_splits") != plane.n_splits
        or receipt.get("fold_ids_sha256") != plane.manifest["fold_ids_sha256"]
        or receipt.get("row_identity_sequence_sha256")
        != plane.manifest["row_identity_sequence_sha256"]
        or receipt.get("matched_rows_folds_target") is not True
        or receipt.get("full_household_welfare_claim") is not False
    ):
        raise C7ContractError("c8_c7_matched_parent_mismatch")
    if (
        c7_config.get("arm") != "L11"
        or c7_config.get("feature_profile_id") != config.feature_profile_id
        or c7_config.get("splits") != config.raw.get("splits")
        or (c7_config.get("estimators") or {}).get("presence")
        != (config.raw.get("estimators") or {}).get("presence")
        or (c7_config.get("estimators") or {}).get("positive_amount")
        != (config.raw.get("estimators") or {}).get("positive_amount")
    ):
        raise C7ContractError("c8_c7_estimator_or_cohort_policy_mismatch")
    required = {
        "c7_0_expected_income.npy", "c7_1_expected_income.npy",
        "c7_0_p_positive.npy", "c7_1_p_positive.npy",
        "metrics.json", "resolved_config.json",
    }
    inventory = receipt.get("artifacts") or {}
    if not required.issubset(inventory):
        raise C7ContractError("c8_c7_comparator_artifact_missing")
    for name, data in inventory.items():
        if not (root / name).is_file() or _sha256(root / name) != data.get("sha256"):
            raise C7ContractError(f"c8_c7_comparator_artifact_tamper:{name}")
    for name in ("c7_0_expected_income.npy", "c7_1_expected_income.npy"):
        if np.load(root / name, mmap_mode="r").shape != (plane.row_count,):
            raise C7ContractError(f"c8_c7_comparator_prediction_shape:{name}")
    return receipt


def _time_oof_nested(
    x: np.ndarray, label: np.ndarray, y: np.ndarray, fold: np.ndarray,
    config: LongitudinalConfig, transition_names: tuple[str, ...],
    terminal_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Strict nested stack for time layer: even inner holdout LABOR cannot
    enter a transition model that generated terminal-fit rows' inputs.

    Outer train is the only input. For each time-inner holdout, fit:
    1. a transition OOF INSIDE its 3-fold fit population;
    2. a full transition model on that same population to score holdout;
    3. terminal hurdle on the inner-fit OOF probabilities, score holdout.
    Thus no observed holdout labor or income influences predicted holdout
    components, directly or through labor features on fit rows.
    """
    groups = np.asarray(fold, dtype=int)
    truth = np.asarray(y, dtype=float)
    labels = np.asarray(label, dtype=np.int8)
    probability = np.full(len(x), np.nan)
    amount = np.full(len(x), np.nan)
    audit = []
    for inner in sorted(set(groups.tolist())):
        fit, held = groups != inner, groups == inner
        if len(np.unique(groups[fit])) < 2:
            raise C7ContractError("c8_nested_time_labor_needs_two_fit_folds")
        _, q_fit, trace = crossfit_transitions(
            x[fit], labels[fit], groups[fit], config, transition_names,
            include_t0=False,
        )
        q_held, model = t1_conditional(
            x[fit], labels[fit], x[held], config, transition_names,
        )
        fitted_features = np.column_stack((x[fit, :-1], q_fit))
        heldout_features = np.column_stack((x[held, :-1], q_held))
        estimator = _hurdle_estimator(
            config, terminal_names, include_stale=False,
        )
        estimator.fit(fitted_features, truth[fit])
        scored = estimator.predict_components(heldout_features)
        probability[held] = scored.p_positive
        amount[held] = scored.positive_amount
        audit.append({
            "inner_time_fold": inner,
            "time_fit_rows": int(fit.sum()), "time_holdout_rows": int(held.sum()),
            "inner_holdout_labor_or_income_used_in_fit": 0,
            "transition_training_within_time_fit_only": True,
            "transition_fit_oof": trace, "transition_holdout_model": model,
        })
        del q_fit, q_held, fitted_features, heldout_features, estimator, scored
        gc.collect()
    if not np.isfinite(probability).all() or not np.isfinite(amount).all():
        raise C7ContractError("c8_nested_time_oof_incomplete")
    return probability, amount, audit


def _identity(
    plane: C7PanelPlane, config: LongitudinalConfig,
    transition_root: Path, c7_root: Path,
) -> dict[str, Any]:
    return {
        "contract": C8_WELFARE_CONTRACT,
        "panel_release_id": plane.root.name,
        "panel_manifest_sha256": plane.manifest_sha256,
        "transition_release_id": transition_root.name,
        "transition_manifest_sha256": _sha256(transition_root / "transition_manifest.json"),
        "matched_c7_release_id": c7_root.name,
        "matched_c7_manifest_sha256": _sha256(c7_root / "run_manifest.json"),
        "config_digest": config.digest,
        "row_identity_sequence_sha256": plane.manifest["row_identity_sequence_sha256"],
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
        "terminal_features": list(plane.baseline_features) + list(PROBABILITY_FEATURES),
        "policy": "c8_raw_t1_nested_transition_and_nested_time_v1",
    }


def _run_id(identity: Mapping[str, Any]) -> str:
    return "longitudinal-c8-l12-welfare-" + hashlib.sha256(
        _canonical_json(dict(identity)).encode()
    ).hexdigest()[:16]


def _fold_identity(identity: Mapping[str, Any], fold: int) -> dict[str, Any]:
    return {
        "schema": "research.encuestador-c8-welfare-fold/v1",
        "run_id": _run_id(identity),
        "panel_manifest_sha256": identity["panel_manifest_sha256"],
        "transition_manifest_sha256": identity["transition_manifest_sha256"],
        "matched_c7_manifest_sha256": identity["matched_c7_manifest_sha256"],
        "config_digest": identity["config_digest"],
        "fold_ids_sha256": identity["fold_ids_sha256"],
        "outer_fold": fold,
    }


def _checkpoint_valid(root: Path, expected: Mapping[str, Any]) -> bool:
    try:
        info = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if any(info.get(k) != value for k, value in expected.items()):
        return False
    return all(
        (root / name).is_file()
        and _sha256(root / name) == (info.get("artifacts", {}).get(name) or {}).get("sha256")
        for name in ("indices.npy", "p_positive.npy", "positive_amount.npy")
    )


def _execute_fold(
    plane: C7PanelPlane, config: LongitudinalConfig,
    transition_root: Path, identity: Mapping[str, Any],
    work: Path, fold: int,
) -> Path:
    dest = work / "checkpoints" / f"fold_{fold}"
    expected = _fold_identity(identity, fold)
    if _checkpoint_valid(dest, expected):
        return dest
    if dest.exists():
        raise C7ContractError(f"c8_welfare_fold_identity_conflict:{fold}")
    transition_checkpoint = transition_root / "checkpoints" / f"fold_{fold}"
    train_idx = np.load(transition_checkpoint / "train_indices.npy", mmap_mode="r")
    test_idx = np.load(transition_checkpoint / "holdout_indices.npy", mmap_mode="r")
    features = np.load(plane.root / "features.npy", mmap_mode="r")
    target = np.load(plane.root / "target.npy", mmap_mode="r")
    labels = np.load(plane.root / "target_labor.npy", mmap_mode="r")
    folds = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    periods = np.load(plane.root / "period_codes.npy", mmap_mode="r")
    if (
        not len(train_idx) or not len(test_idx)
        or np.any(folds[train_idx] == fold)
        or np.any(folds[test_idx] != fold)
        or len(train_idx) + len(test_idx) != plane.row_count
    ):
        raise C7ContractError("c8_welfare_transition_fold_indices_drift")
    x_train = np.asarray(features[train_idx], dtype=float)
    x_test = np.asarray(features[test_idx], dtype=float)
    q_train = _normalize(np.load(
        transition_checkpoint / "t1_train_oof.npy", mmap_mode="r"
    ).copy())
    q_test = _normalize(np.load(
        transition_checkpoint / "t1_holdout.npy", mmap_mode="r"
    ).copy())
    if q_train.shape != (len(train_idx), 3) or q_test.shape != (len(test_idx), 3):
        raise C7ContractError("c8_welfare_transition_probability_shape_mismatch")
    y_train = np.asarray(target[train_idx], dtype=float)
    train_labor = np.asarray(labels[train_idx], dtype=np.int8)
    train_folds = np.asarray(folds[train_idx], dtype=int)
    if not np.isfinite(y_train).all() or (y_train < 0).any():
        raise C7ContractError("c8_welfare_invalid_target")
    train_x = np.column_stack((x_train[:, :-1], q_train))
    test_x = np.column_stack((x_test[:, :-1], q_test))
    terminal_names = tuple(identity["terminal_features"])
    # This is the categorical-handling correction to September Q4:
    # original composition category positions remain unchanged when
    # the three NUMERIC probabilities are appended.
    time_p, time_amount, time_audit = _time_oof_nested(
        x_train, train_labor, y_train, train_folds, config,
        plane.feature_names, terminal_names,
    )
    period_names = np.asarray(plane.manifest["period_labels"], dtype="<U7")
    time_fit = fit_time_layer(
        period_names[np.asarray(periods[train_idx], dtype=int)],
        time_p, time_amount, y_train,
        exception_policy=EXCEPTIONAL_PERIODS,
    )
    estimator = _hurdle_estimator(
        config, terminal_names, include_stale=False,
    )
    estimator.fit(train_x, y_train)
    scored = estimator.predict_components(test_x)
    p, amount = time_fit.apply(
        period_names[np.asarray(periods[test_idx], dtype=int)],
        scored.p_positive, scored.positive_amount,
    )
    if (
        len(p) != len(test_idx) or len(amount) != len(test_idx)
        or not np.isfinite(p).all() or not np.isfinite(amount).all()
    ):
        raise C7ContractError("c8_welfare_prediction_invalid")
    stage = Path(tempfile.mkdtemp(prefix=f".fold_{fold}.", dir=work / "checkpoints"))
    try:
        for name, value in {
            "indices.npy": test_idx,
            "p_positive.npy": p,
            "positive_amount.npy": amount,
        }.items():
            np.save(stage / name, value)
        info = {
            **expected, "holdout_rows": len(test_idx),
            "outer_holdout_current_labor_consumed_in_training": 0,
            "time_inner_holdout_current_labor_consumed_in_training": 0,
            "numeric_probability_features": list(PROBABILITY_FEATURES),
            "original_categorical_features_preserved": list(config.categorical_features),
            "time_layer": time_fit.as_dict(), "nested_time_evidence": time_audit,
            "resource": {
                "peak_rss_gib": _peak_rss_gib(),
                "current_rss_gib": _current_rss_gib(),
                "current_process_swap_gib": _current_swap_gib(),
            },
            "artifacts": {
                name: {"sha256": _sha256(stage / name)}
                for name in ("indices.npy", "p_positive.npy", "positive_amount.npy")
            },
        }
        (stage / "checkpoint.json").write_text(
            _canonical_json(info), encoding="utf-8"
        )
        os.replace(stage, dest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    del train_x, test_x, x_train, x_test, features, estimator, scored, time_fit
    gc.collect()
    return dest


def _finish(
    plane: C7PanelPlane, identity: Mapping[str, Any],
    work: Path, destination: Path, c7_root: Path, transition_root: Path,
) -> Path:
    n = plane.row_count
    p = np.lib.format.open_memmap(
        work / "p_positive.npy", mode="w+", dtype=float, shape=(n,)
    )
    amount = np.lib.format.open_memmap(
        work / "positive_amount.npy", mode="w+", dtype=float, shape=(n,)
    )
    p[:], amount[:] = np.nan, np.nan
    seen = np.zeros(n, dtype=np.uint8)
    checks = []
    for fold in range(plane.n_splits):
        root = work / "checkpoints" / f"fold_{fold}"
        if not _checkpoint_valid(root, _fold_identity(identity, fold)):
            raise C7ContractError(f"c8_welfare_fold_missing:{fold}")
        indices = np.load(root / "indices.npy", mmap_mode="r")
        if seen[indices].any():
            raise C7ContractError("c8_welfare_target_row_reused")
        seen[indices] = 1
        p[indices] = np.load(root / "p_positive.npy", mmap_mode="r")
        amount[indices] = np.load(root / "positive_amount.npy", mmap_mode="r")
        checks.append({
            "outer_fold": fold,
            "checkpoint_sha256": _sha256(root / "checkpoint.json"),
            "holdout_rows": len(indices),
        })
    if not seen.all() or not np.isfinite(p).all() or not np.isfinite(amount).all():
        raise C7ContractError("c8_welfare_oof_incomplete")
    expected = np.lib.format.open_memmap(
        work / "expected_income.npy", mode="w+", dtype=float, shape=(n,)
    )
    expected[:] = p * amount
    expected.flush()
    p.flush()
    amount.flush()
    y = np.load(plane.root / "target.npy", mmap_mode="r")
    g = np.load(plane.root / "gap.npy", mmap_mode="r")
    prior = np.load(plane.root / "stale_labor.npy", mmap_mode="r")
    folds = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    groups = np.load(plane.root / "target_household_codes.npy", mmap_mode="r")
    c7_0 = np.load(c7_root / "c7_0_expected_income.npy", mmap_mode="r")
    c7_1 = np.load(c7_root / "c7_1_expected_income.npy", mmap_mode="r")
    if not np.isfinite(expected).all():
        raise C7ContractError("c8_expected_income_nonfinite")
    valid_positive = y > 0
    report: dict[str, Any] = {
        "row_count": n, "welfare_target": "later_P47T_real",
        "full_household_welfare_claim": False,
        "C8-1": {
            "person_unconditional": _point_metrics(y, expected),
            "presence": _binary_presence_diagnostics(
                valid_positive.astype(int), p
            ),
            "positive_amount": _point_metrics(
                y[valid_positive], amount[valid_positive]
            ) if valid_positive.any() else None,
            "selected_panel_households": _selected_household_metrics(
                groups, y, expected
            ),
            "feature_names": identity["terminal_features"],
        },
        "paired": {},
        "foldwise": [],
        "gap_by_stale_state": [],
        "raw_transition_parent": str(transition_root.name),
    }
    for baseline, values in (("C7-0", c7_0), ("C7-1", c7_1)):
        delta = np.abs(y - values) - np.abs(y - expected)
        report["paired"][f"C8-1_vs_{baseline}"] = {
            "person_mae_gain_positive_favors_C8": float(delta.mean()),
            "person_rmse_gain_positive_favors_C8": float(
                np.sqrt(np.mean((y - values)**2))
                - np.sqrt(np.mean((y - expected)**2))
            ),
        }
    for fold in range(plane.n_splits):
        mask = folds == fold
        report["foldwise"].append({
            "outer_fold": fold, "n": int(mask.sum()),
            "c7_0_mae": float(np.abs(y[mask] - c7_0[mask]).mean()),
            "c7_1_mae": float(np.abs(y[mask] - c7_1[mask]).mean()),
            "c8_1_mae": float(np.abs(y[mask] - expected[mask]).mean()),
        })
    for gap in (1, 3):
        for state in (1, 2, 3):
            mask = (g == gap) & (prior == state)
            if mask.any():
                report["gap_by_stale_state"].append({
                    "gap": gap, "stale_labor": str(state), "n": int(mask.sum()),
                    "c7_0_mae": float(np.abs(y[mask] - c7_0[mask]).mean()),
                    "c7_1_mae": float(np.abs(y[mask] - c7_1[mask]).mean()),
                    "c8_1_mae": float(np.abs(y[mask] - expected[mask]).mean()),
                })
    (work / "welfare_metrics.json").write_text(
        _canonical_json(report), encoding="utf-8"
    )
    (work / "resource.json").write_text(_canonical_json({
        "peak_rss_gib": _peak_rss_gib(),
        "current_rss_gib": _current_rss_gib(),
        "current_process_swap_gib": _current_swap_gib(),
        "checkpoint_count": len(checks),
        "resource_target_peak_rss_gib": 10.0,
    }), encoding="utf-8")
    names = (
        "p_positive.npy", "positive_amount.npy", "expected_income.npy",
        "welfare_metrics.json", "resource.json",
    )
    artifacts = {
        name: {"sha256": _sha256(work / name), "bytes": (work / name).stat().st_size}
        for name in names
    }
    manifest = {
        **dict(identity), "run_id": destination.name,
        "arm": "C8-1_raw_L12_probability_features",
        "class_order": list(CLASS_ORDER),
        "row_count": n, "n_splits": plane.n_splits, "fold_policy": FOLD_POLICY,
        "matched_row_fold_and_target_c7": True,
        "nested_transition_meta_train": True,
        "nested_inner_time_transition": True,
        "categorical_feature_preservation": True,
        "true_current_labor_used_only_as_transition_target": True,
        "anchored": False, "measurement_mode": True,
        "forecasting_authorized": False, "scientific_promotion_authorized": False,
        "long_horizon_census_transport_authorized": False,
        "full_household_welfare_claim": False,
        "checkpoints": checks, "artifacts": artifacts,
    }
    (work / "run_manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    (work / "work_manifest.json").unlink()
    os.replace(work, destination)
    return destination


def run_c8_welfare(
    plane: C7PanelPlane, config: LongitudinalConfig,
    transition_root: Path, c7_run_root: Path,
    output_root: Path, *, outer_fold: int | None = None,
) -> Path:
    """Explicit C8 stage-two invocation: stage one never triggers it automatically."""
    config = resolve_c8_config(config, plane)
    transition_root = Path(transition_root).expanduser().resolve()
    c7_run_root = Path(c7_run_root).expanduser().resolve()
    load_c8_transition(transition_root, plane, config)
    _c7_comparator(c7_run_root, plane, config)
    identity = _identity(plane, config, transition_root, c7_run_root)
    rid = _run_id(identity)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / rid
    if destination.exists():
        info = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
        if any(info.get(k) != v for k, v in identity.items()):
            raise C7ContractError("c8_existing_welfare_run_identity_conflict")
        for name, metadata in (info.get("artifacts") or {}).items():
            if _sha256(destination / name) != metadata.get("sha256"):
                raise C7ContractError(f"c8_existing_welfare_artifact_tamper:{name}")
        return destination
    work = output_root / f".{rid}.work"
    work_identity = {"schema": "research.encuestador-c8-welfare-work/v1", **identity}
    if work.exists():
        info = json.loads((work / "work_manifest.json").read_text(encoding="utf-8"))
        if info != work_identity:
            raise C7ContractError("c8_welfare_work_identity_conflict")
    else:
        work.mkdir()
        (work / "checkpoints").mkdir()
        (work / "work_manifest.json").write_text(
            _canonical_json(work_identity), encoding="utf-8"
        )
    selected = (outer_fold,) if outer_fold is not None else tuple(range(plane.n_splits))
    for fold in selected:
        if fold < 0 or fold >= plane.n_splits:
            raise C7ContractError(f"c8_welfare_outer_fold_invalid:{fold}")
        _execute_fold(plane, config, transition_root, identity, work, fold)
    if outer_fold is not None:
        return work
    return _finish(plane, identity, work, destination, c7_run_root, transition_root)
