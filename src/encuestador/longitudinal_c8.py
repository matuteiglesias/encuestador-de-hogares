"""C8 transition science: train-only empirical T0 and conditional T1.

Only observed 1Q/3Q C7 pairs are supported. E/U/I order is immutable (1/2/3).
No global Gate-B transition probabilities may enter any OOF model.
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

from .longitudinal_c6 import (
    FOLD_POLICY,
    _canonical_json,
    _current_rss_gib,
    _current_swap_gib,
    _peak_rss_gib,
    _sha256,
)
from .longitudinal_c7 import C7PanelPlane
from .longitudinal_c7_contract import C7_COMPOSITION_PROFILE, C7ContractError
from .longitudinal_runtime import LongitudinalConfig, _transition_factory

C8_TRANSITION_CONTRACT = "research.encuestador-c8-raw-transition/v1"
C8_WELFARE_CONTRACT = "research.encuestador-c8-raw-l12-welfare/v1"
CLASS_ORDER = ("1", "2", "3")
PROBABILITY_FEATURES = (
    "latent.current_labor.p[1]",
    "latent.current_labor.p[2]",
    "latent.current_labor.p[3]",
)
T0_PRIOR_STRENGTH = 3.0
T1_PROBABILITY_FLOOR = 1e-9


def resolve_c8_config(
    config: LongitudinalConfig, plane: C7PanelPlane,
) -> LongitudinalConfig:
    if config.arm != "L12" or config.composition_source != "canonical_parent":
        raise C7ContractError("c8_requires_canonical_l12_config")
    if (
        config.feature_profile_id != C7_COMPOSITION_PROFILE
        or plane.manifest["feature_profile_id"] != C7_COMPOSITION_PROFILE
        or config.n_splits != plane.n_splits
    ):
        raise C7ContractError("c8_c7_profile_or_fold_mismatch")
    if config.anchor_enabled:
        raise C7ContractError("c8_raw_transition_must_not_anchor")
    if config.target_field != "P47T_real" or tuple(config.allowed_panel_gaps) != (1, 3):
        raise C7ContractError("c8_target_or_gap_contract_changed")
    if (
        config.stale_labor_field != "stale_labor_state"
        or config.current_labor_field != "target_current_labor_state"
        or config.elapsed_quarters_field != "elapsed_quarters"
    ):
        raise C7ContractError("c8_labor_state_fields_changed")
    if tuple(plane.feature_names[-2:]) != ("elapsed_quarters", "stale_labor_state"):
        raise C7ContractError("c8_c7_feature_clock_changed")
    if plane.manifest.get("fold_policy") != FOLD_POLICY:
        raise C7ContractError("c8_group_fold_policy_changed")
    return replace(
        config,
        composition_features=tuple(plane.manifest["composition_features"]),
        categorical_features=tuple(plane.manifest["categorical_features"]),
    )


def _normalize(p: np.ndarray) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    if (
        values.ndim != 2 or values.shape[1] != 3
        or not np.isfinite(values).all() or (values < 0).any()
    ):
        raise C7ContractError("c8_transition_probability_invalid")
    values = np.maximum(values, T1_PROBABILITY_FLOOR)
    values /= values.sum(axis=1, keepdims=True)
    if not np.allclose(values.sum(axis=1), 1, atol=1e-12):
        raise C7ContractError("c8_probability_not_normalized")
    return values


def _state_codes(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, (1, 2, 3)).all():
        raise C7ContractError("c8_later_state_outside_reviewed_classes")
    return values


def _pair_codes(features: np.ndarray) -> np.ndarray:
    pairs = np.asarray(features[:, -2:], dtype=float)
    if (
        pairs.ndim != 2 or pairs.shape[1] != 2
        or not np.isfinite(pairs).all()
        or not np.isin(pairs[:, 0], (1, 3)).all()
        or not np.isin(pairs[:, 1], (1, 2, 3)).all()
    ):
        raise C7ContractError("c8_stale_state_or_gap_invalid")
    return pairs.astype(np.int8)


def t0_empirical(
    fit_pairs: np.ndarray, fit_labels: np.ndarray, score_pairs: np.ndarray,
) -> np.ndarray:
    """P(later class | earlier class, gap), estimated on FIT ROWS ONLY.

    Dirichlet shrinkage to a training-only global prior stabilizes rare/unseen
    (earlier class, gap) cells without importing full Gate-B marginals.
    """
    fit_pair, score_pair = _pair_codes(fit_pairs), _pair_codes(score_pairs)
    y = _state_codes(fit_labels)
    if not len(y) or len(fit_pair) != len(y):
        raise C7ContractError("c8_t0_training_empty_or_misaligned")
    global_counts = np.bincount(y, minlength=4)[1:].astype(float)
    prior = (global_counts + 1.0) / (len(y) + 3.0)
    result = np.empty((len(score_pair), 3), dtype=float)
    for gap in (1, 3):
        for earlier in (1, 2, 3):
            scored = (score_pair[:, 0] == gap) & (score_pair[:, 1] == earlier)
            if not scored.any():
                continue
            source = (fit_pair[:, 0] == gap) & (fit_pair[:, 1] == earlier)
            counts = np.bincount(y[source], minlength=4)[1:].astype(float)
            result[scored] = (
                counts + T0_PRIOR_STRENGTH * prior
            ) / (int(source.sum()) + T0_PRIOR_STRENGTH)
    return _normalize(result)


def t1_conditional(
    fit_features: np.ndarray, fit_labels: np.ndarray,
    score_features: np.ndarray, config: LongitudinalConfig,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse C4's explicit categorical HGB transition factory and class order."""
    y = _state_codes(fit_labels)
    if len(y) != len(fit_features) or not len(y):
        raise C7ContractError("c8_t1_training_empty_or_misaligned")
    if len(np.unique(y)) < 2:
        return (
            t0_empirical(fit_features, y, score_features),
            {"model": "train_only_t0_fallback", "classes_seen": np.unique(y).tolist()},
        )
    model = _transition_factory(config, feature_names)()
    model.fit(np.asarray(fit_features, dtype=float), y.astype(str))
    scores = model.predict_proba(np.asarray(score_features, dtype=float))
    ordered = np.zeros((len(score_features), 3), dtype=float)
    seen = tuple(str(item) for item in model.classes_)
    for index, label in enumerate(CLASS_ORDER):
        if label in seen:
            ordered[:, index] = scores[:, seen.index(label)]
    return _normalize(ordered), {"model": "c4_hgb_multiclass", "classes_seen": list(seen)}


def crossfit_transitions(
    features: np.ndarray, labels: np.ndarray, group_fold: np.ndarray,
    config: LongitudinalConfig, feature_names: tuple[str, ...],
    *, include_t0: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, list[dict[str, Any]]]:
    """Internal OOF; the corresponding held-out labor label never trains its model.

    Input folds are inherited from C7 panel-household groups; no fresh
    person-level random split is introduced. Outer holdout is never passed in.
    """
    x = np.asarray(features, dtype=float)
    y = _state_codes(labels)
    folds = np.asarray(group_fold, dtype=int)
    if len(x) != len(y) or len(y) != len(folds):
        raise C7ContractError("c8_transition_training_vector_mismatch")
    distinct = sorted(set(folds.tolist()))
    if len(distinct) < 2:
        raise C7ContractError("c8_crossfit_needs_two_household_group_folds")
    t0 = np.full((len(x), 3), np.nan) if include_t0 else None
    t1 = np.full((len(x), 3), np.nan)
    trace: list[dict[str, Any]] = []
    for fold in distinct:
        train, test = folds != fold, folds == fold
        if not train.any() or not test.any():
            raise C7ContractError(f"c8_inner_fold_empty:{fold}")
        if include_t0:
            assert t0 is not None
            t0[test] = t0_empirical(x[train], y[train], x[test])
        t1[test], model = t1_conditional(
            x[train], y[train], x[test], config, feature_names
        )
        trace.append({
            "inner_fold": fold, "fit_rows": int(train.sum()),
            "score_rows": int(test.sum()), "group_overlap": 0,
            "fitted_current_labor_labels_from_scored_rows": 0,
            **model,
        })
    _normalize(t1)
    if t0 is not None:
        _normalize(t0)
    return t0, t1, trace


def transition_metrics(
    labels: np.ndarray, probabilities: np.ndarray,
    *, calibration_bins: int = 10,
) -> dict[str, Any]:
    """Multiclass Brier is mean sum over THREE squared class errors."""
    y = _state_codes(labels)
    p = _normalize(np.asarray(probabilities, dtype=float).copy())
    if not len(y) or len(y) != len(p):
        raise C7ContractError("c8_transition_metric_length_invalid")
    indices = y - 1
    one_hot = np.eye(3)[indices]
    classified = p.argmax(axis=1) + 1
    result: dict[str, Any] = {
        "n": len(y), "class_order": list(CLASS_ORDER),
        "log_loss": float(-np.log(p[np.arange(len(y)), indices]).mean()),
        "multiclass_brier": float(np.mean(np.sum((p - one_hot)**2, axis=1))),
        "accuracy": float((classified == y).mean()),
        "classes": {},
    }
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for k, label in enumerate(CLASS_ORDER):
        truth = y == (k + 1)
        bins = []
        for j in range(calibration_bins):
            mask = ((p[:, k] >= edges[j]) &
                    ((p[:, k] <= edges[j + 1]) if j == calibration_bins - 1
                     else (p[:, k] < edges[j + 1])))
            if mask.any():
                bins.append({
                    "bin": j, "n": int(mask.sum()),
                    "mean_probability": float(p[mask, k].mean()),
                    "observed_fraction": float(truth[mask].mean()),
                })
        result["classes"][label] = {
            "support": int(truth.sum()),
            "recall": float((classified[truth] == (k + 1)).mean())
            if truth.any() else None,
            "mean_probability": float(p[:, k].mean()),
            "calibration": bins,
        }
    return result


def _identity(plane: C7PanelPlane, config: LongitudinalConfig) -> dict[str, Any]:
    return {
        "contract": C8_TRANSITION_CONTRACT,
        "panel_release_id": plane.root.name,
        "panel_manifest_sha256": plane.manifest_sha256,
        "config_digest": config.digest,
        "row_identity_sequence_sha256": plane.manifest["row_identity_sequence_sha256"],
        "fold_ids_sha256": plane.manifest["fold_ids_sha256"],
        "class_order": list(CLASS_ORDER),
        "transition_policy": "outer_group_oof_train_only_T0_T1_v1",
        "T0_prior_strength": T0_PRIOR_STRENGTH,
    }


def _run_id(identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(identity)).encode()).hexdigest()[:16]
    return "longitudinal-c8-raw-transition-" + digest


def _fold_identity(identity: Mapping[str, Any], outer_fold: int) -> dict[str, Any]:
    return {
        "schema": "research.encuestador-c8-transition-fold/v1",
        "run_id": _run_id(identity),
        "panel_manifest_sha256": identity["panel_manifest_sha256"],
        "config_digest": identity["config_digest"],
        "fold_ids_sha256": identity["fold_ids_sha256"],
        "outer_fold": outer_fold,
    }


def _fold_valid(root: Path, expected: Mapping[str, Any]) -> bool:
    try:
        record = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if any(record.get(k) != v for k, v in expected.items()):
        return False
    return all(
        (root / name).is_file()
        and _sha256(root / name) == (record.get("artifacts", {}).get(name) or {}).get("sha256")
        for name in ("train_indices.npy", "holdout_indices.npy",
                     "t1_train_oof.npy", "t0_holdout.npy", "t1_holdout.npy")
    )


def _execute_fold(
    plane: C7PanelPlane, config: LongitudinalConfig,
    identity: Mapping[str, Any], work: Path, fold: int,
) -> Path:
    expected = _fold_identity(identity, fold)
    destination = work / "checkpoints" / f"fold_{fold}"
    if _fold_valid(destination, expected):
        return destination
    if destination.exists():
        raise C7ContractError(f"c8_transition_checkpoint_conflict:{fold}")
    x = np.load(plane.root / "features.npy", mmap_mode="r")
    y = np.load(plane.root / "target_labor.npy", mmap_mode="r")
    folds = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    train_idx, test_idx = (
        np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    )
    if not len(train_idx) or not len(test_idx):
        raise C7ContractError(f"c8_transition_empty_fold:{fold}")
    x_train, x_test = np.asarray(x[train_idx]), np.asarray(x[test_idx])
    y_train = _state_codes(y[train_idx])
    fold_train = np.asarray(folds[train_idx], dtype=int)
    t0_oof, t1_oof, trace = crossfit_transitions(
        x_train, y_train, fold_train, config, plane.feature_names,
    )
    assert t0_oof is not None
    p0 = t0_empirical(x_train, y_train, x_test)
    p1, model = t1_conditional(x_train, y_train, x_test, config, plane.feature_names)
    stage = Path(tempfile.mkdtemp(prefix=f".fold_{fold}.", dir=work / "checkpoints"))
    try:
        for name, value in {
            "train_indices.npy": train_idx, "holdout_indices.npy": test_idx,
            "t1_train_oof.npy": t1_oof,
            "t0_holdout.npy": p0, "t1_holdout.npy": p1,
        }.items():
            np.save(stage / name, value)
        record = {
            **expected, "train_rows": len(train_idx), "holdout_rows": len(test_idx),
            "inner_oof": trace, "outer_model": model,
            "outer_train_only": True, "outer_holdout_labor_labels_consumed": 0,
            "class_order": list(CLASS_ORDER),
            "resource": {
                "peak_rss_gib": _peak_rss_gib(),
                "current_rss_gib": _current_rss_gib(),
                "current_process_swap_gib": _current_swap_gib(),
            },
            "artifacts": {
                name: {"sha256": _sha256(stage / name)}
                for name in ("train_indices.npy", "holdout_indices.npy",
                             "t1_train_oof.npy", "t0_holdout.npy", "t1_holdout.npy")
            },
        }
        (stage / "checkpoint.json").write_text(
            _canonical_json(record), encoding="utf-8"
        )
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    del x_train, x_test, t0_oof, t1_oof, x, y
    gc.collect()
    return destination


def _finish(
    plane: C7PanelPlane, identity: Mapping[str, Any],
    work: Path, destination: Path,
) -> Path:
    n = plane.row_count
    q0 = np.lib.format.open_memmap(
        work / "t0_oof.npy", mode="w+", dtype=np.float64, shape=(n, 3)
    )
    q1 = np.lib.format.open_memmap(
        work / "t1_oof.npy", mode="w+", dtype=np.float64, shape=(n, 3)
    )
    q0[:], q1[:] = np.nan, np.nan
    checkpoints = []
    seen = np.zeros(n, dtype=np.uint8)
    for fold in range(plane.n_splits):
        root = work / "checkpoints" / f"fold_{fold}"
        if not _fold_valid(root, _fold_identity(identity, fold)):
            raise C7ContractError(f"c8_transition_missing_fold:{fold}")
        indices = np.load(root / "holdout_indices.npy", mmap_mode="r")
        if seen[indices].any():
            raise C7ContractError("c8_transition_holdout_reused")
        seen[indices] = 1
        q0[indices] = np.load(root / "t0_holdout.npy", mmap_mode="r")
        q1[indices] = np.load(root / "t1_holdout.npy", mmap_mode="r")
        checkpoints.append({
            "outer_fold": fold, "checkpoint_sha256": _sha256(root / "checkpoint.json"),
            "holdout_rows": len(indices),
        })
    if not seen.all():
        raise C7ContractError("c8_transition_holdout_incomplete")
    _normalize(q0)
    _normalize(q1)
    q0.flush()
    q1.flush()
    y = np.load(plane.root / "target_labor.npy", mmap_mode="r")
    gap = np.load(plane.root / "gap.npy", mmap_mode="r")
    stale = np.load(plane.root / "stale_labor.npy", mmap_mode="r")
    fold_ids = np.load(plane.root / "fold_ids.npy", mmap_mode="r")
    report = {
        "row_count": n, "class_order": list(CLASS_ORDER),
        "scope": "unweighted_selected_gate_b_labor_transition_pairs_with_valid_income",
        "T0": transition_metrics(y, q0),
        "T1": transition_metrics(y, q1),
        "gap_by_prior_labor": [],
        "foldwise": [],
        "conditional_minus_empirical_log_loss_gain": None,
    }
    report["conditional_minus_empirical_log_loss_gain"] = (
        report["T0"]["log_loss"] - report["T1"]["log_loss"]
    )
    for fold in range(plane.n_splits):
        sel = fold_ids == fold
        report["foldwise"].append({
            "outer_fold": fold,
            "T0": transition_metrics(y[sel], q0[sel]),
            "T1": transition_metrics(y[sel], q1[sel]),
        })
    for g in (1, 3):
        for state in (1, 2, 3):
            sel = (gap == g) & (stale == state)
            if sel.any():
                report["gap_by_prior_labor"].append({
                    "gap": g, "earlier_state": str(state),
                    "T0": transition_metrics(y[sel], q0[sel]),
                    "T1": transition_metrics(y[sel], q1[sel]),
                })
    (work / "transition_metrics.json").write_text(
        _canonical_json(report), encoding="utf-8"
    )
    artifacts = {
        name: {"sha256": _sha256(work / name), "bytes": (work / name).stat().st_size}
        for name in ("t0_oof.npy", "t1_oof.npy", "transition_metrics.json")
    }
    receipt = {
        **dict(identity),
        "run_id": destination.name,
        "panel_source_parents": {
            key: plane.manifest[key] for key in (
                "l2_release_id", "l2_manifest_sha256",
                "gate_b_release_id", "gate_b_receipt_sha256",
                "c6_release_id", "c6_manifest_sha256",
            )
        },
        "row_count": n, "n_splits": plane.n_splits,
        "fold_policy": FOLD_POLICY,
        "selection": "gate_b_plus_valid_later_income",
        "T0": "within_fold_dirichlet_shrunk_empirical_stale_and_gap",
        "T1": "C4_HGB_conditional_on_donor_X_stale_labor_gap_target_context",
        "raw_only": True, "anchored": False,
        "outer_holdout_labels_used_for_training": 0,
        "measurement_mode": True, "forecasting_authorized": False,
        "scientific_promotion_authorized": False,
        "long_horizon_census_transport_authorized": False,
        "checkpoints": checkpoints, "artifacts": artifacts,
        "resource": {"peak_rss_gib": _peak_rss_gib()},
    }
    (work / "transition_manifest.json").write_text(
        _canonical_json(receipt), encoding="utf-8"
    )
    (work / "work_manifest.json").unlink()
    os.replace(work, destination)
    return destination


def run_c8_transition(
    plane: C7PanelPlane, config: LongitudinalConfig,
    output_root: Path, *, outer_fold: int | None = None,
) -> Path:
    config = resolve_c8_config(config, plane)
    identity = _identity(plane, config)
    rid = _run_id(identity)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / rid
    if destination.exists():
        record = json.loads(
            (destination / "transition_manifest.json").read_text(encoding="utf-8")
        )
        if any(record.get(k) != v for k, v in identity.items()):
            raise C7ContractError("c8_transition_existing_identity_mismatch")
        for name, data in record["artifacts"].items():
            if _sha256(destination / name) != data["sha256"]:
                raise C7ContractError(f"c8_transition_existing_artifact_tamper:{name}")
        return destination
    work = output_root / f".{rid}.work"
    work_identity = {"schema": "research.encuestador-c8-transition-work/v1", **identity}
    if work.exists():
        record = json.loads(
            (work / "work_manifest.json").read_text(encoding="utf-8")
        )
        if record != work_identity:
            raise C7ContractError("c8_transition_work_identity_mismatch")
    else:
        work.mkdir()
        (work / "work_manifest.json").write_text(
            _canonical_json(work_identity), encoding="utf-8"
        )
        (work / "checkpoints").mkdir()
    selected = (
        (outer_fold,) if outer_fold is not None else tuple(range(plane.n_splits))
    )
    for fold in selected:
        if fold < 0 or fold >= plane.n_splits:
            raise C7ContractError(f"c8_transition_fold_invalid:{fold}")
        _execute_fold(plane, config, identity, work, fold)
    if outer_fold is not None:
        return work
    return _finish(plane, identity, work, destination)


def load_c8_transition(
    root: Path, plane: C7PanelPlane, config: LongitudinalConfig,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    try:
        receipt = json.loads(
            (root / "transition_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise C7ContractError("c8_transition_receipt_unavailable") from exc
    resolved = resolve_c8_config(config, plane)
    identity = _identity(plane, resolved)
    if root.name != _run_id(identity) or any(
        receipt.get(k) != v for k, v in identity.items()
    ):
        raise C7ContractError("c8_transition_wrong_c7_parent_or_config")
    for name, data in receipt.get("artifacts", {}).items():
        if _sha256(root / name) != data.get("sha256"):
            raise C7ContractError(f"c8_transition_artifact_hash_mismatch:{name}")
    if set(receipt.get("artifacts", {})) != {
        "t0_oof.npy", "t1_oof.npy", "transition_metrics.json"
    }:
        raise C7ContractError("c8_transition_artifact_inventory_invalid")
    for fold in range(plane.n_splits):
        checkpoint = root / "checkpoints" / f"fold_{fold}"
        if not _fold_valid(checkpoint, _fold_identity(identity, fold)):
            raise C7ContractError(f"c8_transition_fold_tampered:{fold}")
    for name in ("t0_oof.npy", "t1_oof.npy"):
        data = np.load(root / name, mmap_mode="r")
        if data.shape != (plane.row_count, 3):
            raise C7ContractError(f"c8_transition_shape_invalid:{name}")
        _normalize(data)
    return receipt
