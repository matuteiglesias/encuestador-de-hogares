"""Cloud fixture proof for C8 transition, nested L12 and exact C7 comparison."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_longitudinal_c7 import fixture

from encuestador.longitudinal_c6 import (
    _household_fold,
    load_longitudinal_model_plane,
)
from encuestador.longitudinal_c7 import load_c7_panel, materialize_c7_panel
from encuestador.longitudinal_c7_contract import C7ContractError
from encuestador.longitudinal_c7_run import run_resource_safe_c7
from encuestador.longitudinal_c8 import (
    C8_TRANSITION_CONTRACT,
    CLASS_ORDER,
    _normalize,
    crossfit_transitions,
    load_c8_transition,
    resolve_c8_config,
    run_c8_transition,
    t0_empirical,
    transition_metrics,
)
from encuestador.longitudinal_c8_run import (
    _c7_comparator,
    _time_oof_nested,
    run_c8_welfare,
)
from encuestador.longitudinal_runtime import (
    _categorical_positions,
    load_longitudinal_config,
)


def make_config(tmp_path: Path, n_splits: int) -> Path:
    src = Path(__file__).resolve().parents[1] / "configs/longitudinal/c8_real_p1r.yaml"
    raw = src.read_text(encoding="utf-8")
    raw = raw.replace("n_splits: 5", f"n_splits: {n_splits}")
    raw = raw.replace("max_iter: 60", "max_iter: 4")
    raw = raw.replace("min_samples_leaf: 20", "min_samples_leaf: 2")
    path = tmp_path / "c8_fixture.yaml"
    path.write_text(raw, encoding="utf-8")
    return path


def five_fold_fixture(tmp_path: Path):
    source, gate, c6, c7_cfg_path, persons = fixture(tmp_path)
    # The inherited C7 test uses three folds for quick parity. C8 nested
    # time cross-fitting requires five (as in the real commissioned config).
    new = np.asarray(
        [_household_fold(str(row["panel_household_id"]), 5) for row in persons],
        dtype=np.uint8,
    )
    assert set(new.tolist()) == set(range(5))
    np.save(c6.root / "fold_ids.npy", new)
    manifest_path = c6.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256((c6.root / "fold_ids.npy").read_bytes()).hexdigest()
    manifest["n_splits"] = 5
    manifest["fold_ids_sha256"] = digest
    manifest["artifacts"]["fold_ids.npy"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    c6 = load_longitudinal_model_plane(c6.root)
    c7_cfg_path.write_text(
        c7_cfg_path.read_text(encoding="utf-8").replace("n_splits: 3", "n_splits: 5"),
        encoding="utf-8",
    )
    root = materialize_c7_panel(source, gate, c6, tmp_path / "panels")
    panel = load_c7_panel(root)
    c8_cfg = load_longitudinal_config(make_config(tmp_path, 5))
    return panel, c7_cfg_path, c8_cfg


def test_c8_T0_train_only_smoothing_and_three_class_order() -> None:
    features = np.asarray(
        [[0, 1, 1], [0, 1, 1], [0, 3, 1],
         [0, 1, 2], [0, 3, 2], [0, 1, 3]], dtype=float,
    )
    observed = np.asarray([1, 1, 2, 2, 3, 3])
    score = np.asarray([[0, 1, 1], [0, 3, 1], [0, 3, 3]], dtype=float)
    q = t0_empirical(features, observed, score)
    assert q.shape == (3, 3)
    assert np.isfinite(q).all() and (q > 0).all()
    assert np.allclose(q.sum(axis=1), 1.0)
    assert q[0, 0] > q[0, 1]
    metrics = transition_metrics(np.asarray([1, 2, 3]), q)
    assert metrics["class_order"] == list(CLASS_ORDER)
    assert metrics["n"] == 3
    assert metrics["multiclass_brier"] >= 0
    with pytest.raises(C7ContractError, match="transition_probability_invalid"):
        _normalize(np.asarray([[0.1, 0.9, float("nan")]]))


def test_c8_inner_oof_labor_labels_are_never_consumed_by_their_fold(
    tmp_path: Path,
) -> None:
    config = load_longitudinal_config(make_config(tmp_path, 5))
    config = replace(
        config, composition_features=("P02",),
        categorical_features=("P02",),
    )
    names = ("P02", "elapsed_quarters", "stale_labor_state")
    indices = np.arange(75)
    x = np.column_stack((
        1 + indices % 2, np.where(indices % 2, 1, 3), 1 + indices % 3,
    )).astype(float)
    y = np.asarray(1 + (indices // 2) % 3)
    folds = indices % 5
    q0, q1, audit = crossfit_transitions(x, y, folds, config, names)
    assert q0 is not None and q0.shape == q1.shape == (75, 3)
    assert np.allclose(q0.sum(axis=1), 1) and np.allclose(q1.sum(axis=1), 1)
    assert all(record["group_overlap"] == 0 for record in audit)
    changed = y.copy()
    changed[folds == 2] = 3
    q0_again, q1_again, _ = crossfit_transitions(x, changed, folds, config, names)
    assert q0_again is not None
    assert np.allclose(q0[folds == 2], q0_again[folds == 2])
    assert np.allclose(q1[folds == 2], q1_again[folds == 2])


def test_c8_transition_restartable_and_requires_exact_parent(tmp_path: Path) -> None:
    panel, _, config = five_fold_fixture(tmp_path)
    part = run_c8_transition(panel, config, tmp_path / "transitions", outer_fold=0)
    checkpoint = part / "checkpoints" / "fold_0" / "checkpoint.json"
    import hashlib

    original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    full = run_c8_transition(panel, config, tmp_path / "transitions")
    assert hashlib.sha256(
        (full / "checkpoints" / "fold_0" / "checkpoint.json").read_bytes()
    ).hexdigest() == original_sha
    receipt = load_c8_transition(full, panel, config)
    metrics = json.loads((full / "transition_metrics.json").read_text(encoding="utf-8"))
    assert receipt["contract"] == C8_TRANSITION_CONTRACT
    assert receipt["raw_only"] and not receipt["anchored"]
    assert receipt["outer_holdout_labels_used_for_training"] == 0
    assert metrics["T1"]["class_order"] == list(CLASS_ORDER)
    assert len(metrics["foldwise"]) == panel.n_splits
    assert np.allclose(np.load(full / "t1_oof.npy").sum(axis=1), 1)
    assert run_c8_transition(panel, config, tmp_path / "transitions") == full

    with pytest.raises(C7ContractError, match="wrong_c7_parent_or_config"):
        changed = replace(config, digest="different-config")
        load_c8_transition(full, panel, changed)
    with (full / "t1_oof.npy").open("ab") as output:
        output.write(b"tamper")
    with pytest.raises(C7ContractError, match="artifact_hash_mismatch"):
        load_c8_transition(full, panel, config)


def test_c8_strict_nested_time_oof_and_categorical_contract(tmp_path: Path) -> None:
    panel, _, config = five_fold_fixture(tmp_path)
    resolved = resolve_c8_config(config, panel)
    assert tuple(resolved.categorical_features) == tuple(panel.manifest["categorical_features"])
    terminal = (*panel.baseline_features, *(
        "latent.current_labor.p[1]",
        "latent.current_labor.p[2]",
        "latent.current_labor.p[3]",
    ))
    positions = _categorical_positions(terminal, resolved, include_stale=False)
    assert positions == (0,)
    assert _categorical_positions(panel.feature_names, resolved, include_stale=True) == (
        0, len(panel.feature_names) - 1,
    )
    x = np.load(panel.root / "features.npy")
    labor = np.load(panel.root / "target_labor.npy")
    target = np.load(panel.root / "target.npy")
    folds = np.load(panel.root / "fold_ids.npy")
    outer = folds != 0
    p, amount, audit = _time_oof_nested(
        x[outer], labor[outer], target[outer], folds[outer], resolved,
        panel.feature_names, terminal,
    )
    assert np.isfinite(p).all() and np.isfinite(amount).all()
    assert len(audit) == 4
    assert all(item["inner_holdout_labor_or_income_used_in_fit"] == 0 for item in audit)
    assert all(
        len(item["transition_fit_oof"]) == 3 for item in audit
    )


def test_c8_welfare_requires_completed_C7_then_matched_L12(tmp_path: Path) -> None:
    panel, c7_config_path, c8_config = five_fold_fixture(tmp_path)
    transition = run_c8_transition(panel, c8_config, tmp_path / "transitions")
    with pytest.raises(C7ContractError, match="complete_c7_comparator_required"):
        run_c8_welfare(
            panel, c8_config, transition, tmp_path / "missing_C7",
            tmp_path / "welfare",
        )
    c7_config = load_longitudinal_config(c7_config_path)
    c7 = run_resource_safe_c7(panel, c7_config, tmp_path / "c7")
    _c7_comparator(c7, panel, resolve_c8_config(c8_config, panel))
    partial = run_c8_welfare(
        panel, c8_config, transition, c7, tmp_path / "welfare", outer_fold=0,
    )
    assert partial.name.endswith(".work")
    original = (partial / "checkpoints" / "fold_0" / "checkpoint.json").read_bytes()
    root = run_c8_welfare(panel, c8_config, transition, c7, tmp_path / "welfare")
    assert (root / "checkpoints" / "fold_0" / "checkpoint.json").read_bytes() == original
    receipt = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "welfare_metrics.json").read_text(encoding="utf-8"))
    assert receipt["matched_row_fold_and_target_c7"]
    assert receipt["nested_inner_time_transition"]
    assert not receipt["scientific_promotion_authorized"]
    assert not receipt["full_household_welfare_claim"]
    assert len(metrics["foldwise"]) == panel.n_splits
    assert len(metrics["gap_by_stale_state"]) == 6
    assert all(np.isfinite(np.load(root / name)).all() for name in (
        "p_positive.npy", "positive_amount.npy", "expected_income.npy",
    ))
    assert run_c8_welfare(panel, c8_config, transition, c7, tmp_path / "welfare") == root
    with (c7 / "c7_1_expected_income.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(C7ContractError, match="comparator_artifact_tamper"):
        run_c8_welfare(panel, c8_config, transition, c7, tmp_path / "else")
