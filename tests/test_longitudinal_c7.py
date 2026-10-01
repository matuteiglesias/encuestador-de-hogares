"""C7A tests: real-contract shape, exact Gate-B audit, donor-X, OOF and resume."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from encuestador.longitudinal_c6 import (
    FOLD_POLICY,
    MODEL_PLANE_CONTRACT,
    LABOR_CONTEXT_FIELDS,
    _household_fold,
    load_longitudinal_model_plane,
)
from encuestador.longitudinal_c7 import (
    C7_PLANE_CONTRACT,
    load_c7_panel,
    materialize_c7_panel,
)
from encuestador.longitudinal_c7_contract import C7ContractError
from encuestador.longitudinal_c7_run import (
    C7_RUN_CONTRACT,
    resolve_c7_config,
    run_resource_safe_c7,
)
from encuestador.longitudinal_gate_b import run_gate_b
from encuestador.longitudinal_runtime import (
    build_longitudinal_fold_manifest,
    execute_longitudinal_arm,
    load_longitudinal_config,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path: Path):
    source = tmp_path / "l2-test"
    source.mkdir()
    persons: list[dict] = []
    links: list[dict] = []
    windows = [
        ("2018-Q1", "2018-Q2", 1), ("2019-Q1", "2019-Q4", 3),
        ("2020-Q3", "2020-Q4", 1), ("2021-Q1", "2021-Q4", 3),
        ("2022-Q2", "2022-Q3", 1), ("2023-Q1", "2023-Q4", 3),
        ("2024-Q3", "2024-Q4", 1), ("2025-Q1", "2025-Q4", 3),
    ]
    labels = [
        f"{year}-Q{q}"
        for year in range(2017, 2027)
        for q in range(1, 5) if (year, q) <= (2026, 1)
    ]
    for i in range(45):
        early, later, gap = windows[i % len(windows)]
        household = f"h{i}|1"
        candidate = household + "|1"
        source_state = str(1 + i % 3)
        next_state = str(1 + (i // 2) % 3)
        start_id, end_id = (f"{period}|h{i}|1|1" for period in (early, later))
        late_value = "0" if i % 7 == 0 else str(350 + 19 * i)
        late_status = "zero" if i % 7 == 0 else "positive"
        if i == 8:
            late_value, late_status = "", "missing_or_non_numeric"
        for is_later, row_id, period in (
            (False, start_id, early), (True, end_id, later)
        ):
            state = next_state if is_later else source_state
            amount = late_value if is_later else "100"
            status = late_status if is_later else "positive"
            persons.append({
                "row_id": row_id,
                "person_linkage_candidate_id": candidate,
                "panel_household_id": household,
                "period": period, "region_id": "pampeana",
                "CH04": str(1 + i % 2),
                "CH06": str(25 + i % 30 + int(is_later)),
                "ESTADO": state, "CONDACT": state,
                "P47T_real": amount, "p47t_value_status": status,
                "monetary_reference_period": "2025-11-01",
            })
        links.append({
            "person_linkage_candidate_id": candidate,
            "previous_row_id": start_id, "current_row_id": end_id,
            "previous_period": early, "current_period": later,
            "elapsed_quarters": str(gap),
            "person_linkage_status": "demographically_consistent_component_candidate",
            "rotation_gap_status": "expected_2_2_2_adjacent_gap",
        })
    links.append(dict(links[0]))  # global duplicate later must reconcile
    invalid = dict(links[1], person_linkage_status="demographic_conflict")
    links.append(invalid)
    write_csv(source / "persons.csv", persons)
    write_csv(source / "panel_links.csv", links)
    (source / "qa.json").write_text("{}\n", encoding="utf-8")
    source_manifest = {
        "contract": "research.eph-longitudinal-analysis-frame/v1",
        "release_id": source.name,
        "panel_audit": {"candidate_link_rows": len(links)},
        "monetary_lineage": {"common_reference_period": "2025-11-01"},
        "parents": {"monetary_conversion": {"status": "candidate"}},
        "artifacts": {
            name: {"sha256": sha(source / name), "rows": len(persons) if name == "persons.csv" else len(links)}
            for name in ("persons.csv", "panel_links.csv", "qa.json")
        },
    }
    (source / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    gate = run_gate_b(source, tmp_path / "gate-output")
    receipt = json.loads((gate / "gate_b_receipt.json").read_text())
    assert receipt["eligible_pairs"] == 45
    assert receipt["exclusion_count"] == 2

    c6 = tmp_path / "c6-fixture"
    c6.mkdir()
    composition = ("P02", "P03")
    names = (*composition, *LABOR_CONTEXT_FIELDS)
    feature = np.zeros((len(persons), len(names)), dtype=np.float64)
    target = np.zeros(len(persons), dtype=np.float64)
    period_code = np.zeros(len(persons), dtype=np.int16)
    fold = np.zeros(len(persons), dtype=np.uint8)
    household_codes = np.zeros(len(persons), dtype=np.int32)
    row_ids = []
    for i, row in enumerate(persons):
        period = str(row["period"])
        feature[i, 0] = float(row["CH04"])
        feature[i, 1] = float(row["CH06"])
        feature[i, 2:] = np.asarray([
            44 + labels.index(period) * .05, 7, 11, 1.1, -.2, .3
        ])
        target[i] = float(row["P47T_real"]) if row["P47T_real"] else np.nan
        period_code[i] = labels.index(period)
        fold[i] = _household_fold(str(row["panel_household_id"]), 3)
        household_codes[i] = i
        row_ids.append(str(row["row_id"]))
    assert set(fold) == {0, 1, 2}
    for name, array in {
        "features.npy": feature, "target.npy": target,
        "period_codes.npy": period_code, "fold_ids.npy": fold,
        "household_codes.npy": household_codes,
    }.items():
        np.save(c6 / name, array)
    (c6 / "row_ids.txt").write_text(
        "".join(row_id + "\n" for row_id in row_ids), encoding="utf-8"
    )
    artifacts = {
        name: {"sha256": sha(c6 / name)}
        for name in (
            "features.npy", "target.npy", "period_codes.npy", "fold_ids.npy",
            "household_codes.npy", "row_ids.txt",
        )
    }
    manifest = {
        "contract": MODEL_PLANE_CONTRACT, "release_id": c6.name,
        "profile_id": "P1R_NOLAB_LONG", "row_count": len(persons),
        "feature_names": list(names), "composition_features": list(composition),
        "categorical_features": ["P02"],
        "labor_context_fields": list(LABOR_CONTEXT_FIELDS),
        "target_field": "P47T_real", "fold_policy": FOLD_POLICY,
        "n_splits": 3, "period_labels": labels,
        "row_identity_sequence_sha256": hashlib.sha256(
            (c6 / "row_ids.txt").read_bytes()
        ).hexdigest(),
        "fold_ids_sha256": artifacts["fold_ids.npy"]["sha256"],
        "parents": {"longitudinal_eph": {
            "release_id": source.name,
            "manifest_sha256": sha(source / "manifest.json"),
        }},
        "artifacts": artifacts,
    }
    (c6 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = tmp_path / "c7.yaml"
    raw = (Path(__file__).parent.parent / "configs/longitudinal/c7_real_p1r.yaml").read_text()
    raw = raw.replace("n_splits: 5", "n_splits: 3")
    raw = raw.replace("max_iter: 60", "max_iter: 4")
    raw = raw.replace("min_samples_leaf: 20", "min_samples_leaf: 2")
    config.write_text(raw, encoding="utf-8")
    return source, gate, load_longitudinal_model_plane(c6), config, persons


def test_c7_materializes_donor_x_and_matched_restartable_science(tmp_path: Path):
    source, gate, c6, config_path, persons = fixture(tmp_path)
    output = materialize_c7_panel(source, gate, c6, tmp_path / "panels")
    panel = load_c7_panel(output)
    assert panel.manifest["contract"] == C7_PLANE_CONTRACT
    assert panel.manifest["gate_b_eligible_pairs"] == 45
    assert panel.manifest["row_count"] == 44
    assert panel.manifest["extra_invalid_later_income"] == {
        "target_income_missing_or_invalid": 1,
    }
    assert panel.manifest["gate_b_first_exclusions"] == {
        "later_observation_reused": 1,
        "link_status:demographic_conflict": 1,
    }
    assert panel.feature_names == (
        "P02", "P03", "elapsed_quarters", *LABOR_CONTEXT_FIELDS,
        "stale_labor_state",
    )
    x = np.load(output / "features.npy")
    y = np.load(output / "target.npy")
    gap = np.load(output / "gap.npy")
    stale = np.load(output / "stale_labor.npy")
    assert np.isfinite(y).all() and (y == 0).any()
    # First pair: donor age comes from the EARLIER observation, not target.
    assert x[0, 1] == float(persons[0]["CH06"])
    assert x[0, 1] != float(persons[1]["CH06"])
    assert x[0, 3] == 44 + c6.manifest["period_labels"].index(persons[1]["period"]) * .05
    assert x[0, -1] == int(persons[0]["ESTADO"])
    assert np.array_equal(x[:, 2], gap)
    assert np.array_equal(x[:, -1], stale)

    with pytest.raises(C7ContractError, match="artifact_missing_or_tampered"):
        (output / "features.npy").open("ab").write(b"tamper")
        load_c7_panel(output)

    # Undo only the deliberate derivative tamper in this fixture, never mutate L2.
    loaded = np.load(output / "features.npy", mmap_mode="r")
    assert loaded.shape == x.shape
    with (output / "features.npy").open("rb+") as stream:
        stream.truncate(stream.seek(0, 2) - len(b"tamper"))
    panel = load_c7_panel(output)

    config = load_longitudinal_config(config_path)
    resolved = resolve_c7_config(config, panel)
    assert resolved.categorical_features == ("P02",)
    partial = run_resource_safe_c7(
        panel, config, tmp_path / "runs", outer_fold=0,
    )
    assert partial.name.endswith(".work")
    checkpoint = partial / "checkpoints" / "fold_0" / "checkpoint.json"
    before = sha(checkpoint)
    full = run_resource_safe_c7(panel, config, tmp_path / "runs")
    assert sha(full / "checkpoints" / "fold_0" / "checkpoint.json") == before
    receipt = json.loads((full / "run_manifest.json").read_text())
    metrics = json.loads((full / "metrics.json").read_text())
    assert receipt["contract"] == C7_RUN_CONTRACT
    assert receipt["matched_rows_folds_target"]
    assert not receipt["full_household_welfare_claim"]
    assert not receipt["long_horizon_census_transport_authorized"]
    assert len(receipt["checkpoints"]) == 3
    assert len(metrics["paired"]["foldwise"]) == 3
    assert len(metrics["paired"]["gap_by_stale_labor"]) == 6
    assert metrics["arms"]["C7-0"]["feature_names"] == list(panel.baseline_features)
    assert metrics["arms"]["C7-1"]["feature_names"] == list(panel.feature_names)
    assert np.isfinite(np.load(full / "c7_0_expected_income.npy")).all()
    assert np.isfinite(np.load(full / "c7_1_expected_income.npy")).all()
    assert run_resource_safe_c7(panel, config, tmp_path / "runs") == full


def test_c7_rejects_parent_drift_without_modifying_sources(tmp_path: Path):
    source, gate, c6, _, _ = fixture(tmp_path)
    good = materialize_c7_panel(source, gate, c6, tmp_path / "good")
    assert good.is_dir()
    (source / "panel_links.csv").open("a").write("\n")
    with pytest.raises((C7ContractError, ValueError), match="hash_mismatch"):
        materialize_c7_panel(source, gate, c6, tmp_path / "bad")


def test_c7_rejects_forged_c6_row_order(tmp_path: Path):
    source, gate, c6, _, _ = fixture(tmp_path)
    ids = c6.root / "row_ids.txt"
    lines = ids.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    ids.write_text("\n".join(lines) + "\n")
    # Rebind the fixture C6 artifact SHA while retaining the old row-sequence
    # scientific identity: stream lockstep still must fail closed.
    manifest_path = c6.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["row_ids.txt"]["sha256"] = sha(ids)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    altered = load_longitudinal_model_plane(c6.root)
    with pytest.raises(C7ContractError, match="row_order_mismatch"):
        materialize_c7_panel(source, gate, altered, tmp_path / "bad")


def test_c7_categorical_and_numeric_families_are_distinct(tmp_path: Path):
    _, _, c6, config_path, _ = fixture(tmp_path)
    # C6 / C4 metadata must not lose baseline P02 categorical declaration
    # when C7-1 appends the stale labor category.
    from encuestador.longitudinal_runtime import _categorical_positions
    cfg = replace(
        load_longitudinal_config(config_path),
        composition_features=c6.composition_features,
        categorical_features=c6.categorical_features,
    )
    names = (*c6.composition_features, "elapsed_quarters",
             *LABOR_CONTEXT_FIELDS, "stale_labor_state")
    baseline = _categorical_positions(names[:-1], cfg, include_stale=False)
    l11 = _categorical_positions(names, cfg, include_stale=True)
    assert baseline == (0,)
    assert l11 == (0, len(names) - 1)
