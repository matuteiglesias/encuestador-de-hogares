"""Fixture proof of bounded L2-link Gate-B descriptive evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from encuestador.longitudinal_gate_b import (
    CONTRACT,
    GateBEvidenceError,
    run_gate_b,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "l2-fixture"
    root.mkdir()
    persons = []
    links = []
    ids = {}

    def person(
        label: str,
        period: str,
        estado: str,
        condact: str,
        age: str,
        income: str,
        income_status: str,
    ) -> str:
        row_id = f"{period}|{label}|1|1"
        persons.append(
            {
                "row_id": row_id,
                "person_linkage_candidate_id": f"{label}|1|1",
                "panel_household_id": f"{label}|1",
                "period": period,
                "region_id": "pampeana",
                "CH04": "1",
                "CH06": age,
                "ESTADO": estado,
                "CONDACT": condact,
                "P47T_real": income,
                "p47t_value_status": income_status,
                "monetary_reference_period": "2025-11-01",
            }
        )
        return row_id

    def pair(
        label: str,
        previous: str,
        current: str,
        elapsed: int,
        early_labor: tuple[str, str],
        late_labor: tuple[str, str],
        early_income: tuple[str, str] = ("100", "positive"),
        late_income: tuple[str, str] = ("200", "positive"),
        age: str = "30",
        linkage_status: str = "demographically_consistent_component_candidate",
    ) -> None:
        early_id = person(
            label, previous, *early_labor, age, *early_income
        )
        late_id = person(
            label, current, *late_labor, age, *late_income
        )
        ids[label] = early_id
        links.append(
            {
                "person_linkage_candidate_id": f"{label}|1|1",
                "previous_row_id": early_id,
                "current_row_id": late_id,
                "previous_period": previous,
                "current_period": current,
                "elapsed_quarters": str(elapsed),
                "person_linkage_status": linkage_status,
                "person_linkage_confidence": "supporting_demographic_consistency",
                "rotation_gap_status": (
                    "expected_2_2_2_adjacent_gap"
                    if elapsed in (1, 3)
                    else "irregular_or_attrition_gap"
                ),
            }
        )

    pair(
        "ee", "2021-Q1", "2021-Q2", 1, ("1", "1"), ("1", "1"),
        ("100", "positive"), ("0", "zero"),
    )
    pair(
        "eu", "2022-Q1", "2022-Q4", 3, ("1", "1"), ("2", "2"),
        ("150", "positive"), ("50", "positive"),
    )
    pair(
        "ue", "2023-Q1", "2023-Q2", 1, ("2", "2"), ("1", "1"),
        ("0", "zero"), ("200", "positive"),
    )
    pair(
        "ii", "2024-Q1", "2024-Q4", 3, ("3", "3"), ("3", "3"),
        ("0", "zero"), ("", "missing_or_non_numeric"),
    )
    pair(
        "conflict", "2021-Q1", "2021-Q2", 1, ("1", "1"), ("1", "1"),
        linkage_status="demographic_conflict",
    )
    pair("missing", "2022-Q1", "2022-Q2", 1, ("", ""), ("1", "1"))
    pair("disagree", "2022-Q1", "2022-Q2", 1, ("1", "2"), ("1", "1"))
    pair("minor", "2022-Q1", "2022-Q2", 1, ("1", "1"), ("1", "1"), age="13")
    pair("gap2", "2022-Q1", "2022-Q3", 2, ("1", "1"), ("1", "1"))
    pair("special", "2023-Q1", "2023-Q2", 1, ("4", "0"), ("1", "1"))
    pair("unknown", "2023-Q1", "2023-Q2", 1, ("7", ""), ("1", "1"))
    pair("broken", "2023-Q1", "2023-Q2", 1, ("1", "1"), ("1", "1"))
    links[-1]["previous_row_id"] = "2023-Q1|not-a-person|1|1"
    pair("candidate", "2023-Q1", "2023-Q2", 1, ("1", "1"), ("1", "1"))
    links[-1]["person_linkage_candidate_id"] = "other|1|1"
    # An identical second target may not be used twice.
    links.append(dict(links[0]))
    # Reverse period direction, while leaving an apparently supported gap field.
    backwards = dict(links[1])
    backwards.update(
        previous_row_id=links[1]["current_row_id"],
        current_row_id=links[1]["previous_row_id"],
        previous_period=links[1]["current_period"],
        current_period=links[1]["previous_period"],
    )
    links.append(backwards)

    person_fields = [
        "row_id", "person_linkage_candidate_id", "panel_household_id",
        "period", "region_id", "CH04", "CH06", "ESTADO", "CONDACT",
        "P47T_real", "p47t_value_status", "monetary_reference_period",
    ]
    link_fields = [
        "person_linkage_candidate_id", "previous_row_id", "current_row_id",
        "previous_period", "current_period", "elapsed_quarters",
        "person_linkage_status", "person_linkage_confidence",
        "rotation_gap_status",
    ]
    _csv(root / "persons.csv", person_fields, persons)
    _csv(root / "panel_links.csv", link_fields, links)
    (root / "qa.json").write_text('{"fixture":true}\n', encoding="utf-8")
    manifest = {
        "contract": "research.eph-longitudinal-analysis-frame/v1",
        "release_id": root.name,
        "panel_audit": {"candidate_link_rows": len(links)},
        "monetary_lineage": {"common_reference_period": "2025-11-01"},
        "parents": {"monetary_conversion": {
            "release_id": "candidate-monetary-fixture", "status": "candidate"
        }},
        "artifacts": {
            name: {"sha256": _sha(root / name)}
            for name in ("persons.csv", "panel_links.csv", "qa.json")
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, ids


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_gate_b_exact_selection_matrices_income_and_lineage(tmp_path: Path) -> None:
    parent, _ = _fixture(tmp_path)
    release = run_gate_b(parent, tmp_path / "out")
    receipt = json.loads((release / "gate_b_receipt.json").read_text())
    assert receipt["contract"] == CONTRACT
    assert receipt["status"] == "descriptive_evidence_only"
    assert receipt["l2_manifest_sha256"] == _sha(parent / "manifest.json")
    assert receipt["persons_sha256"] == _sha(parent / "persons.csv")
    assert receipt["panel_links_sha256"] == _sha(parent / "panel_links.csv")
    assert receipt["candidate_link_rows"] == 15
    assert receipt["eligible_pairs"] == 4
    assert receipt["eligible_pairs_by_gap"] == {"1": 2, "3": 2}
    assert receipt["exclusion_count"] == 11
    assert sum(receipt["exclusions_by_first_reason"].values()) == 11
    assert receipt["exclusions_by_first_reason"]["link_status:demographic_conflict"] == 1
    assert receipt["exclusions_by_first_reason"]["unsupported_gap"] == 1
    assert receipt["exclusions_by_first_reason"]["earlier_labor_missing"] == 1
    assert receipt["exclusions_by_first_reason"]["earlier_labor_valid_code_disagreement"] == 1
    assert receipt["exclusions_by_first_reason"]["earlier_labor_special_only"] == 1
    assert receipt["exclusions_by_first_reason"]["earlier_labor_unreviewed_code"] == 1
    assert receipt["exclusions_by_first_reason"]["age14_universe_exclusion"] == 1
    assert receipt["exclusions_by_first_reason"]["identity_join_failure"] == 1
    assert receipt["exclusions_by_first_reason"]["candidate_household_or_period_mismatch"] == 1
    assert receipt["exclusions_by_first_reason"]["later_observation_reused"] == 1
    assert receipt["exclusions_by_first_reason"]["period_direction_or_gap_mismatch"] == 1
    assert receipt["distinct_eligible_candidate_keys"] == 4
    assert receipt["distinct_eligible_later_observations"] == 4
    assert receipt["exceptional_period_touching_eligible_pairs_by_gap"]["3"] == 1

    matrix = _read_csv(release / "transition_matrix.csv")
    assert len(matrix) == 18
    cells = {
        (int(row["elapsed_quarters"]), row["earlier_state"], row["later_state"]): row
        for row in matrix
    }
    for gap, source, target in (
        (1, "employed", "employed"),
        (1, "unemployed", "employed"),
        (3, "employed", "unemployed"),
        (3, "inactive", "inactive"),
    ):
        row = cells[(gap, source, target)]
        assert row["pair_count"] == "1"
        assert float(row["conditional_probability"]) == 1
        assert row["earlier_state_denominator"] == "1"
    for gap in (1, 3):
        for source in ("employed", "unemployed", "inactive"):
            rows = [
                row for row in matrix
                if int(row["elapsed_quarters"]) == gap and row["earlier_state"] == source
            ]
            denominator = int(rows[0]["earlier_state_denominator"])
            assert sum(int(row["pair_count"]) for row in rows) == denominator
            assert (
                sum(float(row["conditional_probability"]) for row in rows) == 1.0
                if denominator else all(row["conditional_probability"] == "" for row in rows)
            )
    support = _read_csv(release / "panel_support.csv")
    assert sum(int(row["eligible_pairs"]) for row in support if row["row_kind"] == "period_region") == 4

    income = {
        (int(row["elapsed_quarters"]), row["earlier_state"]): row
        for row in _read_csv(release / "income_by_prior_state.csv")
    }
    zero = income[(1, "employed")]
    assert zero["later_income_valid_n"] == "1"
    assert zero["later_positive_n"] == "0"
    assert float(zero["mean_later_unconditional_real_income"]) == 0
    assert float(zero["median_later_real_income"]) == 0
    assert float(zero["mean_real_income_change"]) == -100
    assert income[(1, "unemployed")]["positive_rate_valid_denominator"] == "1.0"
    assert float(income[(1, "unemployed")]["median_later_real_income"]) == 200
    invalid = income[(3, "inactive")]
    assert invalid["eligible_pairs"] == "1"
    assert invalid["later_income_valid_n"] == "0"
    assert invalid["later_income_missing_or_invalid_n"] == "1"
    assert invalid["mean_later_unconditional_real_income"] == ""
    assert receipt["monetary_conversion_parent"]["status"] == "candidate"
    for name, artifact in receipt["artifacts"].items():
        assert artifact["sha256"] == _sha(release / name)
    assert not (release / "_work.sqlite").exists()
    assert "descriptive" in (release / "GATE_B_NOTE.md").read_text().lower()

    with pytest.raises(GateBEvidenceError, match="immutable_gate_b_release_exists"):
        run_gate_b(parent, tmp_path / "out")


def test_gate_b_exceptional_pairs_can_be_excluded(tmp_path: Path) -> None:
    parent, _ = _fixture(tmp_path)
    release = run_gate_b(parent, tmp_path / "out", exclude_exceptional=True)
    receipt = json.loads((release / "gate_b_receipt.json").read_text())
    assert receipt["eligible_pairs"] == 3
    assert receipt["exclusions_by_first_reason"]["exceptional_period_pair"] == 1


def test_gate_b_fails_closed_on_parent_hash_drift(tmp_path: Path) -> None:
    parent, _ = _fixture(tmp_path)
    with (parent / "persons.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampering\n")
    with pytest.raises(GateBEvidenceError, match="l2_artifact_hash_mismatch:persons.csv"):
        run_gate_b(parent, tmp_path / "out")


def test_gate_b_rejects_duplicate_person_identity(tmp_path: Path) -> None:
    parent, _ = _fixture(tmp_path)
    persons_path = parent / "persons.csv"
    rows = _read_csv(persons_path)
    rows.append(dict(rows[0]))
    _csv(persons_path, list(rows[0]), rows)
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["persons.csv"]["sha256"] = _sha(persons_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GateBEvidenceError, match="l2_duplicate_person_row_id"):
        run_gate_b(parent, tmp_path / "out")


def test_gate_b_does_not_modify_source_release(tmp_path: Path) -> None:
    parent, _ = _fixture(tmp_path)
    before = {
        name: _sha(parent / name)
        for name in ("manifest.json", "persons.csv", "panel_links.csv", "qa.json")
    }
    run_gate_b(parent, tmp_path / "out")
    after = {name: _sha(parent / name) for name in before}
    assert after == before


def test_gate_b_multi_batch_keeps_exact_accounting(tmp_path: Path, capsys) -> None:
    """Exercise the 2,000-link prefetch boundary without losing a single source row."""
    parent, _ = _fixture(tmp_path)
    links_path = parent / "panel_links.csv"
    links = _read_csv(links_path)
    links.extend(dict(links[0]) for _ in range(2003))
    _csv(links_path, list(links[0]), links)
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["panel_links.csv"]["sha256"] = _sha(links_path)
    manifest["panel_audit"]["candidate_link_rows"] = len(links)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    release = run_gate_b(parent, tmp_path / "out")
    receipt = json.loads((release / "gate_b_receipt.json").read_text())
    assert receipt["candidate_link_rows"] == 2018
    assert receipt["eligible_pairs"] == 4
    assert receipt["eligible_pairs_by_gap"] == {"1": 2, "3": 2}
    assert receipt["exclusions_by_first_reason"]["later_observation_reused"] == 2004
    assert receipt["eligible_pairs"] + receipt["exclusion_count"] == 2018
    assert "indexed 26 person observations" in capsys.readouterr().err


def test_gate_b_keyboard_interrupt_removes_scratch_staging(tmp_path: Path, monkeypatch) -> None:
    """Ctrl-C must not leave the temporary indexed person database behind."""
    parent, _ = _fixture(tmp_path)
    output_root = tmp_path / "out"

    def interrupt_before_batch(_connection, _links):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "encuestador.longitudinal_gate_b._prefetch_persons",
        interrupt_before_batch,
    )
    with pytest.raises(KeyboardInterrupt):
        run_gate_b(parent, output_root)
    assert list(output_root.iterdir()) == []
    assert (parent / "persons.csv").exists()
    assert (parent / "panel_links.csv").exists()
