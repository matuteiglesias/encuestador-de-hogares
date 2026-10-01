"""R0 fixture checks: C7 projects the EXACT Gate-B selection, then income filters."""
from __future__ import annotations

from collections import Counter

import pytest

from encuestador.longitudinal_c7_contract import (
    C7_BASELINE_ADDITIONAL_FEATURES,
    C7_INPUT_POLICY,
    C7_L11_ADDITIONAL_FEATURES,
    C7_NEVER_TERMINAL_INPUTS,
    C7ContractError,
    C7PairProjector,
    validate_gate_b_parent,
)
from encuestador.longitudinal_gate_b import (
    CONTRACT,
    GAP_STATUS,
    SELECTION_POLICY,
    VALID_LINK,
    _income,
    _select_reason,
)


def person(
    label: str, period: str, state: str,
    income: str, status: str,
    *, age: str = "30", second_state: str | None = None,
) -> dict[str, str]:
    row_id = f"{period}|{label}|1|1"
    return {
        "row_id": row_id,
        "candidate": f"{label}|1|1",
        "household": f"{label}|1",
        "period": period,
        "region": "pampeana",
        "age": age,
        "estado": state,
        "condact": state if second_state is None else second_state,
        "income": income,
        "income_status": status,
        "monetary_reference": "2025-11-01",
    }


def pair(
    label: str, previous: str, current: str, gap: int,
    prior: str, now: str, later_income: str = "150",
    later_status: str = "positive",
    *, age: str = "30", link_status: str = VALID_LINK,
    early_second: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    earlier = person(label, previous, prior, "100", "positive", age=age,
                     second_state=early_second)
    later = person(label, current, now, later_income, later_status, age=age)
    link = {
        "person_linkage_candidate_id": earlier["candidate"],
        "previous_row_id": earlier["row_id"],
        "current_row_id": later["row_id"],
        "previous_period": previous,
        "current_period": current,
        "elapsed_quarters": str(gap),
        "person_linkage_status": link_status,
        "rotation_gap_status": GAP_STATUS if gap in (1, 3) else "irregular",
    }
    return link, {earlier["row_id"]: earlier, later["row_id"]: later}


def test_c7_projection_same_gate_b_decisions_then_income_filter() -> None:
    valid_1, cache1 = pair("ee", "2021-Q1", "2021-Q2", 1, "1", "1")
    valid_3, cache3 = pair("eu", "2022-Q1", "2022-Q4", 3, "1", "2", "0", "zero")
    missing_y, cache_missing = pair(
        "ii", "2021-Q1", "2021-Q2", 1, "3", "3", "", "missing"
    )
    conflict, cache_conflict = pair(
        "cf", "2021-Q1", "2021-Q2", 1, "1", "1",
        link_status="demographic_conflict",
    )
    minor, cache_minor = pair(
        "minor", "2021-Q1", "2021-Q2", 1, "1", "1", age="13"
    )
    disagree, cache_disagree = pair(
        "labor", "2021-Q1", "2021-Q2", 1, "1", "1", early_second="2"
    )
    irregular, cache_gap = pair(
        "gap", "2021-Q1", "2021-Q3", 2, "1", "1"
    )
    reversed_link = dict(valid_1)
    reversed_link.update(
        previous_row_id=valid_1["current_row_id"],
        current_row_id=valid_1["previous_row_id"],
        previous_period=valid_1["current_period"],
        current_period=valid_1["previous_period"],
    )
    broken = dict(valid_1)
    broken["previous_row_id"] = "2021-Q1|ghost|1|1"
    candidates = [
        (valid_1, cache1),
        (valid_3, cache3),
        (missing_y, cache_missing),
        (conflict, cache_conflict),
        (minor, cache_minor),
        (disagree, cache_disagree),
        (irregular, cache_gap),
        (valid_1, cache1),  # repeated later row MUST be rejected globally
        (reversed_link, cache1),
        (broken, cache1),
    ]
    projector = C7PairProjector()
    selected = []
    reasons = Counter()
    expected_gate_b_count = 0
    seen_gate_b = set()
    for link, cache in candidates:
        reason, raw = _select_reason(link, cache, False)
        if reason is None:
            later_id = raw[4]["row_id"]
            if later_id in seen_gate_b:
                reason = "later_observation_reused"
            else:
                seen_gate_b.add(later_id)
        decision = projector.select(link, cache)
        assert decision.gate_b_eligible == (reason is None)
        if reason is None:
            expected_gate_b_count += 1
            assert decision.exclusion_reason is None or (
                decision.exclusion_reason == "target_income_missing_or_invalid"
            )
            if _income(raw[4]) is not None:
                assert decision.private_pair is not None
                selected.append(decision.private_pair)
            else:
                assert decision.exclusion_reason == "target_income_missing_or_invalid"
                assert decision.private_pair is None
        else:
            assert decision.exclusion_reason == reason
            assert decision.private_pair is None
            reasons[reason] += 1
    assert expected_gate_b_count == 3
    assert len(selected) == 2
    assert len({r["target_observation_row_id"] for r in selected}) == 2
    assert {r["elapsed_quarters"] for r in selected} == {1, 3}
    assert {r["stale_labor_state"] for r in selected} == {"1"}
    assert {r["target_current_labor_state"] for r in selected} == {"1", "2"}
    assert selected[1]["target_real_income"] == 0.0
    assert selected[1]["target_period"] == "2022-Q4"
    assert reasons["later_observation_reused"] == 1
    assert reasons["period_direction_or_gap_mismatch"] == 1
    assert reasons["identity_join_failure"] == 1
    assert reasons["earlier_labor_valid_code_disagreement"] == 1
    assert C7_INPUT_POLICY.startswith("c7_gate_b_exact_")


def test_c7_earlier_input_semantics_are_frozen() -> None:
    assert C7_BASELINE_ADDITIONAL_FEATURES == ("elapsed_quarters",)
    assert C7_L11_ADDITIONAL_FEATURES == (
        "elapsed_quarters", "stale_labor_state",
    )
    assert "target_current_labor_state" in C7_NEVER_TERMINAL_INPUTS
    assert "target_real_income" in C7_NEVER_TERMINAL_INPUTS


@pytest.fixture
def receipt():
    return {
        "contract": CONTRACT,
        "l2_release_id": "l2-test",
        "l2_manifest_sha256": "a" * 64,
        "persons_sha256": "b" * 64,
        "panel_links_sha256": "c" * 64,
        "selection_policy": SELECTION_POLICY,
        "exclude_exceptional": False,
        "candidate_link_rows": 10,
        "eligible_pairs": 3,
        "exclusion_count": 7,
        "eligible_pairs_by_gap": {"1": 2, "3": 1},
    }


def check_receipt(record):
    return validate_gate_b_parent(
        record,
        l2_release_id="l2-test",
        l2_manifest_sha256="a" * 64,
        persons_sha256="b" * 64,
        panel_links_sha256="c" * 64,
    )


def test_c7_gate_b_parent_custody_and_counts(receipt) -> None:
    check_receipt(receipt)
    for key, invalid in (
        ("contract", "wrong"),
        ("l2_release_id", "wrong"),
        ("persons_sha256", "d" * 64),
        ("panel_links_sha256", "d" * 64),
        ("l2_manifest_sha256", "d" * 64),
        ("selection_policy", "invented_linkage"),
        ("exclude_exceptional", True),
    ):
        altered = dict(receipt, **{key: invalid})
        with pytest.raises(C7ContractError, match=f"c7_gate_b_parent_mismatch:{key}"):
            check_receipt(altered)
    for key, invalid in (
        ("candidate_link_rows", 9),
        ("eligible_pairs", -1),
        ("exclusion_count", 6),
    ):
        altered = dict(receipt, **{key: invalid})
        with pytest.raises(C7ContractError, match="count_accounting_invalid"):
            check_receipt(altered)
    with pytest.raises(C7ContractError, match="gap_accounting_invalid"):
        check_receipt(dict(receipt, eligible_pairs_by_gap={"1": 1, "3": 1}))


def test_c7_exceptional_policy_is_a_separate_source_variant() -> None:
    link, cache = pair("exc", "2024-Q1", "2024-Q4", 3, "1", "2")
    baseline = C7PairProjector()
    assert baseline.select(link, cache).private_pair is not None
    sensitivity = C7PairProjector(exclude_exceptional=True)
    decision = sensitivity.select(link, cache)
    assert decision.exclusion_reason == "exceptional_period_pair"
    assert not decision.gate_b_eligible
