"""C7 experimental pair contract: an exact projection of Gate B, never new linkage.

R0 freezes the scientific data boundary only.  It does NOT construct a
resource-safe C7 model plane or run L11/L12.  C7A owns later private plane
materialization; local C7B owns real-parent commissioning.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .longitudinal_gate_b import (
    CONTRACT as GATE_B_CONTRACT,
)
from .longitudinal_gate_b import (
    SELECTION_POLICY as GATE_B_SELECTION_POLICY,
)
from .longitudinal_gate_b import _income, _select_reason

C7_INPUT_POLICY = "c7_gate_b_exact_donor_x_valid_target_income_v1"
C7_BASELINE_ARM = "C7-0"
C7_STALE_LABOR_ARM = "C7-1"
C7_LABOR_CLASSES = ("1", "2", "3")
C7_ALLOWED_GAPS = (1, 3)
C7_CONTEXT_MODE = "national_regional"
C7_COMPOSITION_PROFILE = "P1R_NOLAB_LONG"

# Identity/clock variables are *not* external model features by default.  C7A
# may encode elapsed_quarters for both arms and must prove the feature difference.
C7_BASELINE_ADDITIONAL_FEATURES = ("elapsed_quarters",)
C7_L11_ADDITIONAL_FEATURES = (*C7_BASELINE_ADDITIONAL_FEATURES, "stale_labor_state")
C7_NEVER_TERMINAL_INPUTS = (
    "target_current_labor_state",
    "target_real_income",
    "target_observation_row_id",
    "target_period_as_individual_feature",
)


class C7ContractError(ValueError):
    """Gate-B source custody or paired-model input invariant failure."""


def validate_gate_b_parent(
    receipt: Mapping[str, Any],
    *,
    l2_release_id: str,
    l2_manifest_sha256: str,
    persons_sha256: str,
    panel_links_sha256: str,
    exclude_exceptional: bool = False,
) -> None:
    """Reject stale/synthetic/mismatched Gate-B provenance before private C7 intake."""
    expected = {
        "contract": GATE_B_CONTRACT,
        "l2_release_id": l2_release_id,
        "l2_manifest_sha256": l2_manifest_sha256,
        "persons_sha256": persons_sha256,
        "panel_links_sha256": panel_links_sha256,
        "selection_policy": GATE_B_SELECTION_POLICY,
        "exclude_exceptional": exclude_exceptional,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise C7ContractError(f"c7_gate_b_parent_mismatch:{key}")
    original = receipt.get("candidate_link_rows")
    accepted = receipt.get("eligible_pairs")
    excluded = receipt.get("exclusion_count")
    if (
        not all(type(n) is int and n >= 0 for n in (original, accepted, excluded))
        or original != accepted + excluded
    ):
        raise C7ContractError("c7_gate_b_parent_count_accounting_invalid")
    by_gap = receipt.get("eligible_pairs_by_gap")
    if (
        not isinstance(by_gap, Mapping)
        or not all(type(by_gap.get(str(gap))) is int for gap in C7_ALLOWED_GAPS)
        or accepted != sum(by_gap[str(gap)] for gap in C7_ALLOWED_GAPS)
    ):
        raise C7ContractError("c7_gate_b_parent_gap_accounting_invalid")


@dataclass(frozen=True)
class C7PairDecision:
    """One exclusive decision per audited source link.

    gate_b_eligible precedes the extra C7 target-income filter, preserving the
    Gate-B population/counts.  Only a successful decision has a private row.
    """

    exclusion_reason: str | None
    gate_b_eligible: bool
    private_pair: dict[str, Any] | None


class C7PairProjector:
    """Turn audited Gate-B candidate links into a donor-X experiment intake.

    Pass pre-fetched person projections exactly as in Gate B.  This class does
    not recompute linkage or review ESTADO/CONDACT; _select_reason is the single
    scientific source of truth for eligibility.  Keep one instance across ALL
    source-link batches to preserve the global unique-later-row constraint.
    """

    def __init__(self, *, exclude_exceptional: bool = False) -> None:
        self.exclude_exceptional = exclude_exceptional
        self.seen_gate_b_target_rows: set[str] = set()

    def select(
        self, link: dict[str, str], person_cache: Mapping[str, Any]
    ) -> C7PairDecision:
        reason, selected = _select_reason(
            link, person_cache, self.exclude_exceptional
        )
        if reason is not None:
            return C7PairDecision(reason, False, None)
        (
            gap, stale_period, target_period, earlier, later,
            stale_labor, target_labor,
        ) = selected
        target_id = later["row_id"]
        if target_id in self.seen_gate_b_target_rows:
            return C7PairDecision("later_observation_reused", False, None)
        self.seen_gate_b_target_rows.add(target_id)

        # Income eligibility is an ADDITIONAL restriction, not a retroactive
        # alteration of the accepted Gate-B labor-transition population.
        target_income = _income(later)
        if target_income is None:
            return C7PairDecision("target_income_missing_or_invalid", True, None)

        stale_id = earlier["row_id"]
        household = earlier["household"]
        record = {
            "pair_id": f"{stale_id}->{target_id}",
            "stale_observation_row_id": stale_id,
            "target_observation_row_id": target_id,
            "person_linkage_candidate_id": link["person_linkage_candidate_id"],
            "panel_household_id": household,
            "stale_period": stale_period,
            "target_period": target_period,
            "elapsed_quarters": gap,
            "stale_labor_state": stale_labor,
            # Later state is available solely as a held-out transition/eval label.
            "target_current_labor_state": target_labor,
            "target_real_income": target_income,
            "target_region_id": later["region"],
            "target_monetary_reference_period": later["monetary_reference"],
        }
        return C7PairDecision(None, True, record)
