from pathlib import Path

import pytest

from encuestador.longitudinal_intake import (
    LongitudinalIntakeError,
    exact_parent_metadata,
    load_labor_context_release,
)


OFFICIAL = Path("/home/matias/data/labor-state-releases/indec-eph-labor-state-52ca6bcb586f2b0b")
OVERLAY = Path("/home/matias/data/labor-state-completions/indec-eph-labor-context-completion-backward_fill-2212ec7a74aa7f16")


@pytest.mark.skipif(not OFFICIAL.exists() or not OVERLAY.exists(), reason="local L1/L1B artifacts unavailable")
def test_incomplete_official_requires_explicit_overlay():
    with pytest.raises(LongitudinalIntakeError, match="coverage_incomplete"):
        load_labor_context_release(OFFICIAL)
    labor = load_labor_context_release(OFFICIAL, OVERLAY)
    derived = [row for row in labor.read_observations() if row.get("value_status") == "derived_bfill"]
    assert len(derived) == 4
    assert labor.completion is not None
    assert labor.completion.release_id.startswith("indec-eph-labor-context-completion-backward_fill-")


@pytest.mark.skipif(not OFFICIAL.exists() or not OVERLAY.exists(), reason="local L1/L1B artifacts unavailable")
def test_parent_metadata_records_official_and_completion_identity():
    labor = load_labor_context_release(OFFICIAL, OVERLAY)
    # The completion identity is attached by exact_parent_metadata when used
    # by longitudinal-run; inspect it through a minimal parent-shaped object.
    class EPH:
        release_id = "eph"
        monetary_release_id = "money"
        monetary_reference_period = "2026-Q1"
        root = OFFICIAL
    parent = exact_parent_metadata(EPH(), labor, composition_metadata={})
    assert parent["labor_context"]["release_id"] == labor.release_id
    assert parent["labor_context_completion"]["release_id"] == labor.completion.release_id
    assert parent["labor_context_completion"]["manifest_sha256"] == labor.completion.manifest_sha256
