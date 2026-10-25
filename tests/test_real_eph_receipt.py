"""WP03-D metadata-only semantic acceptance tests: no real microdata or models."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from emit_real_eph_receipt import (  # noqa: E402
    COMPLETE, COUNTS, HOUSEHOLDS, MATRIX, PAIRS, PERSONS, RELEASE,
    SOURCE, UNAVAILABLE, build_receipt,
)
from export_real_eph_receipt_gcp import ExportError, build_entry  # noqa: E402


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    evidence = tmp_path / "evidence"
    runs = evidence / "encuestador-runs"
    comps = evidence / "comparisons"
    runs.mkdir(parents=True)
    comps.mkdir()
    _write(evidence / "real-eph-target-evidence.json", {
        "release_id": RELEASE, "period": "2024-Q3",
        "persons": PERSONS, "households": HOUSEHOLDS, **SOURCE,
        "target_eligibility": COUNTS,
    })
    parent = {
        "contract": "publicdata.eph-microdata@1",
        "period": "2024-Q3", "release_id": RELEASE,
        "source_archive_sha256": SOURCE["source_archive_sha256"],
        "source_manifest_sha256": SOURCE["source_manifest_sha256"],
        "individual": {"sha256": SOURCE["individual_sha256"], "rows": PERSONS},
        "household": {"sha256": SOURCE["household_sha256"], "rows": HOUSEHOLDS},
    }
    ledger = []
    names = {}
    for (arch, terminal), name in MATRIX.items():
        rid = f"real_eph_2024q3_{name}-fixture001"
        names[name] = rid
        row = {"run_id": rid, "architecture_id": arch,
               "terminal_formulation": terminal, "person_rows": PERSONS,
               "household_observations": HOUSEHOLDS,
               "complete_households": COMPLETE,
               "unavailable_households": UNAVAILABLE}
        ledger.append(row)
        root = runs / rid
        root.mkdir()
        metrics = {
            "fold_counts": {"0": PERSONS},
            "household": {
                "person_observation_count": PERSONS,
                "household_observation_count": HOUSEHOLDS,
                "complete_observed_income_household_count": COMPLETE,
                "unavailable_observed_income_household_count": UNAVAILABLE,
            },
        }
        _write(root / "metrics.json", metrics)
        _write(root / "input_evidence.json", [parent])
        (root / "synthetic-marker.txt").write_text("fixture without model/person data\n")
        artifacts = {
            name: {"sha256": _sha(root / name), "bytes": (root / name).stat().st_size}
            for name in ("metrics.json", "input_evidence.json", "synthetic-marker.txt")
        }
        _write(root / "run_manifest.json", {
            "contract": "research.encuestador-run/v1", "run_id": rid,
            "architecture_id": arch, "config_digest": "a" * 64,
            "terminal": {"formulation": terminal, "target": "P47T"},
            "input_evidence": [parent], "artifacts": artifacts,
        })
    _write(evidence / "real-eph-run-ledger.json", ledger)
    by_name = {name: next(x for x in ledger if x["run_id"] == rid)
               for name, rid in names.items()}
    for a, b, axis in PAIRS.values():
        pairs = [by_name[a], by_name[b]]
        digest = hashlib.sha256("\x1f".join(x["run_id"] for x in pairs).encode()).hexdigest()[:16]
        _write(comps / f"comparison-{digest}.json", {
            "contract": "research.encuestador-comparison/v1",
            "comparison_id": f"comparison-{digest}",
            "changed_axes": [axis], "status": "comparable",
            "runs": pairs,
        })
    return {"evidence": evidence, "runs": runs, "comps": comps}


def _receipt(fixture: dict[str, Path], **kwargs: str) -> dict:
    args = {
        "scheduler_run_id": "39617231503", "attempt": "1",
        "event": "schedule", "job_status": "success",
        "evidence_upload": "success",
        "started_at": "2026-10-21T09:17:00Z",
        "finished_at": "2026-10-21T10:18:00Z",
        "evidence_root": fixture["evidence"],
        "runs_root": fixture["runs"],
        "comparisons_root": fixture["comps"],
    }
    return build_receipt(**(args | kwargs))


def _mutate(path: Path, mutation) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    _write(path, value)


def _assert_bad(fixture: dict[str, Path]) -> None:
    rec = _receipt(fixture)
    assert rec["result"]["artifact_state"] != "healthy"
    assert rec["result"]["terminal_state"] == "failed"
    assert rec["outputs"][0]["state"] != "valid"
    build_entry(rec, "39617231503", "1")


def test_all_four_comparisons_and_bundles_are_verified(fixture):
    rec = _receipt(fixture)
    assert all(v == "success" for v in rec["stages"].values())
    assert rec["result"]["artifact_state"] == "healthy"
    assert rec["result"]["authority_state"] == "hold"
    assert rec["outputs"][0]["durability"] == "run_evidence"
    assert rec["outputs"][0]["state"] == "valid"
    assert len(rec["evidence_refs"]) == 11
    entry = build_entry(rec, "39617231503", "1")
    assert entry["jsonPayload"]["producer"]["observation_mode"] == "semantic_receipt"
    assert "pkl" not in json.dumps(entry).lower()
    assert "person_oof" not in json.dumps(entry).lower()


def test_missing_bundle_fails_closed(fixture):
    shutil.rmtree(next(fixture["runs"].iterdir()))
    _assert_bad(fixture)


def test_corrupted_digest_fails_closed(fixture):
    bundle = next(fixture["runs"].iterdir())
    (bundle / "synthetic-marker.txt").write_text("tampered")
    _assert_bad(fixture)


def test_duplicate_architecture_terminal_pair_fails_closed(fixture):
    ledger_path = fixture["evidence"] / "real-eph-run-ledger.json"
    _mutate(ledger_path, lambda d: d[1].update(
        architecture_id=d[0]["architecture_id"],
        terminal_formulation=d[0]["terminal_formulation"]))
    _assert_bad(fixture)


@pytest.mark.parametrize("location", ["target", "manifest"])
def test_wrong_official_source_digest_rejected(fixture, location):
    if location == "target":
        _mutate(fixture["evidence"] / "real-eph-target-evidence.json",
                lambda d: d.update(source_manifest_sha256="0" * 64))
    else:
        bundle = next(fixture["runs"].iterdir())
        _mutate(bundle / "run_manifest.json",
                lambda d: d["input_evidence"][0].update(source_manifest_sha256="0" * 64))
    _assert_bad(fixture)


def test_target_population_eligibility_drift_rejected(fixture):
    _mutate(fixture["evidence"] / "real-eph-target-evidence.json",
            lambda d: d["target_eligibility"].update(eligible=999))
    _assert_bad(fixture)


def test_ledger_count_drift_rejected(fixture):
    _mutate(fixture["evidence"] / "real-eph-run-ledger.json",
            lambda d: d[0].update(person_rows=999))
    _assert_bad(fixture)


@pytest.mark.parametrize("breakage", ["missing", "failed", "wrong_pairs"])
def test_missing_or_invalid_comparison_is_not_accepted(fixture, breakage):
    comparison = next(fixture["comps"].glob("*.json"))
    if breakage == "missing":
        comparison.unlink()
    elif breakage == "failed":
        _mutate(comparison, lambda d: d.update(status="failed"))
    else:
        _mutate(comparison, lambda d: d["runs"][0].update(run_id="other"))
    _assert_bad(fixture)


def test_early_workflow_failure_yields_partial_truthful_receipt(fixture):
    for path in fixture["evidence"].iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    rec = _receipt(fixture, job_status="failure", evidence_upload="skipped")
    assert rec["result"]["execution_state"] == "failed"
    assert rec["result"]["artifact_state"] != "healthy"
    assert rec["stages"]["official_source"] == "skipped"
    assert rec["stages"]["run_ledger"] == "skipped"
    assert rec["outputs"][0]["state"] == "missing"
    build_entry(rec, "39617231503", "1")


def test_pull_request_receipt_allowed_without_cloud_authority(fixture):
    rec = _receipt(fixture, event="pull_request")
    assert rec["trigger"] == "event"
    assert rec["result"]["scheduler_state"] == "not_applicable"
    assert rec["result"]["authority_state"] == "hold"
    build_entry(rec, "39617231503", "1")


def test_replay_deterministic_and_attempt_changes_identity(fixture):
    first = _receipt(fixture)
    again = _receipt(fixture)
    assert first == again
    a = build_entry(first, "39617231503", "1")
    b = build_entry(again, "39617231503", "1")
    assert a["insertId"] == b["insertId"]
    second = _receipt(fixture, attempt="2")
    assert build_entry(second, "39617231503", "2")["insertId"] != a["insertId"]


def test_exporter_rejects_private_payload_even_if_consumer_misbehaves(fixture):
    rec = _receipt(fixture)
    rec["warnings"].append("/home/matias/private/persons.csv")
    with pytest.raises(ExportError):
        build_entry(rec, "39617231503", "1")
    rec = _receipt(fixture)
    rec["outputs"][0]["ref"] = "/tmp/model.pkl"
    with pytest.raises(ExportError):
        build_entry(rec, "39617231503", "1")


def test_upload_failure_does_not_retroactively_invalidate_computation(fixture):
    rec = _receipt(fixture, job_status="failure", evidence_upload="failure")
    assert all(rec["stages"][k] == "success" for k in rec["stages"]
               if k not in ("workflow", "evidence_upload"))
    assert rec["result"]["artifact_state"] == "degraded"
    assert rec["result"]["execution_state"] == "failed"
    assert rec["outputs"][0]["state"] == "partial"
    build_entry(rec, "39617231503", "1")
