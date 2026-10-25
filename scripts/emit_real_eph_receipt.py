#!/usr/bin/env python3
"""Fail-closed, metadata-only receipt for the real EPH 2024-Q3 science workflow.

This is an OBSERVATION of bounded research evidence, not a welfare release.
No microdata, model, report, person IDs or raw paths enter the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT = "evidence:producer-run-receipt@1"
PRODUCER = "producer.github.welfare-real-eph-acceptance"
OUTPUT = "research.welfare-real-eph-acceptance@1"
REPO = "matuteiglesias/encuestador-de-hogares"
RELEASE = "eph-2024-q3-3b6a7a15c4af"
SOURCE = {
    "source_archive_sha256": "fbeacb866777759e2fcbc64e9643734bae29ec6db480dd6d532dc505505b3204",
    "source_manifest_sha256": "0104a46dc53dfb537a730e6fa323ee5ab0240cc838b66d32459cb2e2c3002f5d",
    "individual_sha256": "b435a2f22c72c5d0e281279f27ade8da674c6db6fe6597b620dc0485d9b27260",
    "household_sha256": "4fc25e8d9e283be1d9c41bc93d237e6c6ad666dbc80bb33559cbe185d17f2fb5",
}
COUNTS = {"rows": 47564, "eligible": 41821, "positive": 25209,
          "zero": 16612, "nonresponse": 5688, "missing": 55}
PERSONS, HOUSEHOLDS, COMPLETE, UNAVAILABLE = 47564, 16650, 12568, 4082
MATRIX = {
    ("direct_v1", "hurdle_gamma"): "direct_gamma",
    ("lean_labor_state_v1", "hurdle_gamma"): "lean_gamma",
    ("direct_v1", "hurdle_log"): "direct_log",
    ("lean_labor_state_v1", "hurdle_log"): "lean_log",
}
PAIRS = {
    "architecture_gamma": ("direct_gamma", "lean_gamma", "architecture"),
    "architecture_log": ("direct_log", "lean_log", "architecture"),
    "terminal_direct": ("direct_gamma", "direct_log", "terminal_formulation"),
    "terminal_lean": ("lean_gamma", "lean_log", "terminal_formulation"),
}
STAGES = ("workflow", "official_source", "target_eligibility", "run_ledger",
          *(f"bundle_{name}" for name in MATRIX.values()),
          *(f"comparison_{name}" for name in PAIRS),
          "evidence_upload")
HEX = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,110}$")


class ReceiptEvidenceError(ValueError):
    """Science evidence is missing, inconsistent or corrupt."""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool) -> None:
    if not condition:
        raise ReceiptEvidenceError("evidence_contract_mismatch")


def _observed(path: Path, validator: Any, *, early_failure: bool) -> str:
    if not path.is_file():
        return "skipped" if early_failure else "failed"
    try:
        validator(_json(path))
        return "success"
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return "failed"


def _source_shape(record: dict[str, Any]) -> bool:
    return (
        isinstance(record, dict)
        and record.get("release_id") == RELEASE
        and record.get("period") == "2024-Q3"
        and record.get("contract", "publicdata.eph-microdata@1") == "publicdata.eph-microdata@1"
        and all(record.get(name) == sha for name, sha in SOURCE.items()
                if name in ("source_archive_sha256", "source_manifest_sha256"))
        and record.get("individual", {}).get("sha256") == SOURCE["individual_sha256"]
        and record.get("household", {}).get("sha256") == SOURCE["household_sha256"]
        and record.get("individual", {}).get("rows") == PERSONS
        and record.get("household", {}).get("rows") == HOUSEHOLDS
    )


def _validate_source(evidence: dict[str, Any]) -> None:
    _require(isinstance(evidence, dict) and evidence.get("release_id") == RELEASE
             and evidence.get("period") == "2024-Q3"
             and evidence.get("persons") == PERSONS
             and evidence.get("households") == HOUSEHOLDS)
    for name, sha in SOURCE.items():
        _require(evidence.get(name) == sha)
    _require(evidence.get("target_eligibility") == COUNTS)


def _validate_ledger(rows: Any) -> dict[str, dict[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == len(MATRIX))
    found: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict))
        name = MATRIX.get((row.get("architecture_id"), row.get("terminal_formulation")))
        run_id = row.get("run_id")
        _require(name is not None and name not in found
                 and isinstance(run_id, str) and IDENTITY.fullmatch(run_id) is not None
                 and run_id not in seen
                 and run_id.startswith("real_eph_2024q3_"))
        _require(row.get("person_rows") == PERSONS
                 and row.get("household_observations") == HOUSEHOLDS
                 and row.get("complete_households") == COMPLETE
                 and row.get("unavailable_households") == UNAVAILABLE)
        found[name] = row
        seen.add(run_id)
    _require(set(found) == set(MATRIX.values()))
    return found


def _validate_bundle(root: Path, row: dict[str, Any]) -> str:
    # The existing governed validator verifies ALL manifest-listed artifact
    # digests including models/person rows; none of their bytes are projected.
    from encuestador.run_bundle import validate_run_bundle

    manifest = validate_run_bundle(root, verify_hashes=True)
    _require(manifest.get("run_id") == row["run_id"]
             and manifest.get("architecture_id") == row["architecture_id"]
             and manifest.get("terminal", {}).get("formulation") == row["terminal_formulation"]
             and manifest.get("terminal", {}).get("target") == "P47T"
             and isinstance(manifest.get("config_digest"), str)
             and HEX.fullmatch(manifest["config_digest"]) is not None)
    inputs = manifest.get("input_evidence")
    _require(isinstance(inputs, list) and len(inputs) == 1 and _source_shape(inputs[0]))
    _require(_json(root / "input_evidence.json") == inputs)
    metrics = _json(root / "metrics.json")
    _require(isinstance(metrics.get("fold_counts"), dict)
             and sum(metrics["fold_counts"].values()) == PERSONS)
    households = metrics.get("household", {})
    _require(households.get("person_observation_count") == PERSONS
             and households.get("household_observation_count") == HOUSEHOLDS
             and households.get("complete_observed_income_household_count") == COMPLETE
             and households.get("unavailable_observed_income_household_count") == UNAVAILABLE)
    return _hash(root / "run_manifest.json")


def _validate_comparison(record: Any, first: dict[str, Any],
                         second: dict[str, Any], axis: str) -> None:
    _require(isinstance(record, dict)
             and record.get("contract") == "research.encuestador-comparison/v1"
             and record.get("status") == "comparable"
             and record.get("changed_axes") == [axis])
    runs = record.get("runs")
    _require(isinstance(runs, list) and len(runs) == 2)
    expected = {first["run_id"], second["run_id"]}
    actual = {item.get("run_id") for item in runs if isinstance(item, dict)}
    _require(len(actual) == 2 and actual == expected)
    for item in runs:
        match = first if item["run_id"] == first["run_id"] else second
        _require(item.get("architecture_id") == match["architecture_id"]
                 and item.get("terminal_formulation") == match["terminal_formulation"])
    digest = hashlib.sha256("\x1f".join(item["run_id"] for item in runs).encode()).hexdigest()[:16]
    _require(record.get("comparison_id") == f"comparison-{digest}")


def _utc(value: str) -> str:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(instant.tzinfo is not None)
    return instant.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_receipt(*, scheduler_run_id: str, attempt: str, event: str,
                  job_status: str, evidence_upload: str, started_at: str,
                  finished_at: str, evidence_root: Path, runs_root: Path,
                  comparisons_root: Path) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9]+", scheduler_run_id) is not None
             and re.fullmatch(r"[1-9][0-9]*", attempt) is not None
             and event in ("schedule", "workflow_dispatch", "pull_request")
             and job_status in ("success", "failure", "cancelled", "skipped")
             and evidence_upload in ("success", "failure", "skipped", "cancelled"))
    start, finish = _utc(started_at), _utc(finished_at)
    _require(start <= finish)
    early = job_status != "success"
    stages = {key: "unknown" for key in STAGES}
    stages["workflow"] = "success" if job_status == "success" else (
        "failed" if job_status == "failure" else "unknown")
    stages["evidence_upload"] = (
        "success" if evidence_upload == "success"
        else "failed" if evidence_upload == "failure" else "skipped")
    source_path = evidence_root / "real-eph-target-evidence.json"
    ledger_path = evidence_root / "real-eph-run-ledger.json"
    stages["official_source"] = _observed(source_path, _validate_source, early_failure=early)
    stages["target_eligibility"] = stages["official_source"]
    # An invalid source must fail both source and target gates.
    ledger: dict[str, dict[str, Any]] | None = None
    if not ledger_path.is_file():
        stages["run_ledger"] = "skipped" if early else "failed"
    else:
        try:
            ledger = _validate_ledger(_json(ledger_path))
            stages["run_ledger"] = "success"
        except (OSError, ValueError, KeyError, TypeError):
            stages["run_ledger"] = "failed"

    bundle_digests: dict[str, str] = {}
    if ledger is not None:
        expected_ids = {row["run_id"] for row in ledger.values()}
        # Extra directories make the run evidence ambiguous.
        actual_ids = {p.name for p in runs_root.iterdir() if p.is_dir()} if runs_root.is_dir() else set()
        if actual_ids != expected_ids:
            stages["run_ledger"] = "failed"
        for name, row in ledger.items():
            root = runs_root / row["run_id"]
            if not root.is_dir():
                stages[f"bundle_{name}"] = "skipped" if early else "failed"
                continue
            try:
                bundle_digests[name] = _validate_bundle(root, row)
                stages[f"bundle_{name}"] = "success"
            except Exception:
                stages[f"bundle_{name}"] = "failed"
    else:
        for name in MATRIX.values():
            stages[f"bundle_{name}"] = "skipped" if early else "failed"

    comparison_digests: dict[str, str] = {}
    if ledger is not None and comparisons_root.is_dir():
        paths = list(comparisons_root.glob("comparison-*.json"))
        # The expected matrix contains exactly four immutable comparisons.
        if len(paths) != len(PAIRS):
            for name in PAIRS:
                stages[f"comparison_{name}"] = "failed" if not early else "skipped"
        else:
            used: set[Path] = set()
            for name, (a, b, axis) in PAIRS.items():
                matches = []
                for path in paths:
                    try:
                        comparison = _json(path)
                        if {x.get("run_id") for x in comparison.get("runs", [])} == {
                            ledger[a]["run_id"], ledger[b]["run_id"]
                        }:
                            matches.append((path, comparison))
                    except (OSError, ValueError, TypeError, AttributeError):
                        continue
                if len(matches) != 1:
                    stages[f"comparison_{name}"] = "failed"
                else:
                    path, comparison = matches[0]
                    try:
                        _validate_comparison(comparison, ledger[a], ledger[b], axis)
                        _require(path.stem == comparison["comparison_id"] and path not in used)
                        comparison_digests[name] = _hash(path)
                        stages[f"comparison_{name}"] = "success"
                        used.add(path)
                    except (OSError, ValueError, KeyError, TypeError):
                        stages[f"comparison_{name}"] = "failed"
    else:
        for name in PAIRS:
            stages[f"comparison_{name}"] = "skipped" if early else "failed"

    science_keys = [x for x in STAGES if x not in ("workflow", "evidence_upload")]
    science_healthy = all(stages[x] == "success" for x in science_keys)
    failed = any(stages[x] == "failed" for x in science_keys)
    # A complete science validation can coexist with a failed upload, but
    # the declared run_evidence output is then not durably published.
    artifact_state = ("healthy" if science_healthy and stages["evidence_upload"] == "success"
                      else "failed" if failed else "degraded" if science_healthy else "unknown")
    output_state = ("valid" if artifact_state == "healthy"
                    else "invalid" if artifact_state == "failed"
                    else "partial" if ledger is not None or stages["official_source"] == "success"
                    else "missing" if early else "unknown")
    execution_state = "healthy" if job_status == "success" else (
        "failed" if job_status == "failure" else "unknown")
    terminal_state = ("healthy" if execution_state == "healthy" and artifact_state == "healthy"
                      else "failed" if execution_state == "failed" or artifact_state == "failed"
                      else "degraded" if artifact_state == "degraded" else "unknown")
    warnings = [f"{name}:{value}" for name, value in stages.items() if value != "success"]
    warnings.append("bounded_science_evidence_not_public_release")
    refs = [f"github-actions:{REPO}:{scheduler_run_id}:attempt:{attempt}",
            f"eph-release:{RELEASE}"]
    if stages["target_eligibility"] == "success":
        refs.append("eph-eligibility:rows-47564:households-16650:eligible-41821:"
                    "positive-25209:zero-16612:nonresponse-5688:missing-55")
    for name in MATRIX.values():
        if name in bundle_digests:
            refs.append(f"run-manifest-sha256:{name}:{bundle_digests[name]}")
    for name in PAIRS:
        if name in comparison_digests:
            refs.append(f"comparison-sha256:{name}:{comparison_digests[name]}")
    return {
        "contract": CONTRACT,
        "run_id": f"github-actions:{REPO}:{scheduler_run_id}:{attempt}",
        "producer_id": PRODUCER,
        "scheduler_run_id": scheduler_run_id,
        "trigger": {"schedule": "schedule", "workflow_dispatch": "manual", "pull_request": "event"}[event],
        "started_at": start,
        "finished_at": finish,
        "inputs": [{"ref": f"publicdata.eph-microdata@1:{RELEASE}",
                    "version": SOURCE["source_manifest_sha256"]}],
        "stages": stages,
        "outputs": [{
            "type": OUTPUT,
            "state": output_state,
            "durability": "run_evidence",
            **({"ref": f"github-actions:{REPO}:{scheduler_run_id}:attempt:{attempt}:artifact:real-eph-2024q3-science-evidence"}
               if stages["evidence_upload"] == "success" else {}),
        }],
        "result": {
            "scheduler_state": "healthy" if event in ("schedule", "workflow_dispatch") else "not_applicable",
            "execution_state": execution_state,
            "artifact_state": artifact_state,
            "terminal_state": terminal_state,
            "authority_state": "hold",
        },
        "warnings": warnings,
        "evidence_refs": refs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--job-status", required=True)
    parser.add_argument("--evidence-upload", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", default=None)
    parser.add_argument("--evidence-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--runs-root", type=Path, default=Path("/tmp/encuestador-runs"))
    parser.add_argument("--comparisons-root", type=Path, default=Path("/tmp/comparisons"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = build_receipt(
            scheduler_run_id=args.scheduler_run_id, attempt=args.attempt,
            event=args.event, job_status=args.job_status,
            evidence_upload=args.evidence_upload,
            started_at=args.started_at,
            finished_at=args.finished_at or datetime.now(timezone.utc).isoformat(),
            evidence_root=args.evidence_root, runs_root=args.runs_root,
            comparisons_root=args.comparisons_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"producer_id": PRODUCER, "run_id": receipt["run_id"],
                          "artifact_state": receipt["result"]["artifact_state"],
                          "terminal_state": receipt["result"]["terminal_state"]},
                         sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError):
        # Fail closed; never print untrusted file contents, paths or raw exception.
        print("real-EPH receipt arguments or timestamps invalid", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
