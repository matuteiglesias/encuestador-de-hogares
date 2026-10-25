#!/usr/bin/env python3
"""Strict metadata projection and keyless Cloud Logging export of WP03-D receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from emit_real_eph_receipt import (
    CONTRACT, MATRIX, OUTPUT, PAIRS, PRODUCER, RELEASE, REPO, SOURCE, STAGES,
)

PROJECT = "estate-observability-510700"
LOG = "estate-producer-run"
OWNER = "repo.encuestador-de-hogares"
RUN_FIELDS = {"contract", "run_id", "producer_id", "scheduler_run_id",
              "trigger", "started_at", "finished_at", "inputs",
              "stages", "outputs", "result", "warnings", "evidence_refs"}
STATES = {"success", "failed", "skipped", "not_applicable", "unknown"}
EXPECTED_RESULT = {"scheduler_state", "execution_state", "artifact_state",
                   "terminal_state", "authority_state"}
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExportError(ValueError):
    pass


def require(value: bool) -> None:
    if not value:
        raise ExportError("unsafe_or_invalid_receipt")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def validate(receipt: dict[str, Any], run_id: str, attempt: str) -> None:
    require(isinstance(receipt, dict) and set(receipt) == RUN_FIELDS)
    require(re.fullmatch(r"[0-9]+", run_id) is not None
            and re.fullmatch(r"[1-9][0-9]*", attempt) is not None)
    require(receipt["contract"] == CONTRACT and receipt["producer_id"] == PRODUCER
            and receipt["run_id"] == f"github-actions:{REPO}:{run_id}:{attempt}"
            and receipt["scheduler_run_id"] == run_id
            and receipt["trigger"] in ("schedule", "manual", "event"))
    require(isinstance(receipt["started_at"], str)
            and isinstance(receipt["finished_at"], str)
            and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", receipt["started_at"]) is not None
            and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", receipt["finished_at"]) is not None
            and receipt["started_at"] <= receipt["finished_at"])
    require(receipt["inputs"] == [{
        "ref": f"publicdata.eph-microdata@1:{RELEASE}",
        "version": SOURCE["source_manifest_sha256"],
    }])
    stages = receipt["stages"]
    require(isinstance(stages, dict) and set(stages) == set(STAGES)
            and all(s in STATES for s in stages.values()))
    result = receipt["result"]
    require(isinstance(result, dict) and set(result) == EXPECTED_RESULT)
    require(result["authority_state"] == "hold"
            and result["scheduler_state"] in ("healthy", "not_applicable")
            and result["execution_state"] in ("healthy", "failed", "unknown")
            and result["artifact_state"] in ("healthy", "degraded", "failed", "unknown")
            and result["terminal_state"] in ("healthy", "failed", "degraded", "unknown"))
    require((result["scheduler_state"] == "not_applicable") == (receipt["trigger"] == "event"))
    semantic = [k for k in STAGES if k not in ("workflow", "evidence_upload")]
    complete = all(stages[k] == "success" for k in semantic)
    expected_execution = {"success": "healthy", "failed": "failed",
                          "unknown": "unknown"}[stages["workflow"]]
    require(result["execution_state"] == expected_execution)
    semantically_failed = any(stages[k] == "failed" for k in semantic)
    expected_artifact = (
        "healthy" if complete and stages["evidence_upload"] == "success"
        else "failed" if semantically_failed
        else "degraded" if complete else "unknown"
    )
    require(result["artifact_state"] == expected_artifact)
    expected_terminal = (
        "healthy" if expected_execution == "healthy" and expected_artifact == "healthy"
        else "failed" if expected_execution == "failed" or expected_artifact == "failed"
        else "degraded" if expected_artifact == "degraded" else "unknown"
    )
    require(result["terminal_state"] == expected_terminal)
    outputs = receipt["outputs"]
    require(isinstance(outputs, list) and len(outputs) == 1)
    o = outputs[0]
    require(isinstance(o, dict) and set(o) in (
        {"type", "state", "durability"}, {"type", "state", "durability", "ref"}))
    require(o["type"] == OUTPUT and o["durability"] == "run_evidence"
            and o["state"] in ("valid", "partial", "invalid", "missing", "unknown"))
    require((o["state"] == "valid") == (result["artifact_state"] == "healthy"))
    expected_ref = f"github-actions:{REPO}:{run_id}:attempt:{attempt}:artifact:real-eph-2024q3-science-evidence"
    require(("ref" in o) == (stages["evidence_upload"] == "success"))
    if "ref" in o:
        require(o["ref"] == expected_ref)
    warnings = receipt["warnings"]
    require(isinstance(warnings, list)
            and warnings == [f"{k}:{v}" for k, v in stages.items() if v != "success"]
            + ["bounded_science_evidence_not_public_release"])
    refs = receipt["evidence_refs"]
    require(isinstance(refs, list) and len(refs) <= 11 and len(refs) >= 2
            and refs[0] == f"github-actions:{REPO}:{run_id}:attempt:{attempt}"
            and refs[1] == f"eph-release:{RELEASE}")
    summary = ("eph-eligibility:rows-47564:households-16650:eligible-41821:"
               "positive-25209:zero-16612:nonresponse-5688:missing-55")
    if stages["target_eligibility"] == "success":
        require(len(refs) >= 3 and refs[2] == summary)
        hashed_refs = refs[3:]
    else:
        hashed_refs = refs[2:]
    expected_prefixes = [f"run-manifest-sha256:{name}:" for name in MATRIX.values()] + [
        f"comparison-sha256:{name}:" for name in PAIRS]
    seen: set[str] = set()
    for ref in hashed_refs:
        prefixes = [p for p in expected_prefixes if ref.startswith(p)]
        require(len(prefixes) == 1 and prefixes[0] not in seen)
        require(DIGEST_PATTERN.fullmatch(ref[len(prefixes[0]):]) is not None)
        stage = ("bundle_" + prefixes[0].split(":")[1]
                 if prefixes[0].startswith("run-manifest") else
                 "comparison_" + prefixes[0].split(":")[1])
        require(stages[stage] == "success")
        seen.add(prefixes[0])
    # References are mandatory exactly for successfully verified bundles and
    # comparisons, not merely validated when the caller happens to provide them.
    for name in MATRIX.values():
        prefix = f"run-manifest-sha256:{name}:"
        require((prefix in seen) == (stages[f"bundle_{name}"] == "success"))
    for name in PAIRS:
        prefix = f"comparison-sha256:{name}:"
        require((prefix in seen) == (stages[f"comparison_{name}"] == "success"))
    # No free-form source values can cross the GCP boundary.
    serialized = canonical_json(receipt).lower()
    require(len(serialized) <= 14000)
    for forbidden in ("/tmp/", "/home/", "model.pkl", "person_oof",
                      "household_oof", "report.md", "raw_", "p47t", "row_id",
                      "password", "access_token", "api_key", "credential",
                      "secret", "microdata_rows", "\\users\\"):
        # P47T is a target name only in source artifacts, never in the receipt.
        require(forbidden not in serialized)


def build_entry(receipt: dict[str, Any], run_id: str, attempt: str) -> dict[str, Any]:
    validate(receipt, run_id, attempt)
    digest = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
    identity = {"producer_id": PRODUCER, "run_id": receipt["run_id"], "receipt_sha256": digest}
    insert_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    values = {receipt["result"][k] for k in ("execution_state", "artifact_state", "terminal_state")}
    severity = "ERROR" if "failed" in values else (
        "WARNING" if values & {"unknown", "degraded"} else "INFO")
    return {
        "logName": f"projects/{PROJECT}/logs/{LOG}",
        "resource": {"type": "global", "labels": {"project_id": PROJECT}},
        "timestamp": receipt["finished_at"],
        "severity": severity,
        "insertId": insert_id,
        "labels": {"producer_id": PRODUCER, "source_contract": CONTRACT, "receipt_sha256": digest},
        "jsonPayload": {
            "contract": CONTRACT,
            "receipt": receipt,
            "receipt_sha256": digest,
            "logical_receipt_id": f"{PRODUCER}:{receipt['run_id']}",
            "producer": {"producer_id": PRODUCER, "owner": OWNER,
                         "scheduler_kind": "github_actions",
                         "observation_mode": "semantic_receipt",
                         "authority_state": "hold"},
        },
    }


def write(entry: dict[str, Any]) -> None:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/logging.write"])
    session = AuthorizedSession(credentials)
    body = {"logName": f"projects/{PROJECT}/logs/{LOG}",
            "resource": {"type": "global", "labels": {"project_id": PROJECT}},
            "entries": [entry], "partialSuccess": False}
    for dry_run in (True, False):
        response = session.post(
            "https://logging.googleapis.com/v2/entries:write",
            params={"dryRun": "true"} if dry_run else None,
            json=body, timeout=30)
        if not response.ok:
            raise ExportError(f"cloud_logging_status_{response.status_code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--scheduler-run-id", required=True)
    parser.add_argument("--attempt", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        entry = build_entry(receipt, args.scheduler_run_id, args.attempt)
        if args.write:
            write(entry)
        print(json.dumps({"insertId": entry["insertId"],
                          "mode": "exported" if args.write else "validated",
                          "run_id": receipt["run_id"]}, sort_keys=True))
        return 0
    except Exception:
        # Never echo untrusted model evidence, cloud error bodies or local paths.
        print("real-EPH receipt export rejected or unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
