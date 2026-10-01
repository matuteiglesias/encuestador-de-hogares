"""C7A: bounded private Gate-B -> donor-clock C7 panel plane.

Selection and income policy are imported directly from Gate B/R0.  The
1.87M-row L2 and C6 inputs are streamed/indexed on disk; only 2,000 links
and their referenced observations are resident at a time.  No Census
inference and no L12 transition model are implemented here.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from .longitudinal_c6 import (
    FOLD_POLICY,
    MODEL_PLANE_CONTRACT,
    LongitudinalModelPlane,
    _canonical_json,
    _household_fold,
    _peak_rss_gib,
    _sha256,
)
from .longitudinal_c7_contract import (
    C7_COMPOSITION_PROFILE,
    C7_CONTEXT_MODE,
    C7_INPUT_POLICY,
    C7ContractError,
    C7PairProjector,
    validate_gate_b_parent,
)
from .longitudinal_gate_b import (
    LINK_FIELDS,
    PERSON_FIELDS,
    STATE_LABELS,
    _csv_rows,
    _prefetch_persons,
    _project_persons,
    _sha,
    _verify_l2,
)
from .longitudinal_runtime import LABOR_CONTEXT_FIELDS

C7_PLANE_CONTRACT = "research.encuestador-c7-donor-clock-panel-plane/v1"
C7_RUN_CONTRACT = "research.encuestador-c7-matched-l11-run/v1"
C7_BASE_FEATURES = ("elapsed_quarters",)
C7_L11_FEATURES = ("stale_labor_state",)
CHUNK = 2000


def _progress(message: str) -> None:
    print(f"[c7] {message}", file=sys.stderr, flush=True)


def _receipt(root: Path) -> dict[str, Any]:
    path = root / "gate_b_receipt.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C7ContractError("c7_gate_b_receipt_unreadable") from exc
    if record.get("release_id") != root.name or record.get("status") != "descriptive_evidence_only":
        raise C7ContractError("c7_gate_b_receipt_identity_invalid")
    for name in ("panel_support.csv", "transition_matrix.csv"):
        info = (record.get("artifacts") or {}).get(name) or {}
        if _sha(root / name) != info.get("sha256"):
            raise C7ContractError(f"c7_gate_b_aggregate_hash_mismatch:{name}")
    return record


def _prefetch_positions(
    connection: sqlite3.Connection, block: list[dict[str, str]]
) -> dict[str, int]:
    ids = sorted({
        row_id
        for link in block
        for row_id in (link["previous_row_id"], link["current_row_id"])
        if row_id
    })
    found: dict[str, int] = {}
    for start in range(0, len(ids), 400):
        keys = ids[start : start + 400]
        query = "SELECT row_id, position FROM c6_position WHERE row_id IN (" + (
            ",".join("?" for _ in keys)
        ) + ")"
        for row in connection.execute(query, keys):
            found[str(row[0])] = int(row[1])
    return found


def _index_c6_positions(
    connection: sqlite3.Connection,
    eph_root: Path,
    plane: LongitudinalModelPlane,
) -> int:
    """Verify EVERY L2 row ID matches C6 at the same position, while indexing.

    Equal cardinality or parent labels alone are not enough: this exact
    lockstep verification prevents silent donor-X misalignment.
    """
    connection.execute(
        "CREATE TABLE c6_position (row_id TEXT PRIMARY KEY, position INTEGER NOT NULL)"
    )
    count = 0
    batch: list[tuple[str, int]] = []
    digest = hashlib.sha256()
    source = _csv_rows(eph_root / "persons.csv", PERSON_FIELDS)
    with plane.row_ids_path.open(encoding="utf-8") as ids:
        for item in zip_longest(source, ids):
            person, text = item
            if person is None or text is None:
                raise C7ContractError("c7_l2_c6_row_count_mismatch")
            row_id = str(person["row_id"])
            if text.rstrip("\n\r") != row_id:
                raise C7ContractError(f"c7_l2_c6_row_order_mismatch:{count}")
            digest.update((row_id + "\n").encode())
            batch.append((row_id, count))
            count += 1
            if len(batch) == 10000:
                connection.executemany("INSERT INTO c6_position VALUES (?,?)", batch)
                batch.clear()
                if count % 200000 == 0:
                    _progress(f"lockstep indexed {count:,} L2/C6 row IDs")
        if batch:
            connection.executemany("INSERT INTO c6_position VALUES (?,?)", batch)
    if count != plane.row_count or digest.hexdigest() != plane.manifest["row_identity_sequence_sha256"]:
        raise C7ContractError("c7_l2_c6_sequence_digest_mismatch")
    connection.commit()
    return count


@dataclass(frozen=True)
class C7PanelPlane:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    @property
    def row_count(self) -> int:
        return int(self.manifest["row_count"])

    @property
    def n_splits(self) -> int:
        return int(self.manifest["n_splits"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["feature_names"])

    @property
    def baseline_features(self) -> tuple[str, ...]:
        return self.feature_names[:-1]


def load_c7_panel(root: Path, *, verify_hashes: bool = True) -> C7PanelPlane:
    root = Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C7ContractError("c7_plane_manifest_unreadable") from exc
    if manifest.get("contract") != C7_PLANE_CONTRACT or manifest.get("release_id") != root.name:
        raise C7ContractError("c7_plane_contract_or_identity_invalid")
    n = manifest.get("row_count")
    if type(n) is not int or n < 1 or manifest.get("n_splits", 0) < 3:
        raise C7ContractError("c7_plane_row_or_fold_count_invalid")
    if manifest.get("feature_names") != (
        list(manifest.get("composition_features") or ())
        + list(LABOR_CONTEXT_FIELDS) + ["elapsed_quarters", "stale_labor_state"]
    ):
        raise C7ContractError("c7_plane_feature_schema_invalid")
    required = {
        "features.npy": (n, len(manifest["feature_names"])),
        "target.npy": (n,),
        "fold_ids.npy": (n,),
        "period_codes.npy": (n,),
        "gap.npy": (n,),
        "stale_labor.npy": (n,),
        "target_labor.npy": (n,),
        "target_household_codes.npy": (n,),
    }
    for name, shape in required.items():
        file = root / name
        info = (manifest.get("artifacts") or {}).get(name) or {}
        if not file.is_file() or (verify_hashes and _sha256(file) != info.get("sha256")):
            raise C7ContractError(f"c7_plane_artifact_missing_or_tampered:{name}")
        if np.load(file, mmap_mode="r").shape != shape:
            raise C7ContractError(f"c7_plane_artifact_shape_invalid:{name}")
    for name in ("pairs.csv",):
        file = root / name
        info = (manifest.get("artifacts") or {}).get(name) or {}
        if not file.is_file() or (verify_hashes and _sha256(file) != info.get("sha256")):
            raise C7ContractError(f"c7_plane_artifact_missing_or_tampered:{name}")
    folds = np.load(root / "fold_ids.npy", mmap_mode="r")
    if (folds >= manifest["n_splits"]).any():
        raise C7ContractError("c7_plane_fold_out_of_range")
    return C7PanelPlane(root, manifest, _sha256(root / "manifest.json"))


def materialize_c7_panel(
    eph_release_root: Path,
    gate_b_root: Path,
    c6_plane: LongitudinalModelPlane,
    output_root: Path,
) -> Path:
    """Private panel projection with exact parent custody and Gate-B reconciliation."""
    eph_root = Path(eph_release_root).expanduser().resolve()
    gate_root = Path(gate_b_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if c6_plane.manifest.get("contract") != MODEL_PLANE_CONTRACT:
        raise C7ContractError("c7_c6_contract_invalid")
    if c6_plane.manifest.get("profile_id") != C7_COMPOSITION_PROFILE:
        raise C7ContractError("c7_requires_p1r_nolab_parent")
    if tuple(c6_plane.manifest.get("labor_context_fields") or ()) != LABOR_CONTEXT_FIELDS:
        raise C7ContractError("c7_c6_labor_context_schema_changed")
    if tuple(c6_plane.feature_names) != (
        *c6_plane.composition_features, *LABOR_CONTEXT_FIELDS
    ):
        raise C7ContractError("c7_c6_feature_order_changed")
    if c6_plane.manifest.get("target_field") != "P47T_real":
        raise C7ContractError("c7_c6_target_field_changed")
    if c6_plane.manifest.get("fold_policy") != FOLD_POLICY:
        raise C7ContractError("c7_c6_fold_policy_changed")
    manifest, hashes, manifest_sha = _verify_l2(eph_root)
    parent = (c6_plane.manifest.get("parents") or {}).get("longitudinal_eph") or {}
    if (parent.get("release_id"), parent.get("manifest_sha256")) != (
        eph_root.name, manifest_sha
    ):
        raise C7ContractError("c7_c6_l2_parent_identity_mismatch")
    receipt = _receipt(gate_root)
    validate_gate_b_parent(
        receipt,
        l2_release_id=eph_root.name,
        l2_manifest_sha256=manifest_sha,
        persons_sha256=hashes["persons.csv"],
        panel_links_sha256=hashes["panel_links.csv"],
        exclude_exceptional=False,
    )
    if receipt.get("person_observations_indexed") != c6_plane.row_count:
        raise C7ContractError("c7_person_parent_count_mismatch")
    payload = {
        "contract": C7_PLANE_CONTRACT,
        "policy": C7_INPUT_POLICY,
        "l2_release_id": eph_root.name,
        "l2_manifest_sha256": manifest_sha,
        "gate_b_release_id": gate_root.name,
        "gate_b_receipt_sha256": _sha256(gate_root / "gate_b_receipt.json"),
        "c6_release_id": c6_plane.release_id,
        "c6_manifest_sha256": c6_plane.manifest_sha256,
        "context_mode": C7_CONTEXT_MODE,
        "feature_profile_id": C7_COMPOSITION_PROFILE,
    }
    release_hash = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]
    destination = output_root / ("longitudinal-c7-panel-" + release_hash)
    if destination.exists():
        existing = load_c7_panel(destination)
        if any(existing.manifest.get(k) != v for k, v in payload.items()):
            raise C7ContractError("c7_existing_plane_identity_conflict")
        return destination
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".longitudinal-c7-panel.", dir=output_root))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(staging / "_scratch.sqlite")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-131072")
        connection.execute("PRAGMA temp_store=MEMORY")
        _progress("constructing audited L2 person index")
        person_count = _project_persons(connection, eph_root / "persons.csv")
        if person_count != c6_plane.row_count:
            raise C7ContractError("c7_l2_person_count_mismatch")
        _progress("checking exact C6/L2 row-ID sequence in lockstep")
        _index_c6_positions(connection, eph_root, c6_plane)
        connection.execute(
            "CREATE TABLE selected (seq INTEGER PRIMARY KEY,"
            " stale_idx INTEGER NOT NULL, target_idx INTEGER NOT NULL,"
            " pair_id TEXT NOT NULL, stale_id TEXT NOT NULL, target_id TEXT NOT NULL,"
            " gap INTEGER NOT NULL, stale_state INTEGER NOT NULL,"
            " target_state INTEGER NOT NULL, target_period TEXT NOT NULL)"
        )
        projector = C7PairProjector()
        exclusions = Counter()
        eligible_by_gap = Counter()
        support = Counter()
        transitions = Counter()
        target_invalid = Counter()
        total = 0
        accepted = 0
        _progress("replaying Gate-B links in bounded batches")
        links = iter(_csv_rows(eph_root / "panel_links.csv", LINK_FIELDS))
        while block := list(islice(links, CHUNK)):
            person_cache = _prefetch_persons(connection, block)
            positions = _prefetch_positions(connection, block)
            selected_batch = []
            for link in block:
                total += 1
                decision = projector.select(link, person_cache)
                if not decision.gate_b_eligible:
                    exclusions[decision.exclusion_reason] += 1
                    continue
                prior_row = person_cache[link["previous_row_id"]]
                later_row = person_cache[link["current_row_id"]]
                gap = int(link["elapsed_quarters"])
                eligible_by_gap[gap] += 1
                support[(gap, prior_row["period"], later_row["period"], later_row["region"])] += 1
                transitions[(gap, prior_row["estado"] or prior_row["condact"], later_row["estado"] or later_row["condact"])] += 1
                if decision.private_pair is None:
                    target_invalid[decision.exclusion_reason] += 1
                    continue
                row = decision.private_pair
                stale_idx = positions.get(row["stale_observation_row_id"])
                target_idx = positions.get(row["target_observation_row_id"])
                if stale_idx is None or target_idx is None:
                    raise C7ContractError("c7_selected_c6_row_position_missing")
                selected_batch.append((
                    stale_idx, target_idx, row["pair_id"],
                    row["stale_observation_row_id"], row["target_observation_row_id"],
                    gap, int(row["stale_labor_state"]), int(row["target_current_labor_state"]),
                    row["target_period"],
                ))
                accepted += 1
            if selected_batch:
                connection.executemany(
                    "INSERT INTO selected (stale_idx,target_idx,pair_id,stale_id,"
                    "target_id,gap,stale_state,target_state,target_period)"
                    " VALUES (?,?,?,?,?,?,?,?,?)", selected_batch
                )
            if total % 20000 == 0:
                connection.commit()
            if total % 50000 == 0:
                _progress(f"audited {total:,} links; Gate-B eligible "
                          f"{sum(eligible_by_gap.values()):,}; valid-income {accepted:,}")
        connection.commit()
        if total != receipt["candidate_link_rows"] or sum(eligible_by_gap.values()) != receipt["eligible_pairs"]:
            raise C7ContractError("c7_gate_b_total_reconciliation_failed")
        if {str(g): eligible_by_gap[g] for g in (1, 3)} != receipt["eligible_pairs_by_gap"]:
            raise C7ContractError("c7_gate_b_gap_reconciliation_failed")
        if dict(sorted(exclusions.items())) != receipt["exclusions_by_first_reason"]:
            raise C7ContractError("c7_gate_b_first_exclusion_reconciliation_failed")
        if len(projector.seen_gate_b_target_rows) != receipt["distinct_eligible_later_observations"]:
            raise C7ContractError("c7_gate_b_unique_target_count_mismatch")
        if total != sum(exclusions.values()) + sum(eligible_by_gap.values()):
            raise C7ContractError("c7_gate_b_exclusive_accounting_failed")
        with (gate_root / "panel_support.csv").open(encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream):
                if record["row_kind"] == "period_region":
                    key = (
                        int(record["elapsed_quarters"]), record["earlier_period"],
                        record["later_period"], record["later_region"],
                    )
                    if support.pop(key, 0) != int(record["eligible_pairs"]):
                        raise C7ContractError("c7_gate_b_support_cell_mismatch")
        if support:
            raise C7ContractError("c7_gate_b_unaccounted_support_cell")
        with (gate_root / "transition_matrix.csv").open(encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream):
                earlier = next((key for key, value in STATE_LABELS.items()
                                if value == record["earlier_state"]), None)
                later = next((key for key, value in STATE_LABELS.items()
                              if value == record["later_state"]), None)
                key = (int(record["elapsed_quarters"]), earlier, later)
                if transitions.pop(key, 0) != int(record["pair_count"]):
                    raise C7ContractError("c7_gate_b_transition_cell_mismatch")
        if transitions:
            raise C7ContractError("c7_gate_b_unaccounted_transition_cell")
        if accepted != sum(eligible_by_gap.values()) - sum(target_invalid.values()):
            raise C7ContractError("c7_extra_income_reconciliation_failed")
        if accepted < c6_plane.n_splits:
            raise C7ContractError("c7_valid_panel_too_small")
        _progress(f"Gate-B aggregates reconciled; projecting {accepted:,} donor-X target pairs")

        comp = len(c6_plane.composition_features)
        names = (
            *c6_plane.composition_features, *LABOR_CONTEXT_FIELDS,
            *C7_BASE_FEATURES, *C7_L11_FEATURES,
        )
        feature_matrix = np.lib.format.open_memmap(
            staging / "features.npy", mode="w+", dtype=np.float64,
            shape=(accepted, len(names)),
        )
        output_arrays = {
            name: np.lib.format.open_memmap(
                staging / (name + ".npy"), mode="w+", dtype=dtype, shape=(accepted,),
            )
            for name, dtype in (
                ("target", np.float64), ("period_codes", np.int16),
                ("fold_ids", np.uint8), ("gap", np.uint8),
                ("stale_labor", np.uint8), ("target_labor", np.uint8),
                ("target_household_codes", np.int32),
            )
        }
        old_features = np.load(c6_plane.features_path, mmap_mode="r")
        old_target = np.load(c6_plane.target_path, mmap_mode="r")
        old_period = np.load(c6_plane.period_codes_path, mmap_mode="r")
        old_fold = np.load(c6_plane.fold_ids_path, mmap_mode="r")
        old_household = np.load(c6_plane.household_codes_path, mmap_mode="r")
        labels = tuple(c6_plane.manifest["period_labels"])
        pair_identity = hashlib.sha256()
        with (staging / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("pair_id", "stale_observation_row_id",
                             "target_observation_row_id", "elapsed_quarters",
                             "stale_labor_state", "target_current_labor_state",
                             "target_period", "fold_id"))
            selected_rows: Iterator[sqlite3.Row] = iter(
                connection.execute("SELECT * FROM selected ORDER BY seq")
            )
            for index, row in enumerate(selected_rows):
                early, late = int(row["stale_idx"]), int(row["target_idx"])
                early_id, later_id = row["stale_id"], row["target_id"]
                period = row["target_period"]
                if labels[int(old_period[late])] != period:
                    raise C7ContractError("c7_selected_target_period_mismatch")
                if int(old_fold[early]) != int(old_fold[late]):
                    raise C7ContractError("c7_panel_household_fold_drift")
                if int(old_fold[early]) >= c6_plane.n_splits:
                    raise C7ContractError("c7_panel_fold_invalid")
                target_amount = float(old_target[late])
                if not math.isfinite(target_amount) or target_amount < 0:
                    raise C7ContractError("c7_selected_c6_income_invalid")
                feature_matrix[index, :comp] = old_features[early, :comp]
                feature_matrix[index, comp:comp+len(LABOR_CONTEXT_FIELDS)] = (
                    old_features[late, comp:]
                )
                feature_matrix[index, -2:] = (int(row["gap"]), int(row["stale_state"]))
                output_arrays["target"][index] = target_amount
                output_arrays["period_codes"][index] = old_period[late]
                output_arrays["fold_ids"][index] = old_fold[late]
                output_arrays["gap"][index] = row["gap"]
                output_arrays["stale_labor"][index] = row["stale_state"]
                output_arrays["target_labor"][index] = row["target_state"]
                output_arrays["target_household_codes"][index] = old_household[late]
                pair_identity.update((row["pair_id"] + "\n").encode())
                writer.writerow((
                    row["pair_id"], early_id, later_id, row["gap"],
                    row["stale_state"], row["target_state"],
                    period, int(old_fold[late]),
                ))
                if index % 200000 == 199999:
                    _progress(f"projected {index + 1:,}/{accepted:,} C7 rows")
        for array in (feature_matrix, *output_arrays.values()):
            array.flush()
        del feature_matrix, output_arrays, old_features, old_target, old_period
        del old_fold, old_household
        gc.collect()
        observed_folds = set(np.load(staging / "fold_ids.npy", mmap_mode="r").tolist())
        if observed_folds != set(range(c6_plane.n_splits)):
            raise C7ContractError("c7_some_outer_folds_empty")
        _progress("rechecking immutable source hashes")
        if _verify_l2(eph_root)[2] != manifest_sha:
            raise C7ContractError("c7_l2_parent_mutated_during_intake")
        if _sha256(c6_plane.root / "manifest.json") != c6_plane.manifest_sha256:
            raise C7ContractError("c7_c6_manifest_mutated_during_intake")
        if _sha256(gate_root / "gate_b_receipt.json") != payload["gate_b_receipt_sha256"]:
            raise C7ContractError("c7_gate_b_parent_mutated_during_intake")
        connection.close()
        connection = None
        (staging / "_scratch.sqlite").unlink()
        artifacts = {}
        for name in (
            "features.npy", "target.npy", "period_codes.npy", "fold_ids.npy",
            "gap.npy", "stale_labor.npy", "target_labor.npy",
            "target_household_codes.npy", "pairs.csv",
        ):
            file = staging / name
            artifacts[name] = {"sha256": _sha256(file), "bytes": file.stat().st_size}
        output_manifest = {
            **payload, "release_id": destination.name,
            "source_person_rows": person_count, "source_candidate_links": total,
            "gate_b_eligible_pairs": sum(eligible_by_gap.values()),
            "gate_b_eligible_pairs_by_gap": {str(g): eligible_by_gap[g] for g in (1, 3)},
            "gate_b_first_exclusions": dict(sorted(exclusions.items())),
            "extra_invalid_later_income": dict(sorted(target_invalid.items())),
            "row_count": accepted, "valid_zero_income_retained": True,
            "row_identity_sequence_sha256": pair_identity.hexdigest(),
            "fold_ids_sha256": artifacts["fold_ids.npy"]["sha256"],
            "fold_policy": FOLD_POLICY, "n_splits": c6_plane.n_splits,
            "feature_names": list(names),
            "composition_features": list(c6_plane.composition_features),
            "categorical_features": list(c6_plane.categorical_features),
            "baseline_feature_names": list(names[:-1]),
            "l11_feature_names": list(names),
            "period_labels": list(labels),
            "source_clocks": {
                "composition": "stale_observation_row_id",
                "observed_labor": "stale_observation_row_id",
                "aggregate_context": "target_observation_row_id",
                "welfare_target": "target_observation_row_id",
            },
            "target": "later_P47T_real",
            "household_aggregation_scope": "eligible_panel_pair_members_only",
            "privacy": "private_row_linkage_artifact_no_public_export",
            "measurement_mode": True, "forecasting_authorized": False,
            "long_horizon_census_transport_authorized": False,
            "resource": {"peak_rss_gib": _peak_rss_gib()},
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            _canonical_json(output_manifest), encoding="utf-8"
        )
        os.replace(staging, destination)
        _progress(f"materialized {destination}")
        return destination
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
