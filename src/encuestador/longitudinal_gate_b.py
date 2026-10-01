"""Bounded descriptive Gate-B evidence on an immutable longitudinal EPH L2 release.

The panel-links artifact is authoritative for candidate linkage/demographic review.
A temporary indexed SQLite projection of persons.csv supplies source observations;
only aggregate evidence survives in the immutable diagnostic output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .longitudinal_runtime import (
    LongitudinalRuntimeError,
    _observed_labor_state,
    _period_index,
)

CONTRACT = "research.encuestador-gate-b-panel-evidence/v1"
L2_CONTRACT = "research.eph-longitudinal-analysis-frame/v1"
SELECTION_POLICY = "l2_consistent_1q3q_age14_reviewed_labor_v1"
VALID_LINK = "demographically_consistent_component_candidate"
GAP_STATUS = "expected_2_2_2_adjacent_gap"
STATES = ("1", "2", "3")
STATE_LABELS = {"1": "employed", "2": "unemployed", "3": "inactive"}
EXCEPTIONAL = frozenset(("2020-Q2", "2024-Q1", "2024-Q2"))
LABOR_CONFIG = SimpleNamespace(observed_labor_fields=("ESTADO", "CONDACT"))

PERSON_FIELDS = (
    "row_id",
    "person_linkage_candidate_id",
    "panel_household_id",
    "period",
    "region_id",
    "CH06",
    "P47T_real",
    "p47t_value_status",
    "monetary_reference_period",
)
LINK_FIELDS = (
    "person_linkage_candidate_id",
    "previous_row_id",
    "current_row_id",
    "previous_period",
    "current_period",
    "elapsed_quarters",
    "person_linkage_status",
    "rotation_gap_status",
)


class GateBEvidenceError(ValueError):
    """A real-source or diagnostic contract failure."""


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path, required: tuple[str, ...]):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = set(required) - columns
            if missing:
                raise GateBEvidenceError(
                    f"{path.name}:missing_columns:{','.join(sorted(missing))}"
                )
            if None in columns:
                raise GateBEvidenceError(f"{path.name}:blank_header")
            for row in reader:
                if None in row:
                    raise GateBEvidenceError(f"{path.name}:malformed_csv_row")
                yield row
    except OSError as exc:
        raise GateBEvidenceError(f"source_unreadable:{path.name}") from exc


def _write_csv(path: Path, fields: tuple[str, ...], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_l2(root: Path) -> tuple[dict[str, Any], dict[str, str], str]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateBEvidenceError("l2_manifest_unreadable") from exc
    if manifest.get("contract") != L2_CONTRACT:
        raise GateBEvidenceError("l2_contract_mismatch")
    if manifest.get("release_id") != root.name:
        raise GateBEvidenceError("l2_release_directory_mismatch")
    inventory = manifest.get("artifacts") or {}
    hashes = {}
    for name in ("persons.csv", "panel_links.csv", "qa.json"):
        expected = (inventory.get(name) or {}).get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise GateBEvidenceError(f"l2_manifest_artifact_hash_missing:{name}")
        path = root / name
        if not path.is_file():
            raise GateBEvidenceError(f"l2_artifact_missing:{name}")
        observed = _sha(path)
        if observed != expected:
            raise GateBEvidenceError(f"l2_artifact_hash_mismatch:{name}")
        hashes[name] = observed
    return manifest, hashes, _sha(manifest_path)


def _progress(message: str) -> None:
    # CLI JSON remains on stdout; visible progress is deliberately stderr only.
    print(f"[gate-b] {message}", file=sys.stderr, flush=True)


def _prefetch_persons(
    connection: sqlite3.Connection, links: list[dict[str, str]],
) -> dict[str, sqlite3.Row]:
    """Use bounded IN queries rather than ~2 Python/SQLite calls per link.

    SQLite still uses the person.row_id primary-key index. Only IDs from the
    current 2,000-link batch are retained in memory, never the 1.87M persons.
    """
    ids = sorted({
        row_id
        for link in links
        if link["person_linkage_status"] == VALID_LINK
        for row_id in (link["previous_row_id"], link["current_row_id"])
        if row_id
    })
    projected: dict[str, sqlite3.Row] = {}
    for start in range(0, len(ids), 400):
        block = ids[start : start + 400]
        query = "SELECT * FROM person WHERE row_id IN (" + ",".join("?" for _ in block) + ")"
        for row in connection.execute(query, block):
            projected[row["row_id"]] = row
    return projected


def _project_persons(connection: sqlite3.Connection, path: Path) -> int:
    connection.execute(
        "CREATE TABLE person ("
        "row_id TEXT PRIMARY KEY, candidate TEXT, household TEXT, period TEXT,"
        "region TEXT, age TEXT, estado TEXT, condact TEXT, income TEXT,"
        "income_status TEXT, monetary_reference TEXT)"
    )
    count = 0
    batch = []
    iterator = _csv_rows(path, PERSON_FIELDS)
    for row in iterator:
        if "ESTADO" not in row and "CONDACT" not in row:
            raise GateBEvidenceError("l2_observed_labor_fields_missing")
        row_id, period = (row.get("row_id") or ""), (row.get("period") or "")
        if not row_id or not period or not row_id.startswith(period + "|"):
            raise GateBEvidenceError("l2_person_row_identity_invalid")
        batch.append(
            (
                row_id,
                row["person_linkage_candidate_id"],
                row["panel_household_id"],
                period,
                row["region_id"],
                row["CH06"],
                row.get("ESTADO") or "",
                row.get("CONDACT") or "",
                row["P47T_real"],
                row["p47t_value_status"],
                row["monetary_reference_period"],
            )
        )
        count += 1
        if len(batch) >= 10000:
            try:
                connection.executemany("INSERT INTO person VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
            except sqlite3.IntegrityError as exc:
                raise GateBEvidenceError("l2_duplicate_person_row_id") from exc
            batch.clear()
            if count % 200000 == 0:
                _progress(f"indexed persons: {count:,}")
    if batch:
        try:
            connection.executemany("INSERT INTO person VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
        except sqlite3.IntegrityError as exc:
            raise GateBEvidenceError("l2_duplicate_person_row_id") from exc
    connection.commit()
    if count == 0:
        raise GateBEvidenceError("l2_persons_empty")
    return count


def _age(value: str) -> int | None:
    # Same bounded numeric age rule used by the L2 _classify_link audit.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    age = int(number)
    return age if 0 <= age <= 120 else None


def _labor(row: sqlite3.Row) -> tuple[str | None, str | None]:
    source = {"ESTADO": row["estado"], "CONDACT": row["condact"]}
    try:
        state = _observed_labor_state(source, LABOR_CONFIG)
    except LongitudinalRuntimeError as exc:
        message = str(exc)
        if message.startswith("observed_labor_state_disagreement:"):
            return None, "labor_valid_code_disagreement"
        if message.startswith("labor_state_outside_reviewed_classes:"):
            return None, "labor_unreviewed_code"
        if message.startswith("observed_labor_state_missing:"):
            return None, "labor_missing"
        raise
    if state is None:
        return None, "labor_special_only"
    return state, None


def _income(row: sqlite3.Row) -> float | None:
    if row["income_status"] not in {"zero", "positive"}:
        return None
    try:
        amount = float(row["income"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount < 0:
        return None
    if row["income_status"] == "zero" and amount != 0:
        return None
    if row["income_status"] == "positive" and amount <= 0:
        return None
    return amount


def _median(
    connection: sqlite3.Connection, gap: int, prior: str, field: str, count: int
) -> float | None:
    if not count:
        return None
    if field not in {"later_income", "income_change"}:
        raise GateBEvidenceError("median_field_invalid")
    query = (
        f"SELECT {field} FROM income_values WHERE gap=? AND prior=? "
        f"AND {field} IS NOT NULL ORDER BY {field} LIMIT ? OFFSET ?"
    )
    offset = (count - 1) // 2
    values = [
        float(row[0]) for row in connection.execute(
            query, (gap, prior, 1 if count % 2 else 2, offset)
        )
    ]
    return sum(values) / len(values)


def _select_reason(
    link: dict[str, str],
    person_cache: dict[str, sqlite3.Row],
    exclude_exceptional: bool,
):
    if link["person_linkage_status"] != VALID_LINK:
        return "link_status:" + (link["person_linkage_status"] or "missing"), None
    try:
        gap = int(link["elapsed_quarters"])
    except (ValueError, TypeError):
        return "elapsed_quarters_invalid", None
    if gap not in (1, 3):
        return "unsupported_gap", None
    if link["rotation_gap_status"] != GAP_STATUS:
        return "rotation_gap_status_inconsistent", None
    previous, current = link["previous_period"], link["current_period"]
    try:
        if _period_index(current) - _period_index(previous) != gap:
            return "period_direction_or_gap_mismatch", None
    except (ValueError, LongitudinalRuntimeError):
        return "period_direction_or_gap_mismatch", None
    if exclude_exceptional and (previous in EXCEPTIONAL or current in EXCEPTIONAL):
        return "exceptional_period_pair", None

    earlier_id, later_id = link["previous_row_id"], link["current_row_id"]
    if not earlier_id or not later_id or earlier_id == later_id:
        return "pair_row_identity_invalid", None
    earlier = person_cache.get(earlier_id)
    later = person_cache.get(later_id)
    if earlier is None or later is None:
        return "identity_join_failure", None
    candidate = link["person_linkage_candidate_id"]
    if (
        not candidate
        or earlier["candidate"] != candidate
        or later["candidate"] != candidate
        or not earlier["household"]
        or earlier["household"] != later["household"]
        or earlier["period"] != previous
        or later["period"] != current
        or not earlier["region"]
        or not later["region"]
    ):
        return "candidate_household_or_period_mismatch", None
    early_age, late_age = _age(earlier["age"]), _age(later["age"])
    if early_age is None or late_age is None or early_age < 14 or late_age < 14:
        return "age14_universe_exclusion", None
    prior, prior_error = _labor(earlier)
    if prior_error:
        return "earlier_" + prior_error, None
    current_state, current_error = _labor(later)
    if current_error:
        return "later_" + current_error, None
    if prior not in STATES or current_state not in STATES:
        return "labor_outside_three_state_universe", None
    return None, (gap, previous, current, earlier, later, prior, current_state)


def run_gate_b(
    eph_release_root: Path,
    output_root: Path,
    *,
    exclude_exceptional: bool = False,
) -> Path:
    """Emit only aggregate L2-linked panel support, transitions and income evidence."""
    root = Path(eph_release_root).expanduser().resolve()
    _progress(f"verifying immutable L2 parent: {root.name}")
    manifest, source_hashes, manifest_sha = _verify_l2(root)
    _progress("source hashes verified; constructing slim SQLite person index")
    payload = {
        "contract": CONTRACT,
        "l2_release_id": root.name,
        "l2_manifest_sha256": manifest_sha,
        "persons_sha256": source_hashes["persons.csv"],
        "panel_links_sha256": source_hashes["panel_links.csv"],
        "selection_policy": SELECTION_POLICY,
        "exclude_exceptional": exclude_exceptional,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    release_id = "gate-b-panel-evidence-" + identity
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / release_id
    if destination.exists():
        raise GateBEvidenceError(f"immutable_gate_b_release_exists:{release_id}")
    staging = Path(tempfile.mkdtemp(prefix=".gate-b.", dir=output_root))
    connection = None
    try:
        connection = sqlite3.connect(staging / "_work.sqlite")
        connection.row_factory = sqlite3.Row
        # This database is ephemeral and deleted on any failure/interruption.
        # Durability of immutable L2 parents and final aggregate outputs is unchanged.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-131072")
        person_count = _project_persons(connection, root / "persons.csv")
        _progress(
            f"indexed {person_count:,} person observations; "
            "evaluating L2 audited links in 2,000-link batches"
        )
        connection.execute(
            "CREATE TABLE eligible_keys (candidate TEXT NOT NULL,"
            " later_id TEXT PRIMARY KEY, gap INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE income_values (gap INTEGER NOT NULL, prior TEXT NOT NULL,"
            " later_income REAL NOT NULL, income_change REAL)"
        )
        support = Counter()
        transitions = Counter()
        exclusions = Counter()
        eligible_gap = Counter()
        exceptional_touch = Counter()
        income_stats = defaultdict(Counter)
        income_sums = defaultdict(lambda: defaultdict(float))
        original_links = 0
        link_iterator = iter(_csv_rows(root / "panel_links.csv", LINK_FIELDS))
        link_start = time.monotonic()
        while block := list(islice(link_iterator, 2000)):
            cache = _prefetch_persons(connection, block)
            for link in block:
                original_links += 1
                reason, pair = _select_reason(link, cache, exclude_exceptional)
                if reason:
                    exclusions[reason] += 1
                    continue
                gap, previous, current, earlier, later, prior, current_state = pair
                try:
                    connection.execute(
                        "INSERT INTO eligible_keys VALUES (?,?,?)",
                        (link["person_linkage_candidate_id"], later["row_id"], gap),
                    )
                except sqlite3.IntegrityError:
                    exclusions["later_observation_reused"] += 1
                    continue
                eligible_gap[gap] += 1
                support[(gap, previous, current, later["region"])] += 1
                transitions[(gap, prior, current_state)] += 1
                if previous in EXCEPTIONAL or current in EXCEPTIONAL:
                    exceptional_touch[gap] += 1

                income_key = (gap, prior)
                income_stats[income_key]["eligible_pairs"] += 1
                later_value = _income(later)
                if later_value is None:
                    income_stats[income_key]["later_income_invalid"] += 1
                else:
                    earlier_value = _income(earlier)
                    change = (
                        later_value - earlier_value if earlier_value is not None else None
                    )
                    income_stats[income_key]["later_income_valid"] += 1
                    income_sums[income_key]["later_income"] += later_value
                    if later_value > 0:
                        income_stats[income_key]["later_positive"] += 1
                        income_sums[income_key]["positive_income"] += later_value
                    if change is not None:
                        income_stats[income_key]["paired_income_valid"] += 1
                        income_sums[income_key]["income_change"] += change
                    connection.execute(
                        "INSERT INTO income_values VALUES (?,?,?,?)",
                        (gap, prior, later_value, change),
                    )
            if original_links % 20000 == 0:
                connection.commit()
            if original_links % 50000 == 0:
                elapsed = max(time.monotonic() - link_start, 0.001)
                _progress(
                    f"audited links: {original_links:,}; eligible: "
                    f"{sum(eligible_gap.values()):,}; "
                    f"excluded: {sum(exclusions.values()):,}; "
                    f"rate: {original_links / elapsed:,.0f} links/s"
                )
        connection.commit()
        _progress(
            f"links complete ({original_links:,}); indexing six income strata "
            "and writing aggregate outputs"
        )

        audit_count = (manifest.get("panel_audit") or {}).get("candidate_link_rows")
        if audit_count is not None and original_links != int(audit_count):
            raise GateBEvidenceError("l2_panel_link_count_manifest_mismatch")
        eligible = sum(eligible_gap.values())
        if original_links != eligible + sum(exclusions.values()):
            raise GateBEvidenceError("gate_b_link_accounting_mismatch")
        connection.execute(
            "CREATE INDEX income_order ON income_values(gap,prior,later_income)"
        )
        connection.execute(
            "CREATE INDEX change_order ON income_values(gap,prior,income_change)"
        )
        unique_candidates = connection.execute(
            "SELECT COUNT(DISTINCT candidate) FROM eligible_keys"
        ).fetchone()[0]
        unique_targets = connection.execute(
            "SELECT COUNT(*) FROM eligible_keys"
        ).fetchone()[0]
        if unique_targets != eligible:
            raise GateBEvidenceError("gate_b_eligible_target_accounting_mismatch")
        unique_by_gap = {
            str(gap): connection.execute(
                "SELECT COUNT(DISTINCT candidate) FROM eligible_keys WHERE gap=?",
                (gap,),
            ).fetchone()[0]
            for gap in (1, 3)
        }

        support_fields = (
            "row_kind", "elapsed_quarters", "earlier_period", "later_period",
            "later_region", "eligible_pairs",
        )
        support_rows = [
            {
                "row_kind": "gap_total",
                "elapsed_quarters": gap,
                "earlier_period": "",
                "later_period": "",
                "later_region": "",
                "eligible_pairs": eligible_gap[gap],
            }
            for gap in (1, 3)
        ]
        support_rows.extend(
            {
                "row_kind": "period_region",
                "elapsed_quarters": gap,
                "earlier_period": previous,
                "later_period": current,
                "later_region": region,
                "eligible_pairs": count,
            }
            for (gap, previous, current, region), count in sorted(support.items())
        )
        _write_csv(staging / "panel_support.csv", support_fields, support_rows)
        matrix_fields = (
            "elapsed_quarters", "earlier_state", "later_state", "pair_count",
            "earlier_state_denominator", "conditional_probability",
        )
        matrix_rows = []
        for gap in (1, 3):
            for prior in STATES:
                denominator = sum(
                    transitions[(gap, prior, current)] for current in STATES
                )
                for current in STATES:
                    count = transitions[(gap, prior, current)]
                    matrix_rows.append(
                        {
                            "elapsed_quarters": gap,
                            "earlier_state": STATE_LABELS[prior],
                            "later_state": STATE_LABELS[current],
                            "pair_count": count,
                            "earlier_state_denominator": denominator,
                            "conditional_probability": (
                                count / denominator if denominator else ""
                            ),
                        }
                    )
        _write_csv(staging / "transition_matrix.csv", matrix_fields, matrix_rows)

        income_fields = (
            "elapsed_quarters", "earlier_state", "eligible_pairs",
            "later_income_valid_n", "later_income_missing_or_invalid_n",
            "later_positive_n", "positive_rate_valid_denominator",
            "mean_later_unconditional_real_income",
            "mean_later_positive_real_income", "median_later_real_income",
            "paired_income_valid_n", "mean_real_income_change",
            "median_real_income_change",
        )
        income_rows = []
        for gap in (1, 3):
            for prior in STATES:
                key = (gap, prior)
                stats = income_stats[key]
                sums = income_sums[key]
                valid = stats["later_income_valid"]
                positive = stats["later_positive"]
                paired = stats["paired_income_valid"]
                income_rows.append(
                    {
                        "elapsed_quarters": gap,
                        "earlier_state": STATE_LABELS[prior],
                        "eligible_pairs": stats["eligible_pairs"],
                        "later_income_valid_n": valid,
                        "later_income_missing_or_invalid_n": stats["later_income_invalid"],
                        "later_positive_n": positive,
                        "positive_rate_valid_denominator": (
                            positive / valid if valid else ""
                        ),
                        "mean_later_unconditional_real_income": (
                            sums["later_income"] / valid if valid else ""
                        ),
                        "mean_later_positive_real_income": (
                            sums["positive_income"] / positive if positive else ""
                        ),
                        "median_later_real_income": (
                            _median(connection, gap, prior, "later_income", valid)
                            if valid else ""
                        ),
                        "paired_income_valid_n": paired,
                        "mean_real_income_change": (
                            sums["income_change"] / paired if paired else ""
                        ),
                        "median_real_income_change": (
                            _median(connection, gap, prior, "income_change", paired)
                            if paired else ""
                        ),
                    }
                )
        _write_csv(staging / "income_by_prior_state.csv", income_fields, income_rows)

        region_counts = Counter()
        period_pair_counts = Counter()
        for (gap, previous, current, region), count in support.items():
            region_counts[region] += count
            period_pair_counts[(previous, current)] += count
        top_region, top_region_count = (
            region_counts.most_common(1)[0] if region_counts else ("", 0)
        )
        top_period_pair, top_period_pair_count = (
            period_pair_counts.most_common(1)[0]
            if period_pair_counts else (("", ""), 0)
        )

        def persistence(gap: int, prior: str) -> str:
            denominator = sum(transitions[(gap, prior, later)] for later in STATES)
            if not denominator:
                return "n/a (n=0)"
            return (
                f"{transitions[(gap, prior, prior)] / denominator:.3f} "
                f"(n={denominator:,})"
            )

        note = [
            "# Gate B — descriptive real-panel evidence",
            "",
            f"- L2 parent: `{root.name}` (SHA-256 `{manifest_sha}`).",
            f"- Selection policy: `{SELECTION_POLICY}`.",
            f"- Exceptional-period pairs excluded: `{exclude_exceptional}`.",
            f"- Original audited candidate links: **{original_links:,}**.",
            f"- Eligible: **{eligible:,}** ({eligible_gap[1]:,} at 1Q; {eligible_gap[3]:,} at 3Q).",
            f"- Excluded: **{sum(exclusions.values()):,}**, with exhaustive reasons in the receipt.",
            f"- Distinct eligible candidate keys: **{unique_candidates:,}**; distinct later observations: **{unique_targets:,}**.",
            "",
            "## Descriptive support and persistence checks",
            "",
            f"- Later regions represented: **{len(region_counts)}**; observed period-pair windows: **{len(period_pair_counts)}**.",
            f"- Largest later-region stratum: **{top_region}**, {top_region_count:,} pairs ({top_region_count / eligible:.1%} of selected pairs)." if eligible else "- No eligible regional stratum.",
            f"- Largest period-pair window: **{top_period_pair[0]}→{top_period_pair[1]}**, {top_period_pair_count:,} pairs ({top_period_pair_count / eligible:.1%})." if eligible else "- No eligible period-pair window.",
            "",
            "## Reading the outputs",
            "",
            "The 18 cells of `transition_matrix.csv` give counts and row-conditional",
            "probabilities separately for 1Q and 3Q. Empty denominator means no",
            "supported observation; no probability is imputed.",
            "",
            "`income_by_prior_state.csv` gives later-income incidence and levels",
            "by earlier state/gap, with valid-target and paired-income denominators.",
            "Zeros are retained; negative/missing/unusable source incomes are not zeros.",
            "",
            "## Interpretation boundary",
            "",
            "These are **unweighted, descriptive candidate-pair associations**, not",
            "independent unique-person estimates, an OOF L11 effect, Census transport",
            "validation or a causal labor effect. Selected rotating-panel pairs may",
            "have linkage/attrition bias. Short 1Q/3Q persistence does not identify",
            "a CPV-2010 to 2024/2026 transition. Monetary conversion parent remains",
            "candidate where declared by L2. L11 needs a matched panel-cohort",
            "L10 baseline with household-safe folds; L12 needs separate OOF science.",
            "",
        ]
        note.append("## Observed diagonal persistence")
        note.append("")
        note.append("Each proportion is conditional on the earlier labor class:")
        note.append("")
        for gap in (1, 3):
            note.append(
                f"- {gap}Q E→E: {persistence(gap, '1')}; "
                f"U→U: {persistence(gap, '2')}; "
                f"I→I: {persistence(gap, '3')}."
            )
        note.append("")
        note.append(
            "These are observed transitions, not an assessment of L11's held-out "
            "incremental value. Inspect the aggregate income table for later-income "
            "association and its explicit valid-denominator counts."
        )
        note.append("")
        (staging / "GATE_B_NOTE.md").write_text("\n".join(note), encoding="utf-8")
        monetary = manifest.get("monetary_lineage") or {}
        monetary_parent = (manifest.get("parents") or {}).get("monetary_conversion") or {}
        artifacts = {
            name: {"sha256": _sha(staging / name), "bytes": (staging / name).stat().st_size}
            for name in (
                "panel_support.csv",
                "transition_matrix.csv",
                "income_by_prior_state.csv",
                "GATE_B_NOTE.md",
            )
        }
        receipt = {
            **payload,
            "release_id": release_id,
            "status": "descriptive_evidence_only",
            "l2_qa_sha256": source_hashes["qa.json"],
            "person_observations_indexed": person_count,
            "candidate_link_rows": original_links,
            "eligible_pairs": eligible,
            "eligible_pairs_by_gap": {str(gap): eligible_gap[gap] for gap in (1, 3)},
            "exclusion_count": sum(exclusions.values()),
            "exclusions_by_first_reason": dict(sorted(exclusions.items())),
            "distinct_eligible_candidate_keys": unique_candidates,
            "distinct_eligible_later_observations": unique_targets,
            "distinct_candidate_keys_by_gap": unique_by_gap,
            "support_diagnostics": {
                "regions_with_eligible_pairs": len(region_counts),
                "observed_period_pair_windows": len(period_pair_counts),
                "largest_region": top_region,
                "largest_region_pair_count": top_region_count,
                "largest_period_pair": list(top_period_pair),
                "largest_period_pair_count": top_period_pair_count,
                "diagonal_persistence": {
                    str(gap): {
                        STATE_LABELS[prior]: (
                            transitions[(gap, prior, prior)] / denominator
                            if denominator else None
                        )
                        for prior in STATES
                        for denominator in (
                            sum(transitions[(gap, prior, later)] for later in STATES),
                        )
                    }
                    for gap in (1, 3)
                },
            },
            "exceptional_period_touching_eligible_pairs_by_gap": {
                str(gap): exceptional_touch[gap] for gap in (1, 3)
            },
            "reviewed_labor_codebook": STATE_LABELS,
            "labor_source_fields": ["ESTADO", "CONDACT"],
            "labor_semantics": (
                "C4 _observed_labor_state: 0/4 special, valid 1/2/3, "
                "conflicting valid ESTADO/CONDACT rejected"
            ),
            "monetary_reference_period": monetary.get("common_reference_period"),
            "monetary_conversion_parent": monetary_parent,
            "monetary_limitation": (
                "candidate monetary parent remains candidate; descriptive real "
                "amounts must not be called approved-mode scientific freeze"
            ),
            "artifacts": artifacts,
            "limitations": [
                "candidate linkage is not permanent person identity",
                "unweighted descriptive selected-panel pairs",
                "no causal interpretation",
                "no OOF L11 or L12 result",
                "no extrapolation of short-gap persistence to CPV-2010",
                "no forecasting/nowcasting",
            ],
        }
        (staging / "gate_b_receipt.json").write_text(_json(receipt), encoding="utf-8")
        connection.close()
        connection = None
        (staging / "_work.sqlite").unlink()
        _progress("aggregate outputs staged; re-verifying immutable source hashes")
        # Immutable parent re-check catches mutations during the streaming read.
        if _sha(root / "manifest.json") != manifest_sha:
            raise GateBEvidenceError("l2_manifest_changed_during_run")
        for name in ("persons.csv", "panel_links.csv", "qa.json"):
            if _sha(root / name) != source_hashes[name]:
                raise GateBEvidenceError(f"l2_artifact_changed_during_run:{name}")
        os.replace(staging, destination)
        _progress(f"complete: {destination}")
        return destination
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
