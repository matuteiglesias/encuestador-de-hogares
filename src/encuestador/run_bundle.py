"""Immutable scientific run packaging for governed welfare transport experiments."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from .experiments import ResolvedExperiment
from .runner import ExperimentExecutionResult, _identity
from .runtime_model import FittedTransportModel
from .terminal import classify_income_target

RUN_CONTRACT = "research.encuestador-run/v1"


class RunBundleError(ValueError):
    """Raised when immutable run evidence cannot be represented safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json(dict(row)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
    }
    try:
        identity["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        identity["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        identity["git_commit"] = None
        identity["git_dirty"] = None
    return identity


def _target_status(value: Any) -> str:
    result = classify_income_target([value])
    if result.positive[0]:
        return "positive"
    if result.zero[0]:
        return "zero"
    if result.nonresponse[0]:
        return "nonresponse"
    if result.missing[0]:
        return "missing"
    raise RunBundleError("terminal_target_status_unclassified")


def _person_oof_rows(
    resolved: ResolvedExperiment,
    result: ExperimentExecutionResult,
    training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    data = resolved.config["data"]
    terminal = resolved.config["terminal"]
    identity_columns = tuple(str(value) for value in data["person_id_columns"]) + tuple(
        str(value) for value in data.get("period_columns", ())
    )
    target = str(terminal["target"])
    expected_ids = tuple(_identity(row, identity_columns) for row in training_rows)
    if result.oof_prediction.row_ids != expected_ids:
        raise RunBundleError("oof_row_identity_mismatch")
    output: list[dict[str, Any]] = []
    hurdle = result.hurdle
    eligibility = classify_income_target([row.get(target) for row in training_rows]) if hurdle else None
    for index, row in enumerate(training_rows):
        record: dict[str, Any] = {
            "row_id": expected_ids[index],
            "fold_id": result.fold_manifest.fold_ids[index],
            "prediction": float(result.oof_prediction.values[index]),
        }
        for column in identity_columns:
            record[column] = row[column]
        if hurdle is not None and eligibility is not None:
            record["target_status"] = _target_status(row.get(target))
            record["observed_income"] = (
                float(eligibility.numeric[index]) if eligibility.valid[index] else None
            )
            record["p_positive"] = float(hurdle.p_positive.values[index, 1])
            record["positive_amount_prediction"] = float(
                hurdle.positive_amount_prediction.values[index]
            )
        else:
            record["observed_income"] = float(row[target])
        output.append(record)
    return output


def _latent_rows(result: ExperimentExecutionResult) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for target, artifact in result.latent_predictions.items():
        if artifact.kind != "probability" or artifact.fold_ids is None:
            raise RunBundleError(f"latent_artifact_not_oof_probability:{target}")
        records: list[dict[str, Any]] = []
        for index, row_id in enumerate(artifact.row_ids):
            records.append(
                {
                    "row_id": row_id,
                    "fold_id": artifact.fold_ids[index],
                    "probabilities": {
                        label: float(artifact.values[index, column])
                        for column, label in enumerate(artifact.class_labels)
                    },
                }
            )
        output[target] = records
    return output


def _household_oof_rows(
    resolved: ResolvedExperiment,
    result: ExperimentExecutionResult,
    training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    data = resolved.config["data"]
    terminal = resolved.config["terminal"]
    household_columns = tuple(str(value) for value in data["household_id_columns"]) + tuple(
        str(value) for value in data.get("period_columns", ())
    )
    target = str(terminal["target"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(training_rows):
        grouped[_identity(row, household_columns)].append(index)
    eligibility = (
        classify_income_target([row.get(target) for row in training_rows])
        if result.hurdle is not None
        else None
    )
    output: list[dict[str, Any]] = []
    for household_observation_id, indices in sorted(grouped.items()):
        first = training_rows[indices[0]]
        record: dict[str, Any] = {
            "household_observation_id": household_observation_id,
            "member_count": len(indices),
            "predicted_household_income": float(
                result.oof_prediction.values[indices].sum()
            ),
        }
        for column in household_columns:
            record[column] = first[column]
        if eligibility is None:
            record["observed_household_income_status"] = "complete"
            record["observed_household_income"] = float(
                sum(float(training_rows[index][target]) for index in indices)
            )
        elif bool(np.all(eligibility.valid[indices])):
            record["observed_household_income_status"] = "complete"
            record["observed_household_income"] = float(
                eligibility.numeric[indices].sum()
            )
        else:
            record["observed_household_income_status"] = "unavailable_member_income"
            record["observed_household_income"] = None
            record["unavailable_member_count"] = int((~eligibility.valid[indices]).sum())
        output.append(record)
    return output


def _scoring_rows(result: ExperimentExecutionResult) -> list[dict[str, Any]]:
    if result.score_prediction is None:
        return []
    output: list[dict[str, Any]] = []
    for index, row_id in enumerate(result.score_prediction.row_ids):
        record: dict[str, Any] = {
            "row_id": row_id,
            "prediction": float(result.score_prediction.values[index]),
        }
        if result.score_hurdle is not None:
            record["p_positive"] = float(result.score_hurdle.p_positive.values[index, 1])
            record["positive_amount_prediction"] = float(
                result.score_hurdle.positive_amount_prediction.values[index]
            )
        output.append(record)
    return output


def _report_markdown(
    resolved: ResolvedExperiment,
    result: ExperimentExecutionResult,
    input_evidence: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
) -> str:
    terminal = resolved.config["terminal"]
    lines = [
        f"# Run {result.run_id}",
        "",
        f"- Experiment: `{result.experiment_id}`",
        f"- Architecture: `{result.architecture_id}`",
        f"- Config digest: `{result.config_digest}`",
        f"- Terminal: `{terminal.get('formulation')}` on `{terminal.get('target')}`",
        f"- Person OOF rows: {len(result.oof_prediction.row_ids):,}",
        f"- Household observations: {result.metrics.get('household', {}).get('household_observation_count', 'n/a')}",
        "",
        "## Input evidence",
        "",
    ]
    for record in input_evidence:
        lines.append(f"- `{record.get('release_id', record.get('id', 'unknown'))}`")
    if result.hurdle is not None:
        eligibility = result.hurdle.target_eligibility
        lines.extend(
            [
                "",
                "## Terminal eligibility",
                "",
                f"- Eligible: {eligibility.get('eligible', 0):,}",
                f"- Zero: {eligibility.get('zero', 0):,}",
                f"- Positive: {eligibility.get('positive', 0):,}",
                f"- Nonresponse: {eligibility.get('nonresponse', 0):,}",
                f"- Missing: {eligibility.get('missing', 0):,}",
            ]
        )
    if limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {value}" for value in limitations)
    return "\n".join(lines) + "\n"


def package_run(
    output_root: Path,
    resolved: ResolvedExperiment,
    result: ExperimentExecutionResult,
    training_rows: Sequence[Mapping[str, Any]],
    deployable_model: FittedTransportModel,
    *,
    input_evidence: Sequence[Mapping[str, Any]],
    limitations: Sequence[str] = (),
) -> Path:
    """Atomically publish one immutable run directory with hashes and fitted state."""
    output_root = Path(output_root).expanduser().resolve()
    destination = output_root / result.run_id
    if destination.exists():
        raise RunBundleError(f"immutable_run_exists:{destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{result.run_id}.", dir=output_root))
    try:
        _write_json(staging / "resolved_config.json", resolved.config)
        _write_json(staging / "input_evidence.json", list(input_evidence))
        _write_json(
            staging / "fold_manifest.json",
            {
                "policy": result.fold_manifest.policy,
                "n_splits": result.fold_manifest.n_splits,
                "rows": [
                    {
                        "row_id": row_id,
                        "household_group_id": household_id,
                        "fold_id": fold_id,
                    }
                    for row_id, household_id, fold_id in zip(
                        result.fold_manifest.row_ids,
                        result.fold_manifest.household_ids,
                        result.fold_manifest.fold_ids,
                        strict=True,
                    )
                ],
            },
        )
        _write_jsonl(
            staging / "person_oof.jsonl",
            _person_oof_rows(resolved, result, training_rows),
        )
        _write_jsonl(
            staging / "household_oof.jsonl",
            _household_oof_rows(resolved, result, training_rows),
        )
        latent_dir = staging / "latents"
        latent_dir.mkdir()
        for target, rows in _latent_rows(result).items():
            _write_jsonl(latent_dir / f"{target}.jsonl", rows)
        _write_json(staging / "metrics.json", result.metrics)
        score_rows = _scoring_rows(result)
        if score_rows:
            _write_jsonl(staging / "scoring_person.jsonl", score_rows)
        if result.household_welfare:
            _write_jsonl(
                staging / "household_welfare.jsonl", list(result.household_welfare)
            )
        _write_json(staging / "limitations.json", list(limitations))
        with (staging / "model.pkl").open("wb") as stream:
            pickle.dump(deployable_model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (staging / "report.md").write_text(
            _report_markdown(resolved, result, input_evidence, limitations),
            encoding="utf-8",
        )

        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "run_manifest.json":
                relative = path.relative_to(staging).as_posix()
                artifacts[relative] = {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
        feature_config = resolved.config["features"]
        manifest = {
            "contract": RUN_CONTRACT,
            "run_id": result.run_id,
            "experiment_id": result.experiment_id,
            "config_digest": result.config_digest,
            "architecture_id": result.architecture_id,
            "terminal": dict(resolved.config["terminal"]),
            "estimators": dict(resolved.config["estimators"]),
            "feature_plane": {
                "id": feature_config.get("id"),
                "semantic_authority": feature_config.get("semantic_authority"),
                "review_status": feature_config.get("review_status"),
                "census_scoring_allowed": feature_config.get(
                    "census_scoring_allowed"
                ),
                "columns": list(result.feature_names),
            },
            "input_evidence": list(input_evidence),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                **_git_identity(),
            },
            "limitations": list(limitations),
            "artifacts": artifacts,
        }
        _write_json(staging / "run_manifest.json", manifest)
        os.replace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_run_bundle(root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunBundleError("run_manifest_invalid") from exc
    if manifest.get("contract") != RUN_CONTRACT:
        raise RunBundleError("unexpected_run_contract")
    if manifest.get("run_id") != root.name:
        raise RunBundleError("run_directory_identity_mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RunBundleError("run_artifact_manifest_missing")
    for relative, record in artifacts.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RunBundleError("run_artifact_path_escapes_root") from exc
        if not path.is_file():
            raise RunBundleError(f"run_artifact_missing:{relative}")
        if verify_hashes and _sha256(path) != record.get("sha256"):
            raise RunBundleError(f"run_artifact_hash_mismatch:{relative}")
    return manifest
