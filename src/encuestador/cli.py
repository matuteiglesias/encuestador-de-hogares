"""Thin command-line surface for governed experiment validation, execution and evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .eph_microdata import (
    EPHMicrodataRelease,
    load_eph_microdata_release,
    read_eph_person_observation_frame,
)
from .experiments import ResolvedExperiment, resolve_experiment
from .run_bundle import package_run, validate_run_bundle
from .runner import WelfareContext, execute_experiment
from .runtime_model import fit_deployable_hurdle_model


class CLIError(ValueError):
    """Raised for explicit user-facing command contract failures."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _config_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "configs":
            return parent
    return resolved.parent


def _resolve(path: str) -> ResolvedExperiment:
    experiment = Path(path).expanduser().resolve()
    return resolve_experiment(experiment, config_root=_config_root(experiment))


def _header(path: Path, *, encoding: str, delimiter: str) -> tuple[str, ...]:
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise CLIError(f"source_table_empty:{path}") from exc


def _required_science_columns(resolved: ResolvedExperiment) -> set[str]:
    config = resolved.config
    columns = {
        str(name)
        for name, spec in config["features"]["columns"].items()
        if spec.get("allowed_as_external_input") is True
        and spec.get("dtype_role") in {"continuous", "categorical", "binary"}
    }
    columns.add(str(config["terminal"]["target"]))
    for node in resolved.architecture.nodes.values():
        columns.update(str(target) for target in node.targets)
    return columns


def _load_real_eph_rows(
    resolved: ResolvedExperiment,
    release_root: Path,
) -> tuple[list[dict[str, str]], EPHMicrodataRelease]:
    data = resolved.config["data"]
    adapter = data.get("source_adapter")
    if adapter != "microdatos_eph_individual_household_join_v1":
        raise CLIError(f"unsupported_real_eph_source_adapter:{adapter}")
    release = load_eph_microdata_release(
        release_root,
        expected_release_id=str(data["source_release"]),
        verify_hashes=True,
    )
    individual_header = set(
        _header(
            release.individual.path,
            encoding=release.individual.encoding,
            delimiter=release.individual.delimiter,
        )
    )
    household_header = set(
        _header(
            release.household.path,
            encoding=release.household.encoding,
            delimiter=release.household.delimiter,
        )
    )
    required = _required_science_columns(resolved)
    identity = {
        *(str(value) for value in data["person_id_columns"]),
        *(str(value) for value in data.get("period_columns", ())),
    }
    individual_columns: list[str] = []
    household_columns: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for column in sorted(required):
        in_individual = column in individual_header
        in_household = column in household_header
        if in_individual and in_household and column not in identity:
            ambiguous.append(column)
        elif in_individual:
            individual_columns.append(column)
        elif in_household:
            household_columns.append(column)
        else:
            missing.append(column)
    if missing:
        raise CLIError(f"real_eph_required_columns_missing:{','.join(missing)}")
    if ambiguous:
        raise CLIError(f"real_eph_source_column_ambiguous:{','.join(ambiguous)}")
    rows = read_eph_person_observation_frame(
        release,
        individual_columns=individual_columns,
        household_columns=household_columns,
    )
    return rows, release


def _release_evidence(release: EPHMicrodataRelease) -> dict[str, Any]:
    return {
        "release_id": release.release_id,
        "contract": "publicdata.eph-microdata@1",
        "extraction_contract": "eph-zip-v2",
        "period": release.period,
        "source_archive_sha256": release.source_archive_sha256,
        "source_manifest_sha256": release.source_manifest_sha256,
        "individual": {
            "sha256": release.individual.sha256,
            "rows": release.individual.rows,
            "columns": release.individual.columns,
        },
        "household": {
            "sha256": release.household.sha256,
            "rows": release.household.rows,
            "columns": release.household.columns,
        },
        "warnings": list(release.warnings),
    }


def _read_scoring_csv(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise CLIError("scoring_frame_empty")
    return rows


def _welfare_context(args: argparse.Namespace) -> WelfareContext | None:
    if args.scoring_csv is None:
        return None
    values = {
        "welfare_period": args.welfare_period,
        "currency": args.currency,
        "price_reference": args.price_reference,
        "welfare_concept": args.welfare_concept,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise CLIError(f"scoring_welfare_context_missing:{','.join(missing)}")
    return WelfareContext(**values)


def _command_validate(args: argparse.Namespace) -> int:
    resolved = _resolve(args.experiment)
    terminal = resolved.config["terminal"]
    output = {
        "status": "valid",
        "experiment_id": resolved.config["experiment"]["id"],
        "config_digest": resolved.digest,
        "architecture_id": resolved.architecture.architecture_id,
        "terminal_formulation": terminal.get("formulation"),
        "source_release": resolved.config["data"].get("source_release"),
        "feature_plane": resolved.config["features"].get("id"),
        "census_scoring_allowed": resolved.config["features"].get(
            "census_scoring_allowed"
        ),
    }
    sys.stdout.write(_canonical_json(output))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    resolved = _resolve(args.experiment)
    data = resolved.config["data"]
    if data.get("source_adapter") != "microdatos_eph_individual_household_join_v1":
        raise CLIError("run_cli_currently_requires_pinned_eph_microdata_adapter")
    if args.eph_release_root is None:
        raise CLIError("run_requires_eph_release_root")
    training_rows, release = _load_real_eph_rows(
        resolved, Path(args.eph_release_root)
    )

    scoring_rows = None
    context = _welfare_context(args)
    if args.scoring_csv is not None:
        features = resolved.config["features"]
        if features.get("census_scoring_allowed") is not True:
            raise CLIError("census_scoring_blocked_pending_reviewed_feature_plane")
        if not features.get("semantic_release_id"):
            raise CLIError("census_scoring_requires_semantic_release_id")
        scoring_rows = _read_scoring_csv(
            Path(args.scoring_csv), args.scoring_delimiter
        )
        required_ids = {"sample_person_id", "sample_household_id"}
        missing_ids = sorted(required_ids - set(scoring_rows[0]))
        if missing_ids:
            raise CLIError(f"scoring_sample_ids_missing:{','.join(missing_ids)}")

    evidence = _release_evidence(release)
    result = execute_experiment(
        resolved,
        training_rows,
        scoring_rows=scoring_rows,
        welfare_context=context,
        input_release_ids=(release.release_id,),
    )
    model = fit_deployable_hurdle_model(resolved, training_rows)
    limitations = [
        "EPH-side feature plane is a Census candidate only until eph-censo-aligner real-vintage review is attached.",
        "No EPH survey/design weight enters fitting or evaluation.",
    ]
    if scoring_rows is None:
        limitations.append(
            "No real Census scoring artifact was supplied to this run."
        )
    run_root = package_run(
        Path(args.output_root),
        resolved,
        result,
        training_rows,
        model,
        input_evidence=(evidence,),
        limitations=limitations,
    )
    output = {
        "status": "complete",
        "run_id": result.run_id,
        "run_root": str(run_root),
        "experiment_id": result.experiment_id,
        "architecture_id": result.architecture_id,
        "person_rows": len(result.oof_prediction.row_ids),
        "household_observations": result.metrics["household"][
            "household_observation_count"
        ],
        "target_eligibility": (
            dict(result.hurdle.target_eligibility) if result.hurdle else None
        ),
    }
    sys.stdout.write(_canonical_json(output))
    return 0


def _candidate_metrics(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    architecture = str(manifest["architecture_id"])
    person_candidates = metrics["person"]
    candidate = "direct" if architecture.startswith("direct") else "deployable"
    person = person_candidates[candidate]
    if "unconditional_income" in person:
        person_point = person["unconditional_income"]["point"]
    else:
        person_point = person["point"]
    household_candidates = metrics["household"]["candidates"]
    household = household_candidates[candidate]
    return {
        "candidate": candidate,
        "person_mae": person_point["mae"],
        "person_rmse": person_point["rmse"],
        "household_mae": household["point"]["mae"],
        "household_rmse": household["point"]["rmse"],
        "household_complete": metrics["household"].get(
            "complete_observed_income_household_count"
        ),
        "household_unavailable": metrics["household"].get(
            "unavailable_observed_income_household_count"
        ),
    }


def _load_run(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = validate_run_bundle(root)
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    return manifest, config, metrics


def _comparison_id(run_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(run_ids).encode()).hexdigest()[:16]
    return f"comparison-{digest}"


def _command_compare(args: argparse.Namespace) -> int:
    roots = [Path(value).expanduser().resolve() for value in args.runs]
    if len(roots) < 2:
        raise CLIError("compare_requires_at_least_two_runs")
    loaded = [_load_run(root) for root in roots]
    manifests = [value[0] for value in loaded]
    configs = [value[1] for value in loaded]
    metrics = [value[2] for value in loaded]

    data_signatures = {
        _canonical_json(
            {
                "source_release": config["data"].get("source_release"),
                "training_periods": config["data"].get("training_periods"),
                "household_id_columns": config["data"].get("household_id_columns"),
                "period_columns": config["data"].get("period_columns", []),
                "weight_policy": config["data"].get("weight_policy"),
                "feature_id": config["features"].get("id"),
                "feature_columns": [
                    name
                    for name, spec in config["features"]["columns"].items()
                    if spec.get("allowed_as_external_input") is True
                ],
                "target": config["terminal"].get("target"),
                "split_strategy": config["splits"].get("strategy"),
                "group_columns": config["splits"].get("group_columns"),
                "n_splits": config["splits"].get("n_splits"),
            }
        )
        for config in configs
    }
    if len(data_signatures) != 1:
        raise CLIError("comparison_contracts_incompatible")

    architectures = {manifest["architecture_id"] for manifest in manifests}
    formulations = {config["terminal"].get("formulation") for config in configs}
    changed_axes: list[str] = []
    if len(architectures) > 1:
        changed_axes.append("architecture")
    if len(formulations) > 1:
        changed_axes.append("terminal_formulation")
    if not changed_axes:
        changed_axes.append("replicate_or_parameterization")

    rows = []
    for root, manifest, metric in zip(roots, manifests, metrics, strict=True):
        rows.append(
            {
                "run_id": manifest["run_id"],
                "run_root": str(root),
                "architecture_id": manifest["architecture_id"],
                "terminal_formulation": manifest["terminal"].get("formulation"),
                **_candidate_metrics(manifest, metric),
            }
        )
    comparison = {
        "contract": "research.encuestador-comparison/v1",
        "comparison_id": _comparison_id([row["run_id"] for row in rows]),
        "changed_axes": changed_axes,
        "runs": rows,
        "status": "comparable" if len(changed_axes) == 1 else "multi_axis_descriptive",
    }
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / f"{comparison['comparison_id']}.json"
        if path.exists():
            raise CLIError(f"immutable_comparison_exists:{path}")
        path.write_text(_canonical_json(comparison), encoding="utf-8")
        report = output_root / f"{comparison['comparison_id']}.md"
        lines = [
            f"# {comparison['comparison_id']}",
            "",
            f"Changed axes: {', '.join(changed_axes)}",
            "",
            "| run | architecture | terminal | person MAE | household MAE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['run_id']} | {row['architecture_id']} | {row['terminal_formulation']} | {row['person_mae']:.6g} | {row['household_mae']:.6g} |"
            )
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        comparison["output_json"] = str(path)
        comparison["output_report"] = str(report)
    sys.stdout.write(_canonical_json(comparison))
    return 0


def _command_report(args: argparse.Namespace) -> int:
    path = Path(args.run_or_comparison).expanduser().resolve()
    if path.is_dir():
        validate_run_bundle(path)
        report = path / "report.md"
        if not report.is_file():
            raise CLIError("run_report_missing")
        sys.stdout.write(report.read_text(encoding="utf-8"))
        return 0
    if path.suffix == ".json" and path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("contract") != "research.encuestador-comparison/v1":
            raise CLIError("comparison_report_contract_invalid")
        markdown = path.with_suffix(".md")
        if markdown.is_file():
            sys.stdout.write(markdown.read_text(encoding="utf-8"))
        else:
            sys.stdout.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        return 0
    raise CLIError("report_target_not_found")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="encuestador")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("experiment")
    validate.set_defaults(handler=_command_validate)

    run = sub.add_parser("run")
    run.add_argument("experiment")
    run.add_argument("--eph-release-root")
    run.add_argument("--output-root", default="runs")
    run.add_argument("--scoring-csv")
    run.add_argument("--scoring-delimiter", default=",")
    run.add_argument("--welfare-period")
    run.add_argument("--currency")
    run.add_argument("--price-reference")
    run.add_argument("--welfare-concept")
    run.set_defaults(handler=_command_run)

    compare = sub.add_parser("compare")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--output-root")
    compare.set_defaults(handler=_command_compare)

    report = sub.add_parser("report")
    report.add_argument("run_or_comparison")
    report.set_defaults(handler=_command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CLIError, ValueError) as exc:
        parser.exit(2, f"encuestador: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
