"""Deterministic synthetic proof for the starter EPH -> Census surveyor.

This module exercises the scientific plumbing only. It does not approve a real
EPH/Census semantic mapping or a production model. The synthetic proof enforces
household-grouped cross-fitting, unweighted estimation/evaluation, no target-
period calibration, person-level predictions and complete-member household
aggregation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "contracts" / "model_specs" / "historical_staged_v1.json"
DEFAULT_SEMANTIC_PLANE = (
    ROOT / "fixtures" / "semantic_plane" / "synthetic_fixture_v1.json"
)
PROOF_CONTRACT = "research.encuestador-synthetic-transport-proof@1"
SEMANTIC_CLASSES_ALLOWED_EXTERNALLY = {"shared_observable", "derived_shared"}
DESIGN_FIELDS = {
    "PONDERA",
    "PONDIH",
    "PONDIIO",
    "PONDII",
    "selection_probability",
    "design_inverse_probability_weight",
}


class TransportProofError(ValueError):
    """Raised when the proof would violate the starter transport contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransportProofError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise TransportProofError(f"expected_mapping:{path}")
    return value


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("spec_id") != "historical_staged_v1":
        raise TransportProofError("unexpected_model_spec")
    if spec.get("prediction_unit") != "person":
        raise TransportProofError("starter_prediction_unit_must_be_person")
    if spec.get("terminal_person_income_target") != "P47T":
        raise TransportProofError("starter_terminal_target_must_be_P47T")

    fold = spec.get("fold_policy", {})
    if fold.get("strategy") != "household_grouped_v1":
        raise TransportProofError("starter_fold_policy_must_be_household_grouped")
    if fold.get("group_columns") != ["CODUSU", "NRO_HOGAR"]:
        raise TransportProofError("starter_household_identity_changed")
    if fold.get("learned_intermediates_for_training") != "out_of_fold_predictions":
        raise TransportProofError("learned_intermediates_must_be_oof")
    if fold.get("observed_intermediate_substitution_forbidden") is not True:
        raise TransportProofError("observed_intermediate_substitution_must_be_forbidden")

    weights = spec.get("eph_design_policy", {})
    for key in ("fit_weight", "calibration_weight", "evaluation_weight"):
        if weights.get(key) is not None:
            raise TransportProofError(f"starter_weight_policy_must_be_none:{key}")
    if weights.get("claim_boundary") != "sample_conditional":
        raise TransportProofError("starter_claim_boundary_must_be_sample_conditional")

    temporal = spec.get("temporal_policy", {})
    if temporal.get("target_period_calibration") != "none":
        raise TransportProofError("starter_temporal_calibration_must_be_none")
    if temporal.get("aggregate_anchor") is not None:
        raise TransportProofError("starter_aggregate_anchor_must_be_absent")
    if temporal.get("donor_field_mutation") is not False:
        raise TransportProofError("donor_fields_must_not_be_mutated")

    stages = spec.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise TransportProofError("historical_staged_v1_requires_four_stages")
    prior_targets: list[str] = []
    for index, stage in enumerate(stages):
        inputs = stage.get("inputs")
        targets = stage.get("targets")
        if not isinstance(inputs, list) or not isinstance(targets, list):
            raise TransportProofError(f"invalid_stage_lists:{index + 1}")
        if index and inputs[-len(prior_targets) :] != prior_targets:
            raise TransportProofError(f"stage_append_contract_changed:{index + 1}")
        prior_targets.extend(targets)

    all_model_fields = {
        field
        for stage in stages
        for field in [*stage["inputs"], *stage["targets"]]
    }
    if all_model_fields & DESIGN_FIELDS:
        raise TransportProofError("sampling_or_survey_design_field_entered_model_spec")


def validate_semantic_plane(spec: dict[str, Any], plane: dict[str, Any]) -> list[str]:
    if plane.get("contract") != "research.eph-census-semantic-feature-plane@1":
        raise TransportProofError("unexpected_semantic_plane_contract")
    if plane.get("status") != "synthetic_fixture_only":
        raise TransportProofError("synthetic_proof_requires_synthetic_semantic_plane")
    if plane.get("real_vintage_approval") is not False:
        raise TransportProofError("synthetic_plane_must_not_claim_real_approval")

    calibration = plane.get("calibration_policy", {})
    if calibration.get("target_period_calibration") != "none":
        raise TransportProofError("semantic_plane_has_hidden_calibration")
    if calibration.get("aggregate_anchors") != []:
        raise TransportProofError("semantic_plane_has_hidden_anchor")
    if calibration.get("donor_field_mutation") is not False:
        raise TransportProofError("semantic_plane_mutates_donor_fields")

    stage1 = spec["stages"][0]
    forbidden = set(spec["external_input_policy"]["forbidden_external_inputs"])
    expected = [field for field in stage1["inputs"] if field not in forbidden]
    approved = plane.get("approved_external_inputs")
    if approved != expected:
        raise TransportProofError("synthetic_external_input_plane_drift")

    features = plane.get("features")
    if not isinstance(features, dict):
        raise TransportProofError("semantic_plane_features_missing")
    for field in expected:
        record = features.get(field, {})
        if record.get("semantic_class") not in SEMANTIC_CLASSES_ALLOWED_EXTERNALLY:
            raise TransportProofError(f"external_input_not_semantically_admitted:{field}")
        if not record.get("transport_time_role"):
            raise TransportProofError(f"transport_time_role_missing:{field}")
    for field in forbidden:
        record = features.get(field, {})
        if record.get("semantic_class") != "research_only":
            raise TransportProofError(f"forbidden_input_not_research_only:{field}")
        if field in approved:
            raise TransportProofError(f"forbidden_input_approved:{field}")
    return expected


def _household_fold(codusu: str, household: str, n_splits: int) -> int:
    digest = hashlib.sha256(f"{codusu}\x1f{household}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_splits


def _synthetic_base_value(field: str, household: int, component: int, members: int) -> float:
    index = int.from_bytes(hashlib.sha256(field.encode()).digest()[:2], "big")
    if field == "IX_TOT":
        return float(members)
    if field == "P02":
        return float(1 + ((household + component) % 2))
    if field == "P03":
        return float(18 + ((household * 5 + component * 7) % 58))
    if field == "CONDACT":
        return float((household + component) % 4)
    if field == "PROP":
        return float(1 + household % 6)
    return float((household * (index % 7 + 1) + component * (index % 5 + 2) + index) % 9)


def _attach_targets(row: dict[str, Any], household: int, component: int) -> None:
    age = float(row["P03"])
    sex = float(row["P02"])
    activity = float(row["CONDACT"])
    household_context = float(row["IX_TOT"])
    latent = ((household * 17 + component * 11) % 13) - 6

    row["CAT_OCUP"] = float((age > 24) and ((household + component) % 4 != 0))
    row["CAT_INAC"] = float((age > 64) or ((household + component) % 7 == 0))
    row["CH07"] = float(1 + ((household + component) % 5))

    row["INGRESO"] = float(row["CAT_OCUP"] > 0 or row["CAT_INAC"] > 0)
    row["INGRESO_NLB"] = float((household + component) % 6 == 0)
    row["INGRESO_JUB"] = float(age >= 62)
    row["INGRESO_SBS"] = float((household + 2 * component) % 9 == 0)

    row["PP07G1"] = float(row["CAT_OCUP"] > 0 and (household + component) % 3 != 0)
    row["PP07G_59"] = float(row["CAT_OCUP"] > 0 and household % 5 == 0)
    row["PP07I"] = float(row["CAT_OCUP"] > 0 and component % 2 == 0)
    row["PP07J"] = float(row["CAT_OCUP"] > 0 and household % 2 == 0)
    row["PP07K"] = float(row["CAT_OCUP"] > 0 and (household + component) % 5 == 0)

    labor_income = (
        300.0
        + 38.0 * age
        + 145.0 * sex
        + 620.0 * row["CAT_OCUP"]
        + 440.0 * row["INGRESO"]
        + 280.0 * row["PP07G1"]
        + 130.0 * row["PP07I"]
        + 55.0 * household_context
        + 21.0 * activity
        + 17.0 * latent
    )
    pension = 720.0 * row["INGRESO_JUB"]
    transfer = 260.0 * row["INGRESO_SBS"]
    nonlabor = 180.0 * row["INGRESO_NLB"]
    row["P21"] = max(0.0, labor_income if row["CAT_OCUP"] else 0.0)
    row["PP08D1"] = max(0.0, 0.78 * row["P21"])
    row["TOT_P12"] = nonlabor
    row["T_VI"] = pension + transfer + nonlabor
    row["V12_M"] = pension
    row["V2_M"] = transfer
    row["V3_M"] = nonlabor
    row["V5_M"] = float(max(0, latent) * 25)
    row["P47T"] = max(
        0.0,
        row["P21"] + pension + transfer + nonlabor + row["V5_M"],
    )


def make_synthetic_people(
    approved_external_inputs: list[str],
    *,
    household_count: int,
    codusu_prefix: str,
    include_census_design: bool,
) -> list[dict[str, Any]]:
    """Create deterministic person rows with complete household membership."""
    rows: list[dict[str, Any]] = []
    for household in range(household_count):
        members = 2 + household % 3
        codusu = f"{codusu_prefix}-{household // 10:02d}"
        nro_hogar = str(household + 1)
        for component in range(1, members + 1):
            row: dict[str, Any] = {
                "CODUSU": codusu,
                "NRO_HOGAR": nro_hogar,
                "COMPONENTE": component,
                "PONDERA": float(50 + household % 11),
                "PONDIH": float(70 + household % 13),
                "PONDIIO": float(60 + component),
                "PONDII": float(65 + component),
                "AGLO_rk": float((household % 10) / 10),
                "Reg_rk": float((household % 6) / 6),
            }
            if include_census_design:
                probability = 0.008 + 0.001 * (household % 4)
                row["selection_probability"] = probability
                row["design_inverse_probability_weight"] = 1.0 / probability
            for field in approved_external_inputs:
                row[field] = _synthetic_base_value(
                    field,
                    household,
                    component,
                    members,
                )
            _attach_targets(row, household, component)
            rows.append(row)
    return rows


def _matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    try:
        matrix = np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportProofError("non_numeric_or_missing_model_input") from exc
    if not np.isfinite(matrix).all():
        raise TransportProofError("non_finite_model_input")
    return matrix


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coefficients


def _folds(rows: list[dict[str, Any]], n_splits: int) -> np.ndarray:
    values = np.asarray(
        [
            _household_fold(str(row["CODUSU"]), str(row["NRO_HOGAR"]), n_splits)
            for row in rows
        ],
        dtype=int,
    )
    if len(set(values.tolist())) != n_splits:
        raise TransportProofError("synthetic_fold_assignment_has_empty_fold")
    household_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row, fold in zip(rows, values, strict=True):
        household_folds[(str(row["CODUSU"]), str(row["NRO_HOGAR"]))].add(int(fold))
    if any(len(folds) != 1 for folds in household_folds.values()):
        raise TransportProofError("household_crosses_folds")
    return values


def _feature_matrix(
    rows: list[dict[str, Any]],
    stage_inputs: list[str],
    approved_external_inputs: list[str],
    prior_predictions: dict[str, np.ndarray],
    forbidden_external_inputs: set[str],
) -> tuple[np.ndarray, list[str]]:
    approved = set(approved_external_inputs)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for field in stage_inputs:
        if field in forbidden_external_inputs:
            continue
        if field in prior_predictions:
            values = np.asarray(prior_predictions[field], dtype=float)
            columns.append(values)
            names.append(f"oof:{field}")
            continue
        if field not in approved:
            raise TransportProofError(f"unapproved_external_or_observed_stage_input:{field}")
        columns.append(_matrix(rows, [field])[:, 0])
        names.append(field)
    if not columns:
        raise TransportProofError("empty_stage_feature_matrix")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise TransportProofError("non_finite_stage_feature_matrix")
    return matrix, names


def _oof_predict_stage(
    rows: list[dict[str, Any]],
    folds: np.ndarray,
    stage_inputs: list[str],
    targets: list[str],
    approved_external_inputs: list[str],
    prior_predictions: dict[str, np.ndarray],
    forbidden_external_inputs: set[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    x, feature_names = _feature_matrix(
        rows,
        stage_inputs,
        approved_external_inputs,
        prior_predictions,
        forbidden_external_inputs,
    )
    y = _matrix(rows, targets)
    predictions = np.full_like(y, np.nan, dtype=float)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        coefficients = _fit_ridge(x[train], y[train])
        predictions[test] = _predict_ridge(x[test], coefficients)
    if not np.isfinite(predictions).all():
        raise TransportProofError("non_finite_oof_prediction")
    return {
        target: predictions[:, index]
        for index, target in enumerate(targets)
    }, feature_names


def _fit_full_stage(
    training_rows: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
    stage_inputs: list[str],
    targets: list[str],
    approved_external_inputs: list[str],
    training_prior_predictions: dict[str, np.ndarray],
    scoring_prior_predictions: dict[str, np.ndarray],
    forbidden_external_inputs: set[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    train_x, feature_names = _feature_matrix(
        training_rows,
        stage_inputs,
        approved_external_inputs,
        training_prior_predictions,
        forbidden_external_inputs,
    )
    score_x, score_names = _feature_matrix(
        scoring_rows,
        stage_inputs,
        approved_external_inputs,
        scoring_prior_predictions,
        forbidden_external_inputs,
    )
    if score_names != feature_names:
        raise TransportProofError("training_scoring_feature_plane_mismatch")
    y = _matrix(training_rows, targets)
    coefficients = _fit_ridge(train_x, y)
    prediction = _predict_ridge(score_x, coefficients)
    if not np.isfinite(prediction).all():
        raise TransportProofError("non_finite_scoring_prediction")
    return {
        target: prediction[:, index]
        for index, target in enumerate(targets)
    }, feature_names


def _direct_oof(
    rows: list[dict[str, Any]],
    folds: np.ndarray,
    approved_external_inputs: list[str],
    target: str,
) -> np.ndarray:
    x = _matrix(rows, approved_external_inputs)
    y = _matrix(rows, [target])
    prediction = np.full(len(rows), np.nan, dtype=float)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        coefficients = _fit_ridge(x[train], y[train])
        prediction[test] = _predict_ridge(x[test], coefficients)[:, 0]
    return prediction


def _direct_score(
    training_rows: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
    approved_external_inputs: list[str],
    target: str,
) -> np.ndarray:
    train_x = _matrix(training_rows, approved_external_inputs)
    score_x = _matrix(scoring_rows, approved_external_inputs)
    y = _matrix(training_rows, [target])
    coefficients = _fit_ridge(train_x, y)
    return _predict_ridge(score_x, coefficients)[:, 0]


def aggregate_households(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    *,
    truth_field: str = "P47T",
) -> list[dict[str, Any]]:
    """Sum all valid member predictions; never hide an invalid member."""
    if not predictions:
        raise TransportProofError("household_aggregation_requires_predictions")
    length = len(rows)
    if any(len(values) != length for values in predictions.values()):
        raise TransportProofError("prediction_length_mismatch")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row["CODUSU"]), str(row["NRO_HOGAR"]))].append(index)

    output: list[dict[str, Any]] = []
    for household_key in sorted(grouped):
        indices = grouped[household_key]
        record: dict[str, Any] = {
            "CODUSU": household_key[0],
            "NRO_HOGAR": household_key[1],
            "member_count": len(indices),
            "truth_household_income": float(
                sum(float(rows[index][truth_field]) for index in indices)
            ),
        }
        incomplete: set[str] = set()
        for model_id, values in predictions.items():
            invalid = [index for index in indices if not math.isfinite(float(values[index]))]
            if invalid:
                record[f"{model_id}_household_income"] = None
                record[f"{model_id}_missing_members"] = [
                    int(rows[index]["COMPONENTE"]) for index in invalid
                ]
                incomplete.add(model_id)
            else:
                record[f"{model_id}_household_income"] = float(
                    sum(float(values[index]) for index in indices)
                )
                record[f"{model_id}_missing_members"] = []
        record["status"] = "incomplete" if incomplete else "complete"
        record["incomplete_models"] = sorted(incomplete)
        output.append(record)
    return output


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(truth) != len(prediction) or not len(truth):
        raise TransportProofError("metric_length_error")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise TransportProofError("metrics_require_finite_values")
    residual = prediction - truth
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_error": float(np.mean(residual)),
    }


def _household_metrics(
    records: list[dict[str, Any]], model_id: str
) -> dict[str, float]:
    complete = [
        record
        for record in records
        if record["status"] == "complete"
        and record[f"{model_id}_household_income"] is not None
    ]
    if len(complete) != len(records):
        raise TransportProofError("proof_metrics_refuse_incomplete_households")
    truth = np.asarray([record["truth_household_income"] for record in complete])
    prediction = np.asarray(
        [record[f"{model_id}_household_income"] for record in complete]
    )
    return _metrics(truth, prediction)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TransportProofError(f"cannot_write_empty_csv:{path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serial = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, list) else value
                for key, value in row.items()
            }
            writer.writerow(serial)


def _person_audit_rows(
    rows: list[dict[str, Any]],
    direct: np.ndarray,
    staged: np.ndarray,
    *,
    include_selection_probability: bool,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "CODUSU": row["CODUSU"],
            "NRO_HOGAR": row["NRO_HOGAR"],
            "COMPONENTE": row["COMPONENTE"],
            "truth_person_income": float(row["P47T"]),
            "direct_person_income": float(direct[index]),
            "staged_person_income": float(staged[index]),
        }
        if include_selection_probability:
            record["selection_probability"] = float(row["selection_probability"])
            record["design_inverse_probability_weight"] = float(
                row["design_inverse_probability_weight"]
            )
        output.append(record)
    return output


def run_synthetic_proof(
    *,
    spec_path: Path = DEFAULT_SPEC,
    semantic_plane_path: Path = DEFAULT_SEMANTIC_PLANE,
    output_dir: Path | None = None,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run direct and staged person-income proofs and compare household totals."""
    spec = _load_json(Path(spec_path))
    plane = _load_json(Path(semantic_plane_path))
    validate_spec(spec)
    approved = validate_semantic_plane(spec, plane)
    forbidden = set(spec["external_input_policy"]["forbidden_external_inputs"])

    eph_rows = make_synthetic_people(
        approved,
        household_count=40,
        codusu_prefix="EPH",
        include_census_design=False,
    )
    census_rows = make_synthetic_people(
        approved,
        household_count=16,
        codusu_prefix="CPV2010",
        include_census_design=True,
    )
    folds = _folds(eph_rows, n_splits)
    terminal_target = spec["terminal_person_income_target"]

    direct_oof = _direct_oof(eph_rows, folds, approved, terminal_target)
    direct_census = _direct_score(eph_rows, census_rows, approved, terminal_target)

    eph_prior: dict[str, np.ndarray] = {}
    census_prior: dict[str, np.ndarray] = {}
    stage_audit: list[dict[str, Any]] = []
    staged_oof: np.ndarray | None = None
    staged_census: np.ndarray | None = None
    for stage in spec["stages"]:
        oof, training_features = _oof_predict_stage(
            eph_rows,
            folds,
            stage["inputs"],
            stage["targets"],
            approved,
            eph_prior,
            forbidden,
        )
        scored, scoring_features = _fit_full_stage(
            eph_rows,
            census_rows,
            stage["inputs"],
            stage["targets"],
            approved,
            eph_prior,
            census_prior,
            forbidden,
        )
        if training_features != scoring_features:
            raise TransportProofError("stage_training_scoring_feature_mismatch")
        eph_prior.update(oof)
        census_prior.update(scored)
        stage_audit.append(
            {
                "stage_id": stage["stage_id"],
                "feature_count": len(training_features),
                "features": training_features,
                "targets": stage["targets"],
                "observed_intermediate_inputs_used": False,
                "weight_used": None,
            }
        )
        if terminal_target in oof:
            staged_oof = oof[terminal_target]
            staged_census = scored[terminal_target]

    if staged_oof is None or staged_census is None:
        raise TransportProofError("terminal_target_not_emitted_by_staged_spec")

    eph_households = aggregate_households(
        eph_rows,
        {"direct": direct_oof, "staged": staged_oof},
    )
    census_households = aggregate_households(
        census_rows,
        {"direct": direct_census, "staged": staged_census},
    )
    eph_truth = _matrix(eph_rows, [terminal_target])[:, 0]
    census_truth = _matrix(census_rows, [terminal_target])[:, 0]

    leakage_households = 0
    group_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row, fold in zip(eph_rows, folds, strict=True):
        group_folds[(str(row["CODUSU"]), str(row["NRO_HOGAR"]))].add(int(fold))
    leakage_households = sum(len(values) > 1 for values in group_folds.values())

    summary = {
        "contract": PROOF_CONTRACT,
        "status": "synthetic_proof_only",
        "model_spec": spec["spec_id"],
        "semantic_plane": {
            "release_id": plane["release_id"],
            "status": plane["status"],
            "real_vintage_approval": plane["real_vintage_approval"],
            "approved_external_inputs": approved,
            "forbidden_external_inputs": sorted(forbidden),
        },
        "training_population": {
            "unit": "person",
            "persons": len(eph_rows),
            "households": len(group_folds),
            "neutral_eph_identity": ["CODUSU", "NRO_HOGAR", "COMPONENTE"],
            "survey_design_fields_present": [
                field
                for field in spec["eph_design_policy"]["preserved_fields"]
                if field in eph_rows[0]
            ],
        },
        "fold_policy": {
            "strategy": "household_grouped_v1",
            "n_splits": n_splits,
            "households_crossing_folds": leakage_households,
            "learned_intermediate_training_source": "out_of_fold_predictions",
        },
        "weight_policy": {
            "fit": None,
            "calibration": None,
            "evaluation": None,
            "claim_boundary": "sample_conditional",
            "census_selection_probability_used_as_model_weight": False,
        },
        "temporal_policy": {
            "target_period_calibration": "none",
            "aggregate_anchor": None,
            "donor_field_mutation": False,
        },
        "household_policy": {
            "operation": "sum_linear_person_income",
            "all_members_required": True,
            "incomplete_on_invalid_member": True,
        },
        "stages": stage_audit,
        "metrics": {
            "eph_oof_person": {
                "direct": _metrics(eph_truth, direct_oof),
                "staged": _metrics(eph_truth, staged_oof),
            },
            "eph_oof_household": {
                "direct": _household_metrics(eph_households, "direct"),
                "staged": _household_metrics(eph_households, "staged"),
            },
            "synthetic_census_person": {
                "direct": _metrics(census_truth, direct_census),
                "staged": _metrics(census_truth, staged_census),
            },
            "synthetic_census_household": {
                "direct": _household_metrics(census_households, "direct"),
                "staged": _household_metrics(census_households, "staged"),
            },
        },
        "census_sampler_lineage": {
            "selection_probability_preserved_in_person_audit": True,
            "design_inverse_probability_weight_preserved_in_person_audit": True,
            "neither_used_for_transport_fitting_or_metrics": True,
        },
        "limitations": [
            "Synthetic data are generated with known truth and cannot establish real transport validity.",
            "No real EPH/CPV-2010 semantic mapping is approved by this proof.",
            "No target-period labor/activity calibration is performed.",
            "No full Census scoring or model promotion is performed.",
        ],
    }
    if leakage_households:
        raise TransportProofError("household_fold_leakage_detected")
    if any(record["status"] != "complete" for record in census_households):
        raise TransportProofError("synthetic_census_household_incomplete")

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        eph_person = _person_audit_rows(
            eph_rows,
            direct_oof,
            staged_oof,
            include_selection_probability=False,
        )
        census_person = _person_audit_rows(
            census_rows,
            direct_census,
            staged_census,
            include_selection_probability=True,
        )
        _write_csv(output / "eph_oof_person_predictions.csv", eph_person)
        _write_csv(output / "eph_oof_household_predictions.csv", eph_households)
        _write_csv(output / "census_person_predictions.csv", census_person)
        _write_csv(output / "census_household_predictions.csv", census_households)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
