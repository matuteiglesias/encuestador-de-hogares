"""Promote Q7/Q8 science into a compact predictive household-welfare release.

The release preserves the validated deployment representation instead of
collapsing it to a point estimate only:

    Y_h = max(0, point_welfare_h + R)

where ``R`` is the exact household-level EPH OOF residual ECDF reconstructed
from the governed P1-R OOF predictions and immutable complete-household cohort.

This builder performs no model training and no Census rescoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ARTIFACT_TYPE = "research.household-welfare-predictive/v1"
REPRESENTATION = "additive_empirical_residual_ecdf"
SUPPORT_POLICY = "floor_at_zero"
WELFARE_CONCEPT = "household_total_family_income"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_complete_households(path: Path) -> set[str]:
    h0 = pd.read_json(path, lines=True)
    observed = h0.loc[h0["observed_household_income"].notna(), "household_observation_id"]
    return {"\x1f".join(str(value).split("\x1f")[:2]) for value in observed}


def _reconstruct_residuals(
    *,
    person_oof_path: Path,
    eph_individual_path: Path,
    baseline_household_oof_path: Path,
) -> pd.DataFrame:
    oof = pd.read_json(person_oof_path, lines=True)
    if "row_id" not in oof.columns or "pred" not in oof.columns:
        raise ValueError("P1-R person OOF must contain row_id and pred")
    if oof["row_id"].duplicated().any():
        raise ValueError("duplicate P1-R OOF row_id")

    ind = pd.read_csv(eph_individual_path, sep=";", dtype=str, keep_default_na=False)
    required = {"CODUSU", "NRO_HOGAR", "COMPONENTE", "P47T"}
    missing = required - set(ind.columns)
    if missing:
        raise ValueError(f"EPH individual file missing columns: {sorted(missing)}")
    ind["row_id"] = ind["CODUSU"] + ":" + ind["NRO_HOGAR"] + ":" + ind["COMPONENTE"]
    ind["household_id"] = ind["CODUSU"] + "\x1f" + ind["NRO_HOGAR"]
    ind["observed_person_income"] = pd.to_numeric(ind["P47T"], errors="coerce")
    ind = ind[ind["observed_person_income"].notna() & (ind["observed_person_income"] >= 0)].copy()

    joined = oof[["row_id", "pred"]].merge(
        ind[["row_id", "household_id", "observed_person_income"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(oof):
        raise ValueError(f"OOF/EPH identity mismatch: {len(oof)} OOF rows, {len(joined)} joined")

    complete = _load_complete_households(baseline_household_oof_path)
    joined = joined[joined["household_id"].isin(complete)].copy()
    hh = joined.groupby("household_id", sort=True).agg(
        observed_welfare=("observed_person_income", "sum"),
        point_welfare=("pred", "sum"),
        member_count=("row_id", "size"),
    )
    if set(hh.index) != complete:
        missing_hh = sorted(complete - set(hh.index))[:20]
        extra_hh = sorted(set(hh.index) - complete)[:20]
        raise ValueError(f"complete-household cohort mismatch missing={missing_hh} extra={extra_hh}")
    hh["residual"] = hh["observed_welfare"] - hh["point_welfare"]
    return hh.reset_index()[["household_id", "residual"]].sort_values("residual").reset_index(drop=True)


def build_release(args: argparse.Namespace) -> Path:
    q8_root = Path(args.q8_root).resolve()
    q2_root = Path(args.q2_root).resolve()
    output_parent = Path(args.output_parent).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)

    q8_households = pd.read_parquet(q8_root / "household_predictions.parquet")
    q8_people = pd.read_parquet(q8_root / "person_predictions.parquet")
    required_h = {"household_id", "predicted_household_income", "support_weak"}
    required_p = {"household_id", "domain_prob"}
    if required_h - set(q8_households.columns):
        raise ValueError(f"Q8 household predictions missing {sorted(required_h - set(q8_households.columns))}")
    if required_p - set(q8_people.columns):
        raise ValueError(f"Q8 person predictions missing {sorted(required_p - set(q8_people.columns))}")
    if q8_households["household_id"].duplicated().any():
        raise ValueError("duplicate Q8 household_id")

    # Q8's research script accidentally persisted household `domain_prob` from
    # support_weak rather than person domain propensity. Recompute the intended
    # diagnostic from the correct person-level Q8 field instead of propagating it.
    hh_domain = q8_people.groupby("household_id", sort=False)["domain_prob"].mean().rename("domain_propensity")
    households = q8_households[["household_id", "predicted_household_income", "support_weak"]].copy()
    households = households.merge(hh_domain, on="household_id", how="left", validate="one_to_one")
    if households["domain_propensity"].isna().any():
        raise ValueError("missing household domain propensity after Q8 person aggregation")

    residuals = _reconstruct_residuals(
        person_oof_path=q2_root / "person_oof.jsonl",
        eph_individual_path=Path(args.eph_individual).resolve(),
        baseline_household_oof_path=Path(args.baseline_household_oof).resolve(),
    )
    if len(residuals) != args.expected_residual_count:
        raise ValueError(f"unexpected residual count {len(residuals)} != {args.expected_residual_count}")

    scale = float(args.monetary_scale)
    if not scale > 0:
        raise ValueError("monetary_scale must be positive")
    households["point_welfare"] = households.pop("predicted_household_income").astype(float) * scale
    households["estimation_status"] = "estimated"
    households["support_status"] = households.pop("support_weak").map({False: "high_support", True: "weak_support"})
    residuals["residual"] = residuals["residual"].astype(float) * scale

    identity_seed = {
        "artifact_type": ARTIFACT_TYPE,
        "census_sample_release_id": args.census_sample_release_id,
        "semantic_plane_release_id": args.semantic_plane_release_id,
        "eph_release_id": args.eph_release_id,
        "q8_input_sha256": sha256(q8_root / "household_predictions.parquet"),
        "q2_oof_sha256": sha256(q2_root / "person_oof.jsonl"),
        "monetary_reference": args.monetary_reference,
        "monetary_scale": scale,
        "support_policy": SUPPORT_POLICY,
    }
    release_digest = hashlib.sha256(json.dumps(identity_seed, sort_keys=True).encode()).hexdigest()[:16]
    release_id = f"household-welfare-predictive-{args.welfare_period.lower().replace('_','-')}-{release_digest}"
    root = output_parent / release_id
    root.mkdir(parents=True, exist_ok=False)

    households = households[[
        "household_id", "point_welfare", "estimation_status", "support_status", "domain_propensity"
    ]].sort_values("household_id").reset_index(drop=True)
    households.to_parquet(root / "household_locations.parquet", index=False)
    residuals[["residual"]].to_parquet(root / "residual_ecdf.parquet", index=False)

    residual_values = residuals["residual"]
    floor_diag = {
        "note": "fraction of empirical residual draws that would floor to zero depends on household location; aggregate diagnostics belong to EPH downstream validation",
        "residual_count": int(len(residual_values)),
        "residual_min": float(residual_values.min()),
        "residual_median": float(residual_values.median()),
        "residual_max": float(residual_values.max()),
    }
    (root / "support_policy.json").write_text(json.dumps({
        "policy": SUPPORT_POLICY,
        "formula": "welfare = max(0, point_welfare + residual)",
        "diagnostics": floor_diag,
    }, indent=2))

    files = {}
    for name in ("household_locations.parquet", "residual_ecdf.parquet", "support_policy.json"):
        path = root / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema": "research-artifact-manifest/v1",
        "artifact_type": ARTIFACT_TYPE,
        "release_id": release_id,
        "status": "research_commissioned_with_transport_caveats",
        "frame_namespace": args.frame_namespace,
        "welfare_period": args.welfare_period,
        "currency": args.currency,
        "monetary_reference": args.monetary_reference,
        "welfare_concept": WELFARE_CONCEPT,
        "representation": REPRESENTATION,
        "support_policy": SUPPORT_POLICY,
        "calibration_source": args.eph_release_id,
        "calibration_method": "EPH household OOF residual ECDF from governed P1-R predictions",
        "household_count": int(len(households)),
        "residual_count": int(len(residuals)),
        "monetary_scale": scale,
        "parents": {
            "census_sample_release_id": args.census_sample_release_id,
            "semantic_plane_release_id": args.semantic_plane_release_id,
            "eph_release_id": args.eph_release_id,
            "q7_evidence": args.q7_evidence_id,
            "q8_evidence": args.q8_evidence_id,
        },
        "warnings": [
            "research_commissioning_not_official_statistics",
            "Q8 transport caveats remain active",
            "collective/private dwelling refinement deferred",
            "aggregate uncertainty not supplied",
        ],
        "files": files,
        "identity_seed": identity_seed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    checks = [f"{sha256(root / name)}  {name}" for name in sorted(files)]
    checks.append(f"{sha256(root / 'manifest.json')}  manifest.json")
    (root / "checksums.sha256").write_text("\n".join(checks) + "\n")
    return root


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--q8-root", required=True)
    p.add_argument("--q2-root", required=True)
    p.add_argument("--eph-individual", required=True)
    p.add_argument("--baseline-household-oof", required=True)
    p.add_argument("--output-parent", required=True)
    p.add_argument("--frame-namespace", required=True)
    p.add_argument("--welfare-period", default="2024-Q3")
    p.add_argument("--currency", default="ARS")
    p.add_argument("--monetary-reference", required=True)
    p.add_argument("--monetary-scale", type=float, required=True)
    p.add_argument("--census-sample-release-id", default="census-sample-2024-0839713eafea8d1b")
    p.add_argument("--semantic-plane-release-id", default="eph-cpv2010-semantic-plane-2024q3-v1")
    p.add_argument("--eph-release-id", default="eph-2024-q3-3b6a7a15c4af")
    p.add_argument("--q7-evidence-id", default="science/2026-09-11/results/q7_predictive_distribution")
    p.add_argument("--q8-evidence-id", default="science/2026-09-11/results/q8_census_commissioning")
    p.add_argument("--expected-residual-count", type=int, default=12568)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(build_release(args))
