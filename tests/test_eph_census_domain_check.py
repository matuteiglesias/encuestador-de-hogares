from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "domain_check",
    ROOT / "science" / "commissioning" / "run_eph_census_domain_check.py",
)
DOMAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DOMAIN)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    handoff = tmp_path / "g2"
    handoff.mkdir()

    eph_rows = []
    raw_rows = []
    for aglo in ("32", "33"):
        for i in range(30):
            codusu = f"E{aglo}{i:03d}"
            row_id = f"{codusu}:1:1"
            household_id = f"{codusu}:1"
            eph_rows.append(
                {
                    "row_id": row_id,
                    "household_id": household_id,
                    "P02": 1 + (i % 2),
                    "P03": 20 + i,
                    "V01": 1 + (i % 2),
                }
            )
            raw_rows.append(
                {
                    "CODUSU": codusu,
                    "NRO_HOGAR": "1",
                    "COMPONENTE": "1",
                    "ANO4": "2024",
                    "TRIMESTRE": "3",
                    "AGLOMERADO": aglo,
                    "PONDERA": str(1 + (i % 3)),
                }
            )

    census_rows = []
    geo_rows = []
    for aglo in ("32", "33"):
        for i in range(40):
            hh = f"C{aglo}{i:03d}"
            census_rows.append(
                {
                    "row_id": f"{hh}:p1",
                    "household_id": hh,
                    "P02": 1 + ((i + 1) % 2),
                    "P03": 25 + i,
                    "V01": 2 if i < 25 else 1,
                }
            )
            geo_rows.append(
                {
                    "sample_household_id": hh,
                    "eph_agglomerate_id": aglo,
                    "mapped_to_eph_frame": True,
                    "outside_eph_frame": False,
                    "design_inverse_probability_weight": 999999.0,
                }
            )
    for i in range(10):
        hh = f"OUT{i:03d}"
        census_rows.append(
            {
                "row_id": f"{hh}:p1",
                "household_id": hh,
                "P02": 1,
                "P03": 40,
                "V01": 1,
            }
        )
        geo_rows.append(
            {
                "sample_household_id": hh,
                "eph_agglomerate_id": None,
                "mapped_to_eph_frame": False,
                "outside_eph_frame": True,
                "design_inverse_probability_weight": 999999.0,
            }
        )

    pd.DataFrame(eph_rows).to_parquet(semantic / "eph_p1.parquet", index=False)
    pd.DataFrame(census_rows).to_parquet(semantic / "census_p1.parquet", index=False)
    manifest = {
        "schema": DOMAIN.SEMANTIC_SCHEMA,
        "release_id": "semantic-fixture",
        "parents": {
            "eph_release_id": "eph-fixture",
            "census_frame_release_id": "frame-fixture",
            "census_sample_release_id": "census-sample-fixture",
        },
        "clocks": {
            "eph_period": "2024-Q3",
            "census_vintage": 2010,
            "sampling_target_year": 2024,
        },
        "p1_r_fields": ["P02", "P03", "V01"],
        "consumer_handoff": {
            "semantic_alignment_only": True,
            "statistical_transport_authorized": False,
            "temporal_roles": {
                "P02": "stable/shared",
                "P03": "target-period-state",
                "V01": "research-only",
            },
        },
    }
    (semantic / "feature_plane_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    raw = tmp_path / "usu_individual_t324.txt"
    pd.DataFrame(raw_rows).to_csv(raw, sep=";", index=False)

    pd.DataFrame(geo_rows).to_parquet(
        handoff / "household_geography.parquet", index=False
    )
    g2_manifest = {
        "contract": DOMAIN.G2_CONTRACT,
        "sample_parent": {
            "release_id": "census-sample-fixture",
            "frame_vintage": 2010,
        },
        "geography_parent": {
            "dataset_id": "arggeo.indec.eph.census2010.agglomerate-footprint",
            "release_version": "fixture-a7",
            "parent_kind": "G1_first_class_agglomerate",
        },
    }
    (handoff / "manifest.json").write_text(json.dumps(g2_manifest), encoding="utf-8")
    return semantic, raw, handoff


def test_feature_tiers_follow_temporal_roles_without_duplicate_policy() -> None:
    manifest = {
        "p1_r_fields": ["P02", "P03", "V01", "CONDACT"],
        "consumer_handoff": {
            "temporal_roles": {
                "P02": "stable/shared",
                "P03": "target-period-state",
                "V01": "research-only",
                "CONDACT": "target-period-state",
            }
        },
    }
    tiers = DOMAIN.feature_tiers(manifest, exclude_fields={"CONDACT"})
    assert tiers == {
        "S": ["P02"],
        "S+T": ["P02", "P03"],
        "S+T+R": ["P02", "P03", "V01"],
    }


def test_thin_domain_check_materializes_oof_and_two_lenses(tmp_path: Path) -> None:
    semantic, raw, handoff = _fixture(tmp_path)
    output = tmp_path / "out"
    manifest = DOMAIN.run(
        semantic_plane=semantic,
        eph_individual=raw,
        census_geography_handoff=handoff,
        output=output,
        numeric_fields={"P03"},
        exclude_fields=set(),
        min_source_persons=10,
        folds=5,
        random_state=42,
    )

    assert manifest["status"] == "diagnostic_only"
    assert manifest["statistical_transport_authorized"] is False
    assert manifest["no_reweighting_or_raking_applied"] is True
    assert manifest["clocks"]["census_vintage"] == 2010
    assert manifest["feature_tiers"]["S"] == ["P02"]
    assert manifest["feature_tiers"]["S+T"] == ["P02", "P03"]
    assert manifest["feature_tiers"]["S+T+R"] == ["P02", "P03", "V01"]

    inventory = pd.read_csv(output / "domain_inventory.csv", dtype={"domain": str})
    pooled = inventory.loc[inventory["domain"] == "EPH_TOTAL"].iloc[0]
    outside = inventory.loc[inventory["domain"] == "OUTSIDE_EPH_FRAME"].iloc[0]
    assert int(pooled["eph_persons"]) == 60
    assert int(pooled["census_persons"]) == 80
    assert int(outside["census_persons"]) == 10

    classifier = pd.read_csv(output / "domain_classifier_oof.csv", dtype={"domain": str})
    assert set(classifier["domain"]) == {"EPH_TOTAL", "32", "33"}
    assert set(classifier["tier"]) == {"S", "S+T", "S+T+R"}
    assert classifier["auc_oof"].between(0, 1).all()
    assert classifier["brier_oof"].between(0, 1).all()
    assert np.isfinite(classifier["log_loss_oof"]).all()
    assert set(classifier["n_splits"]) == {5}

    marginals = pd.read_csv(output / "marginal_comparison.csv", dtype={"domain": str})
    assert set(marginals["lens"]) == {"model_support", "population_composition"}
    pop = marginals[
        (marginals["domain"] == "EPH_TOTAL")
        & (marginals["lens"] == "population_composition")
        & (marginals["concept"] == "P02")
    ].iloc[0]
    model = marginals[
        (marginals["domain"] == "EPH_TOTAL")
        & (marginals["lens"] == "model_support")
        & (marginals["concept"] == "P02")
    ].iloc[0]
    assert float(pop["eph_weight_sum"]) == sum(1 + (i % 3) for i in range(30)) * 2
    assert float(model["eph_weight_sum"]) == 60.0
    # Census design inverse weights are deliberately ignored in both lenses.
    assert float(pop["census_weight_sum"]) == 80.0
    assert float(model["census_weight_sum"]) == 80.0


def test_eph_period_clock_fails_closed(tmp_path: Path) -> None:
    semantic, raw, handoff = _fixture(tmp_path)
    frame = pd.read_csv(raw, sep=";", dtype=str)
    frame["TRIMESTRE"] = "2"
    frame.to_csv(raw, sep=";", index=False)
    with pytest.raises(DOMAIN.DomainCheckError, match="raw_eph_period_mismatch"):
        DOMAIN.run(
            semantic_plane=semantic,
            eph_individual=raw,
            census_geography_handoff=handoff,
            output=tmp_path / "out",
            numeric_fields={"P03"},
            exclude_fields=set(),
            min_source_persons=10,
            folds=5,
            random_state=42,
        )


def test_current_g2_rejects_non_2010_donor(tmp_path: Path) -> None:
    semantic, raw, handoff = _fixture(tmp_path)
    manifest_path = handoff / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sample_parent"]["frame_vintage"] = 2022
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DOMAIN.DomainCheckError, match="current_g2_handoff_requires_cpv2010"):
        DOMAIN.run(
            semantic_plane=semantic,
            eph_individual=raw,
            census_geography_handoff=handoff,
            output=tmp_path / "out",
            numeric_fields={"P03"},
            exclude_fields=set(),
            min_source_persons=10,
            folds=5,
            random_state=42,
        )
