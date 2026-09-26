from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE = (
    Path(__file__).parents[1]
    / "science"
    / "telescope_b"
    / "prepare_nested_residuals.py"
)
SPEC = importlib.util.spec_from_file_location("nested_residual_prepare", MODULE)
assert SPEC and SPEC.loader
NR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NR
SPEC.loader.exec_module(NR)


def test_period_parts_and_inner_folds_are_deterministic_household_safe():
    assert NR.period_parts("2025-Q3") == (2025, 3)
    households = pd.Series(["b", "a", "b", "c", "a", "d", "e"])
    first = NR.assign_inner_folds(households, 5)
    second = NR.assign_inner_folds(households.sample(frac=1, random_state=7).sort_index(), 5)
    assert first.equals(second)
    frame = pd.DataFrame({"hh": households, "inner": first})
    assert (frame.groupby("hh").inner.nunique() == 1).all()


def test_household_residuals_filters_to_exact_complete_cohort():
    validation = pd.DataFrame(
        {
            "hh": ["a", "a", "b", "b", "c"],
            "y": [10.0, 20.0, 5.0, 15.0, 100.0],
        }
    )
    predictions = np.array([8.0, 18.0, 7.0, 13.0, 90.0])
    out = NR.household_residuals(validation, predictions, {"a", "b"}).set_index("hh")
    assert set(out.index) == {"a", "b"}
    assert out.loc["a", "observed"] == 30.0
    assert out.loc["a", "point"] == 26.0
    assert out.loc["a", "residual"] == 4.0
    assert out.loc["b", "residual"] == 0.0


def test_model_contract_remains_frozen():
    assert NR.FEATURES == [
        "IX_TOT", "P02", "P03", "P05", "P07", "P08", "P09", "P10", "CONDACT",
        "V01", "H05", "H06", "H07", "H08", "H09", "H10", "H12", "H13", "H14",
        "H15", "PROP",
    ]
    assert NR.MODEL_PARAMS == {
        "learning_rate": 0.08,
        "max_iter": 60,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "random_state": 42,
        "early_stopping": False,
    }


def test_canonical_fold_policy_matches_repository_contract():
    raw = pd.DataFrame(
        [
            {
                "CODUSU": "hhA",
                "NRO_HOGAR": "1",
                "COMPONENTE": "1",
                "ANO4": "2024",
                "TRIMESTRE": "3",
            },
            {
                "CODUSU": "hhA",
                "NRO_HOGAR": "1",
                "COMPONENTE": "2",
                "ANO4": "2024",
                "TRIMESTRE": "3",
            },
            {
                "CODUSU": "hhB",
                "NRO_HOGAR": "1",
                "COMPONENTE": "1",
                "ANO4": "2024",
                "TRIMESTRE": "3",
            },
        ]
    )
    payload = NR.build_canonical_fold_manifest(raw)
    assert payload["policy"] == "household_grouped_v1"
    assert payload["n_splits"] == 5
    rows = payload["rows"]
    assert rows[0]["row_id"] == "hhA\x1f1\x1f1\x1f2024\x1f3"
    assert rows[0]["household_group_id"] == "hhA\x1f1"
    assert rows[0]["fold_id"] == rows[1]["fold_id"]
    expected = int.from_bytes(
        __import__("hashlib").sha256("hhA\x1f1".encode()).digest()[:8],
        "big",
    ) % 5
    assert rows[0]["fold_id"] == expected


def test_canonical_json_is_stable():
    payload = {
        "policy": "household_grouped_v1",
        "n_splits": 5,
        "rows": [
            {
                "row_id": "a\x1f1\x1f1\x1f2024\x1f3",
                "household_group_id": "a\x1f1",
                "fold_id": 2,
            }
        ],
    }
    encoded = NR.canonical_json_bytes(payload)
    assert encoded.endswith(b"\n")
    assert b" " not in encoded


def test_outer_oof_uses_each_household_in_one_heldout_fold_and_covers_all_rows():
    rows = []
    for fold in range(5):
        for household_index in range(2):
            hh = f"h{fold}_{household_index}"
            for person_index in range(2):
                row = {
                    "row_id": f"{hh}:{person_index}",
                    "hh": hh,
                    "y": float(10 + fold + person_index),
                    "outer_fold": fold,
                }
                for feature in NR.FEATURES:
                    row[feature] = float(person_index + 1)
                rows.append(row)
    model = pd.DataFrame(rows)

    def fake_predict(train, test):
        heldout = set(test.outer_fold.unique())
        assert len(heldout) == 1
        assert not set(train.outer_fold.unique()) & heldout
        return np.full(len(test), 3.0)

    with patch.object(NR, "fit_hurdle_predict", side_effect=fake_predict):
        out = NR.outer_oof_predictions(model)

    assert len(out) == len(model)
    assert out.row_id.nunique() == len(model)
    assert set(out.fold) == {0, 1, 2, 3, 4}
    assert (out.groupby("hh").fold.nunique() == 1).all()
    assert (out.pred == 3.0).all()


def test_historical_comparison_is_non_blocking_sensitivity(tmp_path):
    canonical = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "hh": ["ha", "hb"],
            "y": [1.0, 2.0],
            "fold": [0, 1],
            "pred": [1.5, 2.5],
        }
    )
    historical = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "fold": [4, 1],
            "pred": [1.0, 3.0],
        }
    )
    path = tmp_path / "old.jsonl"
    historical.to_json(path, orient="records", lines=True)
    summary = NR.compare_historical_oof(canonical, path)
    assert summary["status"] == "sensitivity_only"
    assert summary["fold_agreement"] == 0.5
    assert summary["max_abs_prediction_delta"] == 0.5
