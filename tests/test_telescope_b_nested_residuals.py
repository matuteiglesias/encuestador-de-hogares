from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
