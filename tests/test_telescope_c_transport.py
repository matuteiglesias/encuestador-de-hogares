from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).parents[1] / "science" / "telescope_c" / "prepare.py"
SPEC = importlib.util.spec_from_file_location("telescope_c_prepare", MODULE)
assert SPEC and SPEC.loader
TC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TC
SPEC.loader.exec_module(TC)


def frame(n_households: int, *, target: bool = False) -> pd.DataFrame:
    rows = []
    for h in range(n_households):
        for member in range(2):
            i = 2 * h + member
            row = {
                "row_id": f"{'T' if target else 'S'}:{h}:{member}",
                "household_id": f"{'T' if target else 'S'}:{h}",
            }
            for feature in TC.FEATURES:
                if feature == "IX_TOT":
                    row[feature] = 2.0
                elif feature == "P03":
                    row[feature] = float(18 + (i % 55))
                elif feature == "H15":
                    row[feature] = float(i % 4)
                else:
                    row[feature] = float((i + len(feature)) % 3)
            if not target:
                row["outer_fold"] = h % 5
                row["y"] = 0.0 if i % 3 == 0 else float(100 + 7 * i)
            rows.append(row)
    return pd.DataFrame(rows)


def test_stable_household_fold_is_deterministic_and_bounded():
    values = [TC.stable_household_fold(f"hh-{i}") for i in range(100)]
    assert values == [TC.stable_household_fold(f"hh-{i}") for i in range(100)]
    assert set(values) <= set(range(5))
    assert len(set(values)) == 5


def test_hard_support_flags_unseen_target_category():
    source = frame(30)
    target = frame(10, target=True)
    target.loc[target.index[0], "V01"] = 99.0
    weak = TC.hard_support_weak(source, target)
    assert weak.dtype == bool
    assert bool(weak[0])
    assert not bool(weak[1:].all())


def test_crossfit_domain_support_is_complete_and_household_safe():
    source = frame(50)
    target = frame(60, target=True)
    source_scores, target_scores, summary = TC.crossfit_domain_support(source, target)
    assert len(source_scores) == len(source)
    assert len(target_scores) == len(target)
    assert source_scores.target_probability_equal_prior.between(0, 1).all()
    assert target_scores.target_probability_equal_prior.between(0, 1).all()
    assert np.isfinite(summary["cross_fitted_auc"])
    assert 0 <= summary["cross_fitted_auc"] <= 1
    assert target_scores.groupby("household_id").domain_fold.nunique().max() == 1


def test_matched_outer_models_reproduce_persisted_oof():
    source = frame(50)
    target = frame(10, target=True)
    expected = []
    for fold in range(5):
        train = source[source.outer_fold != fold]
        test = source[source.outer_fold == fold]
        pred = TC.fit_hurdle_predict(train, test)
        expected.append(
            pd.DataFrame(
                {
                    "row_id": test.row_id.to_numpy(),
                    "fold": np.full(len(test), fold, dtype=int),
                    "pred": pred,
                }
            )
        )
    persisted = pd.concat(expected, ignore_index=True)
    eph_scores, census_scores, diagnostic = TC.matched_model_scores(
        source, target, persisted
    )
    assert diagnostic["all_reproduced"]
    assert len(eph_scores) == len(source)
    assert len(census_scores) == 5 * len(target)
    assert census_scores.groupby("row_id").outer_fold.nunique().eq(5).all()


def test_matched_outer_models_reject_b_to_c_fold_mismatch():
    source = frame(30)
    target = frame(10, target=True)
    expected = []
    for fold in range(5):
        train = source[source.outer_fold != fold]
        test = source[source.outer_fold == fold]
        pred = TC.fit_hurdle_predict(train, test)
        expected.append(
            pd.DataFrame(
                {
                    "row_id": test.row_id.to_numpy(),
                    "fold": np.full(len(test), fold, dtype=int),
                    "pred": pred,
                }
            )
        )
    persisted = pd.concat(expected, ignore_index=True)
    persisted.loc[persisted.index[0], "fold"] = (
        int(persisted.loc[persisted.index[0], "fold"]) + 1
    ) % 5
    try:
        TC.matched_model_scores(source, target, persisted)
    except TC.TelescopeCUpstreamError as exc:
        assert "B->C outer-fold identity mismatch" in str(exc)
    else:
        raise AssertionError("expected B->C fold identity failure")
