import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from encuestador.predictive_distribution import (
    PredictiveDistributionError,
    fit_nested_empirical_residual_distribution,
)
from encuestador.science_diagnostics import (
    oracle_presence_diagnostic,
    select_household_evaluation,
    target_status_positive_upper_bound,
    welfare_diagnostics,
)


def test_nested_distribution_uses_disjoint_households_and_returns_cdf():
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = x[:, 0] + np.tile([-1.0, 1.0], 6)
    result = fit_nested_empirical_residual_distribution(
        x, y, [f"train-{i}" for i in range(12)], ["test-a", "test-b"],
        LinearRegression, inner_folds=3,
    )
    probabilities = result.probability_below([[2.0, 3.0], [30.0, 31.0]], [4.0, 25.0])
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert result.metadata["experimental"] is True
    assert result.metadata["global_oof_residual_reuse_valid"] is False
    assert result.metadata["survey_weights"] == "not_used"


def test_nested_distribution_hard_fails_on_household_overlap():
    with pytest.raises(PredictiveDistributionError, match="overlap"):
        fit_nested_empirical_residual_distribution(
            [[0], [1], [2], [3]], [0, 1, 2, 3],
            ["a", "b", "c", "d"], ["d"], LinearRegression, inner_folds=2,
        )


def test_science_diagnostic_surface_and_oracle_labels():
    report = welfare_diagnostics(
        [0, 2, 4, 8], [1, 2, 3, 9],
        np.arange(1, 21), np.arange(1, 21)[::-1],
        person_fold_ids=[0, 0, 1, 1], household_fold_ids=np.tile([0, 1], 10),
    )
    assert report["weighting"] == "none"
    assert report["positive_amount_head"]["count"] == 3
    assert report["household"]["spearman_rank"] == pytest.approx(-1)
    assert len(report["transitions"]["quintile"]["matrix"]) == 5
    assert len(report["transitions"]["decile"]["matrix"]) == 10
    assert len(report["observed_decile_bias"]) == 10
    assert "person_fold_stability" in report and "household_fold_stability" in report
    assert oracle_presence_diagnostic([1, 2], [1, 2])["deployable"] is False
    assert target_status_positive_upper_bound([0, 2], [1, 2], [False, True])["deployable"] is False


def test_household_completeness_selection_is_explicit():
    rows = [{"status": "complete", "id": 1}, {"status": "incomplete", "id": 2}]
    selected = select_household_evaluation(rows)
    assert selected["selection"] == "complete_only"
    assert [row["id"] for row in selected["records"]] == [1]
    assert len(select_household_evaluation(rows, allow_incomplete=True)["records"]) == 2
