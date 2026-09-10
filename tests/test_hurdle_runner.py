from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from encuestador.experiments import resolve_experiment
from encuestador.runner import execute_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for household in range(72):
        members = 1 + household % 3
        for component in range(1, members + 1):
            age = 17 + (household * 3 + component * 7) % 65
            estado = 1 + (household + component) % 3
            index = len(rows)
            if index % 19 == 0:
                income: object = -9
            elif index % 31 == 0:
                income = None
            elif index % 4 == 0:
                income = 0
            else:
                income = float(50000 + 3500 * age + 12000 * estado + 7000 * component)
            rows.append(
                {
                    "CODUSU": f"Q3-{household:04d}",
                    "NRO_HOGAR": "1",
                    "COMPONENTE": component,
                    "ANO4": 2024,
                    "TRIMESTRE": 3,
                    "CH04": 1 + (component % 2),
                    "CH06": age,
                    "CH07": 1 + (household + component) % 5,
                    "CH09": 1 + (age < 18),
                    "CH10": 1 + (age > 24),
                    "CH12": 1 + (age // 12) % 7,
                    "CH13": 1 + (household % 2),
                    "CH15": 1 + (household % 3),
                    "IX_TOT": members,
                    "ESTADO": estado,
                    "P47T": income,
                }
            )
    return rows


def _resolved(tmp_path: Path, name: str):
    source = CONFIG_ROOT / "experiments" / name
    target = tmp_path / name
    target.write_text(
        source.read_text(encoding="utf-8")
        + """
overrides:
  splits:
    n_splits: 3
  estimators:
    roles:
      latent_categorical:
        params:
          early_stopping: false
          random_state: 42
          max_iter: 4
          max_leaf_nodes: 7
          min_samples_leaf: 3
          learning_rate: 0.08
      terminal_presence:
        params:
          early_stopping: false
          random_state: 42
          max_iter: 5
          max_leaf_nodes: 7
          min_samples_leaf: 3
          learning_rate: 0.08
      terminal_positive:
        params:
          early_stopping: false
          random_state: 42
          max_iter: 5
          max_leaf_nodes: 7
          min_samples_leaf: 3
          learning_rate: 0.08
""",
        encoding="utf-8",
    )
    return resolve_experiment(target, config_root=tmp_path)


@pytest.mark.parametrize(
    "name,formulation",
    [
        ("real_eph_2024q3_direct_hurdle_log_v1.yaml", "log"),
        ("real_eph_2024q3_direct_hurdle_gamma_v1.yaml", "gamma"),
        ("real_eph_2024q3_lean_hurdle_log_v1.yaml", "log"),
        ("real_eph_2024q3_lean_hurdle_gamma_v1.yaml", "gamma"),
    ],
)
def test_committed_real_eph_hurdle_contracts_execute_on_real_shaped_rows(
    tmp_path: Path, name: str, formulation: str
) -> None:
    result = execute_experiment(_resolved(tmp_path, name), _rows())
    assert result.hurdle is not None
    assert result.hurdle.formulation == formulation
    assert result.oof_prediction is result.hurdle.unconditional_expected_income
    assert result.oof_prediction.fold_ids == result.fold_manifest.fold_ids
    assert np.all(result.hurdle.p_positive.values >= 0)
    assert np.all(result.hurdle.p_positive.values <= 1)
    assert np.all(result.hurdle.positive_amount_prediction.values > 0)
    assert np.all(result.oof_prediction.values >= 0)
    eligibility = result.hurdle.target_eligibility
    assert eligibility["positive"] > 0
    assert eligibility["zero"] > 0
    assert eligibility["nonresponse"] > 0
    assert eligibility["missing"] > 0
    household = result.metrics["household"]
    assert household["unavailable_observed_income_household_count"] > 0
    assert household["complete_observed_income_household_count"] < household["household_observation_count"]


def test_direct_and_lean_hurdle_use_identical_household_outer_folds(tmp_path: Path) -> None:
    rows = _rows()
    direct = execute_experiment(
        _resolved(tmp_path / "direct", "real_eph_2024q3_direct_hurdle_gamma_v1.yaml"),
        rows,
    )
    lean = execute_experiment(
        _resolved(tmp_path / "lean", "real_eph_2024q3_lean_hurdle_gamma_v1.yaml"),
        rows,
    )
    assert direct.fold_manifest.row_ids == lean.fold_manifest.row_ids
    assert direct.fold_manifest.household_ids == lean.fold_manifest.household_ids
    assert direct.fold_manifest.fold_ids == lean.fold_manifest.fold_ids
    assert set(lean.latent_predictions) == {"ESTADO"}
    assert set(lean.metrics["person"]) == {"direct", "oracle", "deployable"}
    assert "capture_ratio" in lean.metrics["cascade_gains"]["mae"]
