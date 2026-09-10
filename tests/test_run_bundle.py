from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from encuestador.experiments import resolve_experiment
from encuestador.run_bundle import RunBundleError, package_run, validate_run_bundle
from encuestador.runner import execute_experiment
from encuestador.runtime_model import fit_deployable_hurdle_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
SOURCE = CONFIG_ROOT / "experiments" / "real_eph_2024q3_direct_hurdle_gamma_v1.yaml"


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for household in range(36):
        for component in (1, 2):
            i = len(rows)
            income: object
            if i % 17 == 0:
                income = -9
            elif i % 5 == 0:
                income = 0
            else:
                income = 100000.0 + 2500.0 * i
            rows.append(
                {
                    "CODUSU": f"H{household:03d}",
                    "NRO_HOGAR": "1",
                    "COMPONENTE": component,
                    "ANO4": 2024,
                    "TRIMESTRE": 3,
                    "CH04": 1 + component % 2,
                    "CH06": 20 + (i % 55),
                    "CH07": 1 + i % 5,
                    "CH09": 1 + i % 2,
                    "CH10": 1 + (i // 2) % 2,
                    "CH12": 1 + i % 7,
                    "CH13": 1 + i % 2,
                    "CH15": 1 + i % 3,
                    "IX_TOT": 2,
                    "P47T": income,
                }
            )
    return rows


def _resolved(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "direct.yaml"
    path.write_text(
        SOURCE.read_text(encoding="utf-8")
        + """
overrides:
  splits:
    n_splits: 3
  estimators:
    roles:
      terminal_presence:
        params: {early_stopping: false, random_state: 42, max_iter: 4, max_leaf_nodes: 7, min_samples_leaf: 3}
      terminal_positive:
        params: {early_stopping: false, random_state: 42, max_iter: 4, max_leaf_nodes: 7, min_samples_leaf: 3}
""",
        encoding="utf-8",
    )
    return resolve_experiment(path, config_root=tmp_path)


def test_run_bundle_is_hash_verified_immutable_and_contains_pickled_deployable_model(
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path / "config")
    rows = _rows()
    result = execute_experiment(
        resolved,
        rows,
        input_release_ids=("eph-2024-q3-3b6a7a15c4af",),
    )
    model = fit_deployable_hurdle_model(resolved, rows)
    output = tmp_path / "runs"
    run = package_run(
        output,
        resolved,
        result,
        rows,
        model,
        input_evidence=(
            {
                "release_id": "eph-2024-q3-3b6a7a15c4af",
                "source_archive_sha256": "a" * 64,
                "source_manifest_sha256": "b" * 64,
            },
        ),
        limitations=("fixture data; packaging test only",),
    )
    manifest = validate_run_bundle(run)
    assert manifest["run_id"] == result.run_id
    assert manifest["contract"] == "research.encuestador-run/v1"
    assert (run / "resolved_config.json").is_file()
    assert (run / "fold_manifest.json").is_file()
    assert (run / "person_oof.jsonl").is_file()
    assert (run / "household_oof.jsonl").is_file()
    assert (run / "metrics.json").is_file()
    assert (run / "model.pkl").is_file()
    assert (run / "report.md").is_file()

    with (run / "model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    x = np.asarray(
        [[float(row[name]) for name in result.feature_names] for row in rows[:3]],
        dtype=float,
    )
    predicted = restored.predict_components(x).unconditional_income
    assert predicted.shape == (3,)
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted >= 0)

    household_rows = [
        json.loads(line)
        for line in (run / "household_oof.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["observed_household_income_status"] == "unavailable_member_income"
        for row in household_rows
    )
    assert any(
        row["observed_household_income_status"] == "complete"
        for row in household_rows
    )

    with pytest.raises(RunBundleError, match="immutable_run_exists"):
        package_run(
            output,
            resolved,
            result,
            rows,
            model,
            input_evidence=(),
        )
