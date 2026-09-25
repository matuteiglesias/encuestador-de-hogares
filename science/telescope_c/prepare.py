#!/usr/bin/env python3
"""Prepare matched-model EPH->Census evidence for Telescope C."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

FEATURES = [
    "IX_TOT", "P02", "P03", "P05", "P07", "P08", "P09", "P10", "CONDACT",
    "V01", "H05", "H06", "H07", "H08", "H09", "H10", "H12", "H13", "H14",
    "H15", "PROP",
]
NUMERIC_FEATURES = ("IX_TOT", "P03", "H15")
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}
DOMAIN_PARAMS = {
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "random_state": 42,
    "early_stopping": False,
}
DEFAULT_FOLD_SHA = "7e1d7fa75684d6ebdeb694176a4dcb07363f77597425405e955b62dfaa84247d"


class TelescopeCUpstreamError(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(frame: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TelescopeCUpstreamError(f"{label} missing columns: {missing}")


def stable_household_fold(household_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(str(household_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def categorical_indices(features=FEATURES) -> list[int]:
    return [features.index(name) for name in features if name not in NUMERIC_FEATURES]


def _matrix(frame: pd.DataFrame, features=FEATURES) -> np.ndarray:
    require(frame, features, "feature frame")
    x = frame[list(features)].to_numpy(float)
    if not np.isfinite(x).all():
        raise TelescopeCUpstreamError("feature frame contains non-finite values")
    return x


def fit_hurdle_predict(train: pd.DataFrame, test: pd.DataFrame, features=FEATURES) -> np.ndarray:
    if train.empty or test.empty:
        raise TelescopeCUpstreamError("hurdle train/test must be nonempty")
    require(train, [*features, "y"], "EPH training frame")
    x = _matrix(train, features)
    xt = _matrix(test, features)
    y = train["y"].to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise TelescopeCUpstreamError("EPH hurdle target must be finite and nonnegative")
    cats = categorical_indices(list(features))
    clf = HistGradientBoostingClassifier(
        **MODEL_PARAMS, categorical_features=cats
    ).fit(x, y > 0)
    positive = y > 0
    if not positive.any():
        raise TelescopeCUpstreamError("EPH training fold has no positive incomes")
    reg = HistGradientBoostingRegressor(
        loss="gamma", **MODEL_PARAMS, categorical_features=cats
    ).fit(x[positive], y[positive])
    positive_class = list(clf.classes_).index(True)
    p_positive = clf.predict_proba(xt)[:, positive_class]
    amount = reg.predict(xt)
    out = p_positive * amount
    if not np.isfinite(out).all() or (out < 0).any():
        raise TelescopeCUpstreamError("non-finite/negative P1-R prediction")
    return out


def load_eph(
    eph_persons_path: Path,
    eph_p1_path: Path,
    fold_manifest_path: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(
        eph_persons_path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        usecols=["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "P47T"],
    )
    raw["row_id"] = raw.CODUSU + ":" + raw.NRO_HOGAR + ":" + raw.COMPONENTE
    raw["household_id"] = raw.CODUSU + "\x1f" + raw.NRO_HOGAR
    raw["fold_row"] = (
        raw.CODUSU + "\x1f" + raw.NRO_HOGAR + "\x1f" + raw.COMPONENTE + "\x1f"
        + raw.ANO4 + "\x1f" + raw.TRIMESTRE
    )
    raw["y"] = pd.to_numeric(raw.P47T, errors="coerce")
    fold_rows = json.loads(fold_manifest_path.read_text())["rows"]
    fold_map = {str(row["row_id"]): int(row["fold_id"]) for row in fold_rows}
    raw["outer_fold"] = raw.fold_row.map(fold_map)
    raw = raw[raw.y.notna() & (raw.y >= 0)].copy()
    if raw.outer_fold.isna().any():
        raise TelescopeCUpstreamError("eligible EPH row missing governed outer fold")
    raw["outer_fold"] = raw.outer_fold.astype(int)
    if raw.row_id.duplicated().any():
        raise TelescopeCUpstreamError("duplicate eligible EPH row_id")
    p1 = pd.read_parquet(eph_p1_path)
    require(p1, ["row_id", *FEATURES], "EPH P1")
    if p1.row_id.duplicated().any():
        raise TelescopeCUpstreamError("duplicate EPH P1 row_id")
    out = p1.merge(
        raw[["row_id", "household_id", "outer_fold", "y"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(out) != len(raw):
        raise TelescopeCUpstreamError(
            f"eligible EPH/P1 identity mismatch: eligible={len(raw)} joined={len(out)}"
        )
    household_fold_counts = out.groupby("household_id").outer_fold.nunique()
    if (household_fold_counts != 1).any():
        raise TelescopeCUpstreamError("EPH household members cross outer folds")
    _matrix(out)
    return out


def load_census(census_p1_path: Path) -> pd.DataFrame:
    census = pd.read_parquet(census_p1_path)
    require(census, ["row_id", "household_id", *FEATURES], "Census P1")
    if census.row_id.duplicated().any():
        raise TelescopeCUpstreamError("duplicate Census P1 row_id")
    if census.household_id.isna().any():
        raise TelescopeCUpstreamError("Census P1 missing household_id")
    _matrix(census)
    return census


def matched_model_scores(
    eph: pd.DataFrame,
    census: pd.DataFrame,
    persisted_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require(persisted_oof, ["row_id", "pred"], "persisted P1-R OOF")
    if persisted_oof.row_id.duplicated().any():
        raise TelescopeCUpstreamError("duplicate persisted OOF row_id")
    persisted = persisted_oof.set_index("row_id")["pred"].astype(float)

    eph_rows: list[pd.DataFrame] = []
    census_rows: list[pd.DataFrame] = []
    fold_diag = []
    folds = sorted(int(value) for value in eph.outer_fold.unique())
    if folds != list(range(5)):
        raise TelescopeCUpstreamError(f"expected outer folds 0..4, got {folds}")

    for fold in folds:
        train = eph[eph.outer_fold != fold]
        test = eph[eph.outer_fold == fold]
        eph_pred = fit_hurdle_predict(train, test)
        census_pred = fit_hurdle_predict(train, census)

        e = test[["row_id", "household_id", "outer_fold"]].copy()
        e["pred"] = eph_pred
        eph_rows.append(e)

        c = census[["row_id", "household_id"]].copy()
        c["outer_fold"] = fold
        c["pred"] = census_pred
        census_rows.append(c[["row_id", "household_id", "outer_fold", "pred"]])

        expected = persisted.reindex(test.row_id)
        if expected.isna().any():
            raise TelescopeCUpstreamError(f"persisted P1-R OOF missing fold {fold} rows")
        delta = eph_pred - expected.to_numpy(float)
        scale = np.maximum(np.abs(expected.to_numpy(float)), 1.0)
        max_abs = float(np.max(np.abs(delta)))
        max_rel = float(np.max(np.abs(delta) / scale))
        reproduced = bool(np.allclose(
            eph_pred, expected.to_numpy(float), rtol=1e-10, atol=1e-7
        ))
        fold_diag.append({
            "outer_fold": fold,
            "train_persons": int(len(train)),
            "heldout_persons": int(len(test)),
            "census_persons": int(len(census)),
            "max_abs_delta_vs_persisted_oof": max_abs,
            "max_relative_delta_vs_persisted_oof": max_rel,
            "reproduced": reproduced,
        })
        if not reproduced:
            raise TelescopeCUpstreamError(
                f"outer model {fold} does not reproduce persisted P1-R OOF "
                f"(max_abs={max_abs}, max_rel={max_rel})"
            )

    eph_out = pd.concat(eph_rows, ignore_index=True)
    census_out = pd.concat(census_rows, ignore_index=True)
    if len(eph_out) != len(eph):
        raise TelescopeCUpstreamError("matched EPH scoring did not cover eligible source")
    if len(census_out) != 5 * len(census):
        raise TelescopeCUpstreamError("matched Census scoring did not produce five model copies")
    return eph_out, census_out, {"folds": fold_diag, "all_reproduced": True}


def hard_support_weak(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    weak = np.zeros(len(target), dtype=bool)
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            lo = float(source[feature].min())
            hi = float(source[feature].max())
            values = target[feature].to_numpy(float)
            weak |= (values < lo) | (values > hi)
        else:
            seen = set(source[feature].dropna().unique())
            weak |= ~target[feature].isin(seen).to_numpy()
    return weak


def crossfit_domain_support(
    eph: pd.DataFrame,
    census: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = eph[["row_id", "household_id", "outer_fold", *FEATURES]].copy()
    source["domain_fold"] = source.outer_fold.astype(int)
    target = census[["row_id", "household_id", *FEATURES]].copy()
    target["domain_fold"] = target.household_id.map(stable_household_fold).astype(int)

    source_prob = pd.Series(np.nan, index=source.index, dtype=float)
    target_prob = pd.Series(np.nan, index=target.index, dtype=float)
    cats = categorical_indices()
    fold_diag = []

    for fold in range(5):
        source_train = source[source.domain_fold != fold]
        target_train = target[target.domain_fold != fold]
        source_test = source[source.domain_fold == fold]
        target_test = target[target.domain_fold == fold]
        if min(len(source_train), len(target_train), len(source_test), len(target_test)) == 0:
            raise TelescopeCUpstreamError(f"empty domain-classifier split at fold {fold}")

        train = pd.concat([
            source_train.assign(_domain=0),
            target_train.assign(_domain=1),
        ], ignore_index=True)
        x = _matrix(train)
        y = train._domain.to_numpy(int)
        n_source = int((y == 0).sum())
        n_target = int((y == 1).sum())
        weights = np.where(
            y == 0,
            0.5 / n_source,
            0.5 / n_target,
        )
        clf = HistGradientBoostingClassifier(
            **DOMAIN_PARAMS, categorical_features=cats
        ).fit(x, y, sample_weight=weights)
        target_class = list(clf.classes_).index(1)
        source_prob.loc[source_test.index] = clf.predict_proba(
            _matrix(source_test)
        )[:, target_class]
        target_prob.loc[target_test.index] = clf.predict_proba(
            _matrix(target_test)
        )[:, target_class]
        fold_diag.append({
            "fold": fold,
            "source_train": len(source_train),
            "target_train": len(target_train),
            "source_test": len(source_test),
            "target_test": len(target_test),
        })

    if source_prob.isna().any() or target_prob.isna().any():
        raise TelescopeCUpstreamError("cross-fitted domain probabilities incomplete")
    if not ((source_prob >= 0) & (source_prob <= 1)).all():
        raise TelescopeCUpstreamError("invalid source domain probabilities")
    if not ((target_prob >= 0) & (target_prob <= 1)).all():
        raise TelescopeCUpstreamError("invalid target domain probabilities")

    source_out = source[["row_id", "household_id", "domain_fold"]].copy()
    source_out["target_probability_equal_prior"] = source_prob.to_numpy()
    source_out["support_weak"] = False

    target_out = target[["row_id", "household_id", "domain_fold"]].copy()
    target_out["target_probability_equal_prior"] = target_prob.to_numpy()
    target_out["support_weak"] = hard_support_weak(eph, census)

    labels = np.r_[np.zeros(len(source_out), dtype=int), np.ones(len(target_out), dtype=int)]
    scores = np.r_[
        source_out.target_probability_equal_prior.to_numpy(),
        target_out.target_probability_equal_prior.to_numpy(),
    ]
    auc = float(roc_auc_score(labels, scores))
    summary = {
        "method": "five-fold household-safe domain classifier with equal total source/target training weight",
        "cross_fitted_auc": auc,
        "folds": fold_diag,
        "source_score_quantiles": {
            str(q): float(source_out.target_probability_equal_prior.quantile(q))
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "target_score_quantiles": {
            str(q): float(target_out.target_probability_equal_prior.quantile(q))
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "target_support_weak_person_fraction": float(target_out.support_weak.mean()),
        "note": "diagnostic only; probabilities are not production analysis weights",
    }
    return source_out, target_out, summary


def run(args: argparse.Namespace) -> dict:
    fold_manifest = Path(args.fold_manifest).resolve()
    if args.expected_fold_manifest_sha:
        observed = sha256(fold_manifest)
        if observed != args.expected_fold_manifest_sha:
            raise TelescopeCUpstreamError(
                f"fold manifest SHA mismatch: {observed} != {args.expected_fold_manifest_sha}"
            )

    eph = load_eph(
        Path(args.eph_persons).resolve(),
        Path(args.eph_p1).resolve(),
        fold_manifest,
    )
    census = load_census(Path(args.census_p1).resolve())
    persisted = pd.read_json(Path(args.person_oof).resolve(), lines=True)

    eph_scores, census_scores, reproduction = matched_model_scores(
        eph, census, persisted
    )
    eph_support, census_support, support = crossfit_domain_support(eph, census)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eph_scores.to_parquet(output / "eph_matched_person_scores.parquet", index=False)
    census_scores.to_parquet(output / "census_matched_person_scores.parquet", index=False)
    eph_support.to_parquet(output / "eph_support_scores.parquet", index=False)
    census_support.to_parquet(output / "census_support_scores.parquet", index=False)
    (output / "model_reproduction.json").write_text(json.dumps(reproduction, indent=2))
    (output / "support_summary.json").write_text(json.dumps(support, indent=2))

    manifest = {
        "status": "RESEARCH_TRANSPORT_DIAGNOSTIC_NOT_OFFICIAL_STATISTICS",
        "scope": "Telescope C matched P1-R outer-model scoring and support diagnostics",
        "features": FEATURES,
        "numeric_features": list(NUMERIC_FEATURES),
        "model_params": MODEL_PARAMS,
        "domain_params": DOMAIN_PARAMS,
        "eligible_eph_persons": int(len(eph)),
        "census_persons": int(len(census)),
        "census_households": int(census.household_id.nunique()),
        "outer_folds": [0, 1, 2, 3, 4],
        "model_reproduction": reproduction,
        "support": support,
        "parents": {
            "eph_persons_sha256": sha256(Path(args.eph_persons).resolve()),
            "eph_p1_sha256": sha256(Path(args.eph_p1).resolve()),
            "census_p1_sha256": sha256(Path(args.census_p1).resolve()),
            "fold_manifest_sha256": sha256(fold_manifest),
            "person_oof_sha256": sha256(Path(args.person_oof).resolve()),
        },
        "warnings": [
            "support classifier is diagnostic only",
            "no domain adaptation or transport weighting applied",
            "poverty lines/classification belong downstream",
        ],
    }
    files = [
        "eph_matched_person_scores.parquet",
        "census_matched_person_scores.parquet",
        "eph_support_scores.parquet",
        "census_support_scores.parquet",
        "model_reproduction.json",
        "support_summary.json",
    ]
    manifest["files"] = {
        name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
        for name in files
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Telescope C matched transport evidence")
    p.add_argument("--eph-persons", required=True)
    p.add_argument("--eph-p1", required=True)
    p.add_argument("--census-p1", required=True)
    p.add_argument("--fold-manifest", required=True)
    p.add_argument("--person-oof", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-fold-manifest-sha", default=DEFAULT_FOLD_SHA)
    return p


def main() -> None:
    args = parser().parse_args()
    manifest = run(args)
    print(json.dumps({
        "status": manifest["status"],
        "eligible_eph_persons": manifest["eligible_eph_persons"],
        "census_persons": manifest["census_persons"],
        "census_households": manifest["census_households"],
        "cross_fitted_domain_auc": manifest["support"]["cross_fitted_auc"],
        "output": str(Path(args.output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
