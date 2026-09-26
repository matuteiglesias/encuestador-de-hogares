#!/usr/bin/env python3
"""Thin EPH↔Census domain check on governed semantic and agglomerate surfaces.

This runner is diagnostic only. It does not mutate Census, rake weights, perform
domain adaptation, or authorize statistical transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEMANTIC_SCHEMA = "research.eph-census-semantic-feature-plane/v1"
G2_CONTRACT = "research.census-eph-agglomerate-handoff/v1"
BUNDLE_SCHEMA = "research.eph-census-domain-check/v1"
DEFAULT_NUMERIC_FIELDS = ("IX_TOT", "P03", "H15")
DEFAULT_EXCLUDE_FIELDS = ("CONDACT",)
ROLE_ORDER = {
    "stable/shared": "S",
    "target-period-state": "S+T",
    "research-only": "S+T+R",
}


class DomainCheckError(ValueError):
    """Raised when a governed input or diagnostic invariant fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainCheckError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise DomainCheckError(f"expected_json_object:{path}")
    return value


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_agglomerate(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].astype(int).map(lambda value: f"{value:02d}")
    return out


def load_semantic_plane(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    manifest = load_json(root / "feature_plane_manifest.json")
    if manifest.get("schema") != SEMANTIC_SCHEMA:
        raise DomainCheckError("unexpected_semantic_plane_schema")
    handoff = manifest.get("consumer_handoff") or {}
    if handoff.get("semantic_alignment_only") is not True:
        raise DomainCheckError("semantic_plane_missing_alignment_only_boundary")
    if handoff.get("statistical_transport_authorized") is not False:
        raise DomainCheckError("semantic_plane_must_not_pre_authorize_transport")

    eph = pd.read_parquet(root / "eph_p1.parquet")
    census = pd.read_parquet(root / "census_p1.parquet")
    for label, frame in (("eph", eph), ("census", census)):
        required = {"row_id", "household_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DomainCheckError(f"{label}_semantic_identity_missing:{missing}")
        if frame["row_id"].astype(str).duplicated().any():
            raise DomainCheckError(f"{label}_semantic_person_identity_not_unique")
        if frame["household_id"].isna().any():
            raise DomainCheckError(f"{label}_semantic_household_identity_missing")
    if list(eph.columns) != list(census.columns):
        raise DomainCheckError("semantic_plane_schema_disagreement")
    return eph, census, manifest


def _read_eph_individual(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, low_memory=False)


def attach_eph_metadata(
    semantic: pd.DataFrame,
    individual_path: Path,
    clocks: dict[str, Any],
) -> pd.DataFrame:
    raw = _read_eph_individual(individual_path)
    required = {
        "CODUSU",
        "NRO_HOGAR",
        "COMPONENTE",
        "ANO4",
        "TRIMESTRE",
        "AGLOMERADO",
        "PONDERA",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DomainCheckError(f"raw_eph_metadata_missing:{missing}")

    raw = raw.loc[:, sorted(required)].copy()
    raw["row_id"] = (
        raw["CODUSU"].astype(str)
        + ":"
        + raw["NRO_HOGAR"].astype(str)
        + ":"
        + raw["COMPONENTE"].astype(str)
    )
    raw["source_household_id"] = (
        raw["CODUSU"].astype(str) + ":" + raw["NRO_HOGAR"].astype(str)
    )
    if raw["row_id"].duplicated().any():
        raise DomainCheckError("raw_eph_person_identity_not_unique")

    period = str(clocks.get("eph_period") or "")
    try:
        year_text, quarter_text = period.split("-Q", 1)
        expected_year = int(year_text)
        expected_quarter = int(quarter_text)
    except (TypeError, ValueError) as exc:
        raise DomainCheckError("semantic_eph_period_invalid") from exc

    years = set(pd.to_numeric(raw["ANO4"], errors="raise").astype(int).unique())
    quarters = set(pd.to_numeric(raw["TRIMESTRE"], errors="raise").astype(int).unique())
    if years != {expected_year} or quarters != {expected_quarter}:
        raise DomainCheckError(
            "raw_eph_period_mismatch:"
            f"years={sorted(years)}:quarters={sorted(quarters)}:"
            f"expected={expected_year}-Q{expected_quarter}"
        )

    raw["eph_agglomerate_id"] = normalize_agglomerate(raw["AGLOMERADO"])
    if raw["eph_agglomerate_id"].isna().any():
        raise DomainCheckError("raw_eph_agglomerate_missing")
    raw["PONDERA"] = pd.to_numeric(raw["PONDERA"], errors="raise")
    if (
        raw["PONDERA"].isna().any()
        or (~np.isfinite(raw["PONDERA"])).any()
        or (raw["PONDERA"] <= 0).any()
    ):
        raise DomainCheckError("raw_eph_pondera_invalid")

    joined = semantic.merge(
        raw[
            [
                "row_id",
                "source_household_id",
                "eph_agglomerate_id",
                "PONDERA",
            ]
        ],
        on="row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise DomainCheckError("semantic_eph_rows_missing_from_raw_parent")
    joined = joined.drop(columns=["_merge"])
    if not joined["household_id"].astype(str).equals(
        joined["source_household_id"].astype(str)
    ):
        raise DomainCheckError("semantic_eph_household_identity_mismatch")
    return joined.drop(columns=["source_household_id"])


def load_census_geography_handoff(
    root: Path,
    expected_sample_release: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    if manifest.get("contract") != G2_CONTRACT:
        raise DomainCheckError("unexpected_census_geography_handoff_contract")
    sample_parent = manifest.get("sample_parent") or {}
    if sample_parent.get("release_id") != expected_sample_release:
        raise DomainCheckError("census_geography_sample_parent_mismatch")
    if int(sample_parent.get("frame_vintage", -1)) != 2010:
        raise DomainCheckError("current_g2_handoff_requires_cpv2010")

    geography = pd.read_parquet(root / "household_geography.parquet")
    required = {
        "sample_household_id",
        "eph_agglomerate_id",
        "mapped_to_eph_frame",
        "outside_eph_frame",
    }
    missing = sorted(required - set(geography.columns))
    if missing:
        raise DomainCheckError(f"census_geography_columns_missing:{missing}")
    if geography["sample_household_id"].astype(str).duplicated().any():
        raise DomainCheckError("census_geography_household_identity_not_unique")
    if "design_inverse_probability_weight" in geography.columns:
        # Its presence is expected, but this diagnostic intentionally never uses it.
        pass
    return geography, manifest


def attach_census_geography(
    semantic: pd.DataFrame,
    geography: pd.DataFrame,
) -> pd.DataFrame:
    geo = geography[
        [
            "sample_household_id",
            "eph_agglomerate_id",
            "mapped_to_eph_frame",
            "outside_eph_frame",
        ]
    ].copy()
    geo["sample_household_id"] = geo["sample_household_id"].astype(str)
    geo["eph_agglomerate_id"] = normalize_agglomerate(geo["eph_agglomerate_id"])
    joined = semantic.merge(
        geo,
        left_on=semantic["household_id"].astype(str),
        right_on="sample_household_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise DomainCheckError("semantic_census_household_missing_from_geography_handoff")
    joined = joined.drop(columns=["_merge", "key_0", "sample_household_id"], errors="ignore")
    mapped = joined["mapped_to_eph_frame"].astype(bool)
    if joined.loc[mapped, "eph_agglomerate_id"].isna().any():
        raise DomainCheckError("mapped_census_person_missing_agglomerate")
    if joined.loc[~mapped, "eph_agglomerate_id"].notna().any():
        raise DomainCheckError("outside_census_person_has_agglomerate")
    return joined


def feature_tiers(
    manifest: dict[str, Any],
    *,
    exclude_fields: Iterable[str],
) -> dict[str, list[str]]:
    handoff = manifest.get("consumer_handoff") or {}
    roles = handoff.get("temporal_roles") or {}
    p1r = list(manifest.get("p1_r_fields") or [])
    excluded = set(exclude_fields)
    available = [field for field in p1r if field not in excluded]
    unknown_roles = sorted({roles.get(field) for field in available} - set(ROLE_ORDER))
    if unknown_roles:
        raise DomainCheckError(f"unsupported_temporal_roles:{unknown_roles}")

    tiers = {
        "S": [
            field
            for field in available
            if roles.get(field) == "stable/shared"
        ],
        "S+T": [
            field
            for field in available
            if roles.get(field) in {"stable/shared", "target-period-state"}
        ],
        "S+T+R": [
            field
            for field in available
            if roles.get(field)
            in {"stable/shared", "target-period-state", "research-only"}
        ],
    }
    if not tiers["S"]:
        raise DomainCheckError("stable_shared_tier_is_empty")
    return tiers


def _encode_features(
    frame: pd.DataFrame,
    features: list[str],
    numeric_fields: set[str],
) -> tuple[np.ndarray, list[int]]:
    columns: list[np.ndarray] = []
    categorical_indices: list[int] = []
    for index, field in enumerate(features):
        series = frame[field]
        if field in numeric_fields:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        else:
            categorical_indices.append(index)
            missing = series.isna().to_numpy()
            codes, uniques = pd.factorize(series.astype("string"), sort=True)
            if len(uniques) >= 255:
                raise DomainCheckError(f"categorical_cardinality_too_high:{field}")
            values = codes.astype(float)
            values[missing | (codes < 0)] = np.nan
        columns.append(values)
    return np.column_stack(columns), categorical_indices


def _balanced_source_weights(y: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(y), dtype=float)
    for value in (0, 1):
        mask = y == value
        count = int(mask.sum())
        if count == 0:
            raise DomainCheckError("source_balance_requires_both_sources")
        weights[mask] = len(y) / (2.0 * count)
    return weights


def bounded_household_sample(
    frame: pd.DataFrame,
    *,
    max_persons: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_persons is None or max_persons <= 0 or len(frame) <= max_persons:
        return frame.copy()
    counts = (
        frame.groupby("household_id", sort=False)
        .size()
        .rename("persons")
        .reset_index()
    )
    counts = counts.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    counts["cumulative"] = counts["persons"].cumsum()
    selected = counts.loc[counts["cumulative"] <= max_persons, "household_id"].astype(str)
    if selected.empty:
        selected = counts.head(1)["household_id"].astype(str)
    return frame.loc[frame["household_id"].astype(str).isin(set(selected))].copy()


def cross_fitted_domain_metrics(
    eph: pd.DataFrame,
    census: pd.DataFrame,
    *,
    features: list[str],
    numeric_fields: set[str],
    requested_folds: int,
    random_state: int,
    max_persons_per_source: int | None,
) -> dict[str, Any]:
    eph = bounded_household_sample(
        eph,
        max_persons=max_persons_per_source,
        random_state=random_state,
    )
    census = bounded_household_sample(
        census,
        max_persons=max_persons_per_source,
        random_state=random_state + 10_000,
    )
    left = eph.loc[:, ["household_id", *features]].copy()
    left["source"] = 0
    left["group_id"] = "EPH:" + left["household_id"].astype(str)
    right = census.loc[:, ["household_id", *features]].copy()
    right["source"] = 1
    right["group_id"] = "CENSUS:" + right["household_id"].astype(str)
    mixed = pd.concat([left, right], ignore_index=True, sort=False)

    source_groups = mixed.groupby("source")["group_id"].nunique()
    if set(source_groups.index) != {0, 1}:
        raise DomainCheckError("domain_classifier_requires_both_sources")
    n_splits = min(requested_folds, int(source_groups.min()))
    if n_splits < 2:
        raise DomainCheckError("domain_classifier_requires_two_households_per_source")

    x, categorical_indices = _encode_features(mixed, features, numeric_fields)
    y = mixed["source"].to_numpy(dtype=int)
    groups = mixed["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(mixed), np.nan, dtype=float)
    fold_sizes: list[dict[str, int]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups)):
        if len(np.unique(y[test_idx])) != 2:
            raise DomainCheckError(f"domain_fold_missing_source:{fold}")
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=80,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=random_state + fold,
            early_stopping=False,
            categorical_features=categorical_indices or None,
        )
        train_weights = _balanced_source_weights(y[train_idx])
        model.fit(x[train_idx], y[train_idx], sample_weight=train_weights)
        class_index = list(model.classes_).index(1)
        oof[test_idx] = model.predict_proba(x[test_idx])[:, class_index]
        fold_sizes.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
        )
    if np.isnan(oof).any():
        raise DomainCheckError("domain_oof_predictions_incomplete")

    eval_weights = _balanced_source_weights(y)
    brier = float(np.average((oof - y) ** 2, weights=eval_weights))
    return {
        "n_splits": n_splits,
        "classifier_eph_persons": int((y == 0).sum()),
        "classifier_census_persons": int((y == 1).sum()),
        "auc_oof": float(roc_auc_score(y, oof, sample_weight=eval_weights)),
        "log_loss_oof": float(
            log_loss(y, oof, sample_weight=eval_weights, labels=[0, 1])
        ),
        "brier_oof": brier,
        "mean_domain_probability_eph": float(oof[y == 0].mean()),
        "mean_domain_probability_census": float(oof[y == 1].mean()),
        "fold_sizes": fold_sizes,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = q * cumulative[-1]
    return float(values[np.searchsorted(cumulative, cutoff, side="left")])


def marginal_metric(
    eph: pd.Series,
    census: pd.Series,
    *,
    eph_weights: np.ndarray,
    census_weights: np.ndarray,
    numeric: bool,
) -> dict[str, Any]:
    if numeric:
        e = pd.to_numeric(eph, errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(census, errors="coerce").to_numpy(dtype=float)
        emask = np.isfinite(e)
        cmask = np.isfinite(c)
        e, ew = e[emask], eph_weights[emask]
        c, cw = c[cmask], census_weights[cmask]
        if len(e) == 0 or len(c) == 0:
            return {"kind": "numeric", "status": "no_common_nonmissing_surface"}
        emean = _weighted_mean(e, ew)
        cmean = _weighted_mean(c, cw)
        pooled_sd = math.sqrt(
            max((_weighted_variance(e, ew) + _weighted_variance(c, cw)) / 2.0, 0.0)
        )
        smd = (cmean - emean) / pooled_sd if pooled_sd > 0 else math.nan
        emin, emax = float(np.min(e)), float(np.max(e))
        outside = float(
            np.average(((c < emin) | (c > emax)).astype(float), weights=cw)
        )
        return {
            "kind": "numeric",
            "status": "ok",
            "eph_mean": emean,
            "census_mean": cmean,
            "standardized_mean_difference": float(smd),
            "eph_median": _weighted_quantile(e, ew, 0.5),
            "census_median": _weighted_quantile(c, cw, 0.5),
            "census_outside_eph_support": outside,
        }

    e = eph.astype("string")
    c = census.astype("string")
    emask = e.notna().to_numpy()
    cmask = c.notna().to_numpy()
    e, ew = e.loc[emask], eph_weights[emask]
    c, cw = c.loc[cmask], census_weights[cmask]
    if len(e) == 0 or len(c) == 0:
        return {"kind": "categorical", "status": "no_common_nonmissing_surface"}

    def distribution(values: pd.Series, weights: np.ndarray) -> dict[str, float]:
        frame = pd.DataFrame({"value": values.astype(str).to_numpy(), "weight": weights})
        totals = frame.groupby("value", sort=True)["weight"].sum()
        total = float(totals.sum())
        return {str(key): float(value / total) for key, value in totals.items()}

    ep = distribution(e, ew)
    cp = distribution(c, cw)
    levels = sorted(set(ep) | set(cp))
    tv = 0.5 * sum(abs(ep.get(level, 0.0) - cp.get(level, 0.0)) for level in levels)
    unseen = sum(cp.get(level, 0.0) for level in levels if level not in ep)
    return {
        "kind": "categorical",
        "status": "ok",
        "total_variation_distance": float(tv),
        "census_unseen_mass": float(unseen),
        "eph_levels": len(ep),
        "census_levels": len(cp),
    }


def _domain_subset(
    frame: pd.DataFrame,
    domain: str,
    *,
    census: bool,
) -> pd.DataFrame:
    if domain == "EPH_TOTAL":
        if census:
            return frame.loc[frame["mapped_to_eph_frame"].astype(bool)].copy()
        return frame.copy()
    return frame.loc[frame["eph_agglomerate_id"].astype("string") == domain].copy()


def load_eligible_row_ids(path: Path) -> set[str]:
    path = Path(path).expanduser().resolve()
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "row_id" not in frame.columns:
        raise DomainCheckError("eph_eligible_rows_missing_row_id")
    ids = frame["row_id"].astype(str)
    if ids.duplicated().any():
        raise DomainCheckError("eph_eligible_rows_not_unique")
    return set(ids)


def build_inventory(
    eph_support: pd.DataFrame,
    eph_population: pd.DataFrame,
    census: pd.DataFrame,
) -> pd.DataFrame:
    mapped_census = census.loc[census["mapped_to_eph_frame"].astype(bool)]
    domains = sorted(
        set(eph_population["eph_agglomerate_id"].dropna().astype(str))
        | set(mapped_census["eph_agglomerate_id"].dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []
    for domain in ["EPH_TOTAL", *domains]:
        e_support = _domain_subset(eph_support, domain, census=False)
        e_population = _domain_subset(eph_population, domain, census=False)
        c = _domain_subset(census, domain, census=True)
        rows.append(
            {
                "domain": domain,
                "eph_support_persons": int(len(e_support)),
                "eph_support_households": int(e_support["household_id"].nunique()),
                "eph_population_persons": int(len(e_population)),
                "eph_population_households": int(
                    e_population["household_id"].nunique()
                ),
                "eph_pondera_mass": float(e_population["PONDERA"].sum()),
                "census_persons": int(len(c)),
                "census_households": int(c["household_id"].nunique()),
            }
        )
    rows.append(
        {
            "domain": "OUTSIDE_EPH_FRAME",
            "eph_support_persons": 0,
            "eph_support_households": 0,
            "eph_population_persons": 0,
            "eph_population_households": 0,
            "eph_pondera_mass": 0.0,
            "census_persons": int((~census["mapped_to_eph_frame"].astype(bool)).sum()),
            "census_households": int(
                census.loc[~census["mapped_to_eph_frame"].astype(bool), "household_id"]
                .nunique()
            ),
        }
    )
    return pd.DataFrame(rows)


def run(
    *,
    semantic_plane: Path,
    eph_individual: Path,
    census_geography_handoff: Path,
    output: Path,
    eph_eligible_rows: Path | None,
    numeric_fields: set[str],
    exclude_fields: set[str],
    min_source_persons: int,
    folds: int,
    random_state: int,
    max_classifier_persons_per_source: int | None,
) -> dict[str, Any]:
    semantic_plane = Path(semantic_plane).expanduser().resolve()
    eph_individual = Path(eph_individual).expanduser().resolve()
    census_geography_handoff = Path(census_geography_handoff).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise DomainCheckError("output_directory_must_be_empty")
    output.mkdir(parents=True, exist_ok=True)

    eph_semantic, census_semantic, semantic_manifest = load_semantic_plane(semantic_plane)
    clocks = dict(semantic_manifest.get("clocks") or {})
    parents = dict(semantic_manifest.get("parents") or {})
    eph_population = attach_eph_metadata(eph_semantic, eph_individual, clocks)
    if eph_eligible_rows is None:
        eph_support = eph_population.copy()
        eligibility = {
            "mode": "all_semantic_rows",
            "path": None,
            "sha256": None,
            "rows": int(len(eph_support)),
        }
    else:
        eligible_path = Path(eph_eligible_rows).expanduser().resolve()
        eligible_ids = load_eligible_row_ids(eligible_path)
        semantic_ids = set(eph_population["row_id"].astype(str))
        extra = eligible_ids - semantic_ids
        if extra:
            raise DomainCheckError(
                f"eph_eligible_rows_outside_semantic_plane:{len(extra)}"
            )
        eph_support = eph_population.loc[
            eph_population["row_id"].astype(str).isin(eligible_ids)
        ].copy()
        if len(eph_support) != len(eligible_ids):
            raise DomainCheckError("eph_eligible_rows_coverage_mismatch")
        eligibility = {
            "mode": "explicit_row_id_surface",
            "path": str(eligible_path),
            "sha256": sha256(eligible_path),
            "rows": int(len(eph_support)),
        }
    geography, geography_manifest = load_census_geography_handoff(
        census_geography_handoff,
        expected_sample_release=str(parents.get("census_sample_release_id") or ""),
    )
    census = attach_census_geography(census_semantic, geography)
    tiers = feature_tiers(semantic_manifest, exclude_fields=exclude_fields)

    all_fields = list(dict.fromkeys(tiers["S+T+R"]))
    missing_features = sorted(
        (set(all_fields) - set(eph_population.columns))
        | (set(all_fields) - set(census.columns))
    )
    if missing_features:
        raise DomainCheckError(f"diagnostic_features_missing:{missing_features}")

    inventory = build_inventory(eph_support, eph_population, census)
    inventory_path = output / "domain_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    classifier_rows: list[dict[str, Any]] = []
    eligible_domains = [
        str(row.domain)
        for row in inventory.itertuples()
        if row.domain != "OUTSIDE_EPH_FRAME"
        and row.eph_support_persons >= min_source_persons
        and row.census_persons >= min_source_persons
    ]
    for domain in eligible_domains:
        e = _domain_subset(eph_support, domain, census=False)
        c = _domain_subset(census, domain, census=True)
        for tier, features in tiers.items():
            metrics = cross_fitted_domain_metrics(
                e,
                c,
                features=features,
                numeric_fields=numeric_fields,
                requested_folds=folds,
                random_state=random_state,
                max_persons_per_source=max_classifier_persons_per_source,
            )
            classifier_rows.append(
                {
                    "domain": domain,
                    "tier": tier,
                    "features": ",".join(features),
                    "feature_count": len(features),
                    "eph_persons": len(e),
                    "census_persons": len(c),
                    "eph_households": e["household_id"].nunique(),
                    "census_households": c["household_id"].nunique(),
                    **{key: value for key, value in metrics.items() if key != "fold_sizes"},
                }
            )
    classifier = pd.DataFrame(classifier_rows)
    classifier_path = output / "domain_classifier_oof.csv"
    classifier.to_csv(classifier_path, index=False)

    roles = (semantic_manifest.get("consumer_handoff") or {}).get("temporal_roles") or {}
    marginal_rows: list[dict[str, Any]] = []
    for domain in eligible_domains:
        e_support = _domain_subset(eph_support, domain, census=False)
        e_population = _domain_subset(eph_population, domain, census=False)
        c = _domain_subset(census, domain, census=True)
        for lens in ("model_support", "population_composition"):
            if lens == "model_support":
                e = e_support
                ew = np.ones(len(e), dtype=float)
            else:
                e = e_population
                ew = e["PONDERA"].to_numpy(dtype=float)
            # Census selected persons are the target-year sample surface.
            # Design inverse probabilities are intentionally not analysis weights.
            cw = np.ones(len(c), dtype=float)
            for field in all_fields:
                metric = marginal_metric(
                    e[field],
                    c[field],
                    eph_weights=ew,
                    census_weights=cw,
                    numeric=field in numeric_fields,
                )
                role = roles.get(field)
                marginal_rows.append(
                    {
                        "domain": domain,
                        "lens": lens,
                        "concept": field,
                        "temporal_role": role,
                        "first_tier": ROLE_ORDER.get(role),
                        "eph_weight_sum": float(ew.sum()),
                        "census_weight_sum": float(cw.sum()),
                        **metric,
                    }
                )
    marginals = pd.DataFrame(marginal_rows)
    marginals_path = output / "marginal_comparison.csv"
    marginals.to_csv(marginals_path, index=False)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "diagnostic_only",
        "statistical_transport_authorized": False,
        "no_reweighting_or_raking_applied": True,
        "parents": {
            "semantic_plane_release_id": semantic_manifest.get("release_id"),
            "semantic_plane_manifest_sha256": sha256(
                semantic_plane / "feature_plane_manifest.json"
            ),
            "eph_release_id": parents.get("eph_release_id"),
            "census_frame_release_id": parents.get("census_frame_release_id"),
            "census_sample_release_id": parents.get("census_sample_release_id"),
            "census_geography_handoff_manifest_sha256": sha256(
                census_geography_handoff / "manifest.json"
            ),
            "eph_individual_sha256": sha256(eph_individual),
        },
        "clocks": clocks,
        "geography": {
            "eph_field": "native AGLOMERADO normalized to zero-preserving two digits",
            "census_field": "eph_agglomerate_id from governed G2 household handoff",
            "census_geography_parent": geography_manifest.get("geography_parent"),
            "census_outside_eph_frame_preserved": True,
            "frame_equivalence_assumed": False,
        },
        "feature_tiers": tiers,
        "excluded_fields": sorted(exclude_fields),
        "numeric_fields": sorted(numeric_fields & set(all_fields)),
        "eph_model_support_eligibility": eligibility,
        "lenses": {
            "model_support": {
                "eph_rows": "explicit eligible row_id surface when supplied; otherwise all semantic rows",
                "eph_weight": "unit",
                "census_weight": "unit",
            },
            "population_composition": {
                "eph_weight": "PONDERA",
                "census_weight": "unit selected persons",
                "census_design_inverse_probability_weight_used": False,
            },
        },
        "classifier": {
            "method": "HistGradientBoostingClassifier",
            "cross_fitting": "StratifiedGroupKFold with source-prefixed household groups",
            "fit_weighting": "source-balanced only",
            "evaluation_weighting": "source-balanced only",
            "requested_folds": folds,
            "random_state": random_state,
            "max_persons_per_source_domain": max_classifier_persons_per_source,
            "minimum_persons_per_source_domain": min_source_persons,
        },
        "artifacts": {
            "domain_inventory.csv": sha256(inventory_path),
            "domain_classifier_oof.csv": sha256(classifier_path),
            "marginal_comparison.csv": sha256(marginals_path),
        },
        "legacy_q8_relation": (
            "Supersedes the old global in-sample domain classifier as the primary "
            "domain-separation diagnostic; Q8 remains historical regression evidence."
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--semantic-plane", type=Path, required=True)
    out.add_argument("--eph-individual", type=Path, required=True)
    out.add_argument("--census-geography-handoff", type=Path, required=True)
    out.add_argument("--output", type=Path, required=True)
    out.add_argument(
        "--eph-eligible-rows",
        type=Path,
        help=(
            "Optional CSV/Parquet with unique row_id values defining the exact EPH "
            "training/support universe. Population-composition diagnostics still "
            "use the full semantic EPH plane with PONDERA."
        ),
    )
    out.add_argument(
        "--numeric-fields",
        default=",".join(DEFAULT_NUMERIC_FIELDS),
        help="Comma-separated canonical numeric concepts.",
    )
    out.add_argument(
        "--exclude-fields",
        default=",".join(DEFAULT_EXCLUDE_FIELDS),
        help=(
            "Comma-separated concepts excluded from the general transport diagnostic. "
            "CONDACT is excluded by default because it is a labor target in current work."
        ),
    )
    out.add_argument("--min-source-persons", type=int, default=100)
    out.add_argument("--folds", type=int, default=5)
    out.add_argument("--random-state", type=int, default=42)
    out.add_argument(
        "--max-classifier-persons-per-source",
        type=int,
        default=50000,
        help=(
            "Household-preserving deterministic cap per source/domain for the "
            "cross-fitted classifier only. Marginal diagnostics always use all rows."
        ),
    )
    return out


def main() -> int:
    args = parser().parse_args()
    manifest = run(
        semantic_plane=args.semantic_plane,
        eph_individual=args.eph_individual,
        census_geography_handoff=args.census_geography_handoff,
        output=args.output,
        eph_eligible_rows=args.eph_eligible_rows,
        numeric_fields=set(parse_csv_list(args.numeric_fields)),
        exclude_fields=set(parse_csv_list(args.exclude_fields)),
        min_source_persons=args.min_source_persons,
        folds=args.folds,
        random_state=args.random_state,
        max_classifier_persons_per_source=args.max_classifier_persons_per_source,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
