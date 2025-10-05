# src/encuestador/config.py
from __future__ import annotations
import os, yaml
from pathlib import Path


def _load_yaml(path: str | Path) -> dict:
    """Load YAML and expand env vars / user paths."""
    path = Path(os.path.expandvars(os.path.expanduser(path)))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_paths_config(path: str | Path) -> dict:
    """
    Load paths.yaml.
    Example:
      microdatos_repo: ~/repos/microdatos-EPH-INDEC
      aligner_repo: ~/repos/eph-censo-aligner
      data_root: ./data
    """
    cfg = _load_yaml(path)
    resolved = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            v = os.path.expandvars(os.path.expanduser(v))
        resolved[k] = v
    return resolved


def load_experiment_config(path: str | Path) -> dict:
    """
    Load experiment.yaml with minimal validation.
    Expected fields:
      seed, target, features, models, cv
    """
    cfg = _load_yaml(path)
    required = ["target", "models"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key '{k}' in {path}")

    # ensure structure exists
    cfg.setdefault("seed", 42)
    cfg.setdefault("features", {})
    cfg.setdefault("cv", {"kind": "KFold", "n_splits": 5, "shuffle": True})

    # normalize feature lists
    feats = cfg.get("features", {})
    if isinstance(feats, list):
        cfg["features"] = {"include": feats}

    # normalize models
    models = cfg["models"]
    if not isinstance(models, list):
        raise TypeError("models must be a list of model specs")
    for m in models:
        if "name" not in m or "estimator" not in m:
            raise ValueError(f"Each model must have 'name' and 'estimator': {m}")

    return cfg


def load_eval_config(path: str | Path) -> dict:
    """
    Load eval.yaml.
    Expected fields: test_window, metrics (optional)
    """
    cfg = _load_yaml(path)
    cfg.setdefault("metrics", ["roc_auc", "f1_macro"])
    cfg.setdefault("test_window", None)
    cfg.setdefault("topk_thresholds", [0.5])
    cfg.setdefault("subgroups", [])
    return cfg
