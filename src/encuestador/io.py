# src/encuestador/io.py
from __future__ import annotations
import os, json, hashlib, yaml
from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_paths(paths_yaml: str) -> dict:
    """
    Load YAML with environment expansion and verify key directories exist.
    Example YAML:
      microdatos_repo: /home/matias/repos/microdatos-EPH-INDEC
      aligner_repo: /home/matias/repos/eph-censo-aligner
      data_root: ./data
    """
    with open(paths_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolved = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            v = os.path.expandvars(os.path.expanduser(v))
        resolved[k] = v
        if k.endswith("_repo") or k.endswith("_dir"):
            p = Path(v)
            if not p.exists():
                raise FileNotFoundError(f"Required path missing: {p}")
    return resolved


def load_training_csv(year: int, root: str | Path = "data/training") -> pd.DataFrame:
    """Read EPH training CSV for a given year."""
    path = Path(root) / f"EPHARG_train_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def sha256_file(path: Path, length: int | None = None) -> str:
    """Return hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:length] if length else digest


def sha256_df(df: pd.DataFrame, length: int | None = None) -> str:
    """Compute hash of DataFrame contents and columns."""
    # Avoid expensive full serialization
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(df.columns).encode())
    digest = h.hexdigest()
    return digest[:length] if length else digest


def write_artifact(path: Path, obj, fmt: str = "json") -> Path:
    """Save object to path in the chosen format (json, yaml, csv, or txt)."""
    path = Path(path)
    ensure_dir(path.parent)

    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    elif fmt == "yaml":
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    elif fmt == "csv":
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("write_artifact(fmt='csv') expects a DataFrame.")
        obj.to_csv(path, index=False)
    else:  # fallback
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(obj))
    return path
