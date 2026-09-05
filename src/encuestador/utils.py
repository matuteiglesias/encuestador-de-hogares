import pandas as pd
import re
import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------
# 0) Utilidades genéricas
# ---------------------------

def rename_columns(df: pd.DataFrame, crosswalk: Dict[str, str]) -> pd.DataFrame:
    """Renombra columnas según crosswalk {origen: destino}; ignora faltantes."""
    missing = [c for c in crosswalk if c not in df.columns]
    if missing:
        # no rompe: renombra lo que esté
        pass
    return df.rename(columns={k: v for k, v in crosswalk.items() if k in df.columns})

def collapse_family_one_of(
    df: pd.DataFrame,
    family_prefix: str,
    into: str,
    include_suffix_pattern: str = r"\d+$",
    prefer_first: bool = False,
) -> pd.DataFrame:
    """
    Colapsa columnas tipo V21_01, V21_02, ... → V21 (one_of).
    Estrategia: toma la columna con el mayor valor (argmax) entre dummies/binaries.
    Si prefer_first=True, devuelve el primer sufijo 'activo'.
    """
    pat = re.compile(rf"^{re.escape(family_prefix)}_(\w+)$")
    members = [c for c in df.columns if pat.match(c)]
    if not members:
        return df

    # Asegurar numéricos (0/1/NaN); si hay strings "0"/"1" las convertimos.
    tmp = df[members].apply(pd.to_numeric, errors="coerce")

    if prefer_first:
        # primer índice distinto de 0 → ese código
        codes = tmp.gt(0).values.argmax(axis=1)  # índice del primero True
        any_true = tmp.gt(0).any(axis=1).values
        chosen = np.where(any_true, [members[i] for i in codes], None)
    else:
        # argmax del valor; si todo es NaN/0 → None
        # Reemplazo NaN por 0 para el argmax; si todo 0 ⇒ no activo
        filled = tmp.fillna(0.0)
        codes = filled.values.argmax(axis=1)
        any_pos = (filled.values.max(axis=1) > 0)
        chosen = np.where(any_pos, [members[i] for i in codes], None)

    # Extraer sufijo como '01', '02', ...
    def suffix_or_none(name):
        m = pat.match(name) if isinstance(name, str) else None
        return m.group(1) if m else None

    df[into] = pd.Series([suffix_or_none(x) for x in chosen], index=df.index)
    return df

def collapse_family_multi_any(
    df: pd.DataFrame,
    family_prefix: str,
    into: str,
) -> pd.DataFrame:
    """
    Colapsa columnas tipo V2_01, V2_02, V2_03 → V2 (multi):
    produce una lista de sufijos activos (o [] si ninguno).
    Si preferís booleano: reemplazá la lista por .any().
    """
    pat = re.compile(rf"^{re.escape(family_prefix)}_(\w+)$")
    members = [c for c in df.columns if pat.match(c)]
    if not members:
        return df

    tmp = df[members].apply(pd.to_numeric, errors="coerce").fillna(0)
    active_mask = tmp.gt(0)

    suffixes = [pat.match(c).group(1) for c in members]
    df[into] = [
        [s for s, ok in zip(suffixes, row) if ok] for row in active_mask.values
    ]
    return df

def apply_recode(df: pd.DataFrame, maps: Dict[str, Dict]) -> pd.DataFrame:
    """
    Aplica recodes por columna: maps = {col: {map: {old:new}}}
    Soporta 'map' y deja valores no mapeados tal cual (incluye NaN).
    """
    for col, spec in maps.items():
        if col in df.columns and "map" in spec:
            df[col] = df[col].map(spec["map"]).where(df[col].notna(), df[col])
    return df

def clip_columns(df: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Clip para numéricos: bounds[col] = {min: a, max: b} (cualquiera opcional)."""
    for col, b in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if "min" in b:
                df[col] = df[col].clip(lower=b["min"])
            if "max" in b:
                df[col] = df[col].clip(upper=b["max"])
    return df

def filter_rows(df: pd.DataFrame, rules: List[Tuple[str, str, object]]) -> pd.DataFrame:
    """
    Reglas simples de filtrado: [(col, op, value)], op in {'==','!=','<','<=','>','>='}
    Las reglas se combinan con AND.
    """
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
    }
    mask = pd.Series(True, index=df.index)
    for col, op, val in rules:
        if col in df.columns:
            mask &= ops[op](df[col], val)
    return df.loc[mask]

def conditional_set(
    df: pd.DataFrame,
    condition: Tuple[str, str, object],
    set_map: Dict[str, object]
) -> pd.DataFrame:
    """Si (col op value) entonces setear columnas a valores."""
    col, op, val = condition
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
    }
    if col in df.columns:
        m = ops[op](df[col], val)
        for k, v in set_map.items():
            df.loc[m, k] = v
    return df

def join_region_by_dpto(
    df: pd.DataFrame,
    region_df: pd.DataFrame,
    on: str = "DPTO",
    take: str = "Region",
    into: str = "Region"
) -> pd.DataFrame:
    """Enriquece región por DPTO; deja existente si no hay join."""
    if on in df.columns and on in region_df.columns and take in region_df.columns:
        add = region_df[[on, take]].drop_duplicates()
        df = df.merge(add, on=on, how="left", suffixes=("", "_JOIN"))
        if into != take:
            df = df.rename(columns={take: into})
    return df

def apply_overrides(df: pd.DataFrame, overrides: List[Tuple[Dict, Dict]]) -> pd.DataFrame:
    """
    overrides: lista de (match_dict, set_dict), e.g.
    ({'AGLOMERADO': 33, 'Region': 'Pampeana'}, {'Region': 'Gran Buenos Aires'})
    """
    for match, setter in overrides:
        m = pd.Series(True, index=df.index)
        for k, v in match.items():
            if k in df.columns:
                m &= (df[k] == v)
            else:
                m &= False
        for k, v in setter.items():
            df.loc[m, k] = v
    return df

def cast_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Convierte a numérico (float) con errors='coerce'."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def validate_required(df: pd.DataFrame, required: Iterable[str], where: str = "") -> None:
    """Valida que existan columnas requeridas; lanza ValueError si falta alguna."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        place = f" en {where}" if where else ""
        raise ValueError(f"Faltan columnas requeridas{place}: {missing}")
