# eph_align.py
from __future__ import annotations
from typing import Optional
import pandas as pd


from utils import filter_rows, rename_columns, collapse_family_one_of, collapse_family_multi_any, apply_recode, clip_columns, join_region_by_dpto, apply_overrides, validate_required, conditional_set, cast_numeric


# ------------------------------------
# 1) Crosswalk (subset EPH → Censo)
# ------------------------------------

NAMES_CENSO = [
    'IX_TOT','P02','P03','CONDACT','AGLOMERADO',
    'V01','H05','H06','H07','H08','H09','H10','H11','H12','H16','H15','PROP','H14','H13',
    'P07','P08','P09','P10','P05'
]

NAMES_EPH = [
    'IX_TOT','CH04','CH06','CONDACT','AGLOMERADO',
    'IV1','IV3','IV4','IV5','IV6','IV7','IV8','IV10','IV11','II1','II2','II7','II8','II9',
    'CH09','CH10','CH12','CH13','CH15'
]

CROSSWALK_EPH_TO_CENSO = dict(zip(NAMES_EPH, NAMES_CENSO))

COL_MON = ['P21', 'P47T', 'PP08D1', 'TOT_P12', 'T_VI', 'V12_M', 'V2_M', 'V3_M', 'V5_M']

# ------------------------------------
# 2) Reglas específicas (recode/clip/filters)
# ------------------------------------

RECODE_HOGAR = {
    'IV10': {'map': {1: 1, 2: 2, 3: 2, 0: 0, 9: 9}},  # IV10 → H11 (tras rename)
    'II9':  {'map': {1: 1, 2: 2, 3: 2, 4: 4, 0: 0}},  # II9 → H13
    'II7':  {'map': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6, 0: 0}},  # II7 → PROP
    # opcionales extra (si querés replicar la “mímica” Censo):
    'V01': {'map': {1:1, 2:6, 3:6, 4:2, 5:3, 6:4, 7:5, 8:6}},
    'H06': {'map': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:9}},
    'H09': {'map': {1:1, 2:2, 3:3, 4:4, 5:4, 6:4}},
    'H14': {'map': {1:1, 2:4, 3:2, 4:2, 5:4, 6:3, 7:4, 8:9}},
    'H13': {'map': {1:1, 2:2, 4:0}},
}

CLIP_HOGAR = {'IX_TOT': {'min': 0, 'max': 8}}

FILTERS_HOGAR = [('IV1', '!=', 9)]  # descartar missing de IV1 antes del rename final

# Individual
RECODE_INDIV = {
    'CH15': {'map': {1:1, 2:1, 3:1, 4:2, 5:2, 9:0}},
    'CH09': {'map': {1:1, 2:2, 0:2, 3:2}},
    # En Censo: jardín/especial → no preguntan “terminó”
    # (lo resolvemos más abajo seteando CH13=0 según CH12)
}


# ------------------------------------
# 3) Orquestadores
# ------------------------------------

def harmonize_hogar(
    df: pd.DataFrame,
    region_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Pipeline HOGAR:
    rename → families.collapse → recode → clip/filter/when → join+overrides → types → validate
    """
    out = df.copy()

    # A) filter (sobre el header original)
    if 'IV1' in out.columns:
        out = filter_rows(out, FILTERS_HOGAR)

    # B) rename al subset Censo
    out = rename_columns(out, CROSSWALK_EPH_TO_CENSO)

    # C) families.collapse (si existen variantes split en hogar)
    #   Conviene colapsar antes de recodificar valores.
    #   One-of:
    out = collapse_family_one_of(out, "V21", "V21")
    out = collapse_family_one_of(out, "V22", "V22")
    out = collapse_family_one_of(out, "V5",  "V5")
    out = collapse_family_one_of(out, "V11", "V11")
    #   Multi:
    out = collapse_family_multi_any(out, "V2", "V2")

    # D) recodes y clips
    out = apply_recode(out, RECODE_HOGAR)
    out = clip_columns(out, CLIP_HOGAR)

    # E) join + overrides (Region por DPTO; ajustes AGLO 33/93)
    if region_df is not None:
        out = join_region_by_dpto(out, region_df, on="DPTO", take="Region", into="Region")
        out = apply_overrides(out, [
            ({"AGLOMERADO": 33, "Region": "Pampeana"}, {"Region": "Gran Buenos Aires"}),
            ({"AGLOMERADO": 93, "Region": "Pampeana"}, {"Region": "Patagónica"}),
        ])

    # F) types: columnas monetarias que uses desde hogar (si aplica)
    # (en tu subset monetario están mayormente en individual; dejamos hook)
    # out = cast_numeric(out, SOME_HOME_MON_COLS)

    # G) validate
    # validate_required(out, ['IX_TOT', 'P02', 'P03', 'CONDACT', 'AGLOMERADO'], where="hogar")

    return out


def harmonize_individual(
    df: pd.DataFrame,
    region_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Pipeline INDIVIDUAL:
    rename → families.collapse → recode → clip/filter/when → join+overrides → types → validate
    """
    out = df.copy()

    # A) rename (ESTADO → CONDACT si existe)
    if 'ESTADO' in out.columns and 'CONDACT' not in out.columns:
        out = out.rename(columns={'ESTADO': 'CONDACT'})

    # B) families.collapse (splits _NN_M → _M)
    #   2024Q4+ pueden venir V*_01_M, etc. Colapsamos a V*_M (multi-any).
    for base in ["V2", "V5", "V11", "V21", "V22"]:
        out = collapse_family_multi_any(out, f"{base}", f"{base}_M")  # si no existen, no hace nada
        out = collapse_family_multi_any(out, f"{base}.*_M", f"{base}_M")  # defensa por si ya traen _M

    # C) recodes
    out = apply_recode(out, RECODE_INDIV)

    # D) clip, filter, when
    if 'CH06' in out.columns:
        out = clip_columns(out, {'CH06': {'min': 0}})
        out = conditional_set(out, ('CH06', '<', 14), {'CONDACT': 0})

    #   Jardín/esp. ⇒ CH13 = 0
    if 'CH12' in out.columns and 'CH13' in out.columns:
        out['CH12'] = out['CH12'].replace(99, 0)
        out.loc[out['CH12'].isin([0, 1, 9]), 'CH13'] = 0

    # E) join + overrides (si aplica al nivel persona)
    if region_df is not None and 'DPTO' in out.columns:
        out = join_region_by_dpto(out, region_df, on="DPTO", take="Region", into="Region")
        out = apply_overrides(out, [
            ({"AGLOMERADO": 33, "Region": "Pampeana"}, {"Region": "Gran Buenos Aires"}),
            ({"AGLOMERADO": 93, "Region": "Pampeana"}, {"Region": "Patagónica"}),
        ])

    # F) types (asegurar numéricos en col_mon)
    out = cast_numeric(out, COL_MON)

    # G) validate (mínimos para tu subset)
    validate_required(out, ['CODUSU', 'CH06', 'CH09', 'CONDACT'], where="individual")

    return out
