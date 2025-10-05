# src/encuestador/preprocess.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from eph_align import harmonize_hogar, harmonize_individual
from .io import ensure_dir

def load_and_align(hogar_path: Path, indiv_path: Path, region_df: pd.DataFrame):
    df_h = pd.read_csv(hogar_path, delimiter=";", low_memory=False)
    df_i = pd.read_csv(indiv_path, delimiter=";", low_memory=False)

    h_aligned = harmonize_hogar(df_h, region_df=region_df)
    i_aligned = harmonize_individual(df_i, region_df=region_df)

    return h_aligned, i_aligned


def recompute_ranks(df: pd.DataFrame, by: str) -> pd.DataFrame:
    subset = df.loc[(df["CAT_OCUP"] == 3) & (df["P47T"] >= 100)]
    ranks = subset.groupby(["ANO4", by])["P47T"].mean().rank(pct=True).round(3)
    return ranks.reset_index(name=f"{by}_rk")



def export_ranks(train_dir: Path, info_dir: Path, years: list[int]) -> None:
    aglo_list, regs_list = [], []
    for y in years:
        f = train_dir / f"EPHARG_train_{str(y)[2:]}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["ANO4","AGLOMERADO","Region","CAT_OCUP","P47T"])
        aglo_list.append(recompute_ranks(df, "AGLOMERADO"))
        regs_list.append(recompute_ranks(df, "Region"))

    if aglo_list:
        pd.concat(aglo_list, ignore_index=True).to_csv(info_dir/"AGLO_rk.csv", index=False)
    if regs_list:
        out = pd.concat(regs_list, ignore_index=True)
        regiones = {
            'Gran Buenos Aires': 'gran_buenos_aires',
            'Pampeana': 'pampeana',
            'Noroeste': 'noroeste',
            'Noreste': 'noreste',
            'Patagónica': 'patagonia',
            'Cuyo': 'cuyo'
        }
        out["region_"] = out["Region"].map(regiones)
        out.to_csv(info_dir/"Reg_rk.csv", index=False)





def make_splits(df, cv_cfg):
    from sklearn.model_selection import TimeSeriesSplit, GroupKFold
    if cv_cfg["kind"] == "TimeSeriesSplit":
        splitter = TimeSeriesSplit(n_splits=cv_cfg["n_splits"])
        return list(splitter.split(df))
    elif cv_cfg["kind"] == "GroupKFold":
        groups = df[cv_cfg["group_col"]]
        splitter = GroupKFold(n_splits=cv_cfg["n_splits"])
        return list(splitter.split(df, groups=groups))



def build_training_matrix(years: list[int], paths_cfg: dict, region_df: pd.DataFrame, overwrite=False) -> None:
    data_root = Path(paths_cfg["data_root"])
    out_dir = ensure_dir(data_root / "training")

    for y in years:
        out_path = out_dir / f"EPHARG_train_{str(y)[2:]}.csv"
        if out_path.exists() and not overwrite:
            print(f"→ {y}: already exists, skipping")
            continue

        print(f"→ building {y}")
        # read quarterly files (assume downloaded locally)
        hogar_files = sorted((data_root / "hogar").glob(f"*{y}.txt"))
        indiv_files = sorted((data_root / "individual").glob(f"*{y}.txt"))

        dfh = pd.concat([pd.read_csv(f, delimiter=";") for f in hogar_files], ignore_index=True)
        dfi = pd.concat([pd.read_csv(f, delimiter=";") for f in indiv_files], ignore_index=True)

        # alignment from external repo
        h_aligned = harmonize_hogar(dfh, region_df)
        i_aligned = harmonize_individual(dfi, region_df)

        # merge hogar/individual
        eph = h_aligned.merge(
            i_aligned,
            on=["CODUSU", "ANO4", "TRIMESTRE", "AGLOMERADO"],
            how="inner"
        )

        # optional pooled urban area
        sample = eph.sample(frac=0.05, replace=True, random_state=42)
        sample["AGLOMERADO"] = 0
        eph = pd.concat([eph, sample], ignore_index=True)

        # Quarter date + deflation (simplify for now)
        eph["Q"] = eph.ANO4.astype(str) + ":" + (3*eph.TRIMESTRE).astype(str)
        eph["Q"] = pd.to_datetime(eph["Q"], format="%Y:%m") - pd.DateOffset(months=1) + pd.DateOffset(days=14)

        # Example CPI deflator hook (replace with your aligner CPI function)
        # eph[col_mon] = eph[col_mon] / eph.merge(cpi, on="Q")["index"]

        # Derived targets and ranks
        eph["INGRESO"] = (eph["P47T"] > 100).astype(int)
        eph["INGRESO_NLB"] = (eph["T_VI"] > 100).astype(int)
        eph["INGRESO_JUB"] = (eph["V2_M"] > 100).astype(int)
        eph["INGRESO_SBS"] = (eph["V5_M"] > 100).astype(int)

        eph = eph.sort_values("CODUSU").fillna(0)

        # Persist
        eph.to_csv(out_path, index=False)
        print(f"✓ saved {out_path}")
