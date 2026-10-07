"""Carga y valida la configuracion de un activo (series.csv y config.yaml)."""
from pathlib import Path

import pandas as pd
import yaml

TIPOS = {"precio", "volumen", "nivel", "fundamental", "macro"}
AJUSTES = {"none", "split", "default"}
COLUMNAS = ["columna_cruda", "codigo", "ticker", "campo", "ajuste", "tipo", "relleno",
            "fecha_pub", "rezago_pub_dias", "publica_tarde", "notas"]


def cargar_series(dir_activo: Path) -> pd.DataFrame:
    dir_activo = Path(dir_activo)
    s = pd.read_csv(dir_activo / "series.csv", dtype={"fecha_pub": str}, keep_default_na=False,
                    na_values={"rezago_pub_dias": [""]}, encoding="utf-8")
    faltan = [c for c in COLUMNAS if c not in s.columns]
    if faltan:
        raise ValueError(f"series.csv: faltan columnas {faltan}")
    for campo, validos in (("tipo", TIPOS), ("ajuste", AJUSTES)):
        malos = s.loc[~s[campo].isin(validos), "columna_cruda"].tolist()
        if malos:
            raise ValueError(f"series.csv: {campo} invalido en {malos}")
    if s.columna_cruda.duplicated().any():
        raise ValueError(f"series.csv: columnas duplicadas {s.columna_cruda[s.columna_cruda.duplicated()].tolist()}")
    esperadas = list(pd.read_csv(dir_activo / "historia_congelada.csv", nrows=0).columns[1:])
    if list(s.columna_cruda) != esperadas:
        raise ValueError("series.csv: columna_cruda no coincide (orden o contenido) con historia_congelada.csv")
    macro = s.tipo == "macro"
    if not s.loc[macro, "fecha_pub"].isin(["eco", "rezago"]).all() or s.loc[macro, "rezago_pub_dias"].isna().any():
        raise ValueError("series.csv: toda serie macro necesita fecha_pub (eco|rezago) y rezago_pub_dias")
    s["rezago_pub_dias"] = s["rezago_pub_dias"].astype("Int64")
    s["publica_tarde"] = s["publica_tarde"].eq("si")
    return s[COLUMNAS].reset_index(drop=True)


def cargar_config(dir_activo: Path) -> dict:
    with open(Path(dir_activo) / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
