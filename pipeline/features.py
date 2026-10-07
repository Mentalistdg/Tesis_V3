"""Etapa 4: construye las features con build_dataset.py de la tesis (parchado)."""
import contextlib
import io
import warnings

import pandas as pd

PANDAS_REQUERIDO = "2.3.3"


def construir_features(raw: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    if pd.__version__ != PANDAS_REQUERIDO:
        raise RuntimeError(f"build_dataset requiere pandas {PANDAS_REQUERIDO} (instalado: {pd.__version__}); "
                           "con otra version las features cambian en silencio")
    from pipeline.tesis.build_dataset import build_dataset

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_dataset(df_raw=raw, columnas_fijas=columnas, conservar_sin_target=True, guardar=False)
    return df[columnas].reset_index(drop=True)
