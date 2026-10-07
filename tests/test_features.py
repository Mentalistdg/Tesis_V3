import json

import numpy as np
import pandas as pd
import pytest

from conftest import RAIZ, SPX
from pipeline.features import construir_features

COLUMNAS = json.loads((SPX / "columnas_dataset.json").read_text(encoding="utf-8"))
HIST = pd.read_csv(SPX / "historia_congelada.csv")


def _str_fechas(df):
    return df.assign(date=df["date"].astype(str).str[:10]).reset_index(drop=True)


@pytest.mark.lento
def test_reproduce_dataset_de_entrenamiento():
    df = construir_features(HIST, COLUMNAS)
    ref = pd.read_csv(RAIZ / "data" / "bloomberg_triple_screen_core.csv", low_memory=False)
    assert list(df.columns) == COLUMNAS
    assert len(df) == 6508 and str(df.iloc[-1]["date"])[:10] == "2025-12-12"
    assert np.isnan(df.iloc[-1]["market_forward_excess_returns"])
    pd.testing.assert_frame_equal(_str_fechas(df.iloc[:6507]), _str_fechas(ref), check_exact=False, rtol=1e-12)


@pytest.mark.lento
@pytest.mark.parametrize("corte", ["2015-06-30", "2023-12-29"])
def test_causalidad_agregar_filas_no_cambia_historia(corte):
    completo = _str_fechas(construir_features(HIST, COLUMNAS))
    parcial = _str_fechas(construir_features(HIST[HIST.date <= corte], COLUMNAS))
    n = len(parcial) - 1                                    # la ultima fila cambia al conocerse t+1
    a = parcial.iloc[:n].drop(columns="date_id")
    b = completo[completo.date.isin(a.date)].reset_index(drop=True).drop(columns="date_id")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)


def test_falta_columna_lanza_error():
    with pytest.raises(KeyError):
        construir_features(HIST.drop(columns=["VIX Index"]).iloc[-400:], COLUMNAS)


def test_rechaza_pandas_distinto(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "3.0.1")
    with pytest.raises(RuntimeError, match="pandas 2.3.3"):
        construir_features(pd.DataFrame(), [])
