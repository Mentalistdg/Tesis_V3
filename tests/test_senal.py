import json

import numpy as np
import pandas as pd
import pytest

from conftest import SPX
from pipeline.configuracion import cargar_config
from pipeline.features import construir_features
from pipeline.senal import cargar_modelo, entrenar_meta, generar_senales, metricas, predecir

CFG = cargar_config(SPX)
FIN = CFG["fin_entrenamiento"]
COLUMNAS = json.loads((SPX / "columnas_dataset.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entorno():
    ds = construir_features(pd.read_csv(SPX / "historia_congelada.csv"), COLUMNAS)
    m = cargar_modelo(SPX / "modelo", ds, FIN)
    return ds, m, entrenar_meta(m, ds, FIN)


@pytest.mark.lento
def test_senales_reproducen_backtest_tesis(entorno):
    ds, m, meta = entorno
    s = generar_senales(m, meta, ds, FIN, realista=False)
    mt = metricas(s, CFG, hasta="2025-12-10")
    assert mt["total_return"] == pytest.approx(3.9538, abs=1e-4)
    assert mt["sharpe"] == pytest.approx(1.3192, abs=1e-4)


@pytest.mark.lento
def test_modo_realista(entorno):
    ds, m, meta = entorno
    mt = metricas(generar_senales(m, meta, ds, FIN, realista=True), CFG, hasta="2025-12-10")
    assert mt["total_return"] == pytest.approx(4.125, abs=2e-3)


@pytest.mark.lento
def test_prediccion_coincide_con_la_tesis(entorno):
    ds, m, _ = entorno
    p = predecir(m, ds)
    assert p.loc["2025-12-11"] == pytest.approx(-0.018574, abs=1e-6)


@pytest.mark.lento
def test_senal_del_dia_habil_siguiente(entorno):
    ds, m, meta = entorno
    s = generar_senales(m, meta, ds, FIN)
    ultima = s.iloc[-1]
    assert ultima["date"] == "2025-12-15" and ultima["posicion"] in (0, 1, 3)
    assert np.isnan(ultima["forward_returns"]) and s.iloc[0]["date"] == "2020-10-07"


@pytest.mark.lento
def test_corte_por_fecha_no_por_porcentaje(entorno):
    ds, m, _ = entorno
    corto = predecir(m, ds.iloc[:-300])
    largo = predecir(m, ds)
    comunes = corto.index[:-1]                    # la ultima de "corto" es su dia siguiente
    assert np.allclose(corto.loc[comunes].values, largo.loc[comunes].values)


def test_metricas_entradas_y_retraso():
    s = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=7).strftime("%Y-%m-%d"),
                      "posicion": [0, 1, 1, 0, 3, 3, 0], "forward_returns": [0.01] * 7,
                      "risk_free_rate": [0.0] * 7})
    mt = metricas(s, CFG)
    assert mt["entradas"] == 2 and mt["dias"] == 7
    assert metricas(s, CFG, retraso=1)["entradas"] == 2
