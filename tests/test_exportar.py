import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import SPX
from pipeline.configuracion import cargar_config
from pipeline.exportar import (entrada_app, exportar_app, registrar_emision, reporte_periodo,
                               reporte_texto)
from pipeline.io import escribir_atomico, escribir_csv_atomico
from pipeline.validar import Resultado

CFG = cargar_config(SPX)
DAILY_KEYS = {"category", "params", "dates", "trades", "predictions", "percentiles", "positions",
              "strategy_returns", "market_returns", "equity_curve", "market_equity", "drawdown",
              "daily_costs", "risk_free", "metrics"}


def _senales(pos, desde="2026-01-05"):
    n = len(pos)
    fwd = [0.01, -0.005, 0.002, 0.004, -0.01, 0.003, 0.006, 0.001][:n]
    return pd.DataFrame({"date": pd.bdate_range(desde, periods=n).strftime("%Y-%m-%d"), "posicion": pos,
                         "prediccion": np.linspace(-0.01, 0.01, n), "percentil": np.linspace(10, 90, n),
                         "q_ext": 10, "q_mod": 30, "senal_valida": True,
                         "forward_returns": fwd[:n - 1] + [np.nan], "risk_free_rate": 0.0001})


def test_registrar_es_idempotente_y_solo_agrega_cambios(tmp_path):
    log = tmp_path / "senales_emitidas.csv"
    s = _senales([0, 1, 1, 0, 3, 3, 0])
    assert registrar_emision(log, "spx", s, s, Resultado()) is True
    assert registrar_emision(log, "spx", s, s, Resultado()) is False      # misma senal, misma fecha
    s2 = s.copy(); s2.loc[s2.index[-1], "posicion"] = 1
    assert registrar_emision(log, "spx", s2, s, Resultado("amarillo", ["x"])) is True
    df = pd.read_csv(log)
    assert len(df) == 2 and df.senal_b.tolist() == ["CASH", "SPY"] and df.estado.tolist() == ["verde", "amarillo"]


def test_cambio_vs_ayer(tmp_path):
    log = tmp_path / "e.csv"
    registrar_emision(log, "spx", _senales([0, 0]), _senales([0, 0]), Resultado())
    registrar_emision(log, "spx", _senales([0, 0, 1]), _senales([0, 0, 1]), Resultado())
    assert pd.read_csv(log).cambio_vs_ayer.tolist() == [False, True]


def test_reporte_cuenta_entradas_y_escenarios():
    s = _senales([0, 1, 1, 0, 3, 3, 0, 0])
    rep = reporte_periodo(s, s, CFG, desde="2026-01-05")
    assert rep["B"]["ejecucion_tesis"]["entradas"] == 2
    assert "retraso_1_dia" in rep["B"] and rep["ultima"]["senal_b"] == "CASH"
    assert "Entradas" in reporte_texto(rep)


def test_escritura_atomica_no_deja_parcial(tmp_path, monkeypatch):
    f = tmp_path / "a.csv"
    f.write_text("original", encoding="utf-8")
    def falla(*a, **k):
        raise PermissionError("abierto en Excel")
    monkeypatch.setattr(os, "replace", falla)
    with pytest.raises(PermissionError):
        escribir_atomico(f, "nuevo", reintentos=2, espera_s=0)
    assert f.read_text(encoding="utf-8") == "original" and list(tmp_path.iterdir()) == [f]


def test_escribir_csv_atomico(tmp_path):
    escribir_csv_atomico(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
    assert pd.read_csv(tmp_path / "x.csv").a.tolist() == [1]


def test_vivo_tiene_esquema_de_daily_data_y_fechas_de_realizacion():
    s = _senales([0, 1, 1, 0, 3, 3, 0, 0])
    resumen, daily = entrada_app("LSTM_Attention_vivo_B", s, CFG)
    assert set(daily) == DAILY_KEYS and resumen["model"] == "LSTM_Attention_vivo_B"
    assert daily["dates"][0] == s.date.iloc[1]                          # fecha en que se realiza el retorno
    assert len(daily["dates"]) == len(daily["positions"]) == 7


def test_exportar_app(tmp_path):
    s = _senales([0, 1, 1, 0, 3, 3, 0, 0])
    exportar_app(tmp_path, "spx", s, s, Resultado(), reporte_periodo(s, s, CFG, "2026-01-05"), CFG)
    sj = json.loads((tmp_path / "senales.json").read_text(encoding="utf-8"))
    assert sj["activos"][0]["activo"] == "spx" and sj["activos"][0]["senal_b"] == "CASH"
    vivo = json.loads((tmp_path / "vivo.json").read_text(encoding="utf-8"))
    assert set(vivo["daily"]) == {"LSTM_Attention_vivo_B", "LSTM_Attention_vivo_A"}
