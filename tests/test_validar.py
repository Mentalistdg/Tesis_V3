import numpy as np
import pandas as pd

from conftest import SPX
from fakes import ClienteFalso
from pipeline.configuracion import cargar_config, cargar_series
from pipeline.empalmar import empalmar
from pipeline.extraer import extraer
from pipeline.validar import validar

SERIES = cargar_series(SPX)
CFG = {**cargar_config(SPX), "ancla_inicial": "2025-11-28"}
HIST = pd.read_csv(SPX / "historia_congelada.csv")
PREVIO = HIST[HIST.date <= "2025-11-28"].reset_index(drop=True)
INI = pd.to_datetime(PREVIO.date).iloc[-31].date()


def _correr(cliente, previo=PREVIO):
    d = extraer(SERIES, INI, pd.Timestamp("2025-12-12").date(), cliente)
    nuevo = empalmar(previo, d, SERIES, pd.Timestamp(CFG["ancla_inicial"]), CFG["cola_mutable_dias_habiles"])
    return validar(previo, nuevo, d, SERIES, CFG), nuevo, d


def test_todo_bien_es_verde():
    r, *_ = _correr(ClienteFalso(HIST, SERIES))
    assert r.estado == "verde", r.mensajes


def test_spy_vacio_es_rojo():
    r, *_ = _correr(ClienteFalso(HIST, SERIES, vacio={"SPY US Equity"}))
    assert r.estado == "rojo" and any("SPY" in m for m in r.mensajes)


def test_ticker_con_error_es_rojo_y_nombra_columna():
    r, *_ = _correr(ClienteFalso(HIST, SERIES, vacio={"VVIX Index"}))
    assert r.estado == "rojo" and any("VXEEM Index" in m for m in r.mensajes)


def test_discontinuadas_sin_datos_no_alertan():
    r, *_ = _correr(ClienteFalso(HIST, SERIES, vacio={"US0003M Index", "USSWAP10 Index"}))
    assert r.estado == "verde", r.mensajes


def test_nivel_que_no_calza_en_congelado_es_rojo():
    previo = PREVIO.copy()
    previo.loc[previo.date == "2025-11-20", "VIX Index"] = 50.0
    r, *_ = _correr(ClienteFalso(HIST, SERIES), previo)
    assert r.estado == "rojo" and any("VIX Index" in m for m in r.mensajes)


def test_dato_puntual_faltante_es_amarillo():
    h = HIST.copy()
    h.loc[h.date == "2025-12-10", "VIX Index"] = np.nan
    r, *_ = _correr(ClienteFalso(h, SERIES))
    assert r.estado == "amarillo" and any("VIX Index" in m and "2025-12-10" in m for m in r.mensajes)


def test_salto_fundamental_es_amarillo():
    previo = PREVIO.copy()
    previo.loc[previo.date == "2025-11-28", "CL1 Comdty"] *= 1.05      # SPX PE_RATIO almacenado 5% distinto
    r, *_ = _correr(ClienteFalso(HIST, SERIES), previo)
    assert r.estado == "amarillo" and any("CL1 Comdty" in m for m in r.mensajes)


def test_split_en_solapamiento_es_amarillo():
    previo = PREVIO.copy()
    m = pd.to_datetime(previo.date) >= "2025-11-20"
    previo.loc[m, "XLE US Equity"] = previo.loc[m, "XLE US Equity"] / 4   # IWF almacenado sin ajuste con split 4:1
    r, *_ = _correr(ClienteFalso(HIST, SERIES), previo)
    assert r.estado == "amarillo" and any("split" in m for m in r.mensajes)
