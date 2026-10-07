from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from conftest import SPX
from fakes import ClienteFalso
from pipeline.configuracion import cargar_config, cargar_series
from pipeline.extraer import extraer, guardar_descarga, ultimo_dia_oficial

NY = ZoneInfo("America/New_York")
CFG = cargar_config(SPX)
SERIES = cargar_series(SPX)
HIST = pd.read_csv(SPX / "historia_congelada.csv")


def test_antes_de_hora_final_es_ayer():
    assert ultimo_dia_oficial(datetime(2026, 10, 7, 15, 35, tzinfo=NY), CFG) == date(2026, 10, 6)


def test_despues_de_hora_final_es_hoy():
    assert ultimo_dia_oficial(datetime(2026, 10, 7, 20, 30, tzinfo=NY), CFG) == date(2026, 10, 7)


def test_fin_de_semana_es_viernes():
    assert ultimo_dia_oficial(datetime(2026, 10, 10, 21, 0, tzinfo=NY), CFG) == date(2026, 10, 9)


def test_lunes_temprano_es_viernes():
    assert ultimo_dia_oficial(datetime(2026, 10, 12, 9, 30, tzinfo=NY), CFG) == date(2026, 10, 9)


def test_acepta_hora_de_chile():
    chile = datetime(2026, 10, 7, 21, 30, tzinfo=ZoneInfo("America/Santiago"))  # = 20:30 NY
    assert ultimo_dia_oficial(chile, CFG) == date(2026, 10, 7)


def test_extraer_calendario_es_spy():
    d = extraer(SERIES, date(2025, 11, 3), date(2025, 12, 12), ClienteFalso(HIST, SERIES))
    spy = pd.to_datetime(HIST["date"])
    assert list(d.calendario) == list(spy[(spy >= "2025-11-03") & (spy <= "2025-12-12")])


def test_precios_se_piden_con_split():
    c = ClienteFalso(HIST, SERIES)
    extraer(SERIES, date(2025, 11, 3), date(2025, 12, 12), c)
    ajustes = {t: a for t, campo, a in c.llamadas if campo == "PX_LAST"}
    assert ajustes["IWF US Equity"] == "split" and ajustes["XLK US Equity"] == "split"
    assert ajustes["VIX Index"] == "none"


def test_serie_vacia_queda_en_errores():
    d = extraer(SERIES, date(2025, 11, 3), date(2025, 12, 12), ClienteFalso(HIST, SERIES, vacio={"VIX Index"}))
    assert "VIX Index" in d.errores


def test_publicaciones_macro_y_eco_fuera_de_rango():
    c = ClienteFalso(HIST, SERIES, rezago_pub=5, eco_raro={"NAPMPMI Index"})
    d = extraer(SERIES, date(2025, 6, 2), date(2025, 12, 12), c)
    p = d.publicaciones
    assert set(p.columns) == {"columna_cruda", "periodo", "publicacion", "valor"}
    cpi = p[p.columna_cruda == "RSTAMOM Index"]                       # CPI YoY, fecha_pub eco
    assert ((cpi.publicacion - cpi.periodo).dt.days == 5).all()
    ism = p[p.columna_cruda == "SPY US Equity"]                       # NAPMPMI: eco de 400 dias -> rezago 1
    assert ((ism.publicacion - ism.periodo).dt.days == 1).all()
    gdp = p[p.columna_cruda == "GDP CQOQ Index"]                      # fecha_pub rezago 30
    assert ((gdp.publicacion - gdp.periodo).dt.days == 30).all()


def test_guardar_descarga(tmp_path):
    d = extraer(SERIES, date(2025, 11, 3), date(2025, 12, 12), ClienteFalso(HIST, SERIES, vacio={"VIX Index"}))
    guardar_descarga(d, tmp_path)
    assert (tmp_path / "publicaciones.csv").exists() and (tmp_path / "errores.json").exists()
    assert (tmp_path / "valores.csv").exists()
