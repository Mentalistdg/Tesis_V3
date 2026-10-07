from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import SPX
from fakes import ClienteFalso
from pipeline.configuracion import cargar_series
from pipeline.empalmar import empalmar, encadenar, macro_a, unir_calendario, version_b
from pipeline.extraer import extraer

SERIES = cargar_series(SPX)
HIST = pd.read_csv(SPX / "historia_congelada.csv")
ANCLA = pd.Timestamp("2025-12-12")


def _previo(hasta):
    return HIST[HIST.date <= hasta].reset_index(drop=True)


def _descarga(previo, cliente, fin="2025-12-12"):
    ini = pd.to_datetime(previo.date).iloc[-31].date()
    return extraer(SERIES, ini, pd.Timestamp(fin).date(), cliente)


def test_union_exacta_pierde_fin_de_semana():
    s = pd.Series([35.7, 52.0], index=pd.to_datetime(["2020-02-29", "2020-03-31"]))
    cal = pd.to_datetime(["2020-02-28", "2020-03-02", "2020-03-31"])
    r = unir_calendario(s, cal, relleno=True)
    assert 35.7 not in r.values and np.isnan(r.iloc[0]) and r.iloc[2] == 52.0


def test_union_sin_relleno_deja_nan():
    s = pd.Series([1.0, 3.0], index=pd.to_datetime(["2020-03-02", "2020-03-04"]))
    r = unir_calendario(s, pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-04"]), relleno=False)
    assert np.isnan(r.iloc[1]) and r.iloc[2] == 3.0


def test_encadenar_sobrevive_split():
    f = pd.to_datetime(["2025-12-12", "2025-12-15", "2025-12-16"])
    bbg = pd.Series([50.0, 50.5, 51.005], index=f)      # base nueva (split 2:1 aplicado retroactivamente)
    r = encadenar(200.0, bbg, f[0])
    assert list(r.index) == list(f[1:]) and r.tolist() == pytest.approx([202.0, 204.02])


def test_empalme_reproduce_historia_y_no_toca_congelado():
    previo = _previo("2025-11-28")
    nuevo = empalmar(previo, _descarga(previo, ClienteFalso(HIST, SERIES)), SERIES, pd.Timestamp("2025-11-28"), 5)
    ref = _previo("2025-12-12")
    mercado = SERIES[SERIES.tipo != "macro"].columna_cruda.tolist()
    pd.testing.assert_frame_equal(nuevo[["date"] + mercado], ref[["date"] + mercado], check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(nuevo.iloc[:len(previo)], previo, check_exact=True)


def test_empalme_split_nuevo_en_iwf_sin_salto():
    previo = _previo("2025-11-28")
    c = ClienteFalso(HIST, SERIES, splits={"IWF US Equity": (pd.Timestamp("2025-12-05"), 4.0)})
    nuevo = empalmar(previo, _descarga(previo, c), SERIES, pd.Timestamp("2025-11-28"), 5)
    col = "XLE US Equity"   # columna cruda que contiene IWF
    r_nuevo = nuevo.set_index("date")[col].pct_change().loc["2025-12-01":]
    r_real = HIST.set_index("date")[col].pct_change().loc["2025-12-01":"2025-12-12"]
    assert r_nuevo.values == pytest.approx(r_real.values, rel=1e-9)


def test_cola_mutable_se_reescribe_y_lo_anterior_no():
    previo = _previo("2025-12-12").copy()
    previo.loc[previo.date == "2025-12-11", "VIX Index"] = 99.0   # valor provisorio dentro de la cola
    previo.loc[previo.date == "2025-11-03", "VIX Index"] = 77.0   # fuera de la cola (pero posterior al ancla simulada)
    ancla = pd.Timestamp("2025-10-31")
    nuevo = empalmar(previo, _descarga(previo, ClienteFalso(HIST, SERIES)), SERIES, ancla, 5)
    v = nuevo.set_index("date")["VIX Index"]
    assert v.loc["2025-12-11"] == HIST.set_index("date").loc["2025-12-11", "VIX Index"]
    assert v.loc["2025-11-03"] == 77.0


def test_macro_no_se_guarda_despues_del_ancla():
    previo = _previo("2025-11-28")
    nuevo = empalmar(previo, _descarga(previo, ClienteFalso(HIST, SERIES)), SERIES, pd.Timestamp("2025-11-28"), 5)
    assert nuevo.loc[nuevo.date > "2025-11-28", "NFP TCH Index"].isna().all()


def test_macro_a_reproduce_historia():
    previo = _previo("2025-11-28")
    d = _descarga(previo, ClienteFalso(HIST, SERIES))
    nuevo = empalmar(previo, d, SERIES, pd.Timestamp("2025-11-28"), 5)
    a = macro_a(nuevo, d.publicaciones, SERIES, pd.Timestamp("2025-11-28"))
    macro = SERIES[SERIES.tipo == "macro"].columna_cruda.tolist()
    pd.testing.assert_frame_equal(a[macro], _previo("2025-12-12")[macro], check_exact=False, rtol=1e-12)


def _pub(col, periodos, pubs, valores):
    return pd.DataFrame({"columna_cruda": col, "periodo": pd.to_datetime(periodos),
                         "publicacion": pd.to_datetime(pubs), "valor": valores})


def _raw(fechas, col, valores):
    return pd.DataFrame({"date": [str(f.date()) for f in pd.to_datetime(fechas)], col: valores})


def test_macro_a_observacion_en_sabado_se_pierde():
    fechas = pd.bdate_range("2026-02-26", "2026-03-04")
    raw = _raw(fechas, "SPY US Equity", [48.0, np.nan, np.nan, np.nan, np.nan])
    pub = _pub("SPY US Equity", ["2026-02-28"], ["2026-03-02"], [52.0])
    s = SERIES[SERIES.columna_cruda == "SPY US Equity"]
    a = macro_a(raw, pub, s, pd.Timestamp("2026-02-26"))
    assert a["SPY US Equity"].tolist() == [48.0] * 5


def test_version_b_desplaza_a_publicacion_y_recupera_sabado():
    fechas = pd.bdate_range("2026-02-26", "2026-03-04")
    raw = _raw(fechas, "SPY US Equity", [48.0, 48.0, 48.0, 48.0, 48.0])
    pub = _pub("SPY US Equity", ["2026-01-31", "2026-02-28"], ["2026-02-02", "2026-03-02"], [48.0, 52.0])
    s = SERIES[SERIES.columna_cruda == "SPY US Equity"]
    b = version_b(raw, pub, s, pd.Timestamp("2026-03-04"))
    assert b["SPY US Equity"].tolist() == [48.0, 48.0, 52.0, 52.0, 52.0]


def test_version_b_usa_valor_congelado_en_periodo():
    fechas = pd.bdate_range("2026-01-28", "2026-02-04")
    raw = _raw(fechas, "SPY US Equity", [47.0, 47.0, 49.5, 49.5, 49.5, 49.5])   # 2026-01-30 (viernes) = 49.5
    pub = _pub("SPY US Equity", ["2026-01-30"], ["2026-02-03"], [49.9])         # Bloomberg revisado = 49.9
    s = SERIES[SERIES.columna_cruda == "SPY US Equity"]
    b = version_b(raw, pub, s, pd.Timestamp("2026-02-04"))
    assert b["SPY US Equity"].tolist()[-2:] == [49.5, 49.5] and np.isnan(b["SPY US Equity"].iloc[0])
