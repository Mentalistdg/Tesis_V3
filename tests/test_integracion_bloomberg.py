from datetime import date

import pandas as pd
import pytest

from conftest import SPX
from pipeline.bloomberg import ClienteBloomberg
from pipeline.configuracion import cargar_config, cargar_series
from pipeline.empalmar import empalmar, macro_a
from pipeline.extraer import extraer
from pipeline.validar import validar

SERIES = cargar_series(SPX)
CFG = cargar_config(SPX)
HIST = pd.read_csv(SPX / "historia_congelada.csv")


@pytest.mark.bloomberg
def test_97_series_y_solapamiento_exacto():
    previo = HIST[HIST.date <= "2025-11-28"].reset_index(drop=True)
    d = extraer(SERIES, date(2025, 10, 15), date(2025, 12, 12), ClienteBloomberg())
    cfg = {**CFG, "ancla_inicial": "2025-11-28"}
    nuevo = empalmar(previo, d, SERIES, pd.Timestamp("2025-11-28"), 5)
    r = validar(previo, nuevo, d, SERIES, cfg)
    assert r.estado != "rojo", r.mensajes
    assert set(d.errores) <= {"US0003M Index", "USSWAP10 Index"}, d.errores
    # los dias nuevos reconstruidos desde Bloomberg reproducen la historia congelada (mercado)
    # hasta el 11-dic (ultima fila del entrenamiento); el 12-dic de la historia tiene CVIX aun no publicado
    ref = HIST[(HIST.date > "2025-11-28") & (HIST.date <= "2025-12-11")].reset_index(drop=True)
    got = nuevo[(nuevo.date > "2025-11-28") & (nuevo.date <= "2025-12-11")].reset_index(drop=True)
    exactas = SERIES[SERIES.tipo.isin(["precio", "nivel", "volumen"])].columna_cruda
    exactas = [c for c in exactas if c not in d.errores]
    pd.testing.assert_frame_equal(got[exactas], ref[exactas], check_exact=False, rtol=1e-6)
