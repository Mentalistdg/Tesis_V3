from datetime import date

import pandas as pd
import pytest

from pipeline.bloomberg import ClienteBloomberg, ErrorBloomberg, convertir_eco


class ClienteDePrueba(ClienteBloomberg):
    """Sustituye la unica funcion que toca blpapi."""

    def __init__(self, respuestas=None, fallas=0, **kw):
        super().__init__(espera_s=0, **kw)
        self.llamadas, self.fallas, self.respuestas = [], fallas, respuestas or {}

    def _pedir_lote(self, tickers, campo, inicio, fin, ajuste):
        self.llamadas.append((list(tickers), campo, ajuste))
        if self.fallas:
            self.fallas -= 1
            raise ConnectionError("sin conexion")
        datos = {t: pd.Series([1.0], index=[pd.Timestamp("2025-12-12")]) for t in tickers if t != "MALO Index"}
        errores = {"MALO Index": "Unknown/Invalid security"} if "MALO Index" in tickers else {}
        return datos, errores


def test_divide_en_lotes_de_25():
    c = ClienteDePrueba()
    c.historico([f"T{i} Index" for i in range(60)], "PX_LAST", date(2025, 1, 1), date(2025, 1, 2), "none")
    assert [len(t) for t, _, _ in c.llamadas] == [25, 25, 10]


def test_reintenta_y_luego_falla():
    c = ClienteDePrueba(fallas=3)
    with pytest.raises(ErrorBloomberg):
        c.historico(["A Index"], "PX_LAST", date(2025, 1, 1), date(2025, 1, 2), "none")
    assert len(c.llamadas) == 3


def test_reintento_exitoso():
    c = ClienteDePrueba(fallas=2)
    datos, err = c.historico(["A Index"], "PX_LAST", date(2025, 1, 1), date(2025, 1, 2), "none")
    assert "A Index" in datos and not err and len(c.llamadas) == 3


def test_errores_por_ticker_se_acumulan():
    datos, err = ClienteDePrueba().historico(["A Index", "MALO Index"], "PX_LAST",
                                            date(2025, 1, 1), date(2025, 1, 2), "none")
    assert "A Index" in datos and "MALO Index" in err


def test_ajuste_invalido():
    with pytest.raises(ValueError):
        ClienteDePrueba().historico(["A Index"], "PX_LAST", date(2025, 1, 1), date(2025, 1, 2), "todo")


def test_eco_release_dt_se_parsea():
    assert convertir_eco(20250128.0) == pd.Timestamp("2025-01-28")


@pytest.mark.bloomberg
def test_integracion_spy_cierre():
    s, err = ClienteBloomberg().historico(["SPY US Equity"], "PX_LAST", date(2025, 12, 8), date(2025, 12, 12), "none")
    assert not err and s["SPY US Equity"].loc["2025-12-10"] == pytest.approx(687.57)


@pytest.mark.bloomberg
def test_integracion_eco_release_dt():
    s, err = ClienteBloomberg().historico(["NAPMPMI Index"], "ECO_RELEASE_DT", date(2025, 1, 1), date(2025, 3, 31), "default")
    assert not err and s["NAPMPMI Index"].loc["2025-01-31"] == pd.Timestamp("2025-02-03")
