"""Cliente minimo de la Terminal Bloomberg (blpapi, HistoricalDataRequest)."""
import time
from datetime import date

import pandas as pd


class ErrorBloomberg(Exception):
    """La Terminal no responde o una peticion fallo tras los reintentos."""


AJUSTES_BLP = {
    "none": {"adjustmentNormal": False, "adjustmentAbnormal": False, "adjustmentSplit": False},
    "split": {"adjustmentNormal": False, "adjustmentAbnormal": False, "adjustmentSplit": True},
    "default": {},
}


def convertir_eco(valor: float) -> pd.Timestamp:
    """ECO_RELEASE_DT llega como numero AAAAMMDD.0."""
    return pd.Timestamp(str(int(valor)))


class ClienteBloomberg:
    def __init__(self, host="localhost", puerto=8194, lote=25, reintentos=3, espera_s=5):
        self.host, self.puerto, self.lote = host, puerto, lote
        self.reintentos, self.espera_s = reintentos, espera_s

    def historico(self, tickers: list[str], campo: str, inicio: date, fin: date,
                  ajuste: str) -> tuple[dict[str, pd.Series], dict[str, str]]:
        if ajuste not in AJUSTES_BLP:
            raise ValueError(f"ajuste invalido: {ajuste}")
        datos, errores = {}, {}
        unicos = list(dict.fromkeys(tickers))
        for i in range(0, len(unicos), self.lote):
            grupo = unicos[i:i + self.lote]
            for intento in range(1, self.reintentos + 1):
                try:
                    d, e = self._pedir_lote(grupo, campo, inicio, fin, ajuste)
                    break
                except ErrorBloomberg:
                    raise
                except Exception as exc:  # conexion, timeout de blpapi, etc.
                    if intento == self.reintentos:
                        raise ErrorBloomberg(f"Fallo la peticion {campo} tras {intento} intentos: {exc}") from exc
                    time.sleep(self.espera_s)
            datos.update(d)
            errores.update(e)
        return datos, errores

    def _pedir_lote(self, tickers, campo, inicio, fin, ajuste):
        import blpapi

        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.puerto)
        sesion = blpapi.Session(opts)
        if not sesion.start():
            raise ErrorBloomberg("Terminal Bloomberg no disponible: no se pudo iniciar la sesion de la API")
        try:
            if not sesion.openService("//blp/refdata"):
                raise ErrorBloomberg("Terminal Bloomberg no disponible: no se pudo abrir //blp/refdata")
            req = sesion.getService("//blp/refdata").createRequest("HistoricalDataRequest")
            for t in tickers:
                req.getElement("securities").appendValue(t)
            req.getElement("fields").appendValue(campo)
            req.set("startDate", inicio.strftime("%Y%m%d"))
            req.set("endDate", fin.strftime("%Y%m%d"))
            req.set("periodicitySelection", "DAILY")
            for k, v in AJUSTES_BLP[ajuste].items():
                req.set(k, v)
            req.set("adjustmentFollowDPDF", False)
            sesion.sendRequest(req)
            datos, errores, filas = {}, {}, {}
            while True:
                ev = sesion.nextEvent(120000)
                if ev.eventType() == blpapi.Event.TIMEOUT:
                    raise TimeoutError(f"Bloomberg no respondio en 120 s ({campo})")
                for msg in ev:
                    if not msg.hasElement("securityData"):
                        continue
                    sd = msg.getElement("securityData")
                    tk = sd.getElementAsString("security")
                    if sd.hasElement("securityError"):
                        errores[tk] = sd.getElement("securityError").getElementAsString("message")
                        continue
                    fe = sd.getElement("fieldExceptions") if sd.hasElement("fieldExceptions") else None
                    if fe is not None and fe.numValues():
                        errores[tk] = fe.getValueAsElement(0).getElement("errorInfo").getElementAsString("message")
                    fd = sd.getElement("fieldData")
                    d = filas.setdefault(tk, {})
                    for i in range(fd.numValues()):
                        row = fd.getValueAsElement(i)
                        if row.hasElement(campo):
                            v = row.getElementAsFloat(campo)
                            d[pd.Timestamp(row.getElementAsDatetime("date"))] = convertir_eco(v) if campo == "ECO_RELEASE_DT" else v
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            for tk, d in filas.items():
                s = pd.Series(d).sort_index()
                if len(s):
                    datos[tk] = s
                elif tk not in errores:
                    errores[tk] = "sin datos en el rango"
            for tk in tickers:
                if tk not in datos and tk not in errores:
                    errores[tk] = "sin respuesta"
            return datos, errores
        finally:
            sesion.stop()
