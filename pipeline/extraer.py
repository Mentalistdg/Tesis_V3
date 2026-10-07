"""Etapa 1: decide el modo de corrida y descarga las series desde Bloomberg."""
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.calendario import es_habil_nyse, habil_anterior

NY = ZoneInfo("America/New_York")
INICIO_HISTORIA = date(1999, 12, 1)
VENTANA_MACRO_DIAS = 150          # periodos macro anteriores al inicio cuya publicacion puede caer en la ventana
REZAGO_ECO_VALIDO = (-15, 120)    # dias; fuera de este rango ECO_RELEASE_DT se descarta


@dataclass
class Descarga:
    calendario: pd.DatetimeIndex
    valores: dict[str, pd.Series] = field(default_factory=dict)       # columna_cruda -> serie Bloomberg (mercado)
    publicaciones: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["columna_cruda", "periodo", "publicacion", "valor"]))
    errores: dict[str, str] = field(default_factory=dict)


def _habil_anterior(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def ultimo_dia_oficial(ahora: datetime, config: dict) -> date:
    """Ultimo dia habil cuyos datos de cierre ya son oficiales: hoy si en Nueva York ya paso hora_final_ny."""
    ny = ahora.astimezone(NY)
    if es_habil_nyse(ny.date()) and ny.time() >= _hhmm(config["hora_final_ny"]):
        return ny.date()
    return habil_anterior(ny.date())


def _ajuste_descarga(fila) -> str:
    if fila.tipo == "precio":
        return "split"            # base irrelevante al encadenar; inmune a splits nuevos
    if fila.tipo == "fundamental":
        return "default"
    return fila.ajuste


def _publicaciones(series: pd.DataFrame, px: dict, eco: dict) -> pd.DataFrame:
    partes = []
    for fila in series[series.tipo == "macro"].itertuples():
        s = px.get(fila.ticker)
        if s is None or not len(s):
            continue
        periodo = pd.DatetimeIndex(s.index)
        pub = periodo + pd.to_timedelta(int(fila.rezago_pub_dias), unit="D")
        if fila.fecha_pub == "eco" and fila.ticker in eco:
            e = pd.to_datetime(eco[fila.ticker].reindex(periodo))
            rez = (e - pd.Series(periodo, index=periodo)).dt.days
            ok = e.notna() & (rez >= REZAGO_ECO_VALIDO[0]) & (rez <= REZAGO_ECO_VALIDO[1])
            pub = pd.DatetimeIndex(e.where(ok, pd.Series(pub, index=periodo)).values)
        partes.append(pd.DataFrame({"columna_cruda": fila.columna_cruda, "periodo": periodo,
                                    "publicacion": pub, "valor": s.values}))
    if not partes:
        return Descarga(pd.DatetimeIndex([])).publicaciones
    return pd.concat(partes, ignore_index=True)


def extraer(series: pd.DataFrame, inicio: date, fin: date, cliente) -> Descarga:
    valores, errores = {}, {}
    mercado = series[series.tipo != "macro"]
    grupos = {}
    for fila in mercado.itertuples():
        grupos.setdefault((fila.campo, _ajuste_descarga(fila)), []).append(fila)
    for (campo, ajuste), filas in grupos.items():
        datos, err = cliente.historico([f.ticker for f in filas], campo, inicio, fin, ajuste)
        for f in filas:
            if f.ticker in datos:
                valores[f.columna_cruda] = datos[f.ticker]
            else:
                errores[f.columna_cruda] = err.get(f.ticker, "sin respuesta")
    cal_col = mercado[(mercado.ticker == "SPY US Equity") & (mercado.campo == "PX_LAST")].columna_cruda.iloc[0]
    calendario = pd.DatetimeIndex(valores[cal_col].index) if cal_col in valores else pd.DatetimeIndex([])
    macro = series[series.tipo == "macro"]
    tks = list(dict.fromkeys(macro.ticker))
    ini_macro = inicio - timedelta(days=VENTANA_MACRO_DIAS)
    px, err_px = cliente.historico(tks, "PX_LAST", ini_macro, fin, "default")
    eco, _ = cliente.historico(list(dict.fromkeys(macro[macro.fecha_pub == "eco"].ticker)),
                               "ECO_RELEASE_DT", ini_macro, fin, "default")
    for f in macro.itertuples():
        if f.ticker not in px:
            errores[f.columna_cruda] = err_px.get(f.ticker, "sin respuesta")
    return Descarga(calendario, valores, _publicaciones(series, px, eco), errores)


def extraer_publicaciones_completas(series: pd.DataFrame, cliente, fin: date | None = None) -> pd.DataFrame:
    fin = fin or date.today()
    macro = series[series.tipo == "macro"]
    px, _ = cliente.historico(list(dict.fromkeys(macro.ticker)), "PX_LAST", INICIO_HISTORIA, fin, "default")
    eco, _ = cliente.historico(list(dict.fromkeys(macro[macro.fecha_pub == "eco"].ticker)),
                               "ECO_RELEASE_DT", INICIO_HISTORIA, fin, "default")
    return _publicaciones(series, px, eco)


def guardar_descarga(d: Descarga, carpeta: Path) -> None:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(d.valores).sort_index().to_csv(carpeta / "valores.csv", index_label="date", encoding="utf-8")
    d.publicaciones.to_csv(carpeta / "publicaciones.csv", index=False, encoding="utf-8")
    (carpeta / "errores.json").write_text(json.dumps(d.errores, indent=1, ensure_ascii=False), encoding="utf-8")
