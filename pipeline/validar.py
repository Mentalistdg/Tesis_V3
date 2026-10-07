"""Etapa 3: valida la descarga y el empalme. Estados: verde / amarillo / rojo (spec 8 y adenda 15)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIAS_DISCONTINUADA = 60   # una serie sin datos hace mas de esto en la historia se considera discontinuada


@dataclass
class Resultado:
    estado: str = "verde"
    mensajes: list[str] = field(default_factory=list)

    def amarillo(self, msg: str):
        if self.estado == "verde":
            self.estado = "amarillo"
        self.mensajes.append(msg)

    def rojo(self, msg: str):
        self.estado = "rojo"
        self.mensajes.append(msg)


def _discontinuadas(raw_previo: pd.DataFrame, series: pd.DataFrame) -> set[str]:
    fechas = pd.to_datetime(raw_previo["date"])
    limite = fechas.iloc[-1] - pd.Timedelta(days=DIAS_DISCONTINUADA)
    out = set()
    for c in series.loc[series.tipo != "macro", "columna_cruda"]:
        ultimo = raw_previo[c].last_valid_index()
        if ultimo is None or fechas.iloc[ultimo] < limite:
            out.add(c)
    return out


def validar(raw_previo: pd.DataFrame, raw_nuevo: pd.DataFrame, d, series: pd.DataFrame, config: dict) -> Resultado:
    r = Resultado()
    ancla = pd.Timestamp(config["ancla_inicial"])
    disc = _discontinuadas(raw_previo, series)
    tipos = series.set_index("columna_cruda").tipo

    if len(d.calendario) == 0:
        r.rojo("SPY US Equity sin datos: la Terminal no entrego el calendario (¿sin sesion iniciada?)")
        return r
    for c, msg in d.errores.items():
        if c not in disc:
            r.rojo(f"{c}: Bloomberg no entrego la serie ({msg})")

    nuevo = raw_nuevo.set_index(pd.to_datetime(raw_nuevo["date"]))
    previo = raw_previo.set_index(pd.to_datetime(raw_previo["date"]))
    agregadas = nuevo.index[nuevo.index > previo.index[-1]]
    recalculadas = nuevo.index[nuevo.index > ancla]
    spy = [c for c in series.columna_cruda if c.startswith("SPY US Equity (")]
    for f in recalculadas:
        faltan = [c for c in spy if pd.isna(nuevo.at[f, c])]
        if faltan:
            r.rojo(f"SPY OHLCV ausente el {f.date()}: {faltan}")
    for c in series.loc[series.tipo.isin(["precio", "nivel", "volumen", "fundamental"]), "columna_cruda"]:
        if c in disc or c in d.errores:
            continue
        for f in agregadas:
            if pd.isna(nuevo.at[f, c]):
                r.amarillo(f"{c}: sin dato el {f.date()} (queda NaN, como en la historia)")

    # Solapamiento: calce exigido en filas inmutables (historia congelada y filas posteriores fuera de la cola).
    posteriores = int((previo.index > ancla).sum())
    corte = previo.index[len(previo) - min(config["cola_mutable_dias_habiles"], posteriores) - 1]
    for c, s in d.valores.items():
        if c in disc:
            continue
        s = s[~s.index.duplicated(keep="last")]
        comunes = s.index[(s.index <= corte)].intersection(previo.index)
        if len(comunes) < 2:
            continue
        guardado, bbg = previo.loc[comunes, c].astype(float), s.loc[comunes].astype(float)
        ok = guardado.notna() & bbg.notna()
        guardado, bbg = guardado[ok], bbg[ok]
        if len(guardado) < 2:
            continue
        t = tipos[c]
        if t == "nivel":
            malos = ~np.isclose(guardado, bbg, rtol=1e-4, atol=1e-6)
            if malos.any():
                r.rojo(f"{c}: no calza con la historia congelada en {int(malos.sum())} fechas (p.ej. {guardado.index[malos][0].date()})")
        elif t == "precio":
            rg, rb = guardado.pct_change().iloc[1:], bbg.pct_change().iloc[1:]
            malos = ~np.isclose(rg, rb, rtol=1e-6, atol=1e-9)
            if malos.sum() == 1:
                r.amarillo(f"{c}: posible split o ajuste el {rg.index[malos][0].date()} (encadenado sin salto)")
            elif malos.sum() > 1:
                r.rojo(f"{c}: retornos no calzan con la historia congelada en {int(malos.sum())} fechas")
        elif t == "fundamental":
            razon = bbg / guardado                 # el encadenado fija un nivel; importa que la razon no cambie
            salto = abs(razon.iloc[-1] / razon.iloc[0] - 1)
            if salto > config["umbral_salto_fundamental"]:
                r.amarillo(f"{c}: diferencia de {salto:.1%} con la historia en {guardado.index[-1].date()} (revision; se encadena)")
    return r
