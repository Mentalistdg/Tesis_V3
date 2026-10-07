"""Etapa 2: empalma los datos nuevos sobre la historia (version A) y deriva la macro A y B.

Reglas (spec 6 y adenda 15):
- historia congelada (<= ancla congelada): nunca cambia;
- cola mutable: las ultimas `cola` filas posteriores al ancla se recalculan en cada corrida;
- precio y fundamental: se encadenan desde el ultimo valor valido inmutable;
- nivel y volumen: valor directo, union por fecha exacta, sin relleno;
- macro: no se guarda despues del ancla; se deriva en cada corrida (A en fecha de periodo, B en publicacion).
"""
import numpy as np
import pandas as pd


def unir_calendario(serie: pd.Series, calendario: pd.DatetimeIndex, relleno: bool) -> pd.Series:
    s = serie[~serie.index.duplicated(keep="last")].sort_index()
    r = s.reindex(pd.DatetimeIndex(calendario))
    return r.ffill() if relleno else r


def encadenar(valor_ancla: float, serie_bbg: pd.Series, ancla: pd.Timestamp) -> pd.Series:
    s = serie_bbg.sort_index()
    base = s[s.index <= ancla]
    if base.empty or np.isnan(valor_ancla):
        return s[s.index > ancla] * np.nan
    return valor_ancla * s[s.index > ancla] / base.iloc[-1]


def _fechas(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def empalmar(raw_previo: pd.DataFrame, d, series: pd.DataFrame, ancla_congelada: pd.Timestamp,
             cola: int) -> pd.DataFrame:
    """Devuelve el raw de 98 columnas: filas inmutables de raw_previo + filas recalculadas/nuevas."""
    previo = raw_previo.reset_index(drop=True)
    f_prev = _fechas(previo)
    posteriores = np.flatnonzero(f_prev > ancla_congelada)
    n_inmut = len(previo) - min(cola, len(posteriores))
    inmutable = previo.iloc[:n_inmut]
    corte = _fechas(inmutable)[-1]
    nuevas = d.calendario[d.calendario > corte]
    filas = pd.DataFrame(index=nuevas, columns=previo.columns[1:], dtype=float)
    hist_inm = inmutable.set_index(_fechas(inmutable))
    for f in series.itertuples():
        c = f.columna_cruda
        if f.tipo == "macro" or c not in d.valores:
            continue
        s = d.valores[c]
        if f.tipo in ("precio", "fundamental"):
            validos = hist_inm[c].dropna()
            if validos.empty:
                continue
            ancla, v0 = validos.index[-1], validos.iloc[-1]
            enc = encadenar(float(v0), s, ancla)
            serie = pd.concat([pd.Series([v0], index=[ancla]), enc])
            filas[c] = unir_calendario(serie, pd.DatetimeIndex([ancla]).append(nuevas),
                                       relleno=(f.relleno == "ffill")).reindex(nuevas).values
        else:
            filas[c] = unir_calendario(s, nuevas, relleno=False).values
    filas.insert(0, "date", [x.strftime("%Y-%m-%d") for x in nuevas])
    return pd.concat([inmutable, filas.reset_index(drop=True)], ignore_index=True)[previo.columns]


def macro_a(raw: pd.DataFrame, publicaciones: pd.DataFrame, series: pd.DataFrame,
            ancla_congelada: pd.Timestamp) -> pd.DataFrame:
    """Macro en fecha de periodo (replica del entrenamiento) para filas posteriores al ancla."""
    out = raw.copy()
    cal = _fechas(raw)
    post = cal > ancla_congelada
    for f in series[series.tipo == "macro"].itertuples():
        c = f.columna_cruda
        p = publicaciones[publicaciones.columna_cruda == c]
        obs = pd.Series(p.valor.values, index=pd.DatetimeIndex(p.periodo))
        nuevos = unir_calendario(obs[obs.index > ancla_congelada], cal, relleno=False)
        col = pd.Series(raw[c].values, index=cal, dtype=float)
        col[post] = nuevos[post].values
        out[c] = np.where(post, col.ffill().values, raw[c].values)
    return out


def version_b(raw_a: pd.DataFrame, publicaciones: pd.DataFrame, series: pd.DataFrame,
              ancla_congelada: pd.Timestamp) -> pd.DataFrame:
    """Macro en fecha de publicacion para toda la historia (point-in-time en fechas)."""
    out = raw_a.copy()
    cal = _fechas(raw_a)
    base = pd.DataFrame({"date": cal})
    for f in series[series.tipo == "macro"].itertuples():
        c = f.columna_cruda
        p = publicaciones[publicaciones.columna_cruda == c].sort_values("periodo")
        if p.empty:
            out[c] = np.nan
            continue
        congelado = pd.Series(raw_a[c].values, index=cal)
        periodo = pd.DatetimeIndex(p.periodo)
        val_cong = congelado.reindex(periodo)
        usar_cong = (periodo <= ancla_congelada) & val_cong.notna().values
        valor = np.where(usar_cong, val_cong.values, p.valor.values)
        obs = pd.DataFrame({"pub": pd.DatetimeIndex(p.publicacion), "valor": valor}).sort_values("pub")
        obs = obs.drop_duplicates("pub", keep="last")
        m = pd.merge_asof(base, obs, left_on="date", right_on="pub", direction="backward")
        out[c] = m["valor"].values
    return out
