"""Cliente Bloomberg falso que responde con la historia cruda (para tests sin Terminal)."""
import pandas as pd


class ClienteFalso:
    def __init__(self, historia: pd.DataFrame, series: pd.DataFrame, rezago_pub=5, vacio=(), splits=None,
                 eco_raro=()):
        self.h = historia.assign(date=pd.to_datetime(historia["date"])).set_index("date")
        self.s = series
        self.rezago_pub, self.vacio, self.splits, self.eco_raro = rezago_pub, set(vacio), splits or {}, set(eco_raro)
        self.llamadas = []

    def _columna(self, ticker, campo):
        m = self.s[(self.s.ticker == ticker) & ((self.s.campo == campo) | (campo == "ECO_RELEASE_DT"))]
        return None if m.empty else m.iloc[0]

    def historico(self, tickers, campo, inicio, fin, ajuste):
        datos, errores = {}, {}
        for t in tickers:
            self.llamadas.append((t, campo, ajuste))
            fila = self._columna(t, campo)
            if t in self.vacio or fila is None:
                errores[t] = "sin datos en el rango"
                continue
            s = self.h[fila.columna_cruda].dropna()
            if fila.tipo == "macro":
                s = s[s.ne(s.shift())]                      # observaciones = fechas de cambio
                if campo == "ECO_RELEASE_DT":
                    dias = 400 if t in self.eco_raro else self.rezago_pub
                    s = pd.Series(s.index + pd.Timedelta(days=dias), index=s.index)
            if t in self.splits and fila.tipo == "precio":
                f, factor = self.splits[t]
                s = s.copy()
                # mundo real: desde f el precio negociado cae por el factor; con ajuste split
                # Bloomberg reexpresa toda la historia en la base nueva
                if ajuste == "split":
                    s = s / factor
                else:
                    s[s.index >= f] = s[s.index >= f] / factor
            s = s[(s.index >= pd.Timestamp(inicio)) & (s.index <= pd.Timestamp(fin))]
            if len(s):
                datos[t] = s
            else:
                errores[t] = "sin datos en el rango"
        return datos, errores
