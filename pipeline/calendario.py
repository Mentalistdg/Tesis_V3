"""Calendario de dias habiles de la NYSE (feriados regulares; reglas vigentes desde 2022 con Juneteenth)."""
from datetime import date, timedelta
from functools import lru_cache


def _pascua(anio: int) -> date:
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(anio, mes, dia)


def _n_esimo(anio, mes, dia_semana, n):
    d = date(anio, mes, 1)
    d += timedelta(days=(dia_semana - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _ultimo(anio, mes, dia_semana):
    d = date(anio, mes + 1, 1) - timedelta(days=1) if mes < 12 else date(anio, 12, 31)
    return d - timedelta(days=(d.weekday() - dia_semana) % 7)


def _observado(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def feriados_nyse(anio: int) -> frozenset:
    f = {
        _n_esimo(anio, 1, 0, 3),                 # Martin Luther King Jr.
        _n_esimo(anio, 2, 0, 3),                 # Presidents' Day
        _pascua(anio) - timedelta(days=2),       # Viernes Santo
        _ultimo(anio, 5, 0),                     # Memorial Day
        _observado(date(anio, 7, 4)),            # Independencia
        _n_esimo(anio, 9, 0, 1),                 # Labor Day
        _n_esimo(anio, 11, 3, 4),                # Thanksgiving
        _observado(date(anio, 12, 25)),          # Navidad
    }
    ano_nuevo = date(anio, 1, 1)
    if ano_nuevo.weekday() != 5:                 # si cae sabado la NYSE no lo observa el viernes previo
        f.add(_observado(ano_nuevo))
    if anio >= 2022:
        f.add(_observado(date(anio, 6, 19)))     # Juneteenth
    return frozenset(f)


def es_habil_nyse(d: date) -> bool:
    return d.weekday() < 5 and d not in feriados_nyse(d.year)


def siguiente_habil(d: date) -> date:
    d += timedelta(days=1)
    while not es_habil_nyse(d):
        d += timedelta(days=1)
    return d


def habil_anterior(d: date) -> date:
    d -= timedelta(days=1)
    while not es_habil_nyse(d):
        d -= timedelta(days=1)
    return d
