from datetime import date

from pipeline.calendario import es_habil_nyse, habil_anterior, siguiente_habil


def test_accion_de_gracias():
    assert not es_habil_nyse(date(2026, 11, 26))
    assert siguiente_habil(date(2026, 11, 25)) == date(2026, 11, 27)


def test_viernes_santo():
    assert not es_habil_nyse(date(2026, 4, 3))
    assert siguiente_habil(date(2026, 4, 2)) == date(2026, 4, 6)


def test_navidad_y_ano_nuevo_observados():
    assert not es_habil_nyse(date(2027, 12, 24))      # 25-dic-2027 cae sabado -> se observa el viernes 24
    assert not es_habil_nyse(date(2026, 1, 1))
    assert not es_habil_nyse(date(2026, 6, 19))       # Juneteenth
    assert es_habil_nyse(date(2026, 10, 12))          # Columbus Day: bolsa abierta


def test_habil_anterior_salta_feriado_y_fin_de_semana():
    assert habil_anterior(date(2026, 11, 27)) == date(2026, 11, 25)
    assert habil_anterior(date(2026, 10, 12)) == date(2026, 10, 9)
