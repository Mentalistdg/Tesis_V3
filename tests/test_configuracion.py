import pandas as pd
import pytest

from conftest import SPX
from pipeline.configuracion import AJUSTES, TIPOS, cargar_config, cargar_series


def test_series_cubre_todas_las_columnas_crudas():
    s = cargar_series(SPX)
    cols = list(pd.read_csv(SPX / "historia_congelada.csv", nrows=0).columns[1:])
    assert list(s.columna_cruda) == cols and len(s) == 97


def test_series_valores_validos():
    s = cargar_series(SPX)
    assert set(s.tipo) <= TIPOS and set(s.ajuste) <= AJUSTES
    assert (s.loc[s.tipo.isin(["macro", "fundamental"]), "relleno"] == "ffill").all()
    assert (s.loc[~s.tipo.isin(["macro", "fundamental"]), "relleno"] == "no").all()


def test_series_correcciones_finales():
    s = cargar_series(SPX).set_index("columna_cruda")
    assert s.loc["CONSSENT Index", "ticker"] == "CPMINDX Index"
    assert s.loc["PITLCHNG Index", "ticker"] == "DGNOXTCH Index"
    assert s.loc["XLE US Equity", ["ticker", "ajuste"]].tolist() == ["IWF US Equity", "none"]
    assert s.loc["EFA US Equity", ["ticker", "ajuste"]].tolist() == ["XLK US Equity", "split"]


def test_macro_tiene_rezago_y_fecha_pub():
    s = cargar_series(SPX).set_index("columna_cruda")
    macro = s[s.tipo == "macro"]
    assert len(macro) == 21 and macro.rezago_pub_dias.notna().all()
    assert s.loc["GDP CQOQ Index", ["fecha_pub", "rezago_pub_dias"]].tolist() == ["rezago", 30]
    assert s.loc["CPTICHNG Index", ["ticker", "fecha_pub", "rezago_pub_dias"]].tolist() == ["LEI TOTL Index", "rezago", 20]


def test_publica_tarde_son_las_nueve():
    s = cargar_series(SPX)
    tardias = set(s.loc[s.publica_tarde, "ticker"])
    assert tardias == {"LF98OAS Index", "LF98TRUU Index", "LUACTRUU Index", "LUACOAS Index",
                       "MOVE Index", "CVIX Index", "SKEW Index", "PCUSEQTR Index"}
    assert s.publica_tarde.sum() == 9


def test_cargar_series_rechaza_tipo_invalido(tmp_path):
    for f in ("historia_congelada.csv", "config.yaml"):
        (tmp_path / f).write_bytes((SPX / f).read_bytes())
    s = pd.read_csv(SPX / "series.csv")
    s.loc[0, "tipo"] = "otro"
    s.to_csv(tmp_path / "series.csv", index=False)
    with pytest.raises(ValueError, match="tipo"):
        cargar_series(tmp_path)


def test_config():
    c = cargar_config(SPX)
    assert c["ancla_inicial"] == "2025-12-11" and c["fin_entrenamiento"] == "2020-10-06"
    assert c["cola_mutable_dias_habiles"] == 5 and c["hora_final_ny"] == "20:00"
    assert c["instrumentos"]["UPRO"]["expense_ratio"] == 0.0091
