import json
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from conftest import RAIZ, SPX
from fakes import ClienteFalso
from pipeline import cronos
from pipeline.configuracion import cargar_series
from pipeline.cronos import CorridaEnCurso, actualizar, verificar_regresion

NY = ZoneInfo("America/New_York")
AHORA = datetime(2025, 12, 12, 21, 0, tzinfo=NY)
HIST = pd.read_csv(SPX / "historia_congelada.csv")
SERIES = cargar_series(SPX)
COLUMNAS = json.loads((SPX / "columnas_dataset.json").read_text(encoding="utf-8"))


@pytest.fixture
def raiz(tmp_path):
    a = tmp_path / "activos" / "spx"
    shutil.copytree(SPX / "modelo", a / "modelo")
    for f in ("series.csv", "columnas_dataset.json"):
        shutil.copy(SPX / f, a / f)
    cfg = (SPX / "config.yaml").read_text(encoding="utf-8").replace('ancla_inicial: "2025-12-11"', 'ancla_inicial: "2025-11-28"')
    (a / "config.yaml").write_text(cfg, encoding="utf-8")
    HIST[HIST.date <= "2025-11-28"].to_csv(a / "historia_congelada.csv", index=False)
    (tmp_path / "app" / "backend" / "data").mkdir(parents=True)
    return tmp_path


def _correr(raiz, cliente=None, **kw):
    return actualizar(raiz, "spx", cliente or ClienteFalso(HIST, SERIES), AHORA, verificar=False, **kw)


@pytest.mark.lento
def test_e2e_reproduce_historia(raiz):
    res, info = _correr(raiz)
    assert res.estado == "verde", res.mensajes
    raw = pd.read_csv(raiz / "data/spx/raw_extendido.csv")
    mercado = ["date"] + SERIES[SERIES.tipo != "macro"].columna_cruda.tolist()
    pd.testing.assert_frame_equal(raw[mercado], HIST[mercado], check_exact=False, rtol=1e-9)
    ds = pd.read_csv(raiz / "data/spx/dataset_A.csv", low_memory=False)
    ref = pd.read_csv(RAIZ / "data/bloomberg_triple_screen_core.csv", low_memory=False)
    pd.testing.assert_frame_equal(ds.iloc[:-1].drop(columns="date_id"), ref.drop(columns="date_id"),
                                  check_exact=False, rtol=1e-9)
    assert info["fecha_senal"] == "2025-12-15"
    emit = pd.read_csv(raiz / "logs/senales_emitidas.csv")
    assert emit.fecha.tolist() == ["2025-12-15"]
    assert (raiz / "app/backend/data/senales.json").exists() and (raiz / "logs/reporte_spx.txt").exists()


@pytest.mark.lento
def test_dos_corridas_idempotentes(raiz):
    _correr(raiz)
    raw1 = pd.read_csv(raiz / "data/spx/raw_extendido.csv")
    _correr(raiz)
    raw2 = pd.read_csv(raiz / "data/spx/raw_extendido.csv")
    pd.testing.assert_frame_equal(raw1, raw2, check_exact=False, rtol=1e-12)   # re-encadenar la cola: solo redondeo
    assert len(pd.read_csv(raiz / "logs/senales_emitidas.csv")) == 1


def test_rojo_no_modifica_archivos(raiz):
    res, _ = _correr(raiz, ClienteFalso(HIST, SERIES, vacio={"SPY US Equity"}))
    assert res.estado == "rojo"
    assert not (raiz / "data/spx/raw_extendido.csv").exists()
    assert not (raiz / "app/backend/data/senales.json").exists()
    assert "rojo" in (raiz / "logs/pipeline.log").read_text(encoding="utf-8")


@pytest.mark.lento
def test_interrupcion_en_escritura_deja_previos(raiz, monkeypatch):
    original = cronos.escribir_csv_atomico
    def falla(df, path):
        if path.name == "dataset_B.csv":
            raise OSError("se apago el PC")
        return original(df, path)
    monkeypatch.setattr(cronos, "escribir_csv_atomico", falla)
    with pytest.raises(OSError):
        _correr(raiz)
    assert not (raiz / "data/spx/raw_extendido.csv").exists()
    monkeypatch.setattr(cronos, "escribir_csv_atomico", original)
    res, _ = _correr(raiz)
    assert res.estado == "verde" and (raiz / "data/spx/raw_extendido.csv").exists()


@pytest.mark.lento
def test_ensayo_no_escribe(raiz):
    res, info = _correr(raiz, ensayo=True)
    assert res.estado == "verde" and info["fecha_senal"] == "2025-12-15"
    assert not (raiz / "data").exists() and not (raiz / "app/backend/data/senales.json").exists()


def test_lock_impide_corridas_simultaneas(raiz):
    (raiz / "logs").mkdir()
    (raiz / "logs/cronos.lock").write_text(json.dumps({"pid": 1, "ts": time.time()}), encoding="utf-8")
    with pytest.raises(CorridaEnCurso):
        _correr(raiz)


def test_lock_vencido_se_reemplaza(raiz):
    (raiz / "logs").mkdir()
    (raiz / "logs/cronos.lock").write_text(json.dumps({"pid": 1, "ts": time.time() - 3 * 3600}), encoding="utf-8")
    res, _ = _correr(raiz, ClienteFalso(HIST, SERIES, vacio={"SPY US Equity"}))
    assert res.estado == "rojo" and not (raiz / "logs/cronos.lock").exists()


@pytest.mark.lento
def test_verificar_regresion_en_raiz_real(tmp_path):
    ok, detalle = verificar_regresion(RAIZ, cache=tmp_path / "regresion.json")
    assert ok, detalle
    assert json.loads((tmp_path / "regresion.json").read_text(encoding="utf-8"))["ok"] is True
