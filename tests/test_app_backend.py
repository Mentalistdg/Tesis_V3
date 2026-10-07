import json
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import RAIZ


def _cliente(tmp_path, monkeypatch, con_senales=True):
    datos = tmp_path / "data"
    datos.mkdir()
    shutil.copy(RAIZ / "app/backend/data/models_summary.json", datos)
    shutil.copy(RAIZ / "app/backend/data/market_data.json", datos)
    if con_senales:
        (datos / "senales.json").write_text(json.dumps({"activos": [{"activo": "spx", "senal_b": "SPY", "nombre": "S&P 500 ñ"}]}), encoding="utf-8")
        (datos / "vivo.json").write_text(json.dumps({
            "summary": [{"model": "LSTM_Attention_vivo_B", "activo": "spx", "total_return": 0.1}],
            "daily": {"LSTM_Attention_vivo_B": {"activo": "spx", "dates": ["2026-01-02"], "positions": [1]},
                      "LSTM_Attention_vivo_A": {"activo": "spx", "dates": ["2026-01-02"], "positions": [0]}}}), encoding="utf-8")
    monkeypatch.setenv("CRONOS_APP_DATA", str(datos))
    import importlib
    import app.backend.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_api_senales(tmp_path, monkeypatch):
    r = _cliente(tmp_path, monkeypatch).get("/api/senales")
    assert r.status_code == 200 and r.json()["activos"][0]["activo"] == "spx"
    assert r.json()["activos"][0]["nombre"] == "S&P 500 ñ"


def test_api_historial(tmp_path, monkeypatch):
    r = _cliente(tmp_path, monkeypatch).get("/api/senales/spx/historial")
    assert r.status_code == 200 and set(r.json()) == {"B", "A"}


def test_api_historial_404_activo_inexistente(tmp_path, monkeypatch):
    assert _cliente(tmp_path, monkeypatch).get("/api/senales/btc/historial").status_code == 404


def test_models_incluye_vivo(tmp_path, monkeypatch):
    c = _cliente(tmp_path, monkeypatch)
    nombres = [m["model"] for m in c.get("/api/models").json()["models"]]
    assert "LSTM_Attention_vivo_B" in nombres
    assert c.get("/api/models/LSTM_Attention_vivo_B").json()["positions"] == [1]


def test_sin_senales_json_responde_503(tmp_path, monkeypatch):
    r = _cliente(tmp_path, monkeypatch, con_senales=False).get("/api/senales")
    assert r.status_code == 503 and "corridas" in r.json()["detail"]


def test_estado_corrida_y_senal_vencida(tmp_path, monkeypatch):
    c = _cliente(tmp_path, monkeypatch)
    datos = tmp_path / "data"
    sj = json.loads((datos / "senales.json").read_text(encoding="utf-8"))
    sj["activos"][0]["fecha"] = "2020-01-02"
    (datos / "senales.json").write_text(json.dumps(sj), encoding="utf-8")
    (datos / "estado_corrida.json").write_text(json.dumps({"estado": "rojo", "causa": "Terminal sin sesion", "hora": "x"}), encoding="utf-8")
    r = c.get("/api/senales").json()
    assert r["corrida"]["estado"] == "rojo" and r["activos"][0]["vencida"] is True
