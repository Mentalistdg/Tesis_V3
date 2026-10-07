"""Punto de entrada: python -m pipeline.cronos actualizar [--activo spx] [--ensayo] [--hasta AAAA-MM-DD].

Codigos de salida: 0 = verde/amarillo (o corrida en curso), 2 = rojo, 1 = error inesperado.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.calendario import siguiente_habil
from pipeline.configuracion import cargar_config, cargar_series
from pipeline.empalmar import empalmar, macro_a, version_b
from pipeline.exportar import exportar_app, registrar_emision, reporte_periodo, reporte_texto
from pipeline.extraer import extraer, extraer_publicaciones_completas, guardar_descarga, ultimo_dia_oficial
from pipeline.features import construir_features
from pipeline.io import escribir_atomico, escribir_csv_atomico
from pipeline.senal import POS_A_SENAL, cargar_modelo, entrenar_meta, generar_senales, predecir
from pipeline.validar import Resultado, validar

RAIZ = Path(__file__).resolve().parents[1]
LOCK_VENCE_S = 30 * 60
PRED_REFERENCIA = ("2025-12-11", -0.018574)   # prediccion de la tesis para validar la regresion


class CorridaEnCurso(Exception):
    """Ya hay otra corrida activa (lock vigente)."""


def _log(dir_raiz: Path, msg: str) -> None:
    p = Path(dir_raiz) / "logs" / "pipeline.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


class _Lock:
    def __init__(self, dir_raiz: Path):
        self.path = Path(dir_raiz) / "logs" / "cronos.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                ts = json.loads(self.path.read_text(encoding="utf-8"))["ts"]
            except Exception:
                ts = 0
            try:
                pid = json.loads(self.path.read_text(encoding="utf-8"))["pid"]
            except Exception:
                pid = -1
            if time.time() - ts < LOCK_VENCE_S and _pid_vivo(pid):
                raise CorridaEnCurso(f"otra corrida en curso (pid {pid}, lock {self.path})")
            self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time()}, f)
        return self

    def __exit__(self, *exc):
        if self.path.exists():
            self.path.unlink()


def _pid_vivo(pid: int) -> bool:
    """True si el proceso existe (Windows: OpenProcess + GetExitCodeProcess)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    codigo = ctypes.c_ulong()
    try:
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(codigo))) and codigo.value == 259   # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _estado_corrida(dir_raiz: Path, estado: str, causa: str, info: dict) -> None:
    """Estado de la ultima corrida para la app (se escribe siempre, tambien en rojo o error)."""
    datos = {"hora": datetime.now().isoformat(timespec="seconds"), "estado": estado, "causa": causa, **info}
    escribir_atomico(Path(dir_raiz) / "app" / "backend" / "data" / "estado_corrida.json",
                     json.dumps(datos, ensure_ascii=False, indent=1))


def _hash_entorno(dir_raiz: Path) -> str:
    import sklearn, talib, torch  # noqa: E401
    h = hashlib.sha256()
    archivos = sorted((dir_raiz / "pipeline").rglob("*.py")) + sorted((dir_raiz / "activos").rglob("*"))
    for a in archivos:
        if a.is_file():
            h.update(a.relative_to(dir_raiz).as_posix().encode())
            h.update(a.read_bytes())
    h.update(f"{pd.__version__}|{np.__version__}|{torch.__version__}|{sklearn.__version__}|{talib.__version__}".encode())
    return h.hexdigest()


def verificar_regresion(dir_raiz: Path, cache: Path | None = None) -> tuple[bool, str]:
    """Con la historia congelada, las features deben ser las del entrenamiento y la prediccion la de la tesis.
    Se cachea por hash de codigo + activos + versiones."""
    dir_raiz = Path(dir_raiz)
    cache = Path(cache) if cache else dir_raiz / "logs" / "regresion.json"
    h = _hash_entorno(dir_raiz)
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("hash") == h and c.get("ok"):
            return True, "cacheada"
    spx = dir_raiz / "activos" / "spx"
    columnas = json.loads((spx / "columnas_dataset.json").read_text(encoding="utf-8"))
    ds = construir_features(pd.read_csv(spx / "historia_congelada.csv"), columnas)
    ref = pd.read_csv(dir_raiz / "data" / "bloomberg_triple_screen_core.csv", low_memory=False)
    a = ds.iloc[:len(ref)].drop(columns=["date"]).to_numpy(dtype=float)
    b = ref.drop(columns=["date"]).to_numpy(dtype=float)
    ok_feat = a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True))
    cfg = cargar_config(spx)
    pred = float(predecir(cargar_modelo(spx / "modelo", ds, cfg["fin_entrenamiento"]), ds).loc[PRED_REFERENCIA[0]])
    ok_pred = abs(pred - PRED_REFERENCIA[1]) < 1e-6
    ok = ok_feat and ok_pred
    detalle = f"features identicas={ok_feat}; prediccion {PRED_REFERENCIA[0]}={pred:.6f} (tesis {PRED_REFERENCIA[1]})"
    escribir_atomico(cache, json.dumps({"hash": h, "ok": ok, "detalle": detalle,
                                        "fecha": datetime.now().isoformat(timespec="seconds")}))
    return ok, detalle


def actualizar(dir_raiz: Path, activo: str, cliente, ahora: datetime, ensayo: bool = False,
               hasta: date | None = None, verificar: bool = True) -> tuple[Resultado, dict]:
    dir_raiz = Path(dir_raiz)
    try:
        lock = _Lock(dir_raiz).__enter__()
    except CorridaEnCurso as e:
        _log(dir_raiz, f"[{activo}] corrida omitida: {e}")
        raise
    try:
        _log(dir_raiz, f"[{activo}] inicio corrida{' (ensayo)' if ensayo else ''}")
        try:
            res, info = _actualizar(dir_raiz, activo, cliente, ahora, ensayo, hasta, verificar)
        except Exception as e:
            if not ensayo:
                estado = "rojo" if type(e).__name__ == "ErrorBloomberg" else "error"
                _estado_corrida(dir_raiz, estado, f"{type(e).__name__}: {e}", {"activo": activo})
            raise
        _log(dir_raiz, f"[{activo}] estado={res.estado} {info} {' | '.join(res.mensajes)}")
        if not ensayo:
            _estado_corrida(dir_raiz, res.estado, " | ".join(res.mensajes), {"activo": activo, **info})
        return res, info
    finally:
        lock.__exit__(None, None, None)


def _actualizar(dir_raiz, activo, cliente, ahora, ensayo, hasta, verificar):
    spx = dir_raiz / "activos" / activo
    datos = dir_raiz / "data" / activo
    cfg, series = cargar_config(spx), cargar_series(spx)
    columnas = json.loads((spx / "columnas_dataset.json").read_text(encoding="utf-8"))
    ancla = pd.Timestamp(cfg["ancla_inicial"])
    ruta_raw = datos / "raw_extendido.csv"
    raw_previo = pd.read_csv(ruta_raw if ruta_raw.exists() else spx / "historia_congelada.csv")

    if verificar:
        ok, detalle = verificar_regresion(dir_raiz)
        if not ok:
            return Resultado("rojo", [f"falla la prueba de regresion contra la tesis: {detalle}"]), {}

    fin = hasta or ultimo_dia_oficial(ahora, cfg)
    fechas_prev = pd.to_datetime(raw_previo["date"])
    atras = cfg["solapamiento_dias_habiles"] + cfg["cola_mutable_dias_habiles"]
    inicio = fechas_prev.iloc[max(0, len(fechas_prev) - atras)].date()
    d = extraer(series, inicio, fin, cliente)
    if not ensayo:
        guardar_descarga(d, datos / "descargas" / datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    nuevo = empalmar(raw_previo, d, series, ancla, cfg["cola_mutable_dias_habiles"])
    res = validar(raw_previo, nuevo, d, series, cfg)
    if len(d.calendario) and d.calendario[-1].date() < fin:
        res.rojo(f"Bloomberg aun no publica la barra de SPY del {fin} (ultimo dato {d.calendario[-1].date()}); "
                 "no se emite senal para no mostrar una vencida")
    if res.estado == "rojo":
        return res, {"ultimo_dato": str(nuevo["date"].iloc[-1]), "fin_esperado": str(fin)}

    ruta_pub = datos / "publicaciones_macro.csv"
    if ruta_pub.exists():
        previas = pd.read_csv(ruta_pub, parse_dates=["periodo", "publicacion"], encoding="utf-8")
    else:
        previas = extraer_publicaciones_completas(series, cliente, fin)
    pubs = pd.concat([previas, d.publicaciones], ignore_index=True)
    pubs = pubs.drop_duplicates(["columna_cruda", "periodo"], keep="last").sort_values(["columna_cruda", "periodo"])
    macro = series.loc[series.tipo == "macro", "columna_cruda"]
    primera = pubs.groupby("columna_cruda").periodo.min()
    sin_historia = [c for c in macro if c not in primera.index or primera[c] > ancla - pd.Timedelta(days=365 * 5)]
    if sin_historia:
        res.rojo(f"historia macro incompleta (no se guarda publicaciones_macro.csv): {sin_historia}")
        return res, {"ultimo_dato": str(nuevo["date"].iloc[-1])}

    raw_a = macro_a(nuevo, pubs, series, ancla)
    raw_b = version_b(raw_a, pubs, series, ancla)
    ds_a, ds_b = construir_features(raw_a, columnas), construir_features(raw_b, columnas)
    fin_ent = cfg["fin_entrenamiento"]
    m = cargar_modelo(spx / "modelo", ds_a, fin_ent)
    meta = entrenar_meta(m, ds_a, fin_ent)
    s_a = generar_senales(m, meta, ds_a, fin_ent, realista=True)
    s_b = generar_senales(m, meta, ds_b, fin_ent, realista=True)
    rep = reporte_periodo(s_a, s_b, cfg, desde=cfg["ancla_inicial"])
    esperado = siguiente_habil(pd.Timestamp(nuevo["date"].iloc[-1]).date()).strftime("%Y-%m-%d")
    if s_b.date.iloc[-1] != esperado or s_a.date.iloc[-1] != esperado:
        res.rojo(f"la senal quedo para {s_b.date.iloc[-1]} y deberia ser para {esperado} "
                 "(fila del ultimo dia descartada al construir features)")
        return res, {"ultimo_dato": str(nuevo["date"].iloc[-1])}
    info = {"ultimo_dato": str(nuevo["date"].iloc[-1]), "fecha_senal": s_b.date.iloc[-1],
            "senal_b": POS_A_SENAL[int(s_b.posicion.iloc[-1])], "senal_a": POS_A_SENAL[int(s_a.posicion.iloc[-1])]}
    if ensayo:
        return res, info

    # Orden de escritura: cada archivo es atomico; raw_extendido va ultimo y marca la corrida como completa.
    escribir_csv_atomico(ds_a, datos / "dataset_A.csv")
    escribir_csv_atomico(ds_b, datos / "dataset_B.csv")
    recalc = s_b.assign(senal_b=s_b.posicion.map(POS_A_SENAL), senal_a=s_a.posicion.map(POS_A_SENAL).values)
    escribir_csv_atomico(recalc, datos / "senales_recalculadas.csv")
    escribir_csv_atomico(pubs.assign(periodo=pubs.periodo.dt.strftime("%Y-%m-%d"),
                                     publicacion=pubs.publicacion.dt.strftime("%Y-%m-%d")), ruta_pub)
    escribir_csv_atomico(nuevo, ruta_raw)
    info["emitida"] = registrar_emision(dir_raiz / "logs" / "senales_emitidas.csv", activo, s_b, s_a, res)
    escribir_atomico(dir_raiz / "logs" / f"reporte_{activo}.txt", reporte_texto(rep))
    escribir_atomico(dir_raiz / "logs" / f"reporte_{activo}.json", json.dumps(rep, ensure_ascii=False, indent=1))
    exportar_app(dir_raiz / "app" / "backend" / "data", activo, s_b, s_a, res, rep, cfg)
    return res, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cronos")
    ap.add_argument("comando", choices=["actualizar", "regresion"])
    ap.add_argument("--activo", default="spx")
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--hasta", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    a = ap.parse_args(argv)
    try:
        if a.comando == "regresion":
            ok, detalle = verificar_regresion(RAIZ)
            print(("OK " if ok else "FALLA ") + detalle)
            return 0 if ok else 2
        from pipeline.bloomberg import ClienteBloomberg, ErrorBloomberg
        try:
            res, info = actualizar(RAIZ, a.activo, ClienteBloomberg(), datetime.now().astimezone(), a.ensayo, a.hasta)
        except ErrorBloomberg as e:
            _log(RAIZ, f"[{a.activo}] estado=rojo {e}")
            print(f"ROJO: {e}")
            return 2
        print(f"Estado: {res.estado.upper()} | {info}")
        for msg in res.mensajes:
            print(f"  - {msg}")
        rep = RAIZ / "logs" / f"reporte_{a.activo}.txt"
        if not a.ensayo and res.estado != "rojo" and rep.exists():
            print(rep.read_text(encoding="utf-8"))
        return 2 if res.estado == "rojo" else 0
    except CorridaEnCurso as e:
        print(str(e))
        return 0
    except Exception:
        _log(RAIZ, "ERROR inesperado:\n" + traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
