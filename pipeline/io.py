"""Escritura atomica: archivo temporal en la misma carpeta + os.replace (con reintentos en Windows)."""
import os
import tempfile
import time
from pathlib import Path

import pandas as pd


def escribir_atomico(path: Path, contenido: bytes | str, reintentos: int = 5, espera_s: float = 2.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datos = contenido.encode("utf-8") if isinstance(contenido, str) else contenido
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(datos)
        for intento in range(1, reintentos + 1):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:          # archivo abierto (Excel, lector de la app)
                if intento == reintentos:
                    raise
                time.sleep(espera_s)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def escribir_csv_atomico(df: pd.DataFrame, path: Path) -> None:
    escribir_atomico(path, df.to_csv(index=False, lineterminator="\n"))
