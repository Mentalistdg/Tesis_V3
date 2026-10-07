import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.lento
def test_produccion_reproduce_tesis():
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "produccion_lstm.py"],
        cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    ).stdout
    assert "Predicciones identicas a las de la tesis" in out
    assert "Reproduccion EXACTA de los resultados publicados" in out
    assert "+395.4%" in out
