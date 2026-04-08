# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION COMPLETA DE TODAS LAS TABLAS Y FIGURAS DEL PAPER
================================================================================
Master runner que ejecuta en orden todos los scripts de generacion y
replicacion. Diseñado para que el profesor pueda replicar todos los
numeros del paper con un solo comando.

Ejecuta:
  1. paper/generate_paper_tables.py     -- Genera 8 tablas .tex
  2. paper/generate_paper_figures.py    -- Genera 8 figuras .pdf
  3. paper/scripts/replicar_seccion4_resultados.py  -- Seccion 4
  4. paper/scripts/replicar_seccion5_da_paradox.py   -- Seccion 5
  5. paper/scripts/replicar_seccion6_riesgo.py       -- Seccion 6
  6. paper/scripts/replicar_seccion7_validacion.py   -- Seccion 7
  7. paper/scripts/replicar_seccion8_costos.py       -- Seccion 8

Prerequisitos:
  - Haber ejecutado el pipeline completo (build_dataset, train_models,
    optimize_and_backtest) para que existan los archivos en results/ y models/
  - Haber ejecutado paper/update_backend_data.py para que existan los
    archivos en app/backend/data/ (necesarios para figuras)

USAGE:
    python paper/scripts/replicar_todo.py

================================================================================
Author: David Gonzalez Canon
================================================================================
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER_DIR = os.path.join(BASE_DIR, "paper")
SCRIPTS_DIR = os.path.join(PAPER_DIR, "scripts")

# Python ejecutable (usa el mismo que esta corriendo este script)
PYTHON = sys.executable

# Lista de scripts a ejecutar, en orden
SCRIPTS = [
    ("Tablas LaTeX (.tex)", os.path.join(PAPER_DIR, "generate_paper_tables.py")),
    ("Figuras (.pdf)", os.path.join(PAPER_DIR, "generate_paper_figures.py")),
    ("Seccion 4: Resultados", os.path.join(SCRIPTS_DIR, "replicar_seccion4_resultados.py")),
    ("Seccion 5: DA Paradox", os.path.join(SCRIPTS_DIR, "replicar_seccion5_da_paradox.py")),
    ("Seccion 6: Riesgo", os.path.join(SCRIPTS_DIR, "replicar_seccion6_riesgo.py")),
    ("Seccion 7: Validacion", os.path.join(SCRIPTS_DIR, "replicar_seccion7_validacion.py")),
    ("Seccion 8: Costos", os.path.join(SCRIPTS_DIR, "replicar_seccion8_costos.py")),
]


def run_script(label, script_path):
    """Ejecuta un script Python y retorna (exito, duracion, mensaje)."""
    if not os.path.exists(script_path):
        return False, 0, f"Archivo no encontrado: {script_path}"

    start = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "-u", script_path],
            cwd=BASE_DIR,
            capture_output=False,
            timeout=600,  # 10 minutos maximo
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            return True, elapsed, "OK"
        else:
            return False, elapsed, f"Exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, elapsed, "TIMEOUT (>600s)"
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)


def main():
    print("=" * 90)
    print("REPLICACION COMPLETA DE TABLAS Y FIGURAS DEL PAPER")
    print("=" * 90)
    print(f"Directorio base: {BASE_DIR}")
    print(f"Python: {PYTHON}")
    print(f"Scripts a ejecutar: {len(SCRIPTS)}")
    print()

    # Verificar que existen los archivos del pipeline
    required_files = [
        os.path.join(BASE_DIR, "results", "final_long_only_backtest.json"),
        os.path.join(BASE_DIR, "results", "backtest_detail.pkl"),
        os.path.join(BASE_DIR, "models", "trained_artifacts.pkl"),
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Faltan archivos del pipeline. Ejecute primero:")
        print("  python scripts/build_dataset.py")
        print("  python scripts/train_models.py")
        print("  python scripts/optimize_and_backtest.py")
        print()
        for f in missing:
            print(f"  FALTA: {f}")
        return

    # Ejecutar cada script
    results = []
    total_start = time.time()

    for i, (label, script_path) in enumerate(SCRIPTS, 1):
        print()
        print(f"{'#'*90}")
        print(f"# [{i}/{len(SCRIPTS)}] {label}")
        print(f"# {script_path}")
        print(f"{'#'*90}")
        print()

        success, elapsed, msg = run_script(label, script_path)
        results.append((label, success, elapsed, msg))

        if success:
            print(f"\n>>> [{label}] OK ({elapsed:.1f}s)")
        else:
            print(f"\n>>> [{label}] ERROR: {msg} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # Resumen final
    print()
    print("=" * 90)
    print("RESUMEN DE EJECUCION")
    print("=" * 90)
    print(f"  {'#':>3}  {'Script':<35} {'Status':>8} {'Tiempo':>10}")
    print("  " + "-" * 60)

    n_ok = 0
    n_fail = 0
    for i, (label, success, elapsed, msg) in enumerate(results, 1):
        status = "OK" if success else "FAIL"
        if success:
            n_ok += 1
        else:
            n_fail += 1
        print(f"  {i:>3}  {label:<35} {status:>8} {elapsed:>9.1f}s")

    print("  " + "-" * 60)
    print(f"  Total: {n_ok} exitosos, {n_fail} fallidos, {total_elapsed:.1f}s")
    print()

    # Verificar archivos generados
    generated_files = [
        os.path.join(BASE_DIR, "results", "risk_metrics_detail.json"),
        os.path.join(BASE_DIR, "results", "statistical_validation.json"),
    ]
    print("  Archivos generados:")
    for f in generated_files:
        exists = os.path.exists(f)
        status = "OK" if exists else "FALTA"
        print(f"    [{status}] {os.path.relpath(f, BASE_DIR)}")

    print()
    if n_fail == 0:
        print("  TODOS LOS SCRIPTS EJECUTADOS EXITOSAMENTE.")
    else:
        print(f"  ATENCION: {n_fail} scripts fallaron. Revise los errores arriba.")
    print("=" * 90)


if __name__ == "__main__":
    main()
