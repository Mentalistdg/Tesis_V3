# Sistema de Prediccion de Retornos S&P 500

## Autor: David Gonzalez Canon
## Fecha: Enero 2026

---

# ESTRUCTURA DEL PROYECTO

Tesis_V3/
  requirements.txt                     - Dependencias Python
  scripts/                             - SCRIPTS DEL PIPELINE
    demo_pipeline_prueba.py            - Paso 0: Verificacion anti-leakage
    build_dataset.py                   - Paso 1: Feature engineering
    train_models.py                    - Paso 2: Entrenamiento (23 modelos)
    optimize_model_params.py           - Paso 3: Optimizacion umbrales
    final_long_only_backtest_new.py    - Paso 4: Backtest final
  data/                                - DATOS
    BLOOMBERG_RAW_DATA.csv             - Input (92 variables Bloomberg)
    bloomberg_triple_screen_core.csv   - Output Paso 1 (647 features)
  models/                              - MODELOS ENTRENADOS
    trained_artifacts.pkl              - Output Paso 2 (23 modelos)
    checkpoints/                       - Checkpoints individuales
  results/                             - RESULTADOS
    optimal_model_params.json          - Output Paso 3 (umbrales optimos)
    final_long_only_backtest.json      - Output Paso 4 (metricas)
    long_only_equity_curves.json       - Output Paso 4 (curvas)
  compartir_profesor/                  - Copia standalone para profesor
  _archive/                            - Archivos historicos

---

# FLUJO DE EJECUCION

INSTALACION:
    pip install -r requirements.txt

PASO 1: python scripts/build_dataset.py
PASO 2: python scripts/train_models.py
PASO 3: python scripts/optimize_model_params.py
PASO 4: python scripts/final_long_only_backtest_new.py

---

# REPRODUCIBILIDAD

Semillas fijas (SEED=42) para resultados identicos.
