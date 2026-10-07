# CRONOS — Paquete de Producción (LSTM + Attention)

Paquete autocontenido con el **modelo ganador y estadísticamente significativo** de la tesis
*"Predicción de retornos del S&P 500 con Triple Screen de Elder + Machine Learning"*
(David González Cañón, FEN — Universidad de Chile).

- **Modelo:** LSTM bidireccional (2 capas, hidden 128) + atención Luong (~1,4 M parámetros)
- **Resultados de la tesis (test oct 2020 – dic 2025, 1.301 días):** retorno +395,4%, Sharpe 1,319, max drawdown −20,5% (SPY Buy & Hold en el mismo periodo: +101,3%, Sharpe 0,657)
- **Significancia:** único modelo de los 23 con Diebold-Mariano (p = 0,011) y Clark-West significativos al 5%, y con IC bootstrap del Sharpe que excluye 0

## Cómo ejecutarlo

**Opción A (Windows, automático):** doble clic en `run.bat`
(crea el entorno virtual, instala dependencias y ejecuta; requiere Python 3.12 o superior instalado e internet la primera vez).

**Opción B (manual, cualquier sistema):**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate  (Linux/Mac)
pip install -r requirements.txt
python produccion_lstm.py
```

## Qué hace `produccion_lstm.py`

1. Carga el dataset real de Bloomberg (541 features, 6.507 días, ene 2000 – dic 2025) y replica el split temporal 80/20 de la tesis.
2. Carga los preprocesadores entrenados (`models/preprocessors.joblib`) y el checkpoint real del modelo (`models/LSTM_Attention.pt`) y **recalcula las predicciones con PyTorch** (inferencia genuina, no números guardados).
3. Verifica las predicciones recalculadas contra las originales de la tesis (`reference/resultados_esperados.json`).
4. Entrena el Meta-KNN de umbrales usando **solo datos de entrenamiento** y ejecuta el backtest Long-Only {CASH, SPY, UPRO} con el modelo completo de costos (expense ratio, bid-ask, volatility drag con volatilidad realizada).
5. Compara las métricas obtenidas con las publicadas en la tesis.
6. Emite la **señal de producción** del último día disponible: CASH, SPY (+1) o UPRO (+3).

Tiempo aproximado: 2–4 minutos en CPU.

## Contenido

| Archivo | Descripción |
|---|---|
| `produccion_lstm.py` | Script único de producción (inferencia + Meta-KNN + backtest + señal) |
| `models/LSTM_Attention.pt` | Checkpoint entrenado (state_dict + configuración) |
| `models/preprocessors.joblib` | Imputer + StandardScaler ajustados solo con datos de entrenamiento |
| `data/bloomberg_triple_screen_core.csv` | Dataset de 541 features (incluye las 5 columnas legacy `E20, P12, P13, S11, S12` que el checkpoint espera, restauradas desde el pipeline original) |
| `reference/resultados_esperados.json` | Predicciones y métricas publicadas, para verificación automática |
| `requirements.txt` | Dependencias con versiones fijadas (idénticas al entorno de la tesis) |
| `run.bat` | Lanzador para Windows |

## Notas

- El dataset llega hasta el **11 de diciembre de 2025** (datos Bloomberg congelados). La señal emitida corresponde a esa fecha. Para operar en vivo hay que regenerar el dataset con datos nuevos usando `build_dataset.py` del repositorio completo.
- La lógica del backtest es copia literal de `scripts/optimize_and_backtest.py` del repositorio, para garantizar reproducibilidad exacta. No se usa información futura en ningún punto (umbrales vía Meta-KNN entrenado solo con datos de entrenamiento; percentiles y filtros estrictamente retrospectivos).
- Semilla global 42; con las versiones fijadas de `requirements.txt` los resultados son reproducibles.
