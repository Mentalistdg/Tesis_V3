# Sistema de Prediccion de Retornos S&P 500
# Aplicacion de Machine Learning

## Autor: David Gonzalez Canon
## Fecha: Enero 2026

---

# RESUMEN EJECUTIVO

Este proyecto implementa un sistema de prediccion de retornos del S&P 500 que combina:
- **92 variables financieras** de Bloomberg (precios, tasas, volatilidad, divisas, commodities)
- **647 features** generados mediante feature engineering
- **23 modelos** de Machine Learning y Deep Learning
- **Estrategia de trading** Long-Only con posiciones discretas {0, +1, +3}

El sistema esta disenado con **estrictos controles anti-leakage** para garantizar validez academica.

---

# TABLA DE CONTENIDOS

1. [Instalacion y Ejecucion](#instalacion-y-ejecucion)
2. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
3. [PASO 0: Verificacion Anti-Leakage](#paso-0-verificacion-anti-leakage)
4. [PASO 1: Feature Engineering](#paso-1-feature-engineering)
5. [PASO 2: Entrenamiento de Modelos](#paso-2-entrenamiento-de-modelos)
6. [PASO 3: Optimizacion de Umbrales (DETALLADO)](#paso-3-optimizacion-de-umbrales)
7. [PASO 4: Backtest Final](#paso-4-backtest-final)
8. [Prevencion de Data Leakage](#prevencion-de-data-leakage)
9. [Reproducibilidad](#reproducibilidad)
10. [Referencias Academicas](#referencias-academicas)

---

# INSTALACION Y EJECUCION

## Requisitos
- **Python:** 3.10 o superior
- **Espacio en disco:** ~3-5 GB (principalmente PyTorch)
- **Tiempo total de ejecucion:** ~45-90 minutos

## Comandos

```bash
# 1. Crear y activar entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline en orden
python scripts/demo_pipeline_prueba.py        # Paso 0: Verificacion (opcional)
python scripts/build_dataset.py               # Paso 1: Feature engineering
python scripts/train_models.py                # Paso 2: Entrenamiento (~30-60 min)
python scripts/optimize_model_params.py       # Paso 3: Optimizacion
python scripts/final_long_only_backtest_new.py # Paso 4: Backtest final
```

---

# ARQUITECTURA DEL PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO DE DATOS DEL PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │  BLOOMBERG_RAW_DATA.csv  │   92 variables financieras diarias
    │     (Datos crudos)       │   Periodo: 2015-2025
    └────────────┬─────────────┘
                 │
                 ▼ [PASO 1: build_dataset.py]
    ┌──────────────────────────┐
    │  bloomberg_triple_       │   647 features + target
    │  screen_core.csv         │   Features: indicadores tecnicos, ratios, lags
    │     (Dataset ML)         │   Target: forward_returns (shift -1)
    └────────────┬─────────────┘
                 │
                 ▼ [PASO 2: train_models.py]
    ┌──────────────────────────┐
    │  trained_artifacts.pkl   │   23 modelos entrenados
    │     (Modelos)            │   Predicciones de train y test
    └────────────┬─────────────┘   Metadata del experimento
                 │
                 ▼ [PASO 3: optimize_model_params.py]
    ┌──────────────────────────┐
    │  optimal_model_params    │   Umbrales optimos por modelo
    │     .json                │   Intervalos de confianza (Bootstrap)
    └────────────┬─────────────┘   Caracteristicas de predicciones
                 │
                 ▼ [PASO 4: final_long_only_backtest_new.py]
    ┌──────────────────────────┐
    │  final_long_only_        │   Metricas: Return, Sharpe, MaxDD
    │  backtest.json           │   Comparacion vs SPY Buy & Hold
    │  long_only_equity_       │   Curvas de equity diarias
    │  curves.json             │
    └──────────────────────────┘
```

---

# PASO 0: VERIFICACION ANTI-LEAKAGE

**Script:** `scripts/demo_pipeline_prueba.py`

**Proposito:** Demostrar matematicamente que no existe data leakage en el sistema.

## Que es Data Leakage?

Data leakage ocurre cuando informacion del futuro se filtra al entrenamiento del modelo, invalidando cualquier resultado. Es el error metodologico mas grave en ML financiero.

## Que verifica este script?

1. **Alineacion temporal de features:** Muestra paso a paso como cada feature usa SOLO datos pasados
2. **Calculo del target:** Demuestra que `forward_returns` usa `shift(-1)` correctamente
3. **Diagrama temporal:** Visualiza la separacion entre features (t y anteriores) y target (t+1)

## Ejemplo de verificacion

```
Tiempo:        t-4     t-3     t-2     t-1      t        t+1
               |       |       |       |        |        |
Close:       $100    $101    $102    $103     $104     $105
               |       |       |       |        |        |
               +-------+-------+-------+--------+        |
                              |                          |
               FEATURES: Solo datos hasta t              |
               - SMA_5 = mean(Close[t-4:t])              |
               - RSI = calculo con Close[t-13:t]         |
               - lag_1 = Close[t-1]                      |
                              |                          |
                              +--------------------------+
                                         |
                              TARGET: Retorno de t a t+1
                              = (Close[t+1] - Close[t]) / Close[t]
                              = ($105 - $104) / $104 = 0.96%
```

---

# PASO 1: FEATURE ENGINEERING

**Script:** `scripts/build_dataset.py`

**Input:** `data/BLOOMBERG_RAW_DATA.csv` (92 variables)
**Output:** `data/bloomberg_triple_screen_core.csv` (647 features)

## Variables de Entrada (92)

| Categoria | Variables |
|-----------|-----------|
| **Precios SPX** | Open, High, Low, Close, Volume |
| **Tasas Treasury** | 2Y, 5Y, 10Y, 30Y yields |
| **Volatilidad** | VIX, VIX3M |
| **Commodities** | Oro (GLD), Petroleo (USO) |
| **Divisas** | EUR/USD, USD/JPY, GBP/USD |
| **Indices Globales** | DAX, FTSE, Nikkei, HSI |
| **Sectores** | XLF, XLE, XLK, XLV, XLI, etc. |

## Features Generados (647)

### 1. Retornos (multiples horizontes)
```python
# Retornos pasados (NO LEAKAGE: miran hacia atras)
ret_1d  = close.pct_change(1)    # Retorno 1 dia
ret_5d  = close.pct_change(5)    # Retorno 1 semana
ret_21d = close.pct_change(21)   # Retorno 1 mes
ret_63d = close.pct_change(63)   # Retorno 1 trimestre
```

### 2. Medias Moviles
```python
# SMA y EMA (NO LEAKAGE: ventanas hacia atras)
SMA_5   = close.rolling(5).mean()
SMA_20  = close.rolling(20).mean()
SMA_50  = close.rolling(50).mean()
SMA_200 = close.rolling(200).mean()
EMA_12  = close.ewm(span=12).mean()
EMA_26  = close.ewm(span=26).mean()
```

### 3. Indicadores de Momentum
```python
# RSI (NO LEAKAGE: usa datos pasados)
RSI_14 = calcular_rsi(close, 14)

# MACD (NO LEAKAGE: basado en EMAs pasadas)
MACD = EMA_12 - EMA_26
MACD_signal = MACD.ewm(span=9).mean()

# Stochastic (NO LEAKAGE: basado en high/low pasados)
K = (close - low_14) / (high_14 - low_14)
```

### 4. Volatilidad
```python
# ATR - Average True Range (NO LEAKAGE)
ATR_14 = true_range.rolling(14).mean()

# Bollinger Bands (NO LEAKAGE)
BB_upper = SMA_20 + 2 * close.rolling(20).std()
BB_lower = SMA_20 - 2 * close.rolling(20).std()
```

### 5. Ratios Cross-Asset
```python
# Ratios relativos (NO LEAKAGE)
SPX_vs_VIX = close_spx / vix
Gold_vs_SPX = close_gold / close_spx
Yield_curve = treasury_10y - treasury_2y
```

### 6. Lags (valores pasados)
```python
# Lags explicitos (NO LEAKAGE por definicion)
feature_lag_1 = feature.shift(1)   # Valor de ayer
feature_lag_5 = feature.shift(5)   # Valor de hace 5 dias
```

## Target: Forward Returns

```python
# ESTE ES EL UNICO shift(-1) EN TODO EL SISTEMA
# Representa el retorno FUTURO que queremos predecir
forward_returns = close.pct_change(1).shift(-1)

# En el dia t, forward_returns[t] = retorno de t a t+1
# El modelo NUNCA ve este valor como input
```

---

# PASO 2: ENTRENAMIENTO DE MODELOS

**Script:** `scripts/train_models.py`

**Input:** `data/bloomberg_triple_screen_core.csv`
**Output:** `models/trained_artifacts.pkl`

## Division Temporal de Datos

```
|<------------ TRAIN (80%) ------------>|<----- TEST (20%) ----->|
|                                       |                        |
2015                                   Oct 2020              Dec 2025
                                       (split)

IMPORTANTE: Division TEMPORAL, nunca aleatoria
            El modelo entrena con pasado, predice futuro
```

## Modelos Entrenados (23)

### Machine Learning Clasico (5)
| Modelo | Descripcion | Hiperparametros |
|--------|-------------|-----------------|
| Ridge | Regresion lineal con regularizacion L2 | alpha=1.0 |
| Lasso | Regresion lineal con regularizacion L1 | alpha=0.001 |
| ElasticNet | Combinacion L1+L2 | alpha=0.001, l1_ratio=0.5 |
| RandomForest | Ensemble de arboles | n_estimators=100, max_depth=10 |
| GradientBoosting | Boosting de arboles | n_estimators=100, learning_rate=0.1 |

### Gradient Boosting Avanzado (3)
| Modelo | Descripcion | Hiperparametros |
|--------|-------------|-----------------|
| XGBoost | Extreme Gradient Boosting | n_estimators=100, max_depth=6 |
| LightGBM | Light Gradient Boosting | n_estimators=100, num_leaves=31 |
| CatBoost | Categorical Boosting | iterations=100, depth=6 |

### Series de Tiempo (4)
| Modelo | Descripcion |
|--------|-------------|
| AutoARIMA | ARIMA con seleccion automatica de ordenes |
| ExponentialSmoothing | Suavizado exponencial (ETS) |
| Theta | Metodo Theta de Assimakopoulos |
| SeasonalNaive | Baseline estacional |

### Modelos Especializados (2)
| Modelo | Descripcion |
|--------|-------------|
| Prophet | Modelo de Facebook para series de tiempo |
| GARCH | Modelo de volatilidad condicional |

### Deep Learning - Darts (5)
| Modelo | Arquitectura |
|--------|--------------|
| DLinear | Descomposicion lineal para series de tiempo |
| N-BEATS | Neural Basis Expansion Analysis |
| N-HiTS | Hierarchical interpolation for Time Series |
| TCN | Temporal Convolutional Network |
| TFT | Temporal Fusion Transformer |

### RNN Bidireccionales - Darts (2)
| Modelo | Arquitectura |
|--------|--------------|
| Bi-LSTM | LSTM bidireccional |
| Bi-GRU | GRU bidireccional |

### Hibridos/Atencion - PyTorch (2)
| Modelo | Arquitectura |
|--------|--------------|
| CNN-LSTM | Convolucion + LSTM |
| LSTM+Attention | LSTM con mecanismo de atencion Luong |

## Contenido del Archivo de Salida

```python
trained_artifacts.pkl = {
    'metadata': {
        'forward_returns_test': array,  # Retornos reales del periodo test
        'risk_free_test': array,         # Tasa libre de riesgo diaria
        'y_test': array,                 # Target (para calcular accuracy)
        'test_dates': array,             # Fechas del periodo test
    },
    'models': {
        'Ridge': {
            'model': trained_model,           # Modelo sklearn/pytorch
            'train_predictions': array,       # Predicciones en train
            'test_predictions': array,        # Predicciones en test
        },
        'XGBoost': { ... },
        # ... 23 modelos
    }
}
```

---

# PASO 3: OPTIMIZACION DE UMBRALES

**Script:** `scripts/optimize_model_params.py`

**Input:** `models/trained_artifacts.pkl`
**Output:** `results/optimal_model_params.json`

## Motivacion Teorica

Cada modelo de ML tiene caracteristicas unicas en sus predicciones:
- **Distribucion diferente:** Algunos predicen valores extremos, otros conservadores
- **Sesgo diferente:** Algunos son sistematicamente alcistas o bajistas
- **Precision direccional diferente:** Varia entre 45% y 55%

**Problema:** Usar los mismos umbrales para todos los modelos es suboptimo.

**Solucion:** Encontrar umbrales INDIVIDUALES que maximicen el rendimiento ajustado por riesgo de CADA modelo.

## Conversion de Predicciones a Posiciones

### Sistema de Percentiles Rolling

Las predicciones continuas del modelo se convierten a posiciones discretas usando percentiles:

```
Prediccion continua del modelo (ej: 0.0023)
            │
            ▼
Calcular percentil dentro de ventana rolling de 63 dias
            │
            ▼
Mapear percentil a posicion discreta {0, +1, +3}
```

### Umbrales Long-Only

El sistema usa 2 umbrales para la estrategia Long-Only:

| Umbral | Descripcion | Rango tipico |
|--------|-------------|--------------|
| `q_long_extreme` | Top X% para posicion +3 (UPRO) | 5-30% |
| `q_long_moderate` | Top Y% para posicion +1 (SPY) | 20-60% |

### Ejemplo con q_long_extreme=10, q_long_moderate=30

```
Percentil de la prediccion actual (dentro de ultimos 63 dias):

  100% ─────────────────────────────────────────────────
         │                                              │
   90% ──┼──────────────────────────────────────────────┤ ← thresh_3x = 100 - 10 = 90
         │     POSICION +3 (UPRO - 3x Long)             │
         │     Prediccion en TOP 10%                    │
   70% ──┼──────────────────────────────────────────────┤ ← thresh_1x = 100 - 30 = 70
         │     POSICION +1 (SPY - 1x Long)              │
         │     Prediccion en TOP 30% (pero no top 10%)  │
         │                                              │
    0% ──┼──────────────────────────────────────────────┤
         │     POSICION 0 (CASH - Neutral)              │
         │     Prediccion fuera del TOP 30%             │
         │     NO HAY POSICIONES CORTAS                 │
         └──────────────────────────────────────────────┘
```

### Codigo de la Funcion Principal (Long-Only)

```python
def quantile_position_long_only(predictions, window=63, q_3x=10, q_1x=30):
    """
    Convierte predicciones a posiciones Long-Only {0, 1, 3}
    ANTI-LEAKAGE: La ventana SOLO mira hacia atras
    """
    positions = np.zeros(len(predictions))

    # Calcular umbrales de percentil
    thresh_3x = 100 - q_3x   # ej: 100 - 10 = 90 -> top 10%
    thresh_1x = 100 - q_1x   # ej: 100 - 30 = 70 -> top 30%

    for i in range(len(predictions)):
        # Ventana: SOLO datos pasados + actual (NUNCA futuro)
        start = max(0, i - window + 1)
        window_preds = predictions[start:i+1]

        # Percentil de la prediccion actual
        current_pred = predictions[i]
        percentile = np.mean(window_preds <= current_pred) * 100

        # Mapear a posicion LONG-ONLY
        if percentile >= thresh_3x:
            positions[i] = 3    # Top 10% -> UPRO (3x)
        elif percentile >= thresh_1x:
            positions[i] = 1    # Top 30% -> SPY (1x)
        else:
            positions[i] = 0    # Resto -> CASH (sin posicion)

    return positions
```

## Grid Search: Busqueda de Umbrales Optimos

### Espacio de Busqueda

```python
param_grid = {
    'q_long_extreme':  [5, 10, 15, 20, 25, 30],     # 6 valores (top X% para +3)
    'q_long_moderate': [20, 30, 40, 50, 60],        # 5 valores (top Y% para +1)
}

# Total combinaciones: 6 × 5 = 30 por modelo
# Total evaluaciones: 30 × 23 modelos = 690
```

### Restricciones de Validez

No todas las combinaciones son validas:

```python
# q_long_moderate DEBE ser mayor que q_long_extreme
# (no puedes tener top 30% para +1 si top 10% es para +3)
if q_long_moderate <= q_long_extreme:
    skip  # Combinacion invalida
```

### Funcion Objetivo: Sharpe Ratio Ajustado

```python
def evaluate_params(predictions, forward_returns, risk_free, params):
    # 1. Generar posiciones Long-Only con estos umbrales
    positions = quantile_position_long_only(predictions,
                                            q_long_extreme=params['q_long_extreme'],
                                            q_long_moderate=params['q_long_moderate'])

    # 2. Aplicar filtros de gestion de riesgo
    positions = apply_position_filter(positions)      # Anti-churning
    positions = apply_volatility_targeting(positions) # Vol targeting
    positions = apply_drawdown_control(positions)     # DD control

    # 3. Calcular retornos realistas (con costos)
    returns = calculate_realistic_returns(positions, forward_returns,
                                          risk_free, config)

    # 4. Calcular metricas
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(returns)

    # 5. Score con penalizacion por drawdown excesivo
    score = sharpe
    if max_dd < -0.20:  # Drawdown > 20%
        score -= 0.5    # Penalizar

    return score
```

## Gestion de Riesgo Integrada

### 1. Filtro Anti-Churning

Evita trades frecuentes que generan costos sin valor:

```python
def apply_position_filter(positions, min_change=1):
    """
    Solo ejecuta trade si el cambio de posicion es >= min_change

    Ejemplo:
      - Cambio de 0 a 1: permitido (cambio = 1)
      - Cambio de 1 a 3: permitido (cambio = 2)
      - No hay cambio de 0 a 0 (se mantiene posicion anterior)
    """
```

### 2. Volatility Targeting

Ajusta posiciones para mantener volatilidad constante:

```python
def apply_volatility_targeting(positions, returns, target_vol=0.15):
    """
    Formula: vol_scalar = target_vol / recent_vol

    - Si mercado muy volatil (recent_vol > target): reducir posicion
    - Si mercado calmado (recent_vol < target): aumentar posicion

    Referencia: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
    """
```

### 3. Control de Drawdown

Reduce posicion automaticamente cuando hay perdidas:

```python
def apply_drawdown_control(positions, cumulative_returns, limits):
    """
    Limites por nivel de drawdown:

    | Drawdown | Accion                      |
    |----------|----------------------------|
    | < 10%    | Operar normalmente          |
    | 10-15%   | Maximo 1x (reducir 3x a 1x) |
    | 15-20%   | Maximo 1x (sin apalancamiento)|
    | > 20%    | Ir a cash (proteger capital)|

    ANTI-LEAKAGE: Usamos drawdown hasta AYER, no hasta hoy
    cumulative_lag = [1.0] + cumulative[:-1]
    """
```

## Calculo de Retornos Realistas

### Formula de Retorno por Posicion

```
Para posicion p en el dia t:

    return[t] = rf[t] + p × (market_return[t] - rf[t])

Donde:
    - rf = tasa libre de riesgo (Treasury 3M / 252)
    - p = posicion {0, +1, +3} (Long-Only)
    - market_return = retorno del S&P 500

Ejemplos:
    - p = +3 (UPRO), market = +1%, rf = 0.02%:
      return = 0.02% + 3 × (1% - 0.02%) = 2.96%

    - p = +1 (SPY), market = +1%, rf = 0.02%:
      return = 0.02% + 1 × (1% - 0.02%) = 1.00%

    - p = 0 (CASH):
      return = rf = 0.02%
```

### Costos Incluidos

| Costo | Descripcion | Valores |
|-------|-------------|---------|
| **Expense Ratio** | Costo anual del ETF (prorrateado diario) | UPRO: 0.91%, SPY: 0.09% |
| **Bid-Ask Spread** | Costo por trade | UPRO: 0.05%, SPY: 0.02% |
| **Volatility Drag** | Perdida por rebalanceo diario en ETFs 3x | ~0.5 × (lev² - lev) × σ² |

### Formula Volatility Drag

Los ETFs apalancados rebalancean diariamente, causando una perdida sistematica:

```
vol_drag = 0.5 × (leverage² - |leverage|) × variance

Para leverage = 3:
    vol_drag = 0.5 × (9 - 3) × σ² = 3 × σ²

Si σ diaria = 1%, vol_drag = 3 × 0.0001 = 0.03% diario
Anualizado: ~7.5% de perdida adicional
```

## Bootstrap: Intervalos de Confianza

### Metodologia

El bootstrap es una tecnica estadistica para estimar la incertidumbre de una metrica.

```python
def bootstrap_confidence(returns, n_bootstrap=5000, confidence=0.95):
    """
    Proceso:
    1. Resamplear retornos con reemplazo N veces
    2. Calcular Sharpe Ratio en cada muestra
    3. Obtener distribucion de Sharpes posibles
    4. Extraer percentiles para intervalo de confianza
    """
    sharpes = []

    for _ in range(n_bootstrap):
        # Resamplear con reemplazo
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = returns[sample_idx]

        # Calcular Sharpe del sample
        sharpe = (mean(sample) - rf) / std(sample) * sqrt(252)
        sharpes.append(sharpe)

    # Intervalo de confianza 95%
    ci_lower = percentile(sharpes, 2.5)
    ci_upper = percentile(sharpes, 97.5)

    # Probabilidad de Sharpe > 0
    prob_positive = mean(sharpes > 0)

    return ci_lower, ci_upper, prob_positive
```

### Interpretacion de Resultados

```
Modelo: Ridge
  Sharpe: 0.48
  IC 95%: [-0.30, 1.39]
  P(Sharpe > 0): 78%

Interpretacion:
  - El Sharpe puntual es 0.48
  - Con 95% de confianza, el Sharpe real esta entre -0.30 y 1.39
  - Hay 78% de probabilidad de que el Sharpe sea positivo

NOTA: IC amplio indica alta incertidumbre (tipico en finanzas)
```

## Archivo de Salida

```json
{
  "Ridge": {
    "params": {
      "q_long_extreme": 20,
      "q_long_moderate": 30
    },
    "expected_sharpe": 0.48,
    "expected_return": 0.856,
    "expected_max_dd": -0.386,
    "bootstrap": {
      "sharpe_mean": 0.47,
      "sharpe_ci_lower": -0.30,
      "sharpe_ci_upper": 1.39,
      "sharpe_prob_positive": 0.78
    },
    "characteristics": {
      "directional_accuracy": 0.521,
      "pred_mean": 0.02932,
      "pred_std": 0.04902,
      "pct_positive": 71.4
    }
  },
  // ... 22 modelos mas
}
```

**Nota:** El backtest final solo usa los parametros `q_long_extreme` y `q_long_moderate` para la estrategia Long-Only.

---

# PASO 4: BACKTEST FINAL

**Script:** `scripts/final_long_only_backtest_new.py`

**Inputs:**
- `models/trained_artifacts.pkl`
- `results/optimal_model_params.json`

**Outputs:**
- `results/final_long_only_backtest.json`
- `results/long_only_equity_curves.json`

## Estrategia Long-Only

La estrategia usa exclusivamente posiciones **Long-Only** (sin posiciones cortas):

```
Posicion    Instrumento    Descripcion                          Cuando se usa
─────────────────────────────────────────────────────────────────────────────────
   +3       UPRO          3x Long S&P 500 (muy alcista)        Prediccion en TOP 10%
   +1       SPY           1x Long S&P 500 (alcista)            Prediccion en TOP 30%
    0       CASH          Efectivo (neutral/proteccion)        Prediccion fuera del TOP 30%
```

**Justificacion academica para Long-Only:**
1. El mercado tiene sesgo alcista historico (~7% anual)
2. Las posiciones cortas tienen costos asimetricos (borrow fees, short squeezes)
3. El riesgo de estar corto es teoricamente ilimitado
4. Mayor liquidez y menores costos en ETFs long

## Benchmark: SPY Buy & Hold

Todas las estrategias se comparan contra el benchmark:

```
SPY Buy & Hold: Comprar SPY el primer dia y mantener hasta el final

Metricas del benchmark (periodo test Oct 2020 - Dec 2025):
  - Total Return: +102.2%
  - Sharpe Ratio: 0.882
  - Max Drawdown: -25.4%
```

## Metricas Calculadas

| Metrica | Formula | Interpretacion |
|---------|---------|----------------|
| **Total Return** | ∏(1 + r) - 1 | Retorno compuesto total |
| **Sharpe Ratio** | (r - rf) / σ × √252 | Retorno ajustado por riesgo |
| **Max Drawdown** | max(peak - current) / peak | Maxima caida desde pico |
| **% tiempo en 3x** | dias_3x / total_dias | Exposicion apalancada |
| **% tiempo en cash** | dias_cash / total_dias | Tiempo fuera del mercado |
| **N Trades** | cambios de posicion | Frecuencia de trading |

## Resultados Esperados

```
================================================================================
RESULTADOS FINALES - LONG-ONLY (ordenado por Return)
================================================================================

Rank  Modelo               Return   Sharpe    MaxDD     %3x     %1x   %Cash  Trades
──────────────────────────────────────────────────────────────────────────────────────
1     Ridge               +318.8%*    1.037    38.6%   29.5%    8.5%   62.0%     273
2     GradientBoosting    +260.7%*    0.673    49.3%   80.3%    5.8%   13.9%     175
3     LightGBM            +133.8%*    0.622    43.2%   21.4%   36.7%   41.9%     515
4     RandomForest        +118.3%*    0.664    28.5%   11.5%   35.7%   52.9%     332
5     CatBoost            +111.6%*    0.639    25.9%   15.7%   12.4%   71.9%     449
6     NBEATS              +105.1%*    0.725    39.2%    9.1%   46.2%   44.7%     549
7     Prophet             +102.2%     0.944    24.0%    6.4%   32.1%   61.5%     372
8     BiGRU                +96.5%     0.659    25.5%    7.6%   23.1%   69.3%     314
...
      SPY B&H             +102.2%     0.882    25.4%

[MODELOS QUE BATEN SPY: 6/23]

* = Retorno superior al benchmark
```

**Interpretacion:**
- 6 de 23 modelos superan el benchmark SPY Buy & Hold
- Ridge es el mejor modelo con +318.8% de retorno y Sharpe de 1.037
- La estrategia Long-Only es conservadora: los modelos pasan 40-80% del tiempo en cash

---

# PREVENCION DE DATA LEAKAGE

## Medidas Implementadas

### 1. Features Solo Usan Datos Pasados

```python
# CORRECTO: pct_change mira hacia atras
returns = close.pct_change(5)  # Retorno de hace 5 dias a hoy

# CORRECTO: rolling mira hacia atras
sma = close.rolling(20).mean()  # Promedio de ultimos 20 dias

# CORRECTO: shift positivo mira hacia atras
lag = close.shift(1)  # Precio de ayer
```

### 2. Target Usa shift(-1) Correctamente

```python
# UNICO shift negativo en todo el sistema
forward_returns = close.pct_change(1).shift(-1)

# Esto significa: el target del dia t es el retorno del dia t+1
# El modelo PREDICE el futuro, no lo VE como input
```

### 3. Division Temporal (No Aleatoria)

```python
# CORRECTO: Division temporal
train = data[:'2020-10-01']
test = data['2020-10-01':]

# INCORRECTO: Division aleatoria (causaria leakage)
# train, test = train_test_split(data, shuffle=True)  # MAL!
```

### 4. Drawdown Control Usa Datos de Ayer

```python
# El drawdown de HOY no puede usarse para decidir posicion de HOY
# (seria leakage: usamos info de hoy para decidir hoy)

# CORRECTO: Usar drawdown hasta AYER
cumulative_lag = np.concatenate([[1.0], cumulative[:-1]])
positions = apply_drawdown_control(positions, cumulative_lag, limits)
```

### 5. Percentiles Rolling Miran Hacia Atras

```python
for i in range(n):
    # Ventana: desde (i - window + 1) hasta i (inclusive)
    # NUNCA incluye i+1, i+2, etc.
    start = max(0, i - window + 1)
    window_preds = predictions[start:i+1]  # Solo pasado + hoy
```

---

# REPRODUCIBILIDAD

## Semillas Fijas

Todos los scripts usan semillas fijas para garantizar resultados identicos:

```python
# En train_models.py
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True

# En optimize_model_params.py
SEED = 42
np.random.seed(SEED)  # Para bootstrap

# En todos los modelos sklearn
model = RandomForestRegressor(random_state=42)
```

## Verificacion

Si ejecuta el pipeline 2 veces, obtendra **exactamente** los mismos resultados:
- Mismas predicciones
- Mismos umbrales optimos
- Mismos intervalos de confianza
- Mismas metricas finales

---

# REFERENCIAS ACADEMICAS

## Metodologia

1. **Elder, A.** (1993). *Trading for a Living*. John Wiley & Sons.
   - Estrategia Triple Screen original

2. **Moskowitz, T., Ooi, Y. H., & Pedersen, L. H.** (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228-250.
   - Volatility targeting

3. **Harvey, C. R., Liu, Y., & Zhu, H.** (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
   - Multiple testing en finanzas

## Modelos

4. **Oreshkin, B. N., et al.** (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. *ICLR 2020*.

5. **Challu, C., et al.** (2022). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI 2023*.

6. **Lim, B., et al.** (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*.

## ETFs Apalancados

7. **Cheng, M., & Madhavan, A.** (2009). The Dynamics of Leveraged and Inverse Exchange-Traded Funds. *Journal of Investment Management*, 7(4), 43-62.
   - Volatility drag en ETFs apalancados

---

# ESTRUCTURA DE CARPETAS

```
compartir_profesor/
│
├── requirements.txt                      # Dependencias Python
├── README.md                             # Este documento
│
├── scripts/                              # SCRIPTS DEL PIPELINE
│   ├── demo_pipeline_prueba.py           # Paso 0: Verificacion anti-leakage
│   ├── build_dataset.py                  # Paso 1: Feature engineering
│   ├── train_models.py                   # Paso 2: Entrenamiento de modelos
│   ├── optimize_model_params.py          # Paso 3: Optimizacion de umbrales
│   └── final_long_only_backtest_new.py   # Paso 4: Backtest final
│
├── data/                                 # DATOS
│   ├── BLOOMBERG_RAW_DATA.csv            # Input: 92 variables Bloomberg
│   └── bloomberg_triple_screen_core.csv  # Output Paso 1: 647 features
│
├── models/                               # MODELOS ENTRENADOS
│   └── trained_artifacts.pkl             # Output Paso 2: 23 modelos
│
└── results/                              # RESULTADOS
    ├── optimal_model_params.json         # Output Paso 3: umbrales optimos
    ├── param_grid_search_results.csv     # Output Paso 3: grid search completo
    ├── final_long_only_backtest.json     # Output Paso 4: metricas finales
    └── long_only_equity_curves.json      # Output Paso 4: curvas de equity
```

---

## Contacto

**David Gonzalez Canon**
Enero 2026
