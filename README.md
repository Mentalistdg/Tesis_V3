# Dataset Triple Pantalla Elder - Listo para Entrenamiento ML

## Resumen Ejecutivo

Dataset completo para prediccion de retornos del S&P 500 usando el sistema Triple Pantalla de Alexander Elder, con **647 features** derivados de 92 variables base de Bloomberg, incluyendo estimadores de volatilidad intradiaria estilo hedge fund.

| Metrica | Valor |
|---------|-------|
| **Filas** | 6,507 (dias de trading) |
| **Columnas** | 647 features |
| **Periodo** | 2000-01-03 a 2025-12-12 |
| **Missing** | ~3% |
| **Variables Base Bloomberg** | 92 |
| **Target** | market_forward_excess_returns |

---

## Que se Hizo (Resumen del Pipeline)

### 1. Descarga de Datos de Bloomberg
- 92 variables base descargadas via Bloomberg API
- Variables recuperadas que se habian perdido en pipeline: V3, V12, V13, M12
- Variables de tasas adicionales descargadas: I10-I20

### 2. Feature Engineering (Estilo Hedge Fund)
- 500+ features derivados de variables base
- Retornos multi-horizonte (1d, 5d, 21d, 63d, 126d, 252d)
- Indicadores tecnicos (RSI, MACD, Bollinger, ATR)
- Cross-asset correlations
- Regime features

### 3. Sistema Triple Pantalla de Elder
- Pantalla 1 (Semanal): MACD, tendencia, impulse
- Pantalla 2 (Diario): Force Index, Elder Ray
- Senales combinadas: TS_BUY_SETUP, TS_SELL_SETUP, TS_SIGNAL

### 4. Datos Intradiarios
- Descargados barras de 30 minutos de Bloomberg (limite: 136 dias)
- Implementados **proxy features de volatilidad intradiaria** para periodo historico completo:
  - Parkinson Volatility (1980)
  - Garman-Klass Volatility (1980)
  - Rogers-Satchell Volatility (1991)
  - Yang-Zhang Volatility (2000)
  - Features de microestructura (gaps, velas, posicion del cierre)

### 5. Imputacion Robusta
- Forward-fill para precios (sin data leakage)
- Metodologia que preserva integridad temporal

---

## Estructura de Archivos

```
TRANSFER_TO_TRAINING_PC/
├── README.md                                    # Este archivo
│
├── data/
│   │
│   │  === DATASETS PRINCIPALES ===
│   ├── bloomberg_triple_screen_core.csv        # PRINCIPAL (647 cols, 44 MB)
│   ├── bloomberg_triple_screen_full.csv        # Con intradiarios extra
│   ├── bloomberg_triple_screen_recent.csv      # Solo datos recientes
│   │
│   │  === DATOS RAW DE BLOOMBERG ===
│   ├── hull_dataset_raw.csv                    # Datos crudos (98 cols)
│   ├── RAW_tesis_raw.csv                       # Raw 2015-2025
│   ├── RAW_improved_variables.csv              # Raw commodities 2000-2025
│   ├── RAW_bil_risk_free.csv                   # Tasa libre de riesgo
│   │
│   │  === DATOS INTRADIARIOS ===
│   ├── spy_intraday_30min_max.csv              # Barras 30 min (136 dias)
│   │
│   │  === VERIFICACION ===
│   ├── BLOOMBERG_RAW_VERIFICATION.csv          # Para verificar vs Bloomberg
│   ├── BLOOMBERG_TICKER_MAPPING.csv            # Mapeo codigo -> ticker
│   ├── VERIFICACION_RAPIDA.csv                 # Ultimos 30 dias
│   ├── verification_values.csv                 # Primeros/ultimos valores
│   └── variable_verification_report.json       # Reporte completo
│
└── scripts/
    ├── tesis_data_config.py                    # Configuracion de variables
    ├── feature_engineering_hedge_fund.py       # Feature engineering
    ├── elder_indicators_multifreq.py           # Indicadores Elder
    ├── robust_imputation.py                    # Imputacion
    ├── intraday_proxy_features.py              # Proxy volatilidad intradiaria
    ├── validate_final_dataset.py               # Validacion
    └── audit_complete_pipeline.py              # Auditoria
```

---

## Dataset Principal: bloomberg_triple_screen_core.csv

### Dimensiones
- **Filas**: 6,507 dias de trading
- **Columnas**: 647 features
- **Periodo**: 2000-01-03 a 2025-12-12

### Categorias de Features (647 total)

| Categoria | Prefijo | Cantidad | Descripcion |
|-----------|---------|----------|-------------|
| Mercado | M | ~50 | ETFs, indices, sectores |
| Economicas | E | ~40 | PIB, empleo, inflacion |
| Tasas | I | ~170 | Treasuries, spreads, credit |
| Commodities | P | ~20 | Oil, gold, indices |
| Volatilidad | V | ~40 | VIX, MOVE, term structure |
| Sentimiento | S | ~15 | AAII, put/call, breadth |
| Elder Semanal | W_ | ~15 | MACD, trend, impulse |
| Elder Diario | D_ | ~20 | Force Index, Elder Ray |
| Triple Screen | TS_ | ~7 | Senales combinadas |
| **Intradiario Proxy** | VOL_, INTRA_, GAP_, CANDLE_ | **38** | **Estimadores hedge fund** |
| SPY Features | SPY_ | ~60 | OHLCV y derivados |
| Otros | - | ~200 | Momentum, ratios, etc. |

---

## Features de Volatilidad Intradiaria (Metodologia Hedge Fund)

Cuando no hay datos intradiarios historicos, los hedge funds usan estimadores basados en OHLC:

### Estimadores de Volatilidad Implementados

| Estimador | Eficiencia vs Close-to-Close | Periodo |
|-----------|------------------------------|---------|
| Parkinson (1980) | 5x | 21d, 63d |
| Garman-Klass (1980) | 8x | 21d, 63d |
| Rogers-Satchell (1991) | Robusto a drift | 21d |
| Yang-Zhang (2000) | Maneja gaps overnight | 21d, 63d |

### Features de Microestructura

| Feature | Descripcion |
|---------|-------------|
| INTRA_RANGE_PCT | Rango del dia como % del precio |
| INTRA_CLOSE_POSITION | Donde cerro dentro del rango (0-1) |
| GAP_OVERNIGHT | Gap de apertura vs cierre anterior |
| GAP_VS_ATR | Gap normalizado por ATR |
| CANDLE_BODY_PCT | Tamano del cuerpo vs rango |
| PRICE_EFFICIENCY | Cuanto se movio vs cuanto recorrio |
| CLV | Close Location Value |
| BUYING_PRESSURE_21 | % dias cierre en parte alta |

### Cobertura de Datos Intradiarios

```
2000-01-03 a 2025-06-01: Proxy features (estimadores OHLC)
2025-06-02 a 2025-12-12: Datos reales 30 min + proxy
```

---

## Variable Target

```python
market_forward_excess_returns = retorno_dia_siguiente - tasa_libre_riesgo
```

| Estadistica | Valor |
|-------------|-------|
| Media | 0.028% diario |
| Std | 1.22% |
| Min | -11.9% |
| Max | +10.2% |
| % Positivos | 54.1% |

**IMPORTANTE**: El target ya tiene shift(-1) aplicado. NO hay data leakage.

---

## Variables Base de Bloomberg (92 Total)

### Resumen por Categoria

| Categoria | Variables | Cobertura |
|-----------|-----------|-----------|
| Mercado (M) | M1-M18 | 18/18 OK |
| Economicas (E) | E1-E10 | 10/10 OK |
| Tasas (I) | I1-I20 | 20/20 OK |
| Commodities (P) | P1-P13 | 13/13 OK |
| Volatilidad (V) | V1-V13 | 13/13 OK |
| Sentimiento (S) | S1-S11 | 11/11 OK |
| Dummy (D) | D1-D7 | 7/7 Calculadas |

### Variables con Cobertura Parcial (datos no disponibles antes de cierta fecha)

| Variable | Ticker | Desde | Cobertura |
|----------|--------|-------|-----------|
| V3 | VIX1M Index | 2008 | 69% |
| M12 | FNERTR Index | 2015 | 39% |
| I14 | USSWAP10 Index | 2011 | 53% |
| I15 | CDX IG | 2011 | 55% |
| I16 | CDX HY | 2011 | 55% |

---

## Sistema Triple Pantalla de Elder

### Pantalla 1 - Semanal (Tendencia)
| Feature | Descripcion |
|---------|-------------|
| W_MACD | MACD semanal (12,26,9) |
| W_MACD_HIST | Histograma MACD |
| W_TREND | Direccion tendencia (+1/-1) |
| W_IMPULSE | Sistema Impulse |

### Pantalla 2 - Diario (Osciladores)
| Feature | Descripcion |
|---------|-------------|
| D_FORCE_INDEX_2 | Force Index 2 dias |
| D_FORCE_INDEX_13 | Force Index 13 dias |
| D_BULL_POWER | Elder Ray Bull |
| D_BEAR_POWER | Elder Ray Bear |

### Senales Combinadas
| Feature | Descripcion |
|---------|-------------|
| TS_BUY_SETUP | Tendencia alcista + pullback |
| TS_SELL_SETUP | Tendencia bajista + rally |
| TS_SIGNAL | Senal final (+1, 0, -1) |
| TS_ALIGNMENT | Alineacion de pantallas |

---

## Uso para Entrenamiento

```python
import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/bloomberg_triple_screen_core.csv')
df['date'] = pd.to_datetime(df['date'])

# Columnas a excluir (no son features)
exclude = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
           'market_forward_excess_returns']

# Features y target
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]
y = df['market_forward_excess_returns']

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(df)}")

# Split temporal (NUNCA usar shuffle)
train = df[df['date'] < '2020-01-01']
val = df[(df['date'] >= '2020-01-01') & (df['date'] < '2023-01-01')]
test = df[df['date'] >= '2023-01-01']

print(f"Train: {len(train):,} ({train['date'].min().date()} - {train['date'].max().date()})")
print(f"Val:   {len(val):,} ({val['date'].min().date()} - {val['date'].max().date()})")
print(f"Test:  {len(test):,} ({test['date'].min().date()} - {test['date'].max().date()})")
```

---

## Verificacion de Datos contra Bloomberg

### Archivos para Verificacion

1. **VERIFICACION_RAPIDA.csv** - Ultimos 30 dias con tickers Bloomberg
2. **BLOOMBERG_TICKER_MAPPING.csv** - Mapeo de codigos a tickers
3. **hull_dataset_raw.csv** - Datos crudos sin transformar

### Como Verificar

```
En Bloomberg Terminal:

1. Para SPY: SPY US Equity <GO> → HP <GO>
2. Para VIX: VIX Index <GO> → HP <GO>
3. Para 10Y: USGG10YR Index <GO> → HP <GO>

Comparar valores con VERIFICACION_RAPIDA.csv
```

---

## Scripts de Referencia

| Script | Descripcion |
|--------|-------------|
| `ml_pipeline_bloomberg.py` | **PIPELINE DE ENTRENAMIENTO ML** |
| `tesis_data_config.py` | Definicion de 92 variables y tickers Bloomberg |
| `feature_engineering_hedge_fund.py` | Transformacion de variables a 500+ features |
| `elder_indicators_multifreq.py` | Implementacion Triple Pantalla Elder |
| `intraday_proxy_features.py` | Estimadores volatilidad intradiaria |
| `robust_imputation.py` | Imputacion preservando integridad temporal |
| `validate_final_dataset.py` | Validacion anti-leakage |
| `audit_complete_pipeline.py` | Auditoria completa de variables |

---

## Pipeline de Entrenamiento ML

El archivo `scripts/ml_pipeline_bloomberg.py` contiene el pipeline completo listo para ejecutar.

### Requisitos

```bash
pip install pandas numpy scikit-learn joblib xgboost lightgbm
```

### Ejecucion

```bash
cd TRANSFER_TO_TRAINING_PC
python scripts/ml_pipeline_bloomberg.py
```

### Que Hace el Pipeline

1. **Carga datos**: `bloomberg_triple_screen_core.csv` (647 features)
2. **Valida anti-leakage**: Verifica que no hay contaminacion de datos futuros
3. **Split temporal**: 80% train / 20% test (sin shuffle)
4. **Entrena modelos**:
   - Ridge, Lasso, ElasticNet
   - Random Forest, Gradient Boosting
   - XGBoost, LightGBM (si estan instalados)
5. **Evalua estrategia**: Sharpe ratio, retornos, direccion
6. **Guarda resultados**:
   - `models/best_pipeline_bloomberg.pkl` - Modelo entrenado
   - `models/model_info_bloomberg.json` - Metadata
   - `results/model_comparison_bloomberg.csv` - Comparacion

### Arquitectura Anti-Data Leakage

- Split temporal (no aleatorio)
- TimeSeriesSplit para cross-validation
- Preprocessing fit SOLO en train
- Features calculados solo con datos pasados

---

## Notas Importantes

1. **NO usar shuffle** - Los datos son series temporales, el orden importa

2. **Data Leakage** - El target ya tiene shift(-1), NO aplicar de nuevo

3. **Missing Values** - ~3%, ya imputados con forward-fill

4. **Volatilidad Intradiaria** - Usa proxy features para periodo historico, datos reales solo para ultimos 136 dias

5. **Variables Parciales** - Algunas variables (V3, M12, I14-I16) tienen datos desde fechas posteriores a 2000

6. **TS_SIGNAL** - Puede usarse como feature adicional o como filtro post-modelo

---

## Auditoria Final

| Check | Resultado |
|-------|-----------|
| Variables Bloomberg | 92/92 (100%) |
| Features totales | 647 |
| Data leakage | Ninguno detectado |
| Target verificado | Correlacion 0.9994 con t+1 |
| Proxy intradiarios | 38 features implementados |
| Datos intradiarios reales | 136 dias (30 min bars) |

---

## Generado

- **Fecha**: 2025-12-13
- **Fuente**: Bloomberg Terminal
- **Metodologia**: Triple Pantalla Elder + Proxy Intradiario Hedge Fund
- **Periodo**: 2000-01-03 a 2025-12-12
