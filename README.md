# Predicción del S&P 500 con Deep Learning y Triple Screen de Elder

**Autor:** David González Canón
**Fecha:** Diciembre 2025
**Institución:** Maestría en Inteligencia de Negocios para Finanzas

---

## Descripción del Proyecto

Pipeline de Machine Learning para la predicción de retornos del S&P 500, combinando:

- **Sistema Triple Screen de Alexander Elder** (análisis técnico clásico)
- **Estimadores de volatilidad OHLC** (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
- **21 modelos de ML/DL** incluyendo DLinear, TCN, TFT, N-BEATS, XGBoost, LightGBM
- **92 variables base de Bloomberg** transformadas en **647 características**

---

## Resultados Principales

| Modelo | Retorno | Sharpe | Max Drawdown | Costos Trans. | Retorno Neto |
|--------|---------|--------|--------------|---------------|--------------|
| **DLinear** | 238.0% | 0.865 | -46.8% | 2.96% | 228.1% |
| Ensemble (3 modelos) | 95.7% | 0.931 | -22.8% | - | - |
| Buy & Hold (benchmark) | 96.2% | 0.846 | -25.4% | 0% | 96.2% |

**Período de prueba:** Octubre 2020 - Diciembre 2025 (1,302 días)

---

## Estructura del Proyecto

```
Tesis_V3/
│
├── data/                              # Datos
│   ├── bloomberg_triple_screen_core.csv   # Dataset principal (647 features, 45MB)
│   ├── BLOOMBERG_RAW_DATA.csv             # Datos raw de Bloomberg (92 variables)
│   ├── BLOOMBERG_TICKER_MAPPING.csv       # Mapeo de tickers Bloomberg
│   ├── RAW_bil_risk_free.csv              # Tasa libre de riesgo diaria
│   ├── spy_intraday_30min_max.csv         # Datos intraday para vol OHLC
│   └── dataset_metadata.json              # Metadata del dataset
│
├── scripts/                           # Scripts de Python
│   ├── ml_pipeline_darts_extended.py      # Pipeline principal (19 modelos, posiciones discretas)
│   ├── feature_engineering_hedge_fund.py  # Ingeniería de 647 features
│   ├── elder_indicators_multifreq.py      # Sistema Triple Screen de Elder
│   ├── generate_app_data.py               # Genera datos para app web
│   ├── validate_final_dataset.py          # Validación del dataset
│   ├── tesis_data_config.py               # Configuración de variables
│   └── academic_enhancements.py           # Análisis estadístico (Gu et al. 2020)
│
├── models/                            # Modelos entrenados
│   ├── best_pipeline_bloomberg.pkl        # Mejor modelo sklearn
│   └── model_info_bloomberg.json          # Metadata del modelo
│
├── results/                           # Resultados
│   ├── model_comparison_extended.csv      # Comparación de 21 modelos
│   ├── pipeline_summary_extended.json     # Resumen del pipeline
│   ├── comparison_improvements.csv        # Mejoras (DD control, ensemble)
│   ├── academic_enhancements_results.json # Tests de significancia estadística
│   ├── feature_importance_permutation.csv # Importancia de features
│   └── walk_forward_expanding.csv         # Walk-forward validation
│
├── paper/                             # Paper académico
│   ├── paper_triple_screen_ml.tex         # Paper LaTeX (español, ~50 páginas)
│   ├── paper_triple_screen_ml.pdf         # Paper compilado
│   ├── generate_paper_figures.py          # Script generador de figuras
│   ├── figures/                           # 8 figuras PNG
│   │   ├── fig1_cumulative_returns.png
│   │   ├── fig2_position_distribution.png
│   │   ├── fig3_drawdown.png
│   │   └── ...
│   └── tables/                            # 4 tablas LaTeX
│
├── _archive/                          # Archivos archivados (no esenciales)
│
└── README.md                          # Este archivo
```

---

## Ejecución del Pipeline

### 1. Pipeline Principal (19 modelos)

```bash
python scripts/ml_pipeline_darts_extended.py
```

Entrena 19 modelos con estrategia de posiciones discretas {-3, -1, 0, +1, +3}:
- **Sklearn:** Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM
- **Darts:** DLinear, N-BEATS, TCN, TFT, TiDE, NLinear, TSMixer, N-HiTS
- **Time Series:** AutoARIMA, AutoETS, Prophet, GARCH

### 2. Generar Datos para App Web

```bash
python scripts/generate_app_data.py
```

### 3. Análisis Académico

```bash
python scripts/academic_enhancements.py
```

Ejecuta:
- Bootstrap Sharpe Ratio con intervalos de confianza
- Test de significancia del Alpha (Jensen)
- Walk-forward validation (12 ventanas)
- Análisis de regímenes de mercado
- Importancia de features (SHAP, permutation)

### 4. Generar Figuras del Paper

```bash
cd paper && python generate_paper_figures.py
```

### 5. Compilar Paper

```bash
cd paper
pdflatex paper_triple_screen_ml.tex
pdflatex paper_triple_screen_ml.tex  # Segunda pasada para referencias
```

---

## Dataset: 647 Features

### Variables Base de Bloomberg (92 variables)

| Categoría | Código | Cantidad | Ejemplos |
|-----------|--------|----------|----------|
| Mercado | M1-M18 | 18 | SPY, QQQ, IWM, XLF, TLT |
| Económicas | E1-E10 | 10 | GDP, NFP, CPI, PMI |
| Tasas | I1-I20 | 20 | FDTR, Treasuries 2Y-30Y, Spreads |
| Commodities | P1-P13 | 13 | CL1 (Oil), GC1 (Gold), DXY |
| Volatilidad | V1-V13 | 13 | VIX, MOVE, Term Structure |
| Sentimiento | S1-S11 | 11 | AAII, Put/Call, TRIN |
| Calendario | D1-D7 | 7 | Lunes, Viernes, Fin de mes |

### Transformaciones (647 features totales)

| Tipo | Descripción |
|------|-------------|
| Retornos multi-horizonte | 1d, 5d, 21d, 63d, 126d, 252d |
| Indicadores técnicos | RSI, MACD, Bollinger %B, ATR |
| Triple Screen Elder | MACD semanal, Force Index, Elder Ray, Impulse |
| Volatilidad OHLC | Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| Correlaciones cross-asset | SPY-TLT, SPY-GLD, SPY-VIX |
| Rezagos y estadísticas | lag 1-7, MA 5-63, std móvil |

---

## Costos de Transacción

Se asume **15 basis points (0.15%)** por unidad de cambio de posición.

| Modelo | Turnover Anual | Costos | Retorno Neto |
|--------|----------------|--------|--------------|
| DLinear | 3.8x | 2.96% | 228.1% |
| TiDE | 323.9x | 250.9% | **-84.5%** |

**Hallazgo clave**: El turnover destruye la rentabilidad de modelos complejos. DLinear funciona porque mantiene posiciones estables.

---

## Variable Target

```python
target = market_forward_excess_returns = retorno_dia_siguiente - tasa_libre_riesgo
```

| Estadística | Valor |
|-------------|-------|
| Media | 0.028% diario |
| Std | 1.22% |
| % Positivos | 54.1% |

**Nota**: El target ya tiene `shift(-1)` aplicado. NO hay data leakage.

---

## Uso del Dataset

```python
import pandas as pd

# Cargar datos
df = pd.read_csv('data/bloomberg_triple_screen_core.csv', index_col=0, parse_dates=True)

# Excluir columnas no-features
exclude = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
           'market_forward_excess_returns']
feature_cols = [c for c in df.columns if c not in exclude]

# Features y target
X = df[feature_cols]
y = df['market_forward_excess_returns']

# Split temporal (NUNCA shuffle)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Features: {len(feature_cols)}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

---

## Dependencias

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
darts>=0.26
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
```

---

## Referencias Académicas

- **Fama, E. F. (1970)**. Efficient Capital Markets. *Journal of Finance*.
- **Gu, S., Kelly, B., & Xiu, D. (2020)**. Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*.
- **Elder, A. (1993)**. Trading for a Living. *Wiley*.
- **Zeng, A. et al. (2023)**. Are Transformers Effective for Time Series Forecasting? *AAAI*.
- **Yang, D. & Zhang, Q. (2000)**. Drift-Independent Volatility Estimation. *Journal of Business*.
- **Garman, M. & Klass, M. (1980)**. On the Estimation of Security Price Volatilities. *Journal of Business*.

---

## Archivos Archivados

La carpeta `_archive/` contiene versiones anteriores y archivos temporales que se mantienen por referencia:

**Scripts archivados:**
- `ml_pipeline_bloomberg.py` - Pipeline sklearn anterior (superseded por ml_pipeline_darts_extended.py)
- `unified_model_comparison.py` - Comparación de modelos anterior
- `calculate_proper_returns.py`, `forecasting_quick.py`, `prepare_data.py`
- `robust_imputation.py`, `intraday_proxy_features.py`, `leakage_diagnostic.py`

**Datos archivados:**
- `bloomberg_triple_screen_full.csv` - Versión anterior del dataset
- `hull_dataset_*` - Referencias antiguas

---

## Licencia

Proyecto académico para la Maestría en Inteligencia de Negocios para Finanzas.

---

**Última actualización:** Diciembre 2025
