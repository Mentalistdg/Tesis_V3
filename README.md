# Prediccion de Retornos del S\&P 500 mediante Aprendizaje Automatico

## Autor: David Gonzalez Canon
## Fecha: Abril 2026

---

## Resumen Ejecutivo

Este proyecto implementa un sistema completo de prediccion de retornos diarios del S\&P 500 que combina la metodologia *Triple Screen* de Elder (1993) con tecnicas modernas de *Machine Learning* y *Deep Learning*. El sistema transforma 92 variables financieras de Bloomberg --- precios, tasas, volatilidad, divisas, *commodities* e indices globales --- en 536 *features* mediante ingenieria de caracteristicas, y entrena 23 modelos que abarcan desde regresion lineal regularizada hasta *Transformers* y LSTMs con atencion.

La estrategia de inversion es exclusivamente *Long-Only* con posiciones discretas {0, +1, +3} instrumentadas a traves de CASH, SPY y UPRO respectivamente. Los umbrales de decision para cada modelo se determinan mediante Meta-KNN, un enfoque no-oraculo que utiliza exclusivamente informacion del periodo de entrenamiento para predecir los parametros optimos de cada dia de prueba.

El proyecto incluye ademas un *paper* academico completo en LaTeX con scripts de replicacion para cada seccion, y una aplicacion web interactiva (*dashboard*) construida con React y FastAPI para la visualizacion de resultados.

El resultado principal del sistema indica que 4 de 23 modelos superan el *benchmark* SPY *Buy \& Hold* (+102.2\%) en el periodo de prueba (octubre 2020 -- diciembre 2025). El mejor modelo --- LSTM+Attention --- alcanza un retorno acumulado de +400\% con un *Sharpe ratio* de 1.33.

---

## Tabla de Contenidos

1. [Instalacion y Requisitos](#instalacion-y-requisitos)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Pipeline de ML (3 Pasos)](#pipeline-de-ml-3-pasos)
4. [Prevencion de Data Leakage](#prevencion-de-data-leakage)
5. [Modelos Entrenados (23)](#modelos-entrenados-23)
6. [Resultados Principales](#resultados-principales)
7. [Paper Academico](#paper-academico)
8. [Aplicacion Web (CRONOS)](#aplicacion-web-cronos)
9. [Scripts de Replicacion](#scripts-de-replicacion)
10. [Reproducibilidad](#reproducibilidad)

---

## Instalacion y Requisitos

El sistema requiere Python 3.12 (recomendado), aproximadamente 3--5 GB de espacio en disco (principalmente por PyTorch y los modelos entrenados), y alrededor de 110 minutos de tiempo de ejecucion en CPU para el *pipeline* completo. Para la aplicacion web se necesita Node.js 18+, y para compilar el *paper* se requiere MiKTeX o TeX Live.

### Instalacion

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate            # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# NOTA: TA-Lib requiere una libreria C precompilada.
# En Windows, descargar el .whl correspondiente a tu version de Python desde:
# https://github.com/cgohlke/talib-build/releases
# Luego: pip install TA_Lib-0.4.xx-cpXX-cpXX-win_amd64.whl
```

### Ejecucion del Pipeline Completo

```bash
# Los 3 pasos deben ejecutarse en orden (cada uno depende del anterior)
python scripts/build_dataset.py              # Paso 1: 92 vars -> 536 features (~2 min)
python scripts/train_models.py               # Paso 2: Entrenar 23 modelos (~105 min CPU)
python scripts/optimize_and_backtest.py      # Paso 3: Meta-KNN + backtest (~3 min)

# Script opcional de verificacion anti-leakage
python scripts/demo_pipeline_prueba.py
```

---

## Estructura del Proyecto

```
Tesis_V3/
|
|-- data/                                    DATOS
|   |-- BLOOMBERG_RAW_DATA.csv               Input inmutable (92 variables Bloomberg)
|   |-- bloomberg_triple_screen_core.csv     Output Paso 1 (536 features + target)
|   |-- DICCIONARIO_VARIABLES.md             Documentacion de las 92 variables
|   +-- dataset_metadata.json               Metadata del dataset generado
|
|-- scripts/                                 PIPELINE PRINCIPAL (3 pasos)
|   |-- build_dataset.py                     Paso 1 -- Feature engineering
|   |-- train_models.py                      Paso 2 -- Entrenamiento de 23 modelos
|   |-- optimize_and_backtest.py             Paso 3 -- Meta-KNN + backtest final
|   +-- demo_pipeline_prueba.py              Verificacion anti-leakage (opcional)
|
|-- models/                                  MODELOS ENTRENADOS
|   |-- trained_artifacts.pkl                Metadata + predicciones (23 modelos)
|   +-- checkpoints/                         Checkpoints individuales (gitignored)
|       |-- sklearn/*.joblib
|       |-- darts/*/
|       +-- pytorch/*.pt
|
|-- results/                                 RESULTADOS DEL BACKTEST
|   |-- final_long_only_backtest.json        Metricas finales por modelo
|   |-- long_only_equity_curves.json         Curvas de equity diarias
|   |-- optimal_model_params.json            Umbrales optimos Meta-KNN
|   |-- backtest_detail.pkl                  Detalle diario (posiciones, costos, etc.)
|   |-- risk_metrics_detail.json             Metricas de riesgo (VaR, CVaR, drawdown)
|   +-- statistical_validation.json          Tests estadisticos (bootstrap, DM, CW, MCS)
|
|-- paper/                                   PAPER ACADEMICO
|   |-- paper_triple_screen_ml.tex           Archivo principal LaTeX
|   |-- paper_triple_screen_ml.pdf           PDF compilado
|   |-- compile.ps1                          Script de compilacion (PowerShell)
|   |-- Law.png                              Figura de distribucion de Benford
|   |-- sections/                            Secciones del paper (7 archivos .tex)
|   |-- tables/                              Tablas LaTeX generadas (8 archivos .tex)
|   |-- figures/                             Figuras PDF generadas (8 archivos .pdf)
|   |-- generate_paper_tables.py             Genera tablas .tex desde resultados
|   |-- generate_paper_figures.py            Genera figuras .pdf desde resultados
|   |-- update_backend_data.py               Pipeline -> JSON para la app web
|   +-- scripts/                             Scripts de replicacion (6 archivos)
|       |-- replicar_seccion4_resultados.py
|       |-- replicar_seccion5_da_paradox.py
|       |-- replicar_seccion6_riesgo.py
|       |-- replicar_seccion7_validacion.py
|       |-- replicar_seccion8_costos.py
|       +-- replicar_todo.py                 Ejecuta todos los scripts de replicacion
|
|-- app/                                     APLICACION WEB (CRONOS)
|   |-- CLAUDE.md                            Documentacion tecnica de la app
|   |-- LOGO.png                             Logo CRONOS
|   |-- backend/                             FastAPI (Python)
|   |   |-- main.py                          Servidor API
|   |   |-- requirements.txt                 Dependencias del backend
|   |   +-- data/                            JSON pre-computados para la app
|   |       |-- daily_data.json
|   |       |-- models_summary.json
|   |       |-- market_data.json
|   |       +-- regime_data.json
|   +-- frontend/                            React + TypeScript + Vite
|       |-- package.json
|       +-- src/
|           |-- App.tsx                      Router (6 paginas)
|           |-- pages/                       6 paginas de visualizacion
|           |-- components/                  Componentes reutilizables
|           |-- services/api.ts              Capa de comunicacion con API
|           |-- types/index.ts               Interfaces TypeScript
|           +-- utils/transactionCosts.ts    Modelo de costos de transaccion
|
|-- requirements.txt                         Dependencias Python (versiones fijas)
|-- .gitignore                               Archivos excluidos del repositorio
|-- CLAUDE.md                                Guia tecnica para desarrollo con IA
+-- README.md                                Este documento
```

---

## Pipeline de ML (3 Pasos)

El *pipeline* se compone de tres pasos secuenciales donde cada uno depende de la salida del anterior. El flujo de datos completo se describe a continuacion.

### Flujo de Datos

```
BLOOMBERG_RAW_DATA.csv (92 variables, Ene 2000 - Dic 2025)
         |
         v [Paso 1: build_dataset.py]
bloomberg_triple_screen_core.csv (6,507 obs x 536 features + target)
         |
         v [Paso 2: train_models.py]
trained_artifacts.pkl (23 modelos entrenados, predicciones train/test)
models/checkpoints/ (pesos individuales de cada modelo)
         |
         v [Paso 3: optimize_and_backtest.py]
final_long_only_backtest.json (metricas finales)
long_only_equity_curves.json (curvas de equity)
optimal_model_params.json (umbrales Meta-KNN)
backtest_detail.pkl (detalle diario)
risk_metrics_detail.json (VaR, CVaR, drawdown)
statistical_validation.json (bootstrap, DM, CW, MCS)
```

### Paso 1 -- *Feature Engineering* (`build_dataset.py`)

Este script transforma las 92 variables financieras crudas de Bloomberg en 536 *features* siguiendo la metodologia *Triple Screen*. Las categorias de *features* generados incluyen retornos a multiples horizontes (1, 5, 21 y 63 dias), medias moviles simples y exponenciales de multiples ventanas, indicadores de *momentum* (RSI, MACD, *Stochastic*), medidas de volatilidad (ATR, *Bollinger Bands*, estimadores de rango), ratios *cross-asset* (SPX/VIX, Gold/SPX), *spreads* de la curva de rendimiento del Tesoro, y *lags* de hasta 5 periodos de las variables mas relevantes.

La variable objetivo (*target*) se define como el retorno futuro a un dia --- `forward_returns = close.pct_change(1).shift(-1)` --- y constituye el **unico** `shift(-1)` en todo el sistema. Todos los *features* utilizan exclusivamente datos pasados, garantizando la ausencia de *data leakage*.

Cabe senalar que los primeros registros del *dataset* contienen valores NaN producidos por las ventanas *rolling* (por ejemplo, SMA\_200 requiere 200 dias de historia). Estos valores faltantes no se eliminan, sino que se imputan con la mediana durante el preprocesamiento del Paso 2.

### Paso 2 -- Entrenamiento de Modelos (`train_models.py`)

El segundo paso entrena 23 modelos sobre una division temporal 80/20, donde el periodo de entrenamiento abarca desde enero de 2000 hasta octubre de 2020, y el periodo de prueba comprende desde octubre de 2020 hasta diciembre de 2025.

```
|<------------ TRAIN (80%) ------------>|<----- TEST (20%) ----->|
|                                       |                        |
Ene 2000                               Oct 2020              Dic 2025
                                       (punto de corte)
```

La division es estrictamente temporal --- nunca aleatoria --- y el preprocesamiento (*StandardScaler*, *SimpleImputer*) se ajusta exclusivamente sobre datos de entrenamiento para evitar cualquier contaminacion.

El script implementa un sistema de *checkpoints* mediante el cual cada modelo se guarda individualmente y puede retomarse si se interrumpe el entrenamiento. La memoria se libera despues de cada modelo con `gc.collect()`. Los tiempos aproximados de entrenamiento en CPU son de 5 minutos para los modelos de ML clasico, 3 minutos para *Gradient Boosting*, 5 minutos para series de tiempo, 65 minutos para *Deep Learning* Darts (donde N-BEATS toma ~23 min y TFT ~32 min), y 25 minutos para los modelos PyTorch *custom*.

### Paso 3 -- Optimizacion y *Backtest* (`optimize_and_backtest.py`)

Este paso combina la optimizacion de umbrales, el *backtest* con costos reales y la validacion estadistica en un unico script.

#### Meta-KNN --- Optimizacion No-Oraculo de Umbrales

Cada modelo necesita dos umbrales `(q_ext, q_mod)` para convertir predicciones continuas en posiciones discretas. Las predicciones se transforman a percentiles *rolling* de 63 dias, y si el percentil supera `100 - q_ext` se asigna posicion +3 (UPRO, 3x apalancado), si supera `100 - q_mod` se asigna +1 (SPY, 1x largo), y en caso contrario se permanece en CASH (posicion 0).

El problema fundamental es que los umbrales optimos varian segun las condiciones del mercado, y utilizar umbrales fijos optimizados sobre todo el periodo de prueba constituiria *data leakage* (optimizacion oraculo). La solucion implementada --- denominada Meta-KNN --- consiste en evaluar multiples combinaciones de umbrales en el periodo de entrenamiento, calcular 7 *meta-features* de las predicciones recientes (media, volatilidad, sesgo, autocorrelacion, tendencia, volatilidad del mercado y retorno reciente), y entrenar un KNN (K=5, ponderado por distancia) que asocia estos *meta-features* con los umbrales optimos. Durante el periodo de prueba, para cada dia se calculan los *meta-features* actuales y el KNN predice los umbrales a utilizar, determinandose asi los umbrales exclusivamente con informacion del periodo de entrenamiento.

#### Filtro de Calidad de Senal

Algunos modelos producen predicciones casi constantes, fenomeno que se denomina degeneracion. El filtro detecta estos casos midiendo la desviacion estandar *rolling* de 63 dias de las predicciones, y si esta resulta inferior a 0.001 se clasifica el modelo como degenerado y sus posiciones se fuerzan a CASH.

#### *Backtest* con Costos Reales

La estrategia *Long-Only* se simula incorporando tres tipos de costos realistas. El *expense ratio* anual de cada ETF (0.91\% para UPRO y 0.09\% para SPY) se prorratea diariamente. El *bid-ask spread* se aplica en cada cambio de posicion (5 *bps* para UPRO y 2 *bps* para SPY). Finalmente, el *volatility drag* --- la perdida sistematica por rebalanceo diario en ETFs 3x --- se calcula como `0.5 * 6 * sigma^2` diario.

La formula de retorno es `r = rf + posicion * (market_return - rf) - costos`, donde `rf` representa la tasa libre de riesgo (Treasury 3M / 252).

#### Validacion Estadistica

El Paso 3 ejecuta ademas un conjunto de pruebas estadisticas que incluyen *bootstrap* de 5,000 iteraciones para estimar intervalos de confianza del *Sharpe ratio*, el test de Diebold-Mariano para evaluar superioridad predictiva frente al *benchmark*, el test de Clark-West para capacidad predictiva fuera de muestra, el *Model Confidence Set* (MCS) para identificar el conjunto de mejores modelos, y las correcciones de Bonferroni y FDR para ajustar por pruebas multiples sobre 23 modelos.

---

## Prevencion de *Data Leakage*

El *data leakage* --- filtracion de informacion futura al entrenamiento --- constituye el error metodologico mas grave en *Machine Learning* financiero. Este proyecto implementa cuatro niveles de proteccion.

El primer nivel garantiza que todos los *features* utilicen exclusivamente datos pasados, dado que funciones como `pct_change()`, `rolling()` y `shift(n)` con n positivo solo miran hacia atras.

El segundo nivel impone una division temporal estricta --- nunca aleatoria --- entre los conjuntos de entrenamiento y prueba, con punto de corte en octubre de 2020.

El tercer nivel asegura que el preprocesamiento (*StandardScaler* e *Imputer*) se ajuste unicamente sobre datos de entrenamiento, aplicandose al conjunto de prueba mediante `transform()` --- nunca `fit_transform()`.

El cuarto nivel establece que los umbrales `(q_ext, q_mod)` se predicen con Meta-KNN entrenado exclusivamente sobre datos de entrenamiento, sin utilizar en ningun momento informacion del periodo de prueba para determinar umbrales.

El script `scripts/demo_pipeline_prueba.py` demuestra matematicamente la ausencia de *data leakage*, mostrando paso a paso como cada *feature* y el *target* se alinean temporalmente.

---

## Modelos Entrenados (23)

Se entrenan 23 modelos agrupados en seis categorias. La primera categoria comprende cinco modelos de *Machine Learning* clasico --- Ridge, Lasso, ElasticNet, RandomForest y GradientBoosting --- que implementan regularizacion L2, L1, combinada L1+L2, *ensembles* de arboles de decision y *boosting* de arboles respectivamente.

La segunda categoria agrupa tres implementaciones avanzadas de *Gradient Boosting* --- XGBoost, LightGBM y CatBoost --- que ofrecen optimizaciones sobre el algoritmo base.

La tercera categoria reune cuatro modelos de series de tiempo --- AutoARIMA (seleccion automatica de ordenes), ExponentialSmoothing (suavizado exponencial ETS), Theta (metodo de Assimakopoulos) y SeasonalNaive (*baseline* estacional).

La cuarta categoria incluye dos modelos especializados --- Prophet (modelo de Meta para series de tiempo) y GARCH (volatilidad condicional heteroscedastica).

La quinta categoria abarca cinco arquitecturas de *Deep Learning* implementadas mediante la libreria Darts --- DLinear (descomposicion lineal), N-BEATS (*Neural Basis Expansion Analysis*), N-HiTS (*Neural Hierarchical Interpolation*), TCN (*Temporal Convolutional Network*) y TFT (*Temporal Fusion Transformer*).

La sexta categoria comprende cuatro redes recurrentes con implementacion *custom* en PyTorch --- CNN-LSTM (convolucion temporal + LSTM), LSTM+Attention (LSTM con mecanismo de atencion Luong), BiLSTM (LSTM bidireccional via Darts) y BiGRU (GRU bidireccional via Darts).

Los hiperparametros de cada modelo se documentan en el Apendice A del *paper*, y los umbrales optimos Meta-KNN se documentan en el Apendice B.

---

## Resultados Principales

El periodo de prueba comprende 1,302 dias habiles desde octubre de 2020 hasta diciembre de 2025. El *benchmark* de referencia --- SPY *Buy \& Hold* --- registra un retorno acumulado de +102.2\%, un *Sharpe ratio* de 0.66 y un *Max Drawdown* de -25.4\%.

De los 23 modelos evaluados, cuatro superan al *benchmark*. LSTM+Attention ocupa el primer lugar con un retorno de +400.0\%, *Sharpe* de 1.327, *Sortino* de 2.631, *Calmar* de 1.788 y *Max Drawdown* de -20.5\%. Ridge se ubica en segundo lugar con +263.2\%, *Sharpe* de 0.881 y *Max Drawdown* de -31.8\%. RandomForest alcanza +131.0\% con *Sharpe* de 0.666 y el menor *drawdown* entre los ganadores (-14.7\%). CNN-LSTM cierra el grupo con +129.0\%, *Sharpe* de 0.599 y un *drawdown* mas pronunciado de -35.6\%.

En cuanto a la validacion estadistica, solo LSTM+Attention exhibe significancia al 5\% en los tests de Diebold-Mariano (p=0.011) y Clark-West (p=0.029). Ningun modelo sobrevive la correccion de Bonferroni (alpha ajustado = 0.05/23 = 0.0022), y el *Model Confidence Set* retiene los 24 modelos (23 + SPY) al 5\%, lo que indica que las diferencias de rendimiento no resultan estadisticamente significativas entre si. Estos hallazgos se interpretan en el contexto de la Hipotesis de Mercados Eficientes y Adaptativos en las Secciones 9 y 10 del *paper*.

---

## *Paper* Academico

El *paper* se estructura en 10 secciones mas dos apendices. Las secciones 1 a 3 (Introduccion, Revision de Literatura y Metodologia) y las secciones 9 y 10 (Aportes al Debate sobre Eficiencia y Conclusion) se encuentran *inline* en el archivo principal `paper_triple_screen_ml.tex`. Las secciones 4 a 8 se organizan en archivos independientes --- `section_results_expanded.tex` (Resultados Empiricos), `section_da_paradox.tex` (Paradoja del *Directional Accuracy*), `section_risk_management.tex` (Gestion de Riesgo), `section_statistical_validation.tex` (Validacion Estadistica) y `section_costs_expanded.tex` (Costos de Transaccion). Los apendices A y B documentan los hiperparametros y los parametros optimos Meta-KNN respectivamente.

### Compilacion

```bash
# Generar tablas y figuras primero (si cambiaron los resultados)
python paper/generate_paper_tables.py        # 8 tablas .tex
python paper/generate_paper_figures.py       # 8 figuras .pdf

# Compilar (2 pasadas para TOC y referencias)
powershell.exe -ExecutionPolicy Bypass -File "paper/compile.ps1"
```

### Tablas y Figuras Generadas

El script `generate_paper_tables.py` produce 8 tablas LaTeX --- el resumen ejecutivo de los 23 modelos (Tabla 1), metricas de rendimiento detalladas (Tabla 2), distribucion de posiciones CASH/SPY/UPRO (Tabla 3), rendimiento promedio por categoria (Tabla 4), metricas de riesgo (Tabla 8), rendimiento por regimen de mercado (Tabla 13), desglose de costos de transaccion (Tabla 14) y umbrales optimos Meta-KNN (Tabla A1).

El script `generate_paper_figures.py` produce 8 figuras PDF --- curvas de *equity* de los 5 mejores modelos vs SPY, distribucion de posiciones por modelo, *scatter* de *Sharpe ratio* vs *Max Drawdown*, paradoja DA vs retorno, analisis temporal de *drawdowns*, distribucion de regimenes de mercado, distribucion de P\&L del mejor modelo, y relacion *turnover* vs retorno.

---

## Aplicacion Web (CRONOS)

CRONOS es un *dashboard* interactivo para explorar los resultados del *backtest*. La aplicacion no ejecuta modelos en tiempo real sino que sirve datos pre-computados almacenados en archivos JSON.

### Como Ejecutar

```bash
# 1. Generar datos para la app (si no estan actualizados)
python paper/update_backend_data.py

# 2. Iniciar el backend (FastAPI, puerto 8000)
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --reload --port 8000

# 3. En otra terminal, iniciar el frontend (React, puerto 3000)
npm --prefix app/frontend install       # Solo la primera vez
npm --prefix app/frontend run dev       # Servidor de desarrollo
```

Una vez iniciados ambos servicios, se accede al *dashboard* en `http://localhost:3000`.

### Paginas

La aplicacion ofrece seis paginas de visualizacion. *Overview* presenta el ranking de modelos con curvas de *equity* del top 5 y KPIs agregados. *Detail* permite un analisis profundo de cada modelo individual con graficos sincronizados de *equity*, posiciones y *drawdown*. *Trades* muestra el log de operaciones con capital compuesto y desglose de costos por *trade*. *Risk* ofrece analisis de riesgo incluyendo VaR, volatilidad y metricas *rolling*. *Regime* presenta el rendimiento diferenciado por regimen de mercado (alcista, bajista y lateral). *Costs* documenta el impacto de los costos de transaccion por modelo.

El tema visual --- denominado "Axe Capital" --- emplea un fondo negro puro (#000000) con rojo como color de acento (#c41e3a) y verde para valores positivos (#00c853). Los graficos financieros se implementan con la libreria TradingView *Lightweight Charts*.

---

## Scripts de Replicacion

El directorio `paper/scripts/` contiene seis scripts que replican todos los valores numericos de las secciones 4 a 8 del *paper*, incluyendo tanto las tablas *inline* como los numeros citados en prosa. Estos scripts permiten verificar que cada cifra del *paper* coincide exactamente con los datos producidos por el *pipeline*.

El script maestro `replicar_todo.py` ejecuta primero los generadores de tablas y figuras, luego los cinco scripts de replicacion individuales, y muestra un resumen final con el estado de cada paso. Los scripts individuales se organizan por seccion --- `replicar_seccion4_resultados.py` replica los resultados empiricos incluyendo *benchmarks* y estadisticas de *trading*, `replicar_seccion5_da_paradox.py` verifica la paradoja del *Directional Accuracy* incluyendo la correlacion DA-retorno, `replicar_seccion6_riesgo.py` reproduce todas las metricas de riesgo (*Sharpe*, *Sortino*, *Calmar*, VaR/CVaR, *drawdowns* y *capture ratios*), `replicar_seccion7_validacion.py` replica la validacion estadistica completa (*bootstrap* CI, tests de Diebold-Mariano y Clark-West, MCS y correcciones de Bonferroni), y `replicar_seccion8_costos.py` verifica el analisis de sensibilidad de costos y los multiplicadores de *break-even*.

```bash
# Ejecutar TODOS los scripts de replicacion (recomendado)
python paper/scripts/replicar_todo.py

# O ejecutar scripts individuales
python paper/scripts/replicar_seccion4_resultados.py
python paper/scripts/replicar_seccion5_da_paradox.py
python paper/scripts/replicar_seccion6_riesgo.py
python paper/scripts/replicar_seccion7_validacion.py
python paper/scripts/replicar_seccion8_costos.py
```

---

## Reproducibilidad

Todos los scripts del sistema utilizan la semilla global 42 para garantizar resultados identicos entre ejecuciones. Esta semilla se aplica a NumPy, PyTorch (incluyendo CUDA y cuDNN en modo deterministico) y todos los modelos de scikit-learn.

```python
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
```

El archivo `requirements.txt` utiliza versiones exactas (`==`) para evitar variaciones por actualizaciones de librerias. Las dependencias principales son Python 3.12, PyTorch 2.6 (CPU), Darts 0.41, scikit-learn 1.7.2, y versiones fijadas de XGBoost, LightGBM y CatBoost.

Si se ejecuta el *pipeline* completo dos veces con el mismo entorno y las mismas dependencias, se obtienen exactamente los mismos resultados --- mismas predicciones, mismos umbrales, mismas metricas.

---

## Contacto

**David Gonzalez Canon**
Abril 2026
