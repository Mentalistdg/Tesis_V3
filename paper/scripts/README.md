# Scripts de Calculo del Paper

Scripts organizados por seccion que generan y verifican todos los numeros, tablas
y figuras del paper. Cada script lee datos de `results/` y `models/` y produce
archivos de salida verificables (JSON).

## Como ejecutar

Todos los scripts se ejecutan desde la raiz del repositorio:

```bash
.venv/Scripts/python.exe paper/scripts/seccion_4_resultados/verify_section4_numbers.py
.venv/Scripts/python.exe paper/scripts/seccion_5_da_paradox/verify_section5_numbers.py
.venv/Scripts/python.exe paper/scripts/seccion_6_riesgo/calculate_risk_metrics.py
.venv/Scripts/python.exe paper/scripts/seccion_7_validacion/calculate_statistical_validation.py
.venv/Scripts/python.exe paper/scripts/seccion_8_costos/verify_section8_numbers.py
.venv/Scripts/python.exe paper/scripts/generacion_tablas_figuras/generate_paper_tables.py
.venv/Scripts/python.exe paper/scripts/generacion_tablas_figuras/generate_paper_figures.py
```

## Estructura

```
paper/scripts/
|-- seccion_4_resultados/
|   `-- verify_section4_numbers.py      -> results/section4_verification.json
|-- seccion_5_da_paradox/
|   `-- verify_section5_numbers.py      -> results/section5_verification.json
|-- seccion_6_riesgo/
|   `-- calculate_risk_metrics.py       -> results/risk_metrics_detail.json
|-- seccion_7_validacion/
|   `-- calculate_statistical_validation.py -> results/statistical_validation.json
|-- seccion_8_costos/
|   `-- verify_section8_numbers.py      -> results/section8_verification.json
`-- generacion_tablas_figuras/
    |-- generate_paper_tables.py        -> paper/tables/*.tex (8 tablas LaTeX)
    `-- generate_paper_figures.py       -> paper/figures/*.pdf (8 figuras)
```

## Detalle por Seccion

### Seccion 4 - Resultados Empiricos
**Script:** `seccion_4_resultados/verify_section4_numbers.py`
**Output:** `results/section4_verification.json`
**Verifica:**
- Trading statistics de los 5 modelos principales (Win Rate, Avg Win/Loss, Profit Factor)
- Benchmarks alternativos (UPRO B&H bajo 3 formulas, 60/40 Portfolio)
- 45+ claims inline del paper (retornos, Sharpe, posiciones, drawdowns)
- Conteo de modelos con Sharpe positivo (16/23)
- Rango de retornos (432 pp)

**Nota sobre benchmarks:** UPRO B&H (+246.4% en el paper) y 60/40 (+58.7%) son datos
de fuentes externas. El script computa aproximaciones bajo diferentes supuestos de costos,
pero no se pueden reproducir exactamente sin datos historicos de precio de UPRO y BND.

### Seccion 5 - Paradoja del Directional Accuracy
**Script:** `seccion_5_da_paradox/verify_section5_numbers.py`
**Output:** `results/section5_verification.json`
**Verifica:**
- DA general de los 23 modelos
- DA@3x (directional accuracy en dias de alta conviccion) para 18 modelos
- Correlacion rho(DA, Return) = -0.37
- % de tiempo en posicion 3x por modelo
- 61 checks, todos PASS

### Seccion 6 - Gestion de Riesgo
**Script:** `seccion_6_riesgo/calculate_risk_metrics.py`
**Output:** `results/risk_metrics_detail.json`
**Calcula:**
- Sharpe, Sortino, Calmar (4 winners + SPY)
- VaR/CVaR diario (95%, 99%)
- VaR/CVaR rolling 30 dias y 90 dias (95%)
- MaxDD con fechas, duracion y tiempo de recuperacion
- Capture Ratios (upside, downside, ratio, avg position)
- Conteo de dias alcistas/bajistas del mercado

### Seccion 7 - Validacion Estadistica
**Script:** `seccion_7_validacion/calculate_statistical_validation.py`
**Output:** `results/statistical_validation.json`
**Calcula:**
- Bootstrap Sharpe CI (5,000 iteraciones) + P(S>0) + P(S>SPY)
- Error estandar de Lo (2002) con correccion AR(1)
- Test de Diebold-Mariano (retornos de estrategia vs B&H)
- Test de Clark-West (predicciones vs media historica)
- Model Confidence Set de Hansen (2011) con bootstrap de bloques
- Analisis de overfitting (DA y MSE train vs test)
- Correccion de Bonferroni (23 modelos)
- FDR Benjamini-Hochberg

### Seccion 8 - Costos de Transaccion
**Script:** `seccion_8_costos/verify_section8_numbers.py`
**Output:** `results/section8_verification.json`
**Verifica:**
- Analisis de sensibilidad a costos (+50%, +100%) para 4 winners
- Multiplicador de break-even (cuantos x los costos para igualar SPY)
- Ratio costos/P&L bruto
- Impacto de latencia (T+open, T+close) para LSTM_Attention
- Descomposicion de costos (expense, trading, vol drag) por modelo
- Costos promedio y rango como % del P&L bruto
- 45 claims verificadas, todas PASS

### Tablas y Figuras (multi-seccion)
**Scripts:** `generacion_tablas_figuras/generate_paper_tables.py`, `generate_paper_figures.py`

**Tablas generadas (8):**
| Tabla | Seccion | Datos |
|-------|---------|-------|
| table_model_performance.tex | S4 | Ranking de modelos por retorno |
| table_executive_summary.tex | S4 | Top 5 modelos resumen |
| table_category_comparison.tex | S4 | Promedio por categoria de modelo |
| table_position_distribution.tex | S4 | Distribucion de posiciones |
| table_cost_breakdown.tex | S8 | Desglose de costos |
| table_regime_performance.tex | S9 | Performance por regimen |
| table_optimal_params.tex | Apendice | Parametros optimos KNN |
| table_risk_metrics.tex | S4 | Metricas de riesgo |

**Figuras generadas (8):**
| Figura | Seccion | Contenido |
|--------|---------|-----------|
| equity_curves_top5.pdf | S4 | Curvas de equity top 5 vs SPY |
| pnl_distribution_lstm_attention.pdf | S4 | Histograma P&L LSTM_Attention |
| position_distribution.pdf | S4 | Distribucion de posiciones top 10 |
| turnover_vs_return.pdf | S8 | Scatter trades vs retorno |
| drawdown_analysis.pdf | S6 | Equity + drawdown LSTM_Attention |
| sharpe_vs_maxdd.pdf | S6 | Risk-return scatter |
| regime_distribution.pdf | S9 | Distribucion de regimenes |
| da_vs_return_paradox.pdf | S5 | DA vs retorno + DA@3x |

## Notas

- Semilla global: 42 para reproducibilidad
- Todos los scripts usan rutas relativas al directorio raiz del proyecto
- Los scripts de seccion 4, 5, 8 son de VERIFICACION (comparan paper vs datos)
- Los scripts de seccion 6, 7 son de CALCULO (generan los numeros usados en el paper)
- Los scripts originales tambien se mantienen en `paper/` por compatibilidad
