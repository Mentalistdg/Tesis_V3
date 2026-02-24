# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE TABLAS LaTeX PARA EL PAPER
================================================================================
Extrae datos de los JSON de la app y genera tablas LaTeX consistentes
para incluir en el paper de tesis.

Genera:
- tables/table_model_performance.tex (Tabla principal de rendimiento)
- tables/table_risk_metrics.tex (Metricas de riesgo)
- tables/table_regime_performance.tex (Rendimiento por regimen)
- tables/table_cost_breakdown.tex (Desglose de costos)
- tables/table_trading_stats.tex (Estadisticas de trading)
- tables/table_position_distribution.tex (Distribucion de posiciones)
- tables/table_optimal_params.tex (Parametros optimos por modelo)
================================================================================
"""

import json
import os
import numpy as np
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
APP_DATA_DIR = os.path.join(BASE_DIR, "app", "backend", "data")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 80)
print("GENERADOR DE TABLAS LaTeX PARA EL PAPER")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n[1] Cargando datos JSON...")

with open(os.path.join(APP_DATA_DIR, "models_summary.json"), 'r') as f:
    summary_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "regime_data.json"), 'r') as f:
    regime_data = json.load(f)

models = summary_data['models']
benchmark = summary_data['benchmark']
test_period = summary_data['test_period']

print(f"    Modelos cargados: {len(models)}")
print(f"    Periodo: {test_period['start']} - {test_period['end']} ({test_period['n_days']} dias)")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def format_pct(value, decimals=1):
    """Formatea un valor como porcentaje."""
    if value is None:
        return "--"
    return f"{value * 100:.{decimals}f}\\%"

def format_num(value, decimals=2):
    """Formatea un numero."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"

def format_money(value, decimals=0):
    """Formatea como dinero."""
    if value is None:
        return "--"
    return f"\\${value:,.{decimals}f}"

def bold_if_best(value, best_value, format_func, *args):
    """Aplica bold si es el mejor valor."""
    formatted = format_func(value, *args)
    if value == best_value:
        return f"\\textbf{{{formatted}}}"
    return formatted

def color_value(value, format_func, *args):
    """Colorea valores positivos/negativos."""
    formatted = format_func(value, *args)
    if value > 0:
        return f"\\textcolor{{ForestGreen}}{{{formatted}}}"
    elif value < 0:
        return f"\\textcolor{{BrickRed}}{{{formatted}}}"
    return formatted


# =============================================================================
# TABLA 1: RENDIMIENTO PRINCIPAL DE MODELOS
# =============================================================================
print("\n[2] Generando tabla de rendimiento principal...")

# Ordenar por retorno total
models_sorted = sorted(models, key=lambda x: x['total_return'], reverse=True)

# Encontrar mejores valores
best_return = max(m['total_return'] for m in models)
best_sharpe = max(m['sharpe'] for m in models)
best_sortino = max(m['sortino'] for m in models if m['sortino'] > 0)
best_calmar = max(m['calmar'] for m in models if m['calmar'] < 100)  # Excluir outliers
lowest_dd = max(m['max_drawdown'] for m in models)  # Menos negativo es mejor

latex_performance = r"""\begin{table}[htbp]
\centering
\caption{Rendimiento de los 23 Modelos en el Per\'iodo de Prueba (Oct 2020 - Dic 2025)}
\label{tab:model_performance}
\small
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
\textbf{Rank} & \textbf{Modelo} & \textbf{Categor\'ia} & \textbf{Ret. Total} & \textbf{Ret. Anual} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{Calmar} & \textbf{Max DD} \\
\midrule
"""

for i, m in enumerate(models_sorted):
    rank = i + 1
    name = m['model'].replace('_', '\\_')
    category = m['category']

    # Formatear valores
    ret_total = format_pct(m['total_return'])
    ret_annual = format_pct(m['annual_return'])
    sharpe = format_num(m['sharpe'], 3)
    sortino = format_num(m['sortino'], 3) if m['sortino'] != 0 else "--"
    calmar = format_num(m['calmar'], 3) if m['calmar'] < 100 else "--"
    max_dd = format_pct(m['max_drawdown'])

    # Colorear retornos
    if m['total_return'] > 0:
        ret_total = f"\\textcolor{{ForestGreen}}{{{ret_total}}}"
    else:
        ret_total = f"\\textcolor{{BrickRed}}{{{ret_total}}}"

    # Bold si supera benchmark
    if m['total_return'] > benchmark['total_return']:
        name = f"\\textbf{{{name}}}"

    latex_performance += f"{rank} & {name} & {category} & {ret_total} & {ret_annual} & {sharpe} & {sortino} & {calmar} & {max_dd} \\\\\n"

latex_performance += r"""\midrule
-- & SPY (B\&H) & Benchmark & """ + format_pct(benchmark['total_return']) + r""" & -- & """ + format_num(benchmark['sharpe'], 3) + r""" & -- & -- & """ + format_pct(benchmark['max_drawdown']) + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Los modelos en negrita superan al benchmark SPY Buy \& Hold. Retorno total calculado como producto de (1+r) - 1 sobre el per\'iodo completo. Sharpe ratio anualizado con tasa libre de riesgo promedio del per\'iodo.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_model_performance.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_performance)
print(f"    table_model_performance.tex generada")


# =============================================================================
# TABLA 2: METRICAS DE RIESGO
# =============================================================================
print("\n[3] Generando tabla de metricas de riesgo...")

latex_risk = r"""\begin{table}[htbp]
\centering
\caption{M\'etricas de Riesgo Ajustado por Modelo}
\label{tab:risk_metrics}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{Calmar} & \textbf{Max DD} & \textbf{Win Rate} & \textbf{Dir. Acc.} & \textbf{Pos. Media} \\
\midrule
"""

for m in models_sorted[:15]:  # Top 15 para no hacer la tabla muy larga
    name = m['model'].replace('_', '\\_')
    sharpe = format_num(m['sharpe'], 3)
    sortino = format_num(m['sortino'], 3) if m['sortino'] != 0 else "--"
    calmar = format_num(m['calmar'], 3) if abs(m['calmar']) < 100 else "--"
    max_dd = format_pct(m['max_drawdown'])
    win_rate = format_pct(m['win_rate'])
    dir_acc = format_pct(m['dir_accuracy'])
    mean_pos = format_num(m['mean_position'], 2)

    latex_risk += f"{name} & {sharpe} & {sortino} & {calmar} & {max_dd} & {win_rate} & {dir_acc} & {mean_pos}x \\\\\n"

latex_risk += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Win Rate = proporci\'on de d\'ias con retorno positivo. Dir. Acc. = precisi\'on direccional respecto al mercado. Pos. Media = posici\'on promedio (0=Cash, 1=SPY, 3=UPRO).
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_risk_metrics.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_risk)
print(f"    table_risk_metrics.tex generada")


# =============================================================================
# TABLA 3: DISTRIBUCION DE POSICIONES
# =============================================================================
print("\n[4] Generando tabla de distribucion de posiciones...")

latex_positions = r"""\begin{table}[htbp]
\centering
\caption{Distribuci\'on de Posiciones por Modelo (Estrategia Long-Only)}
\label{tab:position_distribution}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{UPRO (3x)} & \textbf{SPY (1x)} & \textbf{Cash (0)} & \textbf{Total Long} & \textbf{N° Trades} \\
\midrule
"""

for m in models_sorted:
    name = m['model'].replace('_', '\\_')
    pct_3x = f"{m['pct_3x']:.1f}\\%"
    pct_1x = f"{m['pct_1x']:.1f}\\%"
    pct_cash = f"{m['pct_cash']:.1f}\\%"
    pct_long = f"{m['pct_long']:.1f}\\%"
    n_trades = m['n_trades']

    latex_positions += f"{name} & {pct_3x} & {pct_1x} & {pct_cash} & {pct_long} & {n_trades} \\\\\n"

latex_positions += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Porcentajes representan la proporci\'on de d\'ias en cada posici\'on. UPRO = 3x apalancado largo, SPY = 1x largo, Cash = tasa libre de riesgo. N\'umero de trades = cambios de posici\'on.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_position_distribution.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_positions)
print(f"    table_position_distribution.tex generada")


# =============================================================================
# TABLA 4: DESGLOSE DE COSTOS
# =============================================================================
print("\n[5] Generando tabla de desglose de costos...")

latex_costs = r"""\begin{table}[htbp]
\centering
\caption{Impacto de Costos de Transacci\'on por Modelo}
\label{tab:cost_breakdown}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{Capital Final} & \textbf{P\&L Bruto} & \textbf{Tx Costs} & \textbf{Tx/P\&L} & \textbf{Trades} \\
\midrule
"""

for m in models_sorted:
    name = m['model'].replace('_', '\\_')
    final_eq = format_money(m['final_equity'])
    pnl = format_money(m['profit_loss'])
    tx_costs = format_money(m['transaction_costs'])

    # Ratio de costos sobre P&L (si P&L > 0)
    if m['profit_loss'] > 0:
        tx_ratio = f"{(m['transaction_costs'] / m['profit_loss']) * 100:.1f}\\%"
    else:
        tx_ratio = "--"

    n_trades = m['n_trades']

    # Colorear P&L
    if m['profit_loss'] > 0:
        pnl = f"\\textcolor{{ForestGreen}}{{{pnl}}}"
    else:
        pnl = f"\\textcolor{{BrickRed}}{{{pnl}}}"

    latex_costs += f"{name} & {final_eq} & {pnl} & {tx_costs} & {tx_ratio} & {n_trades} \\\\\n"

latex_costs += r"""\midrule
\textit{Promedio} & -- & -- & \$""" + f"{np.mean([m['transaction_costs'] for m in models]):,.0f}" + r""" & -- & """ + f"{np.mean([m['n_trades'] for m in models]):.0f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Capital inicial = \$10,000. Tx Costs incluye bid-ask spreads (UPRO: 10 bps, SPY: 4 bps) y expense ratios. Tx/P\&L = proporci\'on de costos respecto a ganancias brutas.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_cost_breakdown.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_costs)
print(f"    table_cost_breakdown.tex generada")


# =============================================================================
# TABLA 5: RENDIMIENTO POR REGIMEN
# =============================================================================
print("\n[6] Generando tabla de rendimiento por regimen...")

regime_perf = regime_data['regime_performance']
regime_counts = regime_data['regime_counts']

# Calcular dias por regimen (excluyendo unknown)
total_days = sum(v for k, v in regime_counts.items() if k != 'unknown')
regimes = ['bull', 'sideways', 'bear', 'high_vol']
regime_labels = {'bull': 'Bull', 'sideways': 'Sideways', 'bear': 'Bear', 'high_vol': 'High Vol'}

latex_regime = r"""\begin{table}[htbp]
\centering
\caption{Rendimiento por R\'egimen de Mercado (Top 10 Modelos)}
\label{tab:regime_performance}
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{Bull} & \textbf{Sideways} & \textbf{Bear} & \textbf{High Vol} \\
 & \textit{(""" + str(regime_counts.get('bull', 0)) + r""" d\'ias)} & \textit{(""" + str(regime_counts.get('sideways', 0)) + r""" d\'ias)} & \textit{(""" + str(regime_counts.get('bear', 0)) + r""" d\'ias)} & \textit{(""" + str(regime_counts.get('high_vol', 0)) + r""" d\'ias)} \\
\midrule
"""

# Top 10 modelos por retorno total
for m in models_sorted[:10]:
    name = m['model'].replace('_', '\\_')
    model_regime = regime_perf.get(m['model'], {})

    row_values = []
    for reg in regimes:
        if reg in model_regime:
            ret = model_regime[reg]['total_return']
            formatted = format_pct(ret)
            if ret > 0:
                formatted = f"\\textcolor{{ForestGreen}}{{{formatted}}}"
            elif ret < 0:
                formatted = f"\\textcolor{{BrickRed}}{{{formatted}}}"
            row_values.append(formatted)
        else:
            row_values.append("--")

    latex_regime += f"{name} & {' & '.join(row_values)} \\\\\n"

latex_regime += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Bull = retorno acumulado $>$ 10\% y volatilidad $<$ 20\% en ventana de 60 d\'ias. Bear = retorno $<$ -10\%. High Vol = volatilidad $>$ 25\%. Sideways = resto.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_regime_performance.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_regime)
print(f"    table_regime_performance.tex generada")


# =============================================================================
# TABLA 6: PARAMETROS OPTIMOS
# =============================================================================
print("\n[7] Generando tabla de parametros optimos...")

latex_params = r"""\begin{table}[htbp]
\centering
\caption{Par\'ametros \'Optimos de Umbral por Modelo}
\label{tab:optimal_params}
\small
\begin{tabular}{@{}llrrr@{}}
\toprule
\textbf{Modelo} & \textbf{Categor\'ia} & \textbf{$q_{3x}$ (\%)} & \textbf{$q_{1x}$ (\%)} & \textbf{Interpretaci\'on} \\
\midrule
"""

for m in models_sorted:
    name = m['model'].replace('_', '\\_')
    category = m['category']
    params = m.get('params', {})
    q_3x = params.get('q_3x', '--')
    q_1x = params.get('q_1x', '--')

    # Interpretacion
    if q_3x != '--' and q_1x != '--':
        interp = f"Top {q_3x}\\% $\\rightarrow$ 3x, Top {q_1x}\\% $\\rightarrow$ 1x"
    else:
        interp = "--"

    latex_params += f"{name} & {category} & {q_3x} & {q_1x} & {interp} \\\\\n"

latex_params += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: $q_{3x}$ = percentil para posici\'on UPRO (3x). $q_{1x}$ = percentil para posici\'on SPY (1x). Predicciones por debajo del percentil $q_{1x}$ resultan en posici\'on Cash (0). Par\'ametros optimizados mediante grid search sobre conjunto de validaci\'on.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_optimal_params.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_params)
print(f"    table_optimal_params.tex generada")


# =============================================================================
# TABLA 7: RESUMEN EJECUTIVO (Top 5)
# =============================================================================
print("\n[8] Generando tabla resumen ejecutivo...")

latex_summary = r"""\begin{table}[htbp]
\centering
\caption{Resumen Ejecutivo: Top 5 Modelos vs Benchmark}
\label{tab:executive_summary}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{Ret. Total} & \textbf{Sharpe} & \textbf{Max DD} & \textbf{Capital Final} & \textbf{$\alpha$ vs SPY} \\
\midrule
"""

for m in models_sorted[:5]:
    name = m['model'].replace('_', '\\_')
    ret = format_pct(m['total_return'])
    sharpe = format_num(m['sharpe'], 3)
    max_dd = format_pct(m['max_drawdown'])
    final = format_money(m['final_equity'])
    alpha = format_pct(m['excess_return'])

    # Colorear
    if m['total_return'] > benchmark['total_return']:
        ret = f"\\textbf{{\\textcolor{{ForestGreen}}{{{ret}}}}}"
        alpha = f"\\textcolor{{ForestGreen}}{{{alpha}}}"

    latex_summary += f"{name} & {ret} & {sharpe} & {max_dd} & {final} & {alpha} \\\\\n"

latex_summary += r"""\midrule
SPY (B\&H) & """ + format_pct(benchmark['total_return']) + r""" & """ + format_num(benchmark['sharpe'], 3) + r""" & """ + format_pct(benchmark['max_drawdown']) + r""" & """ + format_money(benchmark['spy_final_value']) + r""" & -- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Per\'iodo de prueba: """ + test_period['start'] + r""" a """ + test_period['end'] + r""" (""" + str(test_period['n_days']) + r""" d\'ias). Capital inicial: \$10,000. $\alpha$ = exceso de retorno sobre SPY Buy \& Hold.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_executive_summary.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_summary)
print(f"    table_executive_summary.tex generada")


# =============================================================================
# TABLA 8: COMPARACION POR CATEGORIA
# =============================================================================
print("\n[9] Generando tabla de comparacion por categoria...")

# Agrupar por categoria
categories = {}
for m in models:
    cat = m['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(m)

# Calcular promedios por categoria
cat_stats = {}
for cat, cat_models in categories.items():
    cat_stats[cat] = {
        'n_models': len(cat_models),
        'avg_return': np.mean([m['total_return'] for m in cat_models]),
        'avg_sharpe': np.mean([m['sharpe'] for m in cat_models]),
        'avg_max_dd': np.mean([m['max_drawdown'] for m in cat_models]),
        'best_return': max([m['total_return'] for m in cat_models]),
        'best_model': max(cat_models, key=lambda x: x['total_return'])['model']
    }

# Ordenar por retorno promedio
cat_sorted = sorted(cat_stats.items(), key=lambda x: x[1]['avg_return'], reverse=True)

latex_category = r"""\begin{table}[htbp]
\centering
\caption{Rendimiento Promedio por Categor\'ia de Modelo}
\label{tab:category_comparison}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Categor\'ia} & \textbf{N} & \textbf{Ret. Prom.} & \textbf{Mejor Ret.} & \textbf{Sharpe Prom.} & \textbf{DD Prom.} & \textbf{Mejor Modelo} \\
\midrule
"""

for cat, stats in cat_sorted:
    n = stats['n_models']
    avg_ret = format_pct(stats['avg_return'])
    best_ret = format_pct(stats['best_return'])
    avg_sharpe = format_num(stats['avg_sharpe'], 3)
    avg_dd = format_pct(stats['avg_max_dd'])
    best_model = stats['best_model'].replace('_', '\\_')

    if stats['avg_return'] > 0:
        avg_ret = f"\\textcolor{{ForestGreen}}{{{avg_ret}}}"
    else:
        avg_ret = f"\\textcolor{{BrickRed}}{{{avg_ret}}}"

    latex_category += f"{cat} & {n} & {avg_ret} & {best_ret} & {avg_sharpe} & {avg_dd} & {best_model} \\\\\n"

latex_category += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: N = n\'umero de modelos en la categor\'ia. ML = Machine Learning cl\'asico, GradientBoosting = XGBoost/LightGBM/CatBoost, DeepLearning = redes Darts, RNN = Bi-LSTM/Bi-GRU, TimeSeries = ARIMA/ETS/Theta.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_category_comparison.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_category)
print(f"    table_category_comparison.tex generada")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("TABLAS GENERADAS")
print("=" * 80)
print(f"\nDirectorio: {TABLES_DIR}")
print("\nArchivos generados:")
for f in sorted(os.listdir(TABLES_DIR)):
    if f.endswith('.tex'):
        print(f"  - {f}")

print("\n" + "=" * 80)
print("Para incluir en el paper, usar:")
print("  \\input{tables/table_model_performance}")
print("  \\input{tables/table_risk_metrics}")
print("  etc.")
print("=" * 80)
