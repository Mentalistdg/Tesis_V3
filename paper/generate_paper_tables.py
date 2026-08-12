# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE TABLAS LaTeX PARA EL PAPER
================================================================================
Lee datos EXCLUSIVAMENTE de los JSONs autoritativos del pipeline:
  - results/final_long_only_backtest.json (rendimiento, posiciones, costos)
  - results/optimal_model_params.json (parametros y bootstrap CIs)

Genera:
- tables/table_model_performance.tex
- tables/table_cost_breakdown.tex
- tables/table_position_distribution.tex
- tables/table_optimal_params.tex
- tables/table_executive_summary.tex
- tables/table_category_comparison.tex
================================================================================
"""

import json
import os
import numpy as np
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

# Model category mapping
MODEL_CATEGORIES = {
    'Ridge': "ML cl\\'asico", 'Lasso': "ML cl\\'asico", 'ElasticNet': "ML cl\\'asico",
    'RandomForest': "ML cl\\'asico",
    'GradientBoosting': 'Boosting',
    'XGBoost': 'Boosting', 'LightGBM': 'Boosting',
    'CatBoost': 'Boosting',
    'AutoARIMA': 'Series temporales', 'ExponentialSmoothing': 'Series temporales',
    'Theta': 'Series temporales', 'SeasonalNaive': 'Series temporales',
    'Prophet': 'Series temporales', 'GARCH': 'Series temporales',
    'DLinear': 'Deep learning', 'NBEATS': 'Deep learning',
    'NHiTS': 'Deep learning', 'TCN': 'Deep learning', 'TFT': 'Deep learning',
    'CNN_LSTM': 'Deep learning', 'LSTM_Attention': 'Deep learning',
    'BiLSTM': 'Deep learning', 'BiGRU': 'Deep learning',
}

print("=" * 80)
print("GENERADOR DE TABLAS LaTeX PARA EL PAPER")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS AUTORITATIVOS
# =============================================================================
# Fuentes de datos (todas generadas por el pipeline):
#   - final_long_only_backtest.json: retorno total, Sharpe, Sortino, Calmar, MaxDD,
#     distribuciones de posiciones, costos, n_trades para los 23 modelos
#   - optimal_model_params.json: umbrales (q_ext, q_mod) mas frecuentes del Meta-KNN
# =============================================================================
print("\n[1] Cargando datos JSON autoritativos...")

with open(os.path.join(RESULTS_DIR, "final_long_only_backtest.json"), 'r') as f:
    backtest_data = json.load(f)

with open(os.path.join(RESULTS_DIR, "optimal_model_params.json"), 'r') as f:
    optimal_params = json.load(f)

benchmark = backtest_data['benchmark']
raw_models = backtest_data['models']

# Build unified model list
INITIAL_CAPITAL = 10000
models = []
for name, m in raw_models.items():
    final_capital = m.get('final_capital', INITIAL_CAPITAL * (1 + m['total_return']))
    gross_final = m.get('gross_final', final_capital)  # fallback if not available
    total_costs = m.get('total_costs', gross_final - final_capital)
    gross_pnl = gross_final - INITIAL_CAPITAL
    net_pnl = final_capital - INITIAL_CAPITAL

    params = optimal_params.get(name, {}).get('params', {})
    bootstrap = optimal_params.get(name, {}).get('bootstrap', {})

    models.append({
        'model': name,
        'category': MODEL_CATEGORIES.get(name, 'Other'),
        'total_return': m['total_return'],
        'annual_return': m['annual_return'],
        'sharpe': m['sharpe'],
        'sortino': m['sortino'],
        'calmar': m['calmar'],
        'max_drawdown': m['max_drawdown'],
        'annual_vol': m['annual_vol'],
        'directional_accuracy': m['directional_accuracy'],
        'pct_3x': m['pct_3x'],
        'pct_1x': m['pct_1x'],
        'pct_cash': m['pct_cash'],
        'pct_long': m['pct_3x'] + m['pct_1x'],
        'n_trades': m['n_trades'],
        'final_capital': final_capital,
        'gross_final': gross_final,
        'total_costs': total_costs,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'excess_return': m['total_return'] - benchmark['total_return'],
        'mean_position': (m['pct_3x'] * 3 + m['pct_1x'] * 1) / 100,
        'beat_spy': m['total_return'] > benchmark['total_return'],
        'q_3x': params.get('q_long_extreme', '--'),
        'q_1x': params.get('q_long_moderate', '--'),
        'sharpe_ci_lower': bootstrap.get('sharpe_ci_lower'),
        'sharpe_ci_upper': bootstrap.get('sharpe_ci_upper'),
        'prob_sharpe_positive': bootstrap.get('sharpe_prob_positive'),
    })

# Sort by total return descending
models_sorted = sorted(models, key=lambda x: x['total_return'], reverse=True)

print(f"    Modelos cargados: {len(models)}")
print(f"    Fuente: results/final_long_only_backtest.json ({backtest_data['timestamp']})")
print(f"    Benchmark SPY: +{benchmark['total_return']*100:.1f}%, Sharpe {benchmark['sharpe']:.3f}")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def format_pct(value, decimals=1):
    if value is None:
        return "--"
    return f"{value * 100:.{decimals}f}\\%"

def format_num(value, decimals=2):
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"

def format_money(value, decimals=0):
    if value is None:
        return "--"
    return f"\\${value:,.{decimals}f}"


# =============================================================================
# TABLA 1: RENDIMIENTO PRINCIPAL DE MODELOS (tab:model_performance)
# =============================================================================
# Genera: tables/table_model_performance.tex
# Contenido: Los 23 modelos ordenados por retorno total descendente
# Columnas: Rank, Modelo, Categoria, Ret.Total, Ret.Anual, Sharpe, Sortino, Calmar, MaxDD
# Fuente: final_long_only_backtest.json
# Nota: modelos que superan SPY aparecen en negrita; retornos positivos en verde, negativos en rojo
# =============================================================================
print("\n[2] Generando tabla de rendimiento principal...")

latex_performance = r"""\begin{table}[htbp]
\centering
\caption{Rendimiento de los 23 Modelos en el Per\'iodo de Prueba (Oct 2020 - Dic 2025)}
\label{tab:model_performance}
\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Rank} & \textbf{Modelo} & \textbf{Categor\'ia} & \textbf{Ret. Total} & \textbf{Ret. Anual} & \textbf{Sharpe} & \textbf{Max DD} \\
\midrule
"""

for i, m in enumerate(models_sorted):
    rank = i + 1
    name = m['model'].replace('_', '\\_')
    category = m['category']

    ret_total = format_pct(m['total_return'])
    ret_annual = format_pct(m['annual_return'])
    sharpe = format_num(m['sharpe'], 3)
    max_dd = format_pct(m['max_drawdown'])

    if m['total_return'] > 0:
        ret_total = f"\\textcolor{{ForestGreen}}{{{ret_total}}}"
    else:
        ret_total = f"\\textcolor{{BrickRed}}{{{ret_total}}}"

    if m['beat_spy']:
        name = f"\\textbf{{{name}}}"

    # Marcadores: SeasonalNaive = benchmark minimo (fila resaltada); AutoARIMA = 100% cash
    row_prefix = ""
    if m['model'] == 'SeasonalNaive':
        row_prefix = "\\rowcolor{naivebg} "
        name = f"{name}$^\\dagger$"
    elif m['model'] == 'AutoARIMA':
        name = f"{name}$^\\ddagger$"

    latex_performance += f"{row_prefix}{rank} & {name} & {category} & {ret_total} & {ret_annual} & {sharpe} & {max_dd} \\\\\n"

latex_performance += r"""\midrule
-- & SPY (B\&H) & Benchmark & """ + format_pct(benchmark['total_return']) + r""" & -- & """ + format_num(benchmark['sharpe'], 3) + r""" & """ + format_pct(benchmark['max_drawdown']) + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Los modelos en negrita superan al benchmark SPY Buy \& Hold. \colorbox{naivebg}{$^\dagger$SeasonalNaive} = \textit{benchmark} de complejidad m\'inima. $^\ddagger$AutoARIMA permanece el 100\% del per\'iodo en efectivo (filtro de calidad de se\~nal), por lo que su retorno corresponde a la tasa libre de riesgo y su Sharpe, calculado sobre una volatilidad casi nula (0.14\%), no es comparable con el de los modelos con operaci\'on activa. Retorno total calculado como producto de (1+r) - 1 sobre el per\'iodo completo. Sharpe ratio anualizado con tasa libre de riesgo promedio del per\'iodo.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_model_performance.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_performance)
print(f"    table_model_performance.tex generada")


# =============================================================================
# TABLA 2: DISTRIBUCION DE POSICIONES (tab:position_distribution)
# =============================================================================
# Genera: tables/table_position_distribution.tex
# Contenido: 23 modelos con % dias en UPRO(3x), SPY(1x), Cash(0), Total Long, N trades
# Fuente: final_long_only_backtest.json (campos pct_3x, pct_1x, pct_cash, n_trades)
# =============================================================================
print("\n[4] Generando tabla de distribucion de posiciones...")

latex_positions = r"""\begin{table}[H]
\centering
\caption{Distribuci\'on de Posiciones por Modelo (Estrategia Long-Only)}
\label{tab:position_distribution}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Modelo} & \textbf{UPRO (3x)} & \textbf{SPY (1x)} & \textbf{Cash (0)} & \textbf{Total Long} & \textbf{N\textdegree{} Trades} \\
\midrule
"""

for m in models_sorted:
    name = m['model'].replace('_', '\\_')
    pct_3x = f"{m['pct_3x']:.1f}\\%"
    pct_1x = f"{m['pct_1x']:.1f}\\%"
    pct_cash = f"{m['pct_cash']:.1f}\\%"
    pct_long = f"{m['pct_long']:.1f}\\%"
    n_trades = m['n_trades']

    row_prefix = ""
    if m['model'] == 'SeasonalNaive':
        row_prefix = "\\rowcolor{naivebg} "
        name = f"{name}$^\\dagger$"

    latex_positions += f"{row_prefix}{name} & {pct_3x} & {pct_1x} & {pct_cash} & {pct_long} & {n_trades} \\\\\n"

latex_positions += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Porcentajes representan la proporci\'on de d\'ias en cada posici\'on. UPRO = 3x apalancado largo, SPY = 1x largo, Cash = tasa libre de riesgo. \colorbox{naivebg}{$^\dagger$SeasonalNaive} = \textit{benchmark} de complejidad m\'inima. N\'umero de trades = cambios de posici\'on.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_position_distribution.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_positions)
print(f"    table_position_distribution.tex generada")


# =============================================================================
# TABLA 4: DESGLOSE DE COSTOS (tab:cost_breakdown)
# =============================================================================
# Genera: tables/table_cost_breakdown.tex
# Contenido: 23 modelos con Capital Final, P&L Bruto, Tx Costs, Tx/P&L ratio, N Trades
# Fuente: final_long_only_backtest.json (campos final_capital, gross_final, total_costs)
# Metodologia: Tx/P&L = total_costs / (gross_final - 10000) * 100
# Nota: Capital inicial = $10,000. Incluye expense ratios + bid-ask spreads + vol drag
# =============================================================================
print("\n[5] Generando tabla de desglose de costos...")

latex_costs = r"""\begin{table}[H]
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
    final_eq = format_money(m['final_capital'])
    pnl = format_money(m['gross_pnl'])
    tx_costs = format_money(m['total_costs'])

    if m['gross_pnl'] > 0:
        tx_ratio = f"{(m['total_costs'] / m['gross_pnl']) * 100:.1f}\\%"
    else:
        tx_ratio = "--"

    n_trades = m['n_trades']

    if m['gross_pnl'] > 0:
        pnl = f"\\textcolor{{ForestGreen}}{{{pnl}}}"
    else:
        pnl = f"\\textcolor{{BrickRed}}{{{pnl}}}"

    row_prefix = ""
    if m['model'] == 'SeasonalNaive':
        row_prefix = "\\rowcolor{naivebg} "
        name = f"{name}$^\\dagger$"

    latex_costs += f"{row_prefix}{name} & {final_eq} & {pnl} & {tx_costs} & {tx_ratio} & {n_trades} \\\\\n"

avg_costs = np.mean([m['total_costs'] for m in models])
avg_trades = np.mean([m['n_trades'] for m in models])

latex_costs += r"""\midrule
\textit{Promedio} & -- & -- & \$""" + f"{avg_costs:,.0f}" + r""" & -- & """ + f"{avg_trades:.0f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Capital inicial = \$10,000. \colorbox{naivebg}{$^\dagger$SeasonalNaive} = \textit{benchmark} de complejidad m\'inima. Tx Costs incluye bid-ask spreads (UPRO 5 bps, SPY 2 bps por operaci\'on; 10 y 4 bps ida y vuelta), expense ratios y volatility drag. Tx/P\&L = proporci\'on de costos respecto a ganancias brutas.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_cost_breakdown.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_costs)
print(f"    table_cost_breakdown.tex generada ({len(models)} modelos, avg costs ${avg_costs:,.0f})")


# =============================================================================
# TABLA 5: PARAMETROS OPTIMOS (tab:optimal_params)
# =============================================================================
# Genera: tables/table_optimal_params.tex
# Contenido: 23 modelos con q_3x (percentil UPRO), q_1x (percentil SPY), interpretacion
# Fuente: optimal_model_params.json
# Nota: Los umbrales son los MAS FRECUENTES seleccionados por Meta-KNN dinamico;
#       los umbrales reales varian por dia segun las meta-features
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
    q_3x = m['q_3x']
    q_1x = m['q_1x']

    if q_3x != '--' and q_1x != '--':
        interp = f"Top {q_3x}\\% $\\rightarrow$ 3x, Top {q_1x}\\% $\\rightarrow$ 1x"
    else:
        interp = "--"

    row_prefix = ""
    if m['model'] == 'SeasonalNaive':
        row_prefix = "\\rowcolor{naivebg} "
        name = f"{name}$^\\dagger$"

    latex_params += f"{row_prefix}{name} & {category} & {q_3x} & {q_1x} & {interp} \\\\\n"

latex_params += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: $q_{3x}$ = percentil para posici\'on UPRO (3x). $q_{1x}$ = percentil para posici\'on SPY (1x). Predicciones por debajo del percentil $q_{1x}$ resultan en posici\'on Cash (0). \colorbox{naivebg}{$^\dagger$SeasonalNaive} = \textit{benchmark} de complejidad m\'inima. Par\'ametros mostrados corresponden a la combinaci\'on m\'as frecuente seleccionada por el Meta-KNN din\'amico; los umbrales reales var\'ian por d\'ia.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_optimal_params.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_params)
print(f"    table_optimal_params.tex generada")


# =============================================================================
# TABLA 7: RESUMEN EJECUTIVO (tab:executive_summary)
# =============================================================================
# Genera: tables/table_executive_summary.tex
# Contenido: Top 5 modelos + SPY B&H con Ret.Total, Sharpe, MaxDD, Capital Final, alpha
# Fuente: final_long_only_backtest.json
# alpha = retorno_modelo - retorno_SPY (exceso de retorno sobre el benchmark)
# =============================================================================
print("\n[8] Generando tabla resumen ejecutivo...")

spy_final = INITIAL_CAPITAL * (1 + benchmark['total_return'])

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
    final = format_money(m['final_capital'])
    alpha = format_pct(m['excess_return'])

    if m['beat_spy']:
        ret = f"\\textbf{{\\textcolor{{ForestGreen}}{{{ret}}}}}"
        alpha = f"\\textcolor{{ForestGreen}}{{{alpha}}}"

    latex_summary += f"{name} & {ret} & {sharpe} & {max_dd} & {final} & {alpha} \\\\\n"

latex_summary += r"""\midrule
SPY (B\&H) & """ + format_pct(benchmark['total_return']) + r""" & """ + format_num(benchmark['sharpe'], 3) + r""" & """ + format_pct(benchmark['max_drawdown']) + r""" & """ + format_money(spy_final) + r""" & -- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: Per\'iodo de prueba: Oct 2020 a Dic 2025 (1,302 d\'ias). Capital inicial: \$10,000. $\alpha$ = exceso de retorno sobre SPY Buy \& Hold.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table_executive_summary.tex"), 'w', encoding='utf-8') as f:
    f.write(latex_summary)
print(f"    table_executive_summary.tex generada")


# =============================================================================
# TABLA 8: COMPARACION POR CATEGORIA (tab:category_comparison)
# =============================================================================
# Genera: tables/table_category_comparison.tex
# Contenido: 6 categorias con N modelos, retorno promedio, mejor retorno, Sharpe promedio,
#            DD promedio, mejor modelo
# Categorias: ML (5), GradientBoosting (3), TimeSeries (4), Specialized (2),
#             DeepLearning (5), RNN (4)
# Fuente: final_long_only_backtest.json, agrupado por MODEL_CATEGORIES
# =============================================================================
print("\n[9] Generando tabla de comparacion por categoria...")

categories = {}
for m in models:
    cat = m['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(m)

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

cat_sorted = sorted(cat_stats.items(), key=lambda x: x[1]['avg_return'], reverse=True)

latex_category = r"""\begin{table}[htbp]
\centering
\caption{Rendimiento Promedio por Categor\'ia de Modelo}
\label{tab:category_comparison}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Categor\'ia} & \textbf{N} & \textbf{Ret. Prom.} & \textbf{Mejor Ret.} & \textbf{Sharpe Prom.} & \textbf{Mejor Modelo} \\
\midrule
"""

for cat, stats in cat_sorted:
    n = stats['n_models']
    avg_ret = format_pct(stats['avg_return'])
    best_ret = format_pct(stats['best_return'])
    avg_sharpe = format_num(stats['avg_sharpe'], 3)
    best_model = stats['best_model'].replace('_', '\\_')

    if stats['avg_return'] > 0:
        avg_ret = f"\\textcolor{{ForestGreen}}{{{avg_ret}}}"
    else:
        avg_ret = f"\\textcolor{{BrickRed}}{{{avg_ret}}}"

    latex_category += f"{cat} & {n} & {avg_ret} & {best_ret} & {avg_sharpe} & {best_model} \\\\\n"

latex_category += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Nota: N = n\'umero de modelos en la categor\'ia. ML cl\'asico = Ridge/Lasso/ElasticNet/RandomForest; Boosting = GradientBoosting (sklearn) y las librer\'ias XGBoost/LightGBM/CatBoost; Series temporales = AutoARIMA/ETS/Theta/SeasonalNaive/Prophet/GARCH; Deep learning = redes Darts (DLinear/N-BEATS/N-HiTS/TCN/TFT) y arquitecturas \textit{custom} en PyTorch (CNN-LSTM/LSTM+Attention/BiLSTM/BiGRU).
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

# Sanity checks
print("\n--- SANITY CHECKS ---")
lstm = next(m for m in models if m['model'] == 'LSTM_Attention')
cnn = next(m for m in models if m['model'] == 'CNN_LSTM')
print(f"LSTM_Attention: +{lstm['total_return']*100:.1f}%, Sharpe {lstm['sharpe']:.3f}, MaxDD {lstm['max_drawdown']*100:.1f}%, {lstm['n_trades']} trades")
print(f"CNN_LSTM:       +{cnn['total_return']*100:.1f}%, Sharpe {cnn['sharpe']:.3f}, MaxDD {cnn['max_drawdown']*100:.1f}%, {cnn['n_trades']} trades")
print(f"Avg Tx Costs:   ${np.mean([m['total_costs'] for m in models]):,.0f}")
print(f"SPY B&H:        +{benchmark['total_return']*100:.1f}%, Sharpe {benchmark['sharpe']:.3f}")
print("=" * 80)
