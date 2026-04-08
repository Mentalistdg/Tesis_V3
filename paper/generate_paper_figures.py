# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE FIGURAS PARA EL PAPER
================================================================================
Genera figuras de alta calidad para el paper academico usando los datos
de la app.

Figuras generadas:
- figures/equity_curves_top5.pdf
- figures/pnl_distribution_ridge.pdf
- figures/turnover_vs_return.pdf
- figures/regime_distribution.pdf
- figures/position_distribution.pdf
- figures/drawdown_analysis.pdf
================================================================================
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuracion de estilo para paper academico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
APP_DATA_DIR = os.path.join(BASE_DIR, "app", "backend", "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("GENERADOR DE FIGURAS PARA EL PAPER")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS
# =============================================================================
# Fuentes (todas generadas por paper/update_backend_data.py):
#   - models_summary.json: metricas agregadas de los 23 modelos + benchmark
#   - daily_data.json: arrays diarios (equity, drawdown, positions, returns) por modelo
#   - market_data.json: equity curve y retornos de SPY B&H
#   - regime_data.json: clasificacion y rendimiento por regimen de mercado
# =============================================================================
print("\n[1] Cargando datos JSON...")

with open(os.path.join(APP_DATA_DIR, "models_summary.json"), 'r') as f:
    summary_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "daily_data.json"), 'r') as f:
    daily_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "market_data.json"), 'r') as f:
    market_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "regime_data.json"), 'r') as f:
    regime_data = json.load(f)

models = summary_data['models']
benchmark = summary_data['benchmark']

print(f"    Modelos cargados: {len(models)}")


# =============================================================================
# FIGURA 1: EQUITY CURVES TOP 5 (equity_curves_top5.pdf)
# =============================================================================
# Contenido: Curvas de capital de los 5 mejores modelos + SPY B&H
# Fuente: daily_data.json (equity_curve por modelo), market_data.json (SPY B&H)
# Eje Y: valor del portafolio en dolares (capital inicial $10,000)
# Eje X: fechas (Oct 2020 - Dic 2025)
# =============================================================================
print("\n[2] Generando figura de equity curves...")

top5_models = sorted(models, key=lambda x: x['total_return'], reverse=True)[:5]
INITIAL_CAPITAL = 10000

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#e63946', '#2a9d8f', '#e9c46a', '#264653', '#f4a261']

# Plot top 5 models
for i, model in enumerate(top5_models):
    model_name = model['model']
    if model_name in daily_data:
        data = daily_data[model_name]
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['dates']]
        equity = np.array(data['equity_curve']) * INITIAL_CAPITAL
        ax.plot(dates, equity, label=f"{model_name} (+{model['total_return']*100:.0f}%)",
                color=colors[i], linewidth=1.5)

# Plot benchmark SPY
dates = [datetime.strptime(d, '%Y-%m-%d') for d in market_data['dates']]
spy_equity = np.array(market_data['equity_curve']) * INITIAL_CAPITAL
ax.plot(dates, spy_equity, label=f"SPY B&H (+{benchmark['total_return']*100:.0f}%)",
        color='gray', linewidth=2, linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Equity Curves: Top 5 Models vs SPY Buy & Hold')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Add initial capital line
ax.axhline(y=INITIAL_CAPITAL, color='black', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "equity_curves_top5.pdf"))
plt.close()
print(f"    equity_curves_top5.pdf generada")


# =============================================================================
# FIGURA 2: DISTRIBUCION P&L LSTM_ATTENTION (pnl_distribution_lstm_attention.pdf)
# =============================================================================
# Contenido: Histograma de retornos diarios de exceso (strategy - rf) en dias activos
# Fuente: daily_data.json (strategy_returns, positions, risk_free para LSTM_Attention)
# Nota: Solo dias activos (posicion != 0). Dias positivos en verde, negativos en rojo
# Estadisticas: Win Rate, Avg Win, Avg Loss mostradas en un cuadro
# =============================================================================
print("\n[3] Generando figura de distribucion P&L...")

lstm_data = daily_data.get('LSTM_Attention', None)
if lstm_data:
    positions = np.array(lstm_data['positions'])
    strat_returns = np.array(lstm_data['strategy_returns'])
    rf_daily = np.array(lstm_data['risk_free'])

    # Filter for active days (position != 0) — consistent with tab:trading_stats
    active_mask = positions != 0
    active_returns = (strat_returns[active_mask] - rf_daily[active_mask]) * 100
    n_active = int(np.sum(active_mask))

    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns <= 0]
    win_rate = len(wins) / n_active * 100 if n_active > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    n_bins, bins, patches = ax.hist(active_returns, bins=40, edgecolor='black', alpha=0.7)

    # Color positive/negative
    for i, patch in enumerate(patches):
        if bins[i] >= 0:
            patch.set_facecolor('#2a9d8f')
        else:
            patch.set_facecolor('#e63946')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.axvline(x=np.median(active_returns), color='blue', linestyle='--', linewidth=1.5,
               label=f'Median: {np.median(active_returns):.2f}%')

    ax.set_xlabel('Daily Excess Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'P&L Distribution: LSTM Attention ({n_active} Active Days)')
    ax.legend()

    # Add stats box
    stats_text = f'Win Rate: {win_rate:.1f}%\n'
    stats_text += f'Avg Win: +{np.mean(wins):.2f}%\n'
    stats_text += f'Avg Loss: {np.mean(losses):.2f}%'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pnl_distribution_lstm_attention.pdf"))
    plt.close()
    print(f"    pnl_distribution_lstm_attention.pdf generada")


# =============================================================================
# FIGURA 3: TURNOVER VS RETURN (turnover_vs_return.pdf)
# =============================================================================
# Contenido: Scatter plot de N trades vs retorno total para los 23 modelos
# Fuente: models_summary.json (n_trades, total_return)
# Incluye: linea de regresion lineal con R^2, linea horizontal de SPY B&H
# Conclusión: R^2 ~0.02 demuestra que no hay correlacion significativa
# =============================================================================
print("\n[4] Generando figura turnover vs return...")

fig, ax = plt.subplots(figsize=(8, 6))

n_trades = [m['n_trades'] for m in models]
returns = [m['total_return'] * 100 for m in models]
names = [m['model'] for m in models]

# Scatter plot
scatter = ax.scatter(n_trades, returns, c=returns, cmap='RdYlGn', s=100, alpha=0.7,
                     edgecolors='black', linewidth=0.5)

# Add labels for notable points
for i, name in enumerate(names):
    if returns[i] > 100 or returns[i] < 0 or n_trades[i] > 800:
        ax.annotate(name, (n_trades[i], returns[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

# Add horizontal line at SPY return
ax.axhline(y=benchmark['total_return']*100, color='gray', linestyle='--',
           label=f'SPY B&H ({benchmark["total_return"]*100:.1f}%)')

# Linear regression
z = np.polyfit(n_trades, returns, 1)
p = np.poly1d(z)
x_line = np.linspace(min(n_trades), max(n_trades), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'Trend (R²={np.corrcoef(n_trades, returns)[0,1]**2:.2f})')

ax.set_xlabel('Number of Trades')
ax.set_ylabel('Total Return (%)')
ax.set_title('Turnover vs Return: No Significant Correlation')
ax.legend()

plt.colorbar(scatter, label='Return (%)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "turnover_vs_return.pdf"))
plt.close()
print(f"    turnover_vs_return.pdf generada")


# =============================================================================
# FIGURA 4: DISTRIBUCION DE REGIMENES (regime_distribution.pdf)
# =============================================================================
# Contenido: (izq) Pie chart con distribucion de regimenes, (der) bar chart de rendimiento
# Fuente: regime_data.json (regime_counts, regime_performance)
# Regimenes: Bull, Bear, Sideways, High Vol (clasificados en ventanas de 60 dias)
# =============================================================================
print("\n[5] Generando figura de regimenes...")

regime_counts = regime_data['regime_counts']
# Exclude 'unknown'
regimes = {k: v for k, v in regime_counts.items() if k != 'unknown'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
colors_regime = {'bull': '#2a9d8f', 'bear': '#e63946', 'sideways': '#457b9d', 'high_vol': '#f4a261'}
labels = [f'{k.title()}\n({v} days)' for k, v in regimes.items()]
ax1.pie(regimes.values(), labels=labels, autopct='%1.1f%%',
        colors=[colors_regime[k] for k in regimes.keys()],
        explode=[0.02]*len(regimes), startangle=90)
ax1.set_title('Market Regime Distribution\n(Oct 2020 - Dec 2025)')

# Bar chart of performance by regime for top models
top3 = ['LSTM_Attention', 'Ridge', 'CNN_LSTM']
regime_perf = regime_data['regime_performance']

x = np.arange(len(regimes))
width = 0.25

for i, model in enumerate(top3):
    if model in regime_perf:
        perf = [regime_perf[model].get(r, {}).get('total_return', 0) * 100 for r in regimes.keys()]
        ax2.bar(x + i*width, perf, width, label=model)

ax2.set_xlabel('Market Regime')
ax2.set_ylabel('Return (%)')
ax2.set_title('Model Performance by Regime')
ax2.set_xticks(x + width)
ax2.set_xticklabels([k.title() for k in regimes.keys()])
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "regime_distribution.pdf"))
plt.close()
print(f"    regime_distribution.pdf generada")


# =============================================================================
# FIGURA 5: DISTRIBUCION DE POSICIONES (position_distribution.pdf)
# =============================================================================
# Contenido: Stacked horizontal bar chart de % tiempo en UPRO/SPY/Cash para Top 10
# Fuente: models_summary.json (pct_3x, pct_1x, pct_cash)
# =============================================================================
print("\n[6] Generando figura de posiciones...")

fig, ax = plt.subplots(figsize=(12, 6))

# Top 10 models by return
top10 = sorted(models, key=lambda x: x['total_return'], reverse=True)[:10]
model_names = [m['model'] for m in top10]

pct_3x = [m['pct_3x'] for m in top10]
pct_1x = [m['pct_1x'] for m in top10]
pct_cash = [m['pct_cash'] for m in top10]

x = np.arange(len(model_names))
width = 0.6

ax.barh(x, pct_3x, width, label='UPRO (3x)', color='#2a9d8f')
ax.barh(x, pct_1x, width, left=pct_3x, label='SPY (1x)', color='#457b9d')
ax.barh(x, pct_cash, width, left=[a+b for a,b in zip(pct_3x, pct_1x)], label='Cash', color='#d4d4d4')

ax.set_xlabel('Time Allocation (%)')
ax.set_ylabel('Model')
ax.set_title('Position Distribution: Top 10 Models')
ax.set_yticks(x)
ax.set_yticklabels(model_names)
ax.legend(loc='lower right')
ax.set_xlim(0, 100)

# Add vertical line at 50%
ax.axvline(x=50, color='black', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "position_distribution.pdf"))
plt.close()
print(f"    position_distribution.pdf generada")


# =============================================================================
# FIGURA 6: DRAWDOWN ANALYSIS (drawdown_analysis.pdf)
# =============================================================================
# Contenido: (arriba) Equity curve LSTM_Attention vs SPY, (abajo) drawdown en %
# Fuente: daily_data.json (equity_curve, drawdown para LSTM_Attention), market_data.json
# Nota: Anotacion del MaxDD con flecha en el punto de mayor caida
# =============================================================================
print("\n[7] Generando figura de drawdown...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top model drawdown — LSTM_Attention (best performer)
lstm_dd_data = daily_data.get('LSTM_Attention', None)
if lstm_dd_data:
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in lstm_dd_data['dates']]
    equity = np.array(lstm_dd_data['equity_curve'])
    drawdown = np.array(lstm_dd_data['drawdown']) * 100

    # Upper plot: Equity curve
    ax1.plot(dates, equity * INITIAL_CAPITAL, color='#e63946', linewidth=1.5, label='LSTM Attention')
    ax1.plot(dates, np.array(market_data['equity_curve']) * INITIAL_CAPITAL,
             color='gray', linewidth=1.5, linestyle='--', label='SPY B&H')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('LSTM Attention: Equity Curve and Drawdown Analysis')
    ax1.legend(loc='upper left')

    # Lower plot: Drawdown
    ax2.fill_between(dates, drawdown, 0, alpha=0.7, color='#e63946')
    ax2.plot(dates, drawdown, color='#e63946', linewidth=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')

    # Mark maximum drawdown
    min_dd_idx = np.argmin(drawdown)
    ax2.annotate(f'Max DD: {drawdown[min_dd_idx]:.1f}%',
                 xy=(dates[min_dd_idx], drawdown[min_dd_idx]),
                 xytext=(dates[min_dd_idx], drawdown[min_dd_idx] - 10),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "drawdown_analysis.pdf"))
plt.close()
print(f"    drawdown_analysis.pdf generada")


# =============================================================================
# FIGURA 7: SHARPE VS MAX DRAWDOWN (sharpe_vs_maxdd.pdf)
# =============================================================================
# Contenido: Scatter plot de MaxDD vs Sharpe para 23 modelos, coloreado por retorno
# Fuente: models_summary.json (sharpe, max_drawdown, total_return)
# Incluye: SPY B&H como estrella, lineas de cuadrante en Sharpe=0.5 y DD=-30%
# =============================================================================
print("\n[8] Generando figura Sharpe vs Max DD...")

fig, ax = plt.subplots(figsize=(8, 6))

sharpes = [m['sharpe'] for m in models]
max_dds = [m['max_drawdown'] * 100 for m in models]
names = [m['model'] for m in models]
returns = [m['total_return'] * 100 for m in models]

scatter = ax.scatter(max_dds, sharpes, c=returns, cmap='RdYlGn', s=100,
                     alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels for notable points
for i, name in enumerate(names):
    if sharpes[i] > 0.5 or sharpes[i] < 0 or max_dds[i] < -50:
        ax.annotate(name, (max_dds[i], sharpes[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

# Add benchmark
ax.scatter([benchmark['max_drawdown']*100], [benchmark['sharpe']],
           marker='*', s=200, c='gray', edgecolors='black', label='SPY B&H')

ax.set_xlabel('Maximum Drawdown (%)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Risk-Return Tradeoff: Sharpe Ratio vs Maximum Drawdown')
ax.legend()

# Add quadrant lines
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=-30, color='gray', linestyle=':', alpha=0.5)

plt.colorbar(scatter, label='Total Return (%)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "sharpe_vs_maxdd.pdf"))
plt.close()
print(f"    sharpe_vs_maxdd.pdf generada")


# =============================================================================
# FIGURA 8: DA VS RETURN - LA PARADOJA (da_vs_return_paradox.pdf)
# =============================================================================
# Contenido: (izq) Scatter de DA% vs Retorno% con correlacion rho, (der) bar chart DA vs DA@3x
# Fuente: models_summary.json (directional_accuracy, total_return), daily_data.json (positions),
#         market_data.json (returns para calcular DA@3x)
# DA@3x = hit rate en dias con posicion==3: mide la precision en las apuestas mas agresivas
# Nota: Modelos con DA@3x > 50% pero DA general < 50% son los mejores (paradoja)
# =============================================================================
print("\n[9] Generando figura DA vs Return...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Get DA and returns data
da_values = [m.get('directional_accuracy', m.get('win_rate', 0.5)) * 100 for m in models]
returns = [m['total_return'] * 100 for m in models]
names = [m['model'] for m in models]

# Left plot: DA vs Return scatter
scatter = ax1.scatter(da_values, returns, c=returns, cmap='RdYlGn', s=120,
                     alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels for key models
key_models = ['LSTM_Attention', 'Ridge', 'RandomForest', 'CNN_LSTM', 'GARCH', 'ExponentialSmoothing']
for i, name in enumerate(names):
    if name in key_models or returns[i] > 200 or returns[i] < -10:
        ax1.annotate(name, (da_values[i], returns[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

# Add horizontal line at SPY return
ax1.axhline(y=benchmark['total_return']*100, color='gray', linestyle='--',
           label=f'SPY B&H ({benchmark["total_return"]*100:.1f}%)', alpha=0.7)

# Add correlation line
z = np.polyfit(da_values, returns, 1)
p = np.poly1d(z)
x_line = np.linspace(min(da_values), max(da_values), 100)
corr = np.corrcoef(da_values, returns)[0,1]
ax1.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2,
         label=f'Correlation: ρ = {corr:.2f}')

ax1.set_xlabel('Directional Accuracy (%)')
ax1.set_ylabel('Retorno Total (%)')
ax1.set_title('La Paradoja del Directional Accuracy')
ax1.legend(loc='upper left')

# Add vertical line at 50%
ax1.axvline(x=50, color='black', linestyle=':', alpha=0.3)

# Right plot: Bar chart comparing DA general vs DA@3x for key models
# Calculate DA@3x from daily data
da_3x_data = []
models_for_bar = ['LSTM_Attention', 'Ridge', 'RandomForest', 'CNN_LSTM', 'ExponentialSmoothing', 'BiGRU']

for model_name in models_for_bar:
    if model_name in daily_data:
        data = daily_data[model_name]
        positions = np.array(data.get('positions', []))

        # Get market returns for DA@3x calculation
        mkt_returns = np.array(market_data.get('returns', []))[:len(positions)]

        if len(positions) > 0 and len(mkt_returns) > 0:
            # DA general: use the REAL DA from models_summary (sign(pred) == sign(actual))
            model_data = next((m for m in models if m['model'] == model_name), None)
            da_general = model_data.get('directional_accuracy', 0.5) * 100 if model_data else 50

            # DA@3x (when position == 3) — require min 30 days for reliability
            mask_3x = positions == 3
            n_3x = np.sum(mask_3x)
            if n_3x >= 30:
                correct_3x = np.sum(mkt_returns[mask_3x] > 0)
                da_3x = correct_3x / n_3x * 100
            else:
                da_3x = 50  # default for insufficient sample

            ret = model_data['total_return'] * 100 if model_data else 0

            da_3x_data.append({
                'model': model_name,
                'da_general': da_general,
                'da_3x': da_3x,
                'return': ret
            })

if da_3x_data:
    x = np.arange(len(da_3x_data))
    width = 0.35

    da_gen = [d['da_general'] for d in da_3x_data]
    da_3x_vals = [d['da_3x'] for d in da_3x_data]
    model_labels = [d['model'] for d in da_3x_data]

    bars1 = ax2.bar(x - width/2, da_gen, width, label='DA General', color='#457b9d', alpha=0.8)
    bars2 = ax2.bar(x + width/2, da_3x_vals, width, label='DA@3x (Alta Convicci\u00f3n)', color='#2a9d8f', alpha=0.8)

    ax2.set_xlabel('Modelo')
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('DA General vs DA@3x: Lo Que Realmente Importa')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=50, color='black', linestyle=':', alpha=0.3, label='Random (50%)')

    # Add return annotations on top of bars
    for i, d in enumerate(da_3x_data):
        color = '#2a9d8f' if d['return'] > 100 else '#e63946' if d['return'] < 0 else '#457b9d'
        ax2.annotate(f"{d['return']:+.0f}%",
                    xy=(i, max(d['da_general'], d['da_3x']) + 1),
                    ha='center', fontsize=8, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "da_vs_return_paradox.pdf"))
plt.close()
print(f"    da_vs_return_paradox.pdf generada")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("FIGURAS GENERADAS")
print("=" * 80)
print(f"\nDirectorio: {FIGURES_DIR}")
print("\nArchivos generados:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith('.pdf'):
        print(f"  - {f}")

print("\n" + "=" * 80)
print("Para incluir en el paper, usar:")
print("  \\includegraphics[width=\\textwidth]{figures/equity_curves_top5}")
print("  etc.")
print("=" * 80)
