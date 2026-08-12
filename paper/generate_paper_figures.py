# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE FIGURAS PARA EL PAPER
================================================================================
Genera figuras de alta calidad para el paper academico usando los datos
de la app.

Figuras generadas:
- figures/equity_curves_top5.pdf
- figures/pnl_distribution_lstm_attention.pdf
- figures/turnover_vs_return.pdf
- figures/da_vs_return_paradox.pdf
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

# Nombres de modelos para mostrar (coinciden con la nomenclatura del paper)
DISPLAY_NAMES = {
    'LSTM_Attention': 'LSTM+Attention',
    'CNN_LSTM': 'CNN-LSTM',
    'NBEATS': 'N-BEATS',
    'NHiTS': 'N-HiTS',
}

def disp(name):
    return DISPLAY_NAMES.get(name, name)

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
# =============================================================================
print("\n[1] Cargando datos JSON...")

with open(os.path.join(APP_DATA_DIR, "models_summary.json"), 'r') as f:
    summary_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "daily_data.json"), 'r') as f:
    daily_data = json.load(f)

with open(os.path.join(APP_DATA_DIR, "market_data.json"), 'r') as f:
    market_data = json.load(f)

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
        ax.plot(dates, equity, label=f"{disp(model_name)} (+{model['total_return']*100:.0f}%)",
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
        ax.annotate(disp(name), (n_trades[i], returns[i]), fontsize=8,
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
# FIGURA 4: DA VS RETURN - LA PARADOJA (da_vs_return_paradox.pdf)
# =============================================================================
# Contenido: (izq) Scatter de DA% vs Retorno% con correlacion rho, (der) bar chart DA vs DA@3x
# Fuente: models_summary.json (directional_accuracy, total_return), daily_data.json (positions),
#         market_data.json (returns para calcular DA@3x)
# DA@3x = hit rate en dias con posicion==3: mide la precision en las apuestas mas agresivas
# Nota: Modelos con DA@3x > 50% pero DA general < 50% son los mejores (paradoja)
# =============================================================================
print("\n[5] Generando figura DA vs Return...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Get DA and returns data
da_values = [m.get('directional_accuracy', m.get('win_rate', 0.5)) * 100 for m in models]
returns = [m['total_return'] * 100 for m in models]
names = [m['model'] for m in models]

# Left plot: DA vs Return scatter
scatter = ax1.scatter(da_values, returns, c=returns, cmap='RdYlGn', s=120,
                     alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels for key models with per-model offsets to avoid overlap
display_names = {
    'LSTM_Attention': 'LSTM+Attention',
    'Ridge': 'Ridge',
    'RandomForest': 'RandomForest',
    'CNN_LSTM': 'CNN-LSTM',
    'GARCH': 'GARCH',
    'ExponentialSmoothing': 'ExpSmoothing',
    'BiGRU': 'BiGRU',
}
label_offsets = {
    'LSTM_Attention': (8, -18),
    'Ridge': (-45, 10),
    'RandomForest': (8, 10),
    'CNN_LSTM': (-65, -5),
    'GARCH': (8, 5),
    'ExponentialSmoothing': (8, 8),
    'BiGRU': (8, 5),
}
key_models = set(display_names.keys())
for i, name in enumerate(names):
    if name in key_models or returns[i] > 200 or returns[i] < -10:
        offset = label_offsets.get(name, (5, 5))
        dname = display_names.get(name, name)
        ax1.annotate(dname, (da_values[i], returns[i]), fontsize=7,
                    xytext=offset, textcoords='offset points')

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
ax1.set_ylim(-60, 450)
ax1.legend(loc='upper right', fontsize=8)

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

    # Short display names for x-axis
    short_names = {
        'LSTM_Attention': 'LSTM+Att.',
        'RandomForest': 'RF',
        'CNN_LSTM': 'CNN-LSTM',
        'ExponentialSmoothing': 'ExpSmooth.',
        'Ridge': 'Ridge',
        'BiGRU': 'BiGRU',
    }
    model_labels = [short_names.get(d['model'], d['model']) for d in da_3x_data]

    bars1 = ax2.bar(x - width/2, da_gen, width, label='DA General', color='#457b9d', alpha=0.8)
    bars2 = ax2.bar(x + width/2, da_3x_vals, width, label='DA@3x', color='#2a9d8f', alpha=0.8)

    ax2.set_xlabel('Modelo')
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('DA General vs DA@3x')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.axhline(y=50, color='black', linestyle=':', alpha=0.3)

    # Add return annotations on top of bars
    for i, d in enumerate(da_3x_data):
        color = '#2a9d8f' if d['return'] > 100 else '#e63946' if d['return'] < 0 else '#457b9d'
        ax2.annotate(f"{d['return']:+.0f}%",
                    xy=(i, max(d['da_general'], d['da_3x']) + 2),
                    ha='center', fontsize=7, fontweight='bold', color=color)

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
