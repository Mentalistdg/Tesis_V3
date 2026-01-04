# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE DATOS PARA LA APP - VERSION CORREGIDA
================================================================================
Lee los resultados de la estrategia optimizada y genera datos para la app web.

IMPORTANTE: Este script replica EXACTAMENTE la lógica de evaluate_strategy.py
para asegurar consistencia entre las métricas del CSV y lo que muestra la app.

Incluye:
- Mismas funciones de posicionamiento (quantile_position_asymmetric)
- Mismos ajustes de risk management (drawdown control, position filter, vol targeting)
- Mismas definiciones de métricas (pct_long = positions > 0, etc.)
- Misma lógica de cálculo de retornos realistas

Utiliza:
- results/strategy_realistic_evaluation.csv
- results/bootstrap_confidence.json
- results/optimal_model_params.json
- models/trained_artifacts.pkl

Genera:
- app/backend/data/daily_data.json
- app/backend/data/models_summary.json
- app/backend/data/market_data.json
- app/backend/data/regime_data.json
================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
APP_DATA_DIR = os.path.join(BASE_DIR, "app", "backend", "data")

os.makedirs(APP_DATA_DIR, exist_ok=True)

print("=" * 80)
print("GENERADOR DE DATOS PARA APP - VERSION CORREGIDA")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CONFIGURACION (IDENTICA A evaluate_strategy.py)
# =============================================================================
CONFIG = {
    'initial_capital': 10000,

    'instruments': {
        'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005, 'leverage': 3},
        'SPY': {'expense_ratio': 0.0009, 'bid_ask': 0.0002, 'leverage': 1},
        'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000, 'leverage': 0},
        'SH': {'expense_ratio': 0.0089, 'bid_ask': 0.0004, 'leverage': -1},
        'SPXU': {'expense_ratio': 0.0089, 'bid_ask': 0.0006, 'leverage': -3},
    },

    'drawdown_limits': {
        'level_1': {'threshold': 0.10, 'max_leverage': 2},
        'level_2': {'threshold': 0.15, 'max_leverage': 1},
        'level_3': {'threshold': 0.20, 'max_leverage': 0},
    },

    'min_position_change': 1,
    'volatility_target': 0.15,
    'volatility_lookback': 21,
}


# =============================================================================
# FUNCIONES DE POSICIONAMIENTO (COPIADAS DE evaluate_strategy.py)
# =============================================================================

def quantile_position_asymmetric(predictions, window=63,
                                  q_long_extreme=10, q_long_moderate=30,
                                  q_short_extreme=10, q_short_moderate=30):
    """
    Calcula posiciones basadas en percentiles rolling con umbrales ASIMETRICOS.
    IDENTICA a evaluate_strategy.py
    """
    predictions = np.array(predictions).flatten()
    n = len(predictions)
    positions = np.zeros(n)
    percentiles = np.zeros(n)

    thresh_long_3x = 100 - q_long_extreme
    thresh_long_1x = 100 - q_long_moderate
    thresh_short_3x = q_short_extreme
    thresh_short_1x = q_short_moderate

    for i in range(n):
        start = max(0, i - window + 1)
        window_preds = predictions[start:i+1]

        if len(window_preds) < 5:
            positions[i] = 0
            percentiles[i] = 50
            continue

        current_pred = predictions[i]
        percentile = np.mean(window_preds <= current_pred) * 100
        percentiles[i] = percentile

        if percentile >= thresh_long_3x:
            positions[i] = 3
        elif percentile >= thresh_long_1x:
            positions[i] = 1
        elif percentile <= thresh_short_3x:
            positions[i] = -3
        elif percentile <= thresh_short_1x:
            positions[i] = -1
        else:
            positions[i] = 0

    return positions, percentiles


# =============================================================================
# FUNCIONES DE RISK MANAGEMENT (COPIADAS DE evaluate_strategy.py)
# =============================================================================

def apply_drawdown_control(positions, cumulative_returns, limits):
    """
    Reduce posición cuando el drawdown excede umbrales.
    IMPORTANTE: Solo genera posiciones válidas {-3, -1, 0, +1, +3}

    Lógica de reducción:
    - Drawdown 10-15%: Reduce 3x → 1x (más conservador)
    - Drawdown 15-20%: Reduce todo a máximo 1x
    - Drawdown >20%: Fuerza cash (0)
    """
    positions = np.array(positions).flatten()
    cumulative_returns = np.array(cumulative_returns).flatten()

    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-10)

    adjusted_positions = positions.copy()

    for i in range(len(positions)):
        dd = abs(drawdown[i])
        pos = adjusted_positions[i]

        if dd >= limits['level_3']['threshold']:
            # Drawdown > 20%: Fuerza cash
            adjusted_positions[i] = 0
        elif dd >= limits['level_2']['threshold']:
            # Drawdown 15-20%: Máximo 1x en cualquier dirección
            if abs(pos) > 1:
                adjusted_positions[i] = np.sign(pos) * 1
        elif dd >= limits['level_1']['threshold']:
            # Drawdown 10-15%: Reduce 3x a 1x (no permite 3x leverage)
            if abs(pos) == 3:
                adjusted_positions[i] = np.sign(pos) * 1
        # else: sin drawdown significativo, mantener posición original

    return adjusted_positions


def apply_position_filter(positions, min_change=1):
    """Filtra cambios de posición menores al umbral."""
    positions = np.array(positions).flatten()
    filtered = np.zeros_like(positions)
    filtered[0] = positions[0]

    for i in range(1, len(positions)):
        change = abs(positions[i] - filtered[i-1])
        if change >= min_change:
            filtered[i] = positions[i]
        else:
            filtered[i] = filtered[i-1]

    return filtered


def apply_volatility_targeting(positions, returns, target_vol=0.15, lookback=21):
    """Ajusta posiciones para mantener volatilidad objetivo."""
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()

    adjusted = positions.copy()

    for i in range(lookback, len(positions)):
        recent_vol = np.std(returns[i-lookback:i]) * np.sqrt(252)

        if recent_vol > 0:
            vol_scalar = target_vol / recent_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
            adjusted_continuous = positions[i] * vol_scalar

            if adjusted_continuous <= -2:
                adjusted[i] = -3
            elif adjusted_continuous <= -0.5:
                adjusted[i] = -1
            elif adjusted_continuous < 0.5:
                adjusted[i] = 0
            elif adjusted_continuous < 2:
                adjusted[i] = 1
            else:
                adjusted[i] = 3

    return adjusted


# =============================================================================
# FUNCIONES DE COSTOS Y RETORNOS (COPIADAS DE evaluate_strategy.py)
# =============================================================================

def get_instrument_for_position(position):
    """Retorna el instrumento correspondiente a una posición."""
    if position == 3:
        return 'UPRO'
    elif position == 1:
        return 'SPY'
    elif position == 0:
        return 'CASH'
    elif position == -1:
        return 'SH'
    elif position == -3:
        return 'SPXU'
    else:
        return 'CASH'


def calculate_realistic_returns(positions, forward_returns, risk_free_rate,
                                config, apply_vol_drag=True):
    """Calcula retornos realistas incluyendo todos los costos."""
    positions = np.array(positions).flatten()
    forward_returns = np.array(forward_returns).flatten()
    risk_free_rate = np.array(risk_free_rate).flatten()

    n_days = len(positions)

    gross_returns = np.zeros(n_days)
    expense_costs = np.zeros(n_days)
    trading_costs = np.zeros(n_days)
    vol_drag_costs = np.zeros(n_days)

    if apply_vol_drag:
        lookback = min(21, n_days // 4)
        rolling_variance = pd.Series(forward_returns).rolling(lookback).var().fillna(0).values

    for i in range(n_days):
        pos = positions[i]
        instrument = get_instrument_for_position(pos)
        instrument_config = config['instruments'][instrument]

        # Retorno bruto
        if pos == 0:
            gross_returns[i] = risk_free_rate[i]
        else:
            gross_returns[i] = risk_free_rate[i] + pos * (forward_returns[i] - risk_free_rate[i])

        # Expense ratio (diario)
        daily_expense = instrument_config['expense_ratio'] / 252
        expense_costs[i] = daily_expense

        # Trading cost
        if i > 0 and positions[i] != positions[i-1]:
            prev_instrument = get_instrument_for_position(positions[i-1])
            sell_cost = config['instruments'][prev_instrument]['bid_ask']
            buy_cost = instrument_config['bid_ask']
            trading_costs[i] = sell_cost + buy_cost

        # Volatility drag
        if apply_vol_drag and abs(pos) == 3:
            leverage = pos
            if i >= lookback:
                var = rolling_variance[i]
            else:
                var = np.var(forward_returns[:max(i+1, 5)])
            vol_drag_costs[i] = 0.5 * (leverage**2 - abs(leverage)) * var

    net_returns = gross_returns - expense_costs - trading_costs - vol_drag_costs

    return {
        'gross_returns': gross_returns,
        'net_returns': net_returns,
        'expense_costs': expense_costs,
        'trading_costs': trading_costs,
        'vol_drag_costs': vol_drag_costs,
        'total_costs': np.sum(expense_costs) + np.sum(trading_costs) + np.sum(vol_drag_costs),
    }


# =============================================================================
# FUNCIONES DE MÉTRICAS (CONSISTENTES CON evaluate_strategy.py)
# =============================================================================

def calculate_metrics(returns, positions, risk_free_rate=None, market_returns=None):
    """
    Calcula métricas EXACTAMENTE como evaluate_strategy.py
    IMPORTANTE: pct_long = positions > 0 (incluye 1 y 3)
    """
    returns = np.array(returns).flatten()
    positions = np.array(positions).flatten()

    # Retorno total
    total_return = np.prod(1 + returns) - 1

    # Retorno anualizado
    n_years = len(returns) / 252
    if n_years > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (1/n_years) - 1
    else:
        annualized_return = 0

    # Volatilidad
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)

    # Sharpe
    if annual_vol > 0:
        rf_annual = np.mean(risk_free_rate) * 252 if risk_free_rate is not None else 0.02
        sharpe = (annualized_return - rf_annual) / annual_vol
    else:
        sharpe = 0

    # Sortino
    rf_annual = np.mean(risk_free_rate) * 252 if risk_free_rate is not None else 0.02
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = (annualized_return - rf_annual) / downside_vol if downside_vol > 0 else 0
    else:
        sortino = sharpe * 2

    # Max Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Calmar
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate (días con retorno positivo)
    win_rate = np.mean(returns > 0)

    # Directional accuracy
    # Nota: Esto requiere market_returns para comparar, se calculará aparte

    # Position metrics (CONSISTENTE CON evaluate_strategy.py)
    # pct_long = cualquier posición > 0 (incluye 1 y 3)
    pct_long = np.mean(positions > 0) * 100
    pct_short = np.mean(positions < 0) * 100
    pct_cash = np.mean(positions == 0) * 100
    pct_leveraged_long = np.mean(positions == 3) * 100
    pct_leveraged_short = np.mean(positions == -3) * 100

    # Trades (cambios de posición)
    position_changes = np.abs(np.diff(positions))
    n_trades = int(np.sum(position_changes > 0))

    # Market return y excess return
    if market_returns is not None:
        market_returns = np.array(market_returns).flatten()
        market_return = np.prod(1 + market_returns) - 1
        excess_return = total_return - market_return
    else:
        market_return = 0
        excess_return = 0

    return {
        'total_return': total_return,
        'annual_return': annualized_return,
        'market_return': market_return,
        'excess_return': excess_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'pct_long': pct_long,
        'pct_short': pct_short,
        'pct_cash': pct_cash,
        'pct_3x_long': pct_leveraged_long,
        'pct_3x_short': pct_leveraged_short,
        'mean_position': np.mean(positions),
        'n_trades': n_trades,
        'n_days': len(returns),
    }


def generate_trade_log(positions, returns, dates, market_returns=None, predictions=None,
                       percentiles=None, base_positions=None):
    """
    Genera log de trades basado en cambios de posición.
    Un trade = cada vez que la posición cambia.

    Incluye:
    - market_return: retorno del mercado durante el trade
    - entry_prediction: predicción del modelo al entrar (retorno esperado)
    - entry_percentile: percentil de la predicción en ventana rolling de 63 días
    - base_position: posición ANTES de risk management (solo quantile strategy)
    - entry_position: posición FINAL después de risk management

    El percentil es clave para entender las posiciones:
    - Percentil > 90: posición +3x (predicción muy alta vs histórico reciente)
    - Percentil > 70: posición +1x
    - Percentil < 10: posición -3x (predicción muy baja vs histórico reciente)
    - Percentil < 30: posición -1x
    - Else: cash (0)
    """
    trades = []
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()
    if market_returns is not None:
        market_returns = np.array(market_returns).flatten()
    if predictions is not None:
        predictions = np.array(predictions).flatten()
    if percentiles is not None:
        percentiles = np.array(percentiles).flatten()
    if base_positions is not None:
        base_positions = np.array(base_positions).flatten()

    i = 0
    while i < len(positions):
        # Buscar inicio de trade (posición != 0 o cambio de posición)
        if i == 0 or positions[i] != positions[i-1]:
            entry_idx = i
            entry_pos = positions[i]
            entry_date = str(dates[i])[:10]

            # Acumular retornos hasta que cambie la posición
            trade_returns = []
            trade_market_returns = []
            j = i
            while j < len(positions) and positions[j] == entry_pos:
                if j < len(returns):
                    trade_returns.append(returns[j])
                if market_returns is not None and j < len(market_returns):
                    trade_market_returns.append(market_returns[j])
                j += 1

            exit_idx = j - 1
            exit_date = str(dates[exit_idx])[:10]
            exit_pos = positions[exit_idx] if exit_idx < len(positions) else entry_pos

            if len(trade_returns) > 0:
                total_return = float(np.prod(1 + np.array(trade_returns)) - 1)
                market_return = float(np.prod(1 + np.array(trade_market_returns)) - 1) if trade_market_returns else 0.0

                trade_data = {
                    'entry_idx': int(entry_idx),
                    'entry_date': entry_date,
                    'entry_position': float(entry_pos),
                    'exit_idx': int(exit_idx),
                    'exit_date': exit_date,
                    'exit_position': float(exit_pos),
                    'duration': int(exit_idx - entry_idx + 1),
                    'total_return': total_return,
                    'market_return': market_return,
                }

                # Agregar predicción y percentil si están disponibles
                if predictions is not None and entry_idx < len(predictions):
                    trade_data['entry_prediction'] = float(predictions[entry_idx])
                if percentiles is not None and entry_idx < len(percentiles):
                    trade_data['entry_percentile'] = float(percentiles[entry_idx])
                # Agregar posición base (antes de risk management)
                if base_positions is not None and entry_idx < len(base_positions):
                    trade_data['base_position'] = float(base_positions[entry_idx])

                trades.append(trade_data)

            i = j
        else:
            i += 1

    return trades


def detect_regime(returns, window=60):
    """Detecta el régimen de mercado."""
    regimes = []
    for i in range(len(returns)):
        if i < window:
            regimes.append('unknown')
            continue
        window_returns = returns[i-window:i]
        cum_return = np.prod(1 + window_returns) - 1
        volatility = np.std(window_returns) * np.sqrt(252)

        if cum_return > 0.10 and volatility < 0.20:
            regime = 'bull'
        elif cum_return < -0.10:
            regime = 'bear'
        elif volatility > 0.25:
            regime = 'high_vol'
        else:
            regime = 'sideways'
        regimes.append(regime)
    return regimes


# =============================================================================
# CARGAR RESULTADOS
# =============================================================================
print("\n[1] Cargando resultados...")

# Evaluation CSV
eval_csv_path = os.path.join(RESULTS_DIR, "strategy_realistic_evaluation.csv")
eval_df = pd.read_csv(eval_csv_path)
print(f"    Modelos evaluados: {len(eval_df)}")

# Bootstrap confidence
bootstrap_path = os.path.join(RESULTS_DIR, "bootstrap_confidence.json")
with open(bootstrap_path, 'r') as f:
    bootstrap_data = json.load(f)
print(f"    Bootstrap CIs cargados: {len(bootstrap_data)} modelos")

# Optimal params
params_path = os.path.join(RESULTS_DIR, "optimal_model_params.json")
with open(params_path, 'r') as f:
    optimal_params = json.load(f)
print(f"    Parámetros optimizados: {len(optimal_params)} modelos")

# Summary
summary_path = os.path.join(RESULTS_DIR, "strategy_summary.json")
with open(summary_path, 'r') as f:
    strategy_summary = json.load(f)
print(f"    Periodo de test: {strategy_summary['test_period']['start']} - {strategy_summary['test_period']['end']}")

# Trained artifacts
artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
with open(artifacts_path, 'rb') as f:
    artifacts = pickle.load(f)
print(f"    Artefactos cargados: {len(artifacts['models'])} modelos")


# =============================================================================
# GENERAR DATOS POR MODELO
# =============================================================================
print("\n[2] Generando datos por modelo (con risk management)...")

metadata = artifacts['metadata']
dates_test = pd.to_datetime(metadata['dates_test'])
forward_returns = metadata['forward_returns_test']
risk_free = metadata['risk_free_test']

# Alinear longitudes
market_returns = forward_returns[:-1]
test_dates = dates_test[:-1]
rf_aligned = risk_free[:-1]

MODEL_CATEGORIES = {
    'Ridge': 'ML', 'Lasso': 'ML', 'ElasticNet': 'ML',
    'RandomForest': 'ML', 'GradientBoosting': 'ML',
    'XGBoost': 'GradientBoosting', 'LightGBM': 'GradientBoosting', 'CatBoost': 'GradientBoosting',
    'AutoARIMA': 'TimeSeries', 'ExponentialSmoothing': 'TimeSeries', 'SeasonalNaive': 'TimeSeries',
    'Prophet': 'Prophet', 'GARCH': 'GARCH',
    'DLinear': 'DeepLearning', 'NBEATS': 'DeepLearning', 'NHiTS': 'DeepLearning',
    'TCN': 'DeepLearning', 'TFT': 'DeepLearning'
}

all_models_data = {}
all_models_summary = []

# Regimes
regimes = detect_regime(market_returns)

for _, row in eval_df.iterrows():
    model_name = row['model']
    print(f"    {model_name}...", end=" ")

    if model_name not in artifacts['models']:
        print("[SKIP - no predictions]")
        continue

    predictions = artifacts['models'][model_name]['test_predictions']

    # Obtener parámetros óptimos
    params = bootstrap_data.get(model_name, {}).get('params', {
        'q_long_extreme': 30,
        'q_long_moderate': 50,
        'q_short_extreme': 10,
        'q_short_moderate': 30
    })

    # Alinear longitudes
    min_len = min(len(predictions), len(market_returns), len(rf_aligned))
    pred = predictions[:min_len]
    mkt_ret = market_returns[:min_len]
    rf = rf_aligned[:min_len]

    # ==========================================================================
    # PASO 1: Calcular posiciones base con quantile
    # ==========================================================================
    positions_base, percentiles = quantile_position_asymmetric(
        pred,
        q_long_extreme=params['q_long_extreme'],
        q_long_moderate=params['q_long_moderate'],
        q_short_extreme=params['q_short_extreme'],
        q_short_moderate=params['q_short_moderate']
    )

    # ==========================================================================
    # PASO 2: Aplicar volatility targeting
    # ==========================================================================
    positions_vol = apply_volatility_targeting(
        positions_base, mkt_ret,
        target_vol=CONFIG['volatility_target'],
        lookback=CONFIG['volatility_lookback']
    )

    # ==========================================================================
    # PASO 3: Aplicar position filter (anti-churning)
    # ==========================================================================
    positions_filtered = apply_position_filter(
        positions_vol,
        min_change=CONFIG['min_position_change']
    )

    # ==========================================================================
    # PASO 4: Calcular retornos temporales para drawdown control
    # ==========================================================================
    returns_temp = calculate_realistic_returns(
        positions_filtered, mkt_ret, rf, CONFIG, apply_vol_drag=True
    )
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])

    # ==========================================================================
    # PASO 5: Aplicar drawdown control
    # ==========================================================================
    positions_final = apply_drawdown_control(
        positions_filtered, cumulative_temp, CONFIG['drawdown_limits']
    )

    # ==========================================================================
    # PASO 6: Calcular retornos finales
    # ==========================================================================
    returns_final = calculate_realistic_returns(
        positions_final, mkt_ret, rf, CONFIG, apply_vol_drag=True
    )
    strategy_returns = returns_final['net_returns']

    # Equity curves
    equity_curve = np.cumprod(1 + strategy_returns)
    market_equity = np.cumprod(1 + mkt_ret)

    # Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    # Directional accuracy
    dir_accuracy = np.mean((strategy_returns > 0) == (mkt_ret > 0))

    # Trade log (basado en posiciones finales)
    # Incluye predictions, percentiles y base_positions para transparencia académica
    trade_dates = test_dates[:min_len]
    trade_log = generate_trade_log(
        positions_final, strategy_returns, trade_dates,
        market_returns=mkt_ret,
        predictions=pred,
        percentiles=percentiles,
        base_positions=positions_base  # Posición antes de risk management
    )

    # ==========================================================================
    # CALCULAR MÉTRICAS (consistentes con CSV)
    # ==========================================================================
    metrics = calculate_metrics(strategy_returns, positions_final, rf, mkt_ret)
    metrics['dir_accuracy'] = dir_accuracy

    # IMPORTANTE: Contar solo trades con market exposure (position != 0)
    # Esto es lo que importa para los profesores - trades activos, no periodos en cash
    n_active_trades = sum(1 for t in trade_log if abs(t['entry_position']) >= 0.5)
    metrics['n_trades'] = n_active_trades  # Sobreescribir con el conteo correcto

    # Agregar datos del CSV para Bootstrap CI
    metrics['sharpe_ci_lower'] = float(row.get('sharpe_ci_lower', 0))
    metrics['sharpe_ci_upper'] = float(row.get('sharpe_ci_upper', 0))
    metrics['prob_sharpe_positive'] = float(row.get('prob_sharpe_positive', 0))
    metrics['transaction_costs'] = float(row.get('opt_total_costs', returns_final['total_costs']))

    # Datos originales para comparación
    metrics['orig_return'] = float(row.get('orig_return', 0))
    metrics['orig_sharpe'] = float(row.get('orig_sharpe', 0))
    metrics['return_improvement'] = float(row.get('return_improvement', 0))
    metrics['sharpe_improvement'] = float(row.get('sharpe_improvement', 0))

    # Categoría
    category = MODEL_CATEGORIES.get(model_name, 'Other')

    # Verificar consistencia con CSV
    csv_n_trades = int(row.get('opt_total_trades', 0))
    csv_pct_long = float(row.get('opt_pct_long', 0))

    print(f"n_trades: {metrics['n_trades']} (CSV: {csv_n_trades}), "
          f"pct_long: {metrics['pct_long']:.1f}% (CSV: {csv_pct_long:.1f}%)")

    # Summary para tabla
    summary = {
        'model': model_name,
        'category': category,
        **metrics,
        'optimal_params': params,
        'final_equity': float(row['final_equity']),
        'profit_loss': float(row['profit_loss']),
    }
    all_models_summary.append(summary)

    # Daily data para gráficos
    all_models_data[model_name] = {
        'category': category,
        'dates': [str(d)[:10] for d in trade_dates],
        'predictions': pred.tolist(),
        'percentiles': percentiles.tolist(),  # Percentil rolling para entender decisiones
        'positions': positions_final.tolist(),
        'strategy_returns': strategy_returns.tolist(),
        'market_returns': mkt_ret.tolist(),
        'equity_curve': equity_curve.tolist(),
        'market_equity': market_equity.tolist(),
        'drawdown': drawdown.tolist(),
        'regimes': regimes[:min_len],
        'trades': trade_log,
        'metrics': metrics,
        'optimal_params': params,
    }


# =============================================================================
# GUARDAR DATOS
# =============================================================================
print("\n[3] Guardando datos para la app...")

# 1. Daily data
daily_data_file = os.path.join(APP_DATA_DIR, "daily_data.json")
with open(daily_data_file, 'w') as f:
    json.dump(all_models_data, f)
print(f"    [OK] {daily_data_file}")

# 2. Models summary
summary_file = os.path.join(APP_DATA_DIR, "models_summary.json")
summary = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'initial_capital': strategy_summary['config']['initial_capital'],
        'volatility_target': strategy_summary['config']['volatility_target'],
        'n_bootstrap': strategy_summary['config']['n_bootstrap'],
        'confidence_level': strategy_summary['config']['confidence_level'],
    },
    'test_period': strategy_summary['test_period'],
    'benchmark': strategy_summary['benchmark'],
    'best_model': strategy_summary['best_model'],
    'models_beating_benchmark': strategy_summary['models_beating_benchmark'],
    'total_models': strategy_summary['total_models'],
    'models': sorted(all_models_summary, key=lambda x: x['sharpe'], reverse=True)
}
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    [OK] {summary_file}")

# 3. Market data
market_data_file = os.path.join(APP_DATA_DIR, "market_data.json")
market_equity_full = np.cumprod(1 + market_returns)
market_running_max = np.maximum.accumulate(market_equity_full)
market_drawdown = (market_equity_full - market_running_max) / market_running_max

market_data = {
    'dates': [str(d)[:10] for d in test_dates],
    'returns': market_returns.tolist(),
    'equity_curve': market_equity_full.tolist(),
    'drawdown': market_drawdown.tolist(),
    'regimes': regimes,
    'metrics': {
        'total_return': float(strategy_summary['benchmark']['spy_total_return']),
        'annual_return': float((1 + strategy_summary['benchmark']['spy_total_return']) ** (252 / len(market_returns)) - 1),
        'sharpe': float(np.mean(market_returns) / np.std(market_returns) * np.sqrt(252)),
        'max_drawdown': float(np.min(market_drawdown)),
        'volatility': float(np.std(market_returns) * np.sqrt(252)),
        'final_value': float(strategy_summary['benchmark']['spy_final_value']),
    }
}
with open(market_data_file, 'w') as f:
    json.dump(market_data, f)
print(f"    [OK] {market_data_file}")

# 4. Regime data
regime_file = os.path.join(APP_DATA_DIR, "regime_data.json")
regime_counts = {}
for r in regimes:
    regime_counts[r] = regime_counts.get(r, 0) + 1

regime_performance = {}
for model_name, data in all_models_data.items():
    regime_performance[model_name] = {}
    for regime in ['bull', 'bear', 'sideways', 'high_vol']:
        regime_mask = np.array(data['regimes']) == regime
        if regime_mask.sum() > 0:
            regime_returns = np.array(data['strategy_returns'])[regime_mask[:len(data['strategy_returns'])]]
            regime_performance[model_name][regime] = {
                'total_return': float(np.prod(1 + regime_returns) - 1),
                'n_days': int(regime_mask.sum())
            }

regime_data = {
    'current_regime': regimes[-1] if regimes else 'unknown',
    'regime_counts': regime_counts,
    'regime_performance': regime_performance
}
with open(regime_file, 'w') as f:
    json.dump(regime_data, f, indent=2)
print(f"    [OK] {regime_file}")


# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "=" * 80)
print("[OK] DATOS GENERADOS EXITOSAMENTE")
print("=" * 80)
print(f"\nArchivos generados en: {APP_DATA_DIR}")
print(f"  - daily_data.json ({len(all_models_data)} modelos)")
print(f"  - models_summary.json")
print(f"  - market_data.json")
print(f"  - regime_data.json")

# Verificación de consistencia
print("\n" + "-" * 40)
print("VERIFICACIÓN DE CONSISTENCIA:")
print("-" * 40)
for m in summary['models'][:5]:
    csv_row = eval_df[eval_df['model'] == m['model']].iloc[0] if len(eval_df[eval_df['model'] == m['model']]) > 0 else None
    if csv_row is not None:
        csv_pct_long = csv_row['opt_pct_long']
        app_pct_long = m['pct_long']
        diff = abs(csv_pct_long - app_pct_long)
        status = "OK" if diff < 5 else "DIFF"
        print(f"  {m['model']:20} pct_long: CSV={csv_pct_long:.1f}% App={app_pct_long:.1f}% [{status}]")

print("\n" + "-" * 40)
print("Top 5 Modelos por Sharpe:")
print("-" * 40)
for i, m in enumerate(summary['models'][:5]):
    print(f"  {i+1}. {m['model']:20} Sharpe: {m['sharpe']:.3f}  Return: {m['total_return']*100:.1f}%  P(Sharpe>0): {m['prob_sharpe_positive']:.1%}")
