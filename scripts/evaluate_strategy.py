# -*- coding: utf-8 -*-
"""
================================================================================
EVALUATE_STRATEGY.PY - Evaluacion Realista de Estrategia de Trading
================================================================================

Este script evalua la estrategia de trading con mejoras de nivel hedge fund:

MEJORAS IMPLEMENTADAS:
1. Sigmoid calibrada por modelo (Z-Score + scale=1)
2. Volatility drag de ETFs apalancados (UPRO/SPXU)
3. Costos reales (expense ratios + bid-ask spreads)
4. Control de drawdown dinamico
5. Filtro de churning (minimo cambio de posicion)
6. Volatility targeting
7. Bootstrap confidence intervals

ENTRADA:
- models/trained_artifacts.pkl (generado por train_models.py)

SALIDA:
- results/strategy_realistic_evaluation.csv
- results/strategy_summary.json
- results/bootstrap_confidence.json

USAGE:
    python scripts/evaluate_strategy.py

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

ARTIFACTS_FILE = "trained_artifacts.pkl"
OPTIMAL_PARAMS_FILE = "optimal_model_params.json"

CONFIG = {
    'initial_capital': 10000,  # $10,000 USD

    # Costos reales por instrumento (anualizados donde aplica)
    'instruments': {
        'UPRO': {
            'expense_ratio': 0.0091,  # 0.91% anual
            'bid_ask': 0.0005,        # 0.05% por trade
            'name': 'UPRO (3x Long)',
            'leverage': 3,
        },
        'SPY': {
            'expense_ratio': 0.0009,  # 0.09% anual
            'bid_ask': 0.0002,        # 0.02% por trade
            'name': 'SPY (1x Long)',
            'leverage': 1,
        },
        'CASH': {
            'expense_ratio': 0.0000,
            'bid_ask': 0.0000,
            'name': 'Cash',
            'leverage': 0,
        },
        'SH': {
            'expense_ratio': 0.0089,  # 0.89% anual
            'bid_ask': 0.0004,        # 0.04% por trade
            'name': 'SH (1x Short)',
            'leverage': -1,
        },
        'SPXU': {
            'expense_ratio': 0.0089,  # 0.89% anual
            'bid_ask': 0.0006,        # 0.06% por trade
            'name': 'SPXU (3x Short)',
            'leverage': -3,
        },
    },

    # Risk management
    'drawdown_limits': {
        'level_1': {'threshold': 0.10, 'max_leverage': 2},
        'level_2': {'threshold': 0.15, 'max_leverage': 1},
        'level_3': {'threshold': 0.20, 'max_leverage': 0},
    },

    # Position management
    'min_position_change': 1,      # Minimo cambio para ejecutar trade
    'volatility_target': 0.15,     # 15% volatilidad anual objetivo
    'volatility_lookback': 21,     # Dias para calcular vol

    # Bootstrap
    'n_bootstrap': 10000,
    'confidence_level': 0.95,

    # Sigmoid
    'original_scale': 500,         # Scale original del pipeline
    'calibrated_scale': 1.0,       # Scale para Z-Score normalizado
}

print("="*80)
print("EVALUATE_STRATEGY.PY - Evaluacion Realista de Estrategia")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print(f"Capital inicial: ${CONFIG['initial_capital']:,}")
print()

# =============================================================================
# FUNCIONES DE POSICIONAMIENTO
# =============================================================================

def sigmoid_to_position(predictions, scale=500):
    """
    Convierte predicciones a posiciones usando sigmoid.
    Retorna posiciones continuas en [-3, +3].
    """
    predictions = np.array(predictions).flatten()
    continuous_pos = 6 / (1 + np.exp(-scale * predictions)) - 3
    return continuous_pos

def discretize_position(continuous_pos):
    """
    Discretiza posiciones continuas a {-3, -1, 0, +1, +3}.
    """
    continuous_pos = np.array(continuous_pos).flatten()
    positions = np.zeros_like(continuous_pos)

    positions[continuous_pos <= -2] = -3           # SPXU (3x short)
    positions[(continuous_pos > -2) & (continuous_pos <= -0.5)] = -1   # SH (1x short)
    positions[(continuous_pos > -0.5) & (continuous_pos < 0.5)] = 0    # Cash
    positions[(continuous_pos >= 0.5) & (continuous_pos < 2)] = 1      # SPY (1x long)
    positions[continuous_pos >= 2] = 3             # UPRO (3x long)

    return positions

def quantile_position(predictions, window=63,
                      q_extreme=10, q_moderate=30):
    """
    Calcula posiciones basadas en percentiles rolling (SIMETRICO).

    Args:
        predictions: Array de predicciones del modelo
        window: Ventana rolling para calcular percentiles (default 63 = 3 meses)
        q_extreme: Percentil para posiciones extremas (+/-3) (default 10%)
        q_moderate: Percentil para posiciones moderadas (+/-1) (default 30%)

    Returns:
        positions: Array de posiciones discretas {-3, -1, 0, +1, +3}
        info: Dict con estadisticas de posicionamiento
    """
    # Usar la version asimetrica con parametros simetricos
    return quantile_position_asymmetric(
        predictions, window=window,
        q_long_extreme=q_extreme, q_long_moderate=q_moderate,
        q_short_extreme=q_extreme, q_short_moderate=q_moderate
    )


def quantile_position_asymmetric(predictions, window=63,
                                  q_long_extreme=10, q_long_moderate=30,
                                  q_short_extreme=10, q_short_moderate=30):
    """
    Calcula posiciones basadas en percentiles rolling con umbrales ASIMETRICOS.

    Permite ser mas agresivo en longs que en shorts (o viceversa).

    Args:
        predictions: Array de predicciones del modelo
        window: Ventana rolling para calcular percentiles (default 63 = 3 meses)
        q_long_extreme: Top X% para posicion +3 (default 10 = top 10%)
        q_long_moderate: Top X% para posicion +1 (default 30 = top 30%)
        q_short_extreme: Bottom X% para posicion -3 (default 10 = bottom 10%)
        q_short_moderate: Bottom X% para posicion -1 (default 30 = bottom 30%)

    Returns:
        positions: Array de posiciones discretas {-3, -1, 0, +1, +3}
        info: Dict con estadisticas de posicionamiento

    Ejemplo con q_long_extreme=30, q_long_moderate=50:
        Percentil 70-100%  -> +3 (top 30% = agresivo long)
        Percentil 50-70%   -> +1 (50-70%)
        Percentil 30-50%   ->  0 (neutral)
        Percentil 10-30%   -> -1 (conservador short)
        Percentil 0-10%    -> -3 (bottom 10%)
    """
    predictions = np.array(predictions).flatten()
    n = len(predictions)
    positions = np.zeros(n)
    percentiles = np.zeros(n)

    # Umbrales para LONG (desde arriba)
    thresh_long_3x = 100 - q_long_extreme   # ej: 100-30=70 -> top 30% = +3
    thresh_long_1x = 100 - q_long_moderate  # ej: 100-50=50 -> 50-70% = +1

    # Umbrales para SHORT (desde abajo)
    thresh_short_3x = q_short_extreme       # ej: 10 -> bottom 10% = -3
    thresh_short_1x = q_short_moderate      # ej: 30 -> 10-30% = -1

    for i in range(n):
        # Ventana de predicciones recientes (incluyendo actual)
        start = max(0, i - window + 1)
        window_preds = predictions[start:i+1]

        if len(window_preds) < 5:
            positions[i] = 0
            percentiles[i] = 50
            continue

        # Calcular percentil de la prediccion actual dentro de la ventana
        current_pred = predictions[i]
        percentile = np.mean(window_preds <= current_pred) * 100
        percentiles[i] = percentile

        # Mapear percentil a posicion discreta (ASIMETRICO)
        if percentile >= thresh_long_3x:
            positions[i] = 3   # Top q_long_extreme% -> muy alcista
        elif percentile >= thresh_long_1x:
            positions[i] = 1   # Entre thresh_long_1x y thresh_long_3x -> alcista
        elif percentile <= thresh_short_3x:
            positions[i] = -3  # Bottom q_short_extreme% -> muy bajista
        elif percentile <= thresh_short_1x:
            positions[i] = -1  # Entre thresh_short_3x y thresh_short_1x -> bajista
        else:
            positions[i] = 0   # Zona neutral

    info = {
        'method': 'quantile_asymmetric',
        'window': window,
        'q_long_extreme': q_long_extreme,
        'q_long_moderate': q_long_moderate,
        'q_short_extreme': q_short_extreme,
        'q_short_moderate': q_short_moderate,
        'thresh_long_3x': thresh_long_3x,
        'thresh_long_1x': thresh_long_1x,
        'thresh_short_3x': thresh_short_3x,
        'thresh_short_1x': thresh_short_1x,
        'percentiles_mean': np.mean(percentiles),
        'percentiles_std': np.std(percentiles),
        'pct_pos_3': np.mean(positions == 3) * 100,
        'pct_pos_1': np.mean(positions == 1) * 100,
        'pct_pos_0': np.mean(positions == 0) * 100,
        'pct_pos_neg1': np.mean(positions == -1) * 100,
        'pct_pos_neg3': np.mean(positions == -3) * 100,
    }

    return positions, info

def calibrated_sigmoid(test_predictions, train_predictions, scale=1.0, min_std_threshold=1e-6):
    """
    Aplica sigmoid calibrada usando Z-Score normalization.

    1. Calcula mean y std de train_predictions
    2. Normaliza test_predictions: z = (pred - mean) / std
    3. Aplica sigmoid con scale (default=1 para z-scores)

    IMPORTANTE: Si train_std < min_std_threshold, el modelo produce predicciones
    casi constantes y no tiene poder predictivo real. En este caso:
    - Usa el metodo original (scale=500) como fallback
    - Esto evita errores numericos por division con numeros muy pequeños

    Returns: posiciones continuas en [-3, +3], calibration_info
    """
    train_predictions = np.array(train_predictions).flatten()
    test_predictions = np.array(test_predictions).flatten()

    # Calcular estadisticas de train
    train_mean = np.mean(train_predictions)
    train_std = np.std(train_predictions)

    # Manejar predicciones constantes o NaN
    if np.isnan(train_std) or train_std < min_std_threshold:
        # Modelo sin varianza - usar metodo original como fallback
        # Esto es honesto: si el modelo no tiene varianza, usamos sus predicciones directamente
        continuous_pos = 6 / (1 + np.exp(-500 * test_predictions)) - 3

        calibration_info = {
            'train_mean': train_mean,
            'train_std': train_std,
            'pred_normalized_mean': np.nan,
            'pred_normalized_std': np.nan,
            'fallback_used': True,
            'fallback_reason': 'constant_predictions' if train_std < min_std_threshold else 'nan_std',
        }
        return continuous_pos, calibration_info

    # Normalizar (Z-Score)
    pred_normalized = (test_predictions - train_mean) / train_std

    # Sigmoid con scale calibrada
    continuous_pos = 6 / (1 + np.exp(-scale * pred_normalized)) - 3

    calibration_info = {
        'train_mean': train_mean,
        'train_std': train_std,
        'pred_normalized_mean': np.mean(pred_normalized),
        'pred_normalized_std': np.std(pred_normalized),
        'fallback_used': False,
    }

    return continuous_pos, calibration_info

# =============================================================================
# FUNCIONES DE COSTOS Y RETORNOS REALISTAS
# =============================================================================

def get_instrument_for_position(position):
    """Retorna el instrumento correspondiente a una posicion."""
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

def calculate_volatility_drag(spy_returns, leverage):
    """
    Calcula el decay por volatilidad de ETFs apalancados.

    Los ETFs apalancados rebalancean diariamente, lo que causa decay:
    decay = 0.5 * (leverage^2 - leverage) * variance

    Args:
        spy_returns: Retornos diarios de SPY
        leverage: Factor de apalancamiento (3 para UPRO, -3 para SPXU)

    Returns:
        decay diario
    """
    if abs(leverage) <= 1:
        return 0.0

    # Varianza de los retornos
    variance = np.var(spy_returns)

    # Decay formula
    decay = 0.5 * (leverage**2 - abs(leverage)) * variance

    return decay

def calculate_realistic_returns(positions, forward_returns, risk_free_rate,
                                config, apply_vol_drag=True):
    """
    Calcula retornos realistas incluyendo todos los costos.

    Costos incluidos:
    1. Expense ratios (prorrateados diariamente)
    2. Bid-ask spreads (en cada cambio de posicion)
    3. Volatility drag (para ETFs apalancados)

    Args:
        positions: Array de posiciones discretas {-3, -1, 0, +1, +3}
        forward_returns: Retornos del mercado
        risk_free_rate: Tasa libre de riesgo
        config: Configuracion con costos
        apply_vol_drag: Si aplicar volatility drag

    Returns:
        dict con retornos y desglose de costos
    """
    positions = np.array(positions).flatten()
    forward_returns = np.array(forward_returns).flatten()
    risk_free_rate = np.array(risk_free_rate).flatten()

    n_days = len(positions)

    # Arrays para tracking
    gross_returns = np.zeros(n_days)
    expense_costs = np.zeros(n_days)
    trading_costs = np.zeros(n_days)
    vol_drag_costs = np.zeros(n_days)

    # Calcular volatility drag basado en ventana movil
    if apply_vol_drag:
        lookback = min(21, n_days // 4)
        rolling_variance = pd.Series(forward_returns).rolling(lookback).var().fillna(0).values

    for i in range(n_days):
        pos = positions[i]
        instrument = get_instrument_for_position(pos)
        instrument_config = config['instruments'][instrument]

        # 1. Retorno bruto
        if pos == 0:
            # Cash: solo risk-free
            gross_returns[i] = risk_free_rate[i]
        else:
            # Formula: rf + pos * (market - rf)
            gross_returns[i] = risk_free_rate[i] + pos * (forward_returns[i] - risk_free_rate[i])

        # 2. Expense ratio (diario)
        daily_expense = instrument_config['expense_ratio'] / 252
        expense_costs[i] = daily_expense

        # 3. Trading cost (solo si hay cambio de posicion)
        if i > 0 and positions[i] != positions[i-1]:
            # Costo del instrumento anterior (venta) + nuevo (compra)
            prev_instrument = get_instrument_for_position(positions[i-1])
            sell_cost = config['instruments'][prev_instrument]['bid_ask']
            buy_cost = instrument_config['bid_ask']
            trading_costs[i] = sell_cost + buy_cost

        # 4. Volatility drag (solo para apalancados)
        if apply_vol_drag and abs(pos) == 3:
            leverage = pos
            if i >= lookback:
                var = rolling_variance[i]
            else:
                var = np.var(forward_returns[:max(i+1, 5)])

            vol_drag_costs[i] = 0.5 * (leverage**2 - abs(leverage)) * var

    # Retornos netos
    net_returns = gross_returns - expense_costs - trading_costs - vol_drag_costs

    return {
        'gross_returns': gross_returns,
        'net_returns': net_returns,
        'expense_costs': expense_costs,
        'trading_costs': trading_costs,
        'vol_drag_costs': vol_drag_costs,
        'total_expense_cost': np.sum(expense_costs),
        'total_trading_cost': np.sum(trading_costs),
        'total_vol_drag_cost': np.sum(vol_drag_costs),
        'total_costs': np.sum(expense_costs) + np.sum(trading_costs) + np.sum(vol_drag_costs),
    }

# =============================================================================
# FUNCIONES DE RISK MANAGEMENT
# =============================================================================

def apply_drawdown_control(positions, cumulative_returns, limits):
    """
    Reduce posicion cuando el drawdown excede umbrales.

    Args:
        positions: Posiciones originales
        cumulative_returns: Retornos acumulados (equity curve)
        limits: Dict con umbrales y max_leverage

    Returns:
        Posiciones ajustadas
    """
    positions = np.array(positions).flatten()
    cumulative_returns = np.array(cumulative_returns).flatten()

    # Calcular drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-10)

    adjusted_positions = positions.copy()

    for i in range(len(positions)):
        dd = abs(drawdown[i])

        # Aplicar limites en orden de severidad
        if dd >= limits['level_3']['threshold']:
            max_lev = limits['level_3']['max_leverage']
        elif dd >= limits['level_2']['threshold']:
            max_lev = limits['level_2']['max_leverage']
        elif dd >= limits['level_1']['threshold']:
            max_lev = limits['level_1']['max_leverage']
        else:
            max_lev = 3  # Sin limite

        # Limitar posicion
        if abs(adjusted_positions[i]) > max_lev:
            adjusted_positions[i] = np.sign(adjusted_positions[i]) * max_lev

        # Discretizar al nivel permitido
        if max_lev == 0:
            adjusted_positions[i] = 0
        elif max_lev == 1:
            adjusted_positions[i] = np.sign(adjusted_positions[i]) * min(abs(adjusted_positions[i]), 1)
        elif max_lev == 2:
            if abs(adjusted_positions[i]) == 3:
                adjusted_positions[i] = np.sign(adjusted_positions[i]) * 1

    return adjusted_positions

def apply_position_filter(positions, min_change=1):
    """
    Filtra cambios de posicion menores al umbral para evitar churning.

    Args:
        positions: Posiciones originales
        min_change: Cambio minimo para ejecutar trade

    Returns:
        Posiciones filtradas
    """
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
    """
    Ajusta posiciones para mantener volatilidad objetivo.

    Args:
        positions: Posiciones originales
        returns: Retornos historicos
        target_vol: Volatilidad objetivo anualizada
        lookback: Dias para calcular vol

    Returns:
        Posiciones ajustadas
    """
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()

    adjusted = positions.copy()

    for i in range(lookback, len(positions)):
        # Volatilidad reciente anualizada
        recent_vol = np.std(returns[i-lookback:i]) * np.sqrt(252)

        if recent_vol > 0:
            # Factor de ajuste
            vol_scalar = target_vol / recent_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)

            # Ajustar posicion
            adjusted_continuous = positions[i] * vol_scalar

            # Discretizar
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
# FUNCIONES DE METRICAS
# =============================================================================

def calculate_metrics(returns, risk_free_rate=None, positions=None):
    """
    Calcula metricas completas de la estrategia.
    """
    returns = np.array(returns).flatten()

    # Retorno total (compuesto)
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

    # Sharpe Ratio
    if annual_vol > 0:
        if risk_free_rate is not None:
            rf_annual = np.mean(risk_free_rate) * 252
        else:
            rf_annual = 0.02  # Asumir 2%
        sharpe = (annualized_return - rf_annual) / annual_vol
    else:
        sharpe = 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = (annualized_return - rf_annual) / downside_vol if downside_vol > 0 else 0
    else:
        sortino = sharpe * 2  # Sin downside

    # Max Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Calmar Ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = np.mean(returns > 0)

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'daily_volatility': daily_vol,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_days': len(returns),
    }

    # Metricas de posiciones si disponibles
    if positions is not None:
        positions = np.array(positions).flatten()
        metrics['pct_long'] = np.mean(positions > 0) * 100
        metrics['pct_short'] = np.mean(positions < 0) * 100
        metrics['pct_cash'] = np.mean(positions == 0) * 100
        metrics['pct_leveraged_long'] = np.mean(positions == 3) * 100
        metrics['pct_leveraged_short'] = np.mean(positions == -3) * 100
        metrics['mean_position'] = np.mean(positions)

        # Turnover
        position_changes = np.abs(np.diff(positions))
        metrics['daily_turnover'] = np.mean(position_changes)
        metrics['total_trades'] = np.sum(position_changes > 0)

    return metrics

def bootstrap_confidence(returns, risk_free_rate=None, n_bootstrap=10000, confidence=0.95):
    """
    Calcula intervalos de confianza via bootstrap.

    IMPORTANTE: Si se proporciona risk_free_rate, calcula Sharpe como:
    Sharpe = (mean(returns) - mean(rf)) / std(returns) * sqrt(252)

    Esto es crucial para que modelos en cash (position=0) no tengan
    Sharpe artificialmente alto.
    """
    returns = np.array(returns).flatten()
    n = len(returns)

    # Calcular mean risk-free si se proporciona
    rf_mean = 0.0
    if risk_free_rate is not None:
        rf_mean = np.mean(risk_free_rate)

    # Detectar si el modelo esta 100% en cash (volatilidad muy baja)
    returns_std = np.std(returns)
    is_likely_cash = returns_std < 0.001  # Volatilidad < 0.1% diaria sugiere cash

    sharpes = []
    total_returns = []

    for _ in range(n_bootstrap):
        # Resample con reemplazo
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = returns[sample_idx]

        # Sharpe del sample (con exceso sobre risk-free)
        sample_std = np.std(sample)
        if sample_std > 0.0001:  # Threshold minimo para evitar division por ~0
            # Sharpe = (mean_return - rf) / std * sqrt(252)
            excess_return = np.mean(sample) - rf_mean
            sharpe = excess_return / sample_std * np.sqrt(252)

            # Limitar a rango razonable [-5, 5] para evitar outliers extremos
            sharpe = np.clip(sharpe, -5, 5)
        else:
            # Volatilidad ~0 significa 100% cash o predicciones constantes
            # El Sharpe en este caso es efectivamente 0 (no hay exceso de retorno)
            sharpe = 0
        sharpes.append(sharpe)

        # Total return del sample
        total_ret = np.prod(1 + sample) - 1
        total_returns.append(total_ret)

    sharpes = np.array(sharpes)
    total_returns = np.array(total_returns)

    alpha = 1 - confidence

    return {
        'sharpe_mean': np.mean(sharpes),
        'sharpe_median': np.median(sharpes),
        'sharpe_ci_lower': np.percentile(sharpes, alpha/2 * 100),
        'sharpe_ci_upper': np.percentile(sharpes, (1 - alpha/2) * 100),
        'sharpe_prob_positive': np.mean(sharpes > 0),

        'return_mean': np.mean(total_returns),
        'return_median': np.median(total_returns),
        'return_ci_lower': np.percentile(total_returns, alpha/2 * 100),
        'return_ci_upper': np.percentile(total_returns, (1 - alpha/2) * 100),
        'return_prob_positive': np.mean(total_returns > 0),

        'is_likely_cash': is_likely_cash,
    }

# =============================================================================
# PASO 1: CARGAR ARTEFACTOS
# =============================================================================
print("\n" + "="*80)
print("PASO 1: Cargar Artefactos de Entrenamiento")
print("="*80)

artifacts_path = os.path.join(MODELS_DIR, ARTIFACTS_FILE)
print(f"  + Cargando: {artifacts_path}")

with open(artifacts_path, 'rb') as f:
    artifacts = pickle.load(f)

metadata = artifacts['metadata']
models_data = artifacts['models']

print(f"  + Modelos cargados: {len(models_data)}")
print(f"  + Periodo train: {pd.to_datetime(metadata['dates_train'][0]).date()} a {pd.to_datetime(metadata['dates_train'][-1]).date()}")
print(f"  + Periodo test: {pd.to_datetime(metadata['dates_test'][0]).date()} a {pd.to_datetime(metadata['dates_test'][-1]).date()}")
print(f"  + Train size: {metadata['train_size']:,} | Test size: {metadata['test_size']:,}")

# Extraer datos necesarios
forward_returns_test = metadata['forward_returns_test']
risk_free_test = metadata['risk_free_test']
y_test = metadata['y_test']
dates_test = pd.to_datetime(metadata['dates_test'])

# Cargar parametros optimos por modelo (si existen)
optimal_params_path = os.path.join(RESULTS_DIR, OPTIMAL_PARAMS_FILE)
if os.path.exists(optimal_params_path):
    with open(optimal_params_path, 'r') as f:
        OPTIMAL_MODEL_PARAMS = json.load(f)
    print(f"  + Parametros optimos cargados para {len(OPTIMAL_MODEL_PARAMS)} modelos")
else:
    OPTIMAL_MODEL_PARAMS = {}
    print("  ! No se encontraron parametros optimos - usando defaults")

# =============================================================================
# PASO 2: EVALUAR CADA MODELO
# =============================================================================
print("\n" + "="*80)
print("PASO 2: Evaluar Modelos con Estrategia Realista")
print("="*80)

results_original = {}
results_calibrated = {}
results_quantile = {}
results_optimized = {}
results_full = {}
bootstrap_results = {}
bootstrap_results_quantile = {}
bootstrap_results_optimized = {}

for model_name, model_data in models_data.items():
    print(f"\n  Evaluando: {model_name}")
    print("  " + "-"*50)

    train_predictions = model_data['train_predictions']
    test_predictions = model_data['test_predictions']

    # Alinear longitudes
    n_test = min(len(test_predictions), len(forward_returns_test))
    test_pred = test_predictions[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    # =========================================================================
    # METODO ORIGINAL (scale=500 fijo)
    # =========================================================================
    continuous_orig = sigmoid_to_position(test_pred, scale=CONFIG['original_scale'])
    positions_orig = discretize_position(continuous_orig)

    returns_orig = calculate_realistic_returns(
        positions_orig, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )

    metrics_orig = calculate_metrics(
        returns_orig['net_returns'],
        risk_free_rate=rf,
        positions=positions_orig
    )
    metrics_orig['method'] = 'original'
    metrics_orig['total_costs'] = returns_orig['total_costs']

    results_original[model_name] = metrics_orig

    # =========================================================================
    # METODO CALIBRADO (Z-Score + scale=1)
    # =========================================================================
    continuous_cal, cal_info = calibrated_sigmoid(
        test_pred, train_predictions, scale=CONFIG['calibrated_scale']
    )
    positions_cal = discretize_position(continuous_cal)

    # Aplicar mejoras de risk management
    # 1. Filtro de churning
    positions_filtered = apply_position_filter(positions_cal, CONFIG['min_position_change'])

    # 2. Volatility targeting
    positions_vol = apply_volatility_targeting(
        positions_filtered, fwd_ret,
        target_vol=CONFIG['volatility_target'],
        lookback=CONFIG['volatility_lookback']
    )

    # Calcular retornos para drawdown control
    returns_temp = calculate_realistic_returns(
        positions_vol, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])

    # 3. Drawdown control
    positions_final = apply_drawdown_control(
        positions_vol, cumulative_temp, CONFIG['drawdown_limits']
    )

    # Retornos finales
    returns_cal = calculate_realistic_returns(
        positions_final, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )

    metrics_cal = calculate_metrics(
        returns_cal['net_returns'],
        risk_free_rate=rf,
        positions=positions_final
    )
    metrics_cal['method'] = 'calibrated'
    metrics_cal['total_costs'] = returns_cal['total_costs']
    metrics_cal['calibration'] = cal_info

    results_calibrated[model_name] = metrics_cal

    # =========================================================================
    # METODO QUANTILE (percentiles adaptativos)
    # =========================================================================
    positions_quant, quant_info = quantile_position(
        test_pred, window=63, q_extreme=10, q_moderate=30
    )

    # Aplicar mismas mejoras de risk management
    positions_quant_filtered = apply_position_filter(positions_quant, CONFIG['min_position_change'])
    positions_quant_vol = apply_volatility_targeting(
        positions_quant_filtered, fwd_ret,
        target_vol=CONFIG['volatility_target'],
        lookback=CONFIG['volatility_lookback']
    )

    returns_quant_temp = calculate_realistic_returns(
        positions_quant_vol, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )
    cumulative_quant_temp = np.cumprod(1 + returns_quant_temp['net_returns'])

    positions_quant_final = apply_drawdown_control(
        positions_quant_vol, cumulative_quant_temp, CONFIG['drawdown_limits']
    )

    returns_quant = calculate_realistic_returns(
        positions_quant_final, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )

    metrics_quant = calculate_metrics(
        returns_quant['net_returns'],
        risk_free_rate=rf,
        positions=positions_quant_final
    )
    metrics_quant['method'] = 'quantile'
    metrics_quant['total_costs'] = returns_quant['total_costs']
    metrics_quant['quantile_info'] = quant_info

    results_quantile[model_name] = metrics_quant

    # =========================================================================
    # METODO OPTIMIZADO (parametros especificos por modelo)
    # =========================================================================
    if model_name in OPTIMAL_MODEL_PARAMS:
        opt_params = OPTIMAL_MODEL_PARAMS[model_name]['params']
        positions_opt, opt_info = quantile_position_asymmetric(
            test_pred, window=63,
            q_long_extreme=opt_params['q_long_extreme'],
            q_long_moderate=opt_params['q_long_moderate'],
            q_short_extreme=opt_params['q_short_extreme'],
            q_short_moderate=opt_params['q_short_moderate'],
        )
    else:
        # Si no hay parametros optimos, usar quantile default
        positions_opt, opt_info = quantile_position(test_pred, window=63)

    # Aplicar mismas mejoras de risk management
    positions_opt_filtered = apply_position_filter(positions_opt, CONFIG['min_position_change'])
    positions_opt_vol = apply_volatility_targeting(
        positions_opt_filtered, fwd_ret,
        target_vol=CONFIG['volatility_target'],
        lookback=CONFIG['volatility_lookback']
    )

    returns_opt_temp = calculate_realistic_returns(
        positions_opt_vol, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )
    cumulative_opt_temp = np.cumprod(1 + returns_opt_temp['net_returns'])

    positions_opt_final = apply_drawdown_control(
        positions_opt_vol, cumulative_opt_temp, CONFIG['drawdown_limits']
    )

    returns_opt = calculate_realistic_returns(
        positions_opt_final, fwd_ret, rf, CONFIG, apply_vol_drag=True
    )

    metrics_opt = calculate_metrics(
        returns_opt['net_returns'],
        risk_free_rate=rf,
        positions=positions_opt_final
    )
    metrics_opt['method'] = 'optimized'
    metrics_opt['total_costs'] = returns_opt['total_costs']
    metrics_opt['optimized_info'] = opt_info
    if model_name in OPTIMAL_MODEL_PARAMS:
        metrics_opt['params_used'] = OPTIMAL_MODEL_PARAMS[model_name]['params']

    results_optimized[model_name] = metrics_opt

    # =========================================================================
    # BOOTSTRAP CONFIDENCE (todos los metodos)
    # =========================================================================
    bootstrap_cal = bootstrap_confidence(
        returns_cal['net_returns'],
        risk_free_rate=rf,  # Pasar risk-free para calcular Sharpe correctamente
        n_bootstrap=CONFIG['n_bootstrap'],
        confidence=CONFIG['confidence_level']
    )
    bootstrap_results[model_name] = bootstrap_cal

    bootstrap_quant = bootstrap_confidence(
        returns_quant['net_returns'],
        risk_free_rate=rf,
        n_bootstrap=CONFIG['n_bootstrap'],
        confidence=CONFIG['confidence_level']
    )
    bootstrap_results_quantile[model_name] = bootstrap_quant

    bootstrap_opt = bootstrap_confidence(
        returns_opt['net_returns'],
        risk_free_rate=rf,
        n_bootstrap=CONFIG['n_bootstrap'],
        confidence=CONFIG['confidence_level']
    )
    bootstrap_results_optimized[model_name] = bootstrap_opt

    # =========================================================================
    # RESULTADO COMBINADO
    # =========================================================================
    results_full[model_name] = {
        'original': metrics_orig,
        'calibrated': metrics_cal,
        'quantile': metrics_quant,
        'optimized': metrics_opt,
        'bootstrap_cal': bootstrap_cal,
        'bootstrap_quant': bootstrap_quant,
        'bootstrap_opt': bootstrap_opt,
        'improvement_opt': {
            'return_delta': metrics_opt['total_return'] - metrics_orig['total_return'],
            'sharpe_delta': metrics_opt['sharpe_ratio'] - metrics_orig['sharpe_ratio'],
            'drawdown_improvement': metrics_opt['max_drawdown'] - metrics_orig['max_drawdown'],
        }
    }

    # Imprimir resumen
    print(f"    ORIGINAL:   Return: {metrics_orig['total_return']*100:+7.2f}% | Sharpe: {metrics_orig['sharpe_ratio']:+.2f} | MaxDD: {metrics_orig['max_drawdown']*100:.1f}%")
    print(f"    CALIBRADO:  Return: {metrics_cal['total_return']*100:+7.2f}% | Sharpe: {metrics_cal['sharpe_ratio']:+.2f} | MaxDD: {metrics_cal['max_drawdown']*100:.1f}%")
    print(f"    QUANTILE:   Return: {metrics_quant['total_return']*100:+7.2f}% | Sharpe: {metrics_quant['sharpe_ratio']:+.2f} | MaxDD: {metrics_quant['max_drawdown']*100:.1f}%")
    print(f"    OPTIMIZADO: Return: {metrics_opt['total_return']*100:+7.2f}% | Sharpe: {metrics_opt['sharpe_ratio']:+.2f} | MaxDD: {metrics_opt['max_drawdown']*100:.1f}%")
    print(f"    BOOTSTRAP (Opt):  Sharpe CI: [{bootstrap_opt['sharpe_ci_lower']:.2f}, {bootstrap_opt['sharpe_ci_upper']:.2f}] | P(Sharpe>0): {bootstrap_opt['sharpe_prob_positive']*100:.1f}%")

# =============================================================================
# PASO 3: CALCULAR EQUITY CURVES
# =============================================================================
print("\n" + "="*80)
print("PASO 3: Calcular Equity Curves con $10,000")
print("="*80)

initial_capital = CONFIG['initial_capital']
equity_curves_optimized = {}

for model_name, model_data in models_data.items():
    test_predictions = model_data['test_predictions']

    n_test = min(len(test_predictions), len(forward_returns_test))
    test_pred = test_predictions[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    # Metodo OPTIMIZADO (parametros especificos por modelo)
    if model_name in OPTIMAL_MODEL_PARAMS:
        opt_params = OPTIMAL_MODEL_PARAMS[model_name]['params']
        positions_opt, _ = quantile_position_asymmetric(
            test_pred, window=63,
            q_long_extreme=opt_params['q_long_extreme'],
            q_long_moderate=opt_params['q_long_moderate'],
            q_short_extreme=opt_params['q_short_extreme'],
            q_short_moderate=opt_params['q_short_moderate'],
        )
    else:
        positions_opt, _ = quantile_position(test_pred, window=63)

    positions_opt_filtered = apply_position_filter(positions_opt, CONFIG['min_position_change'])
    positions_opt_vol = apply_volatility_targeting(positions_opt_filtered, fwd_ret, CONFIG['volatility_target'], CONFIG['volatility_lookback'])

    returns_opt_temp = calculate_realistic_returns(positions_opt_vol, fwd_ret, rf, CONFIG, apply_vol_drag=True)
    cumulative_opt_temp = np.cumprod(1 + returns_opt_temp['net_returns'])

    positions_opt_final = apply_drawdown_control(positions_opt_vol, cumulative_opt_temp, CONFIG['drawdown_limits'])
    returns_opt = calculate_realistic_returns(positions_opt_final, fwd_ret, rf, CONFIG, apply_vol_drag=True)

    equity_opt = initial_capital * np.cumprod(1 + returns_opt['net_returns'])
    equity_curves_optimized[model_name] = {
        'equity': equity_opt,
        'final_value': equity_opt[-1],
        'peak': np.max(equity_opt),
        'trough': np.min(equity_opt),
        'profit_loss': equity_opt[-1] - initial_capital,
        'method': 'optimized',
    }

# Usar metodo optimizado para el ranking
equity_curves = equity_curves_optimized
equity_ranking = sorted(equity_curves.items(), key=lambda x: x[1]['final_value'], reverse=True)

print(f"\n  RANKING POR VALOR FINAL - METODO OPTIMIZADO (Capital Inicial: ${initial_capital:,}):")
print("  " + "-"*70)
for i, (name, data) in enumerate(equity_ranking, 1):
    pl_pct = (data['final_value'] / initial_capital - 1) * 100
    # Mostrar parametros usados
    if name in OPTIMAL_MODEL_PARAMS:
        p = OPTIMAL_MODEL_PARAMS[name]['params']
        params_str = f"L({p['q_long_extreme']},{p['q_long_moderate']}) S({p['q_short_extreme']},{p['q_short_moderate']})"
    else:
        params_str = "default"
    print(f"  {i:2d}. {name:20s} ${data['final_value']:>10,.2f} ({pl_pct:+7.2f}%) {params_str}")

# =============================================================================
# PASO 4: COMPARACION CON BENCHMARK
# =============================================================================
print("\n" + "="*80)
print("PASO 4: Comparacion con Benchmark (Buy & Hold SPY)")
print("="*80)

# Buy & Hold SPY
n_test = len(forward_returns_test)
spy_returns = forward_returns_test
spy_cumulative = np.cumprod(1 + spy_returns)
spy_final = initial_capital * spy_cumulative[-1]
spy_total_return = spy_cumulative[-1] - 1

print(f"\n  BUY & HOLD SPY:")
print(f"    Valor Final: ${spy_final:,.2f}")
print(f"    Retorno Total: {spy_total_return*100:+.2f}%")

print(f"\n  MODELOS QUE SUPERAN BUY & HOLD:")
for name, data in equity_ranking:
    if data['final_value'] > spy_final:
        excess = data['final_value'] - spy_final
        print(f"    + {name}: ${data['final_value']:,.2f} (+${excess:,.2f} vs SPY)")

# =============================================================================
# PASO 5: GUARDAR RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("PASO 5: Guardar Resultados")
print("="*80)

# 1. CSV con comparacion detallada (foco en metodo optimizado)
comparison_data = []
for model_name in models_data.keys():
    orig = results_original[model_name]
    opt = results_optimized[model_name]
    boot_opt = bootstrap_results_optimized[model_name]
    eq_opt = equity_curves_optimized[model_name]

    # Parametros usados
    if model_name in OPTIMAL_MODEL_PARAMS:
        params = OPTIMAL_MODEL_PARAMS[model_name]['params']
        q_long_ext = params['q_long_extreme']
        q_long_mod = params['q_long_moderate']
        q_short_ext = params['q_short_extreme']
        q_short_mod = params['q_short_moderate']
    else:
        q_long_ext = q_long_mod = q_short_ext = q_short_mod = None

    row = {
        'model': model_name,
        # Parametros optimizados
        'q_long_extreme': q_long_ext,
        'q_long_moderate': q_long_mod,
        'q_short_extreme': q_short_ext,
        'q_short_moderate': q_short_mod,
        # Original (baseline)
        'orig_return': orig['total_return'],
        'orig_sharpe': orig['sharpe_ratio'],
        'orig_max_dd': orig['max_drawdown'],
        # Optimizado
        'opt_return': opt['total_return'],
        'opt_sharpe': opt['sharpe_ratio'],
        'opt_sortino': opt['sortino_ratio'],
        'opt_max_dd': opt['max_drawdown'],
        'opt_calmar': opt['calmar_ratio'],
        'opt_pct_long': opt.get('pct_long', 0),
        'opt_pct_short': opt.get('pct_short', 0),
        'opt_pct_3x_long': opt.get('pct_leveraged_long', 0),
        'opt_total_trades': opt.get('total_trades', 0),
        'opt_total_costs': opt['total_costs'],
        # Bootstrap
        'sharpe_ci_lower': boot_opt['sharpe_ci_lower'],
        'sharpe_ci_upper': boot_opt['sharpe_ci_upper'],
        'prob_sharpe_positive': boot_opt['sharpe_prob_positive'],
        # Equity
        'final_equity': eq_opt['final_value'],
        'profit_loss': eq_opt['profit_loss'],
        # Improvement vs Original
        'return_improvement': opt['total_return'] - orig['total_return'],
        'sharpe_improvement': opt['sharpe_ratio'] - orig['sharpe_ratio'],
    }
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('opt_sharpe', ascending=False)

csv_path = os.path.join(RESULTS_DIR, 'strategy_realistic_evaluation.csv')
comparison_df.to_csv(csv_path, index=False)
print(f"  + Guardado: {csv_path}")

# 2. JSON con resumen
best_model_name = equity_ranking[0][0]
best_sharpe = results_optimized[best_model_name]['sharpe_ratio']
best_boot = bootstrap_results_optimized[best_model_name]

# Parametros del mejor modelo
if best_model_name in OPTIMAL_MODEL_PARAMS:
    best_params = OPTIMAL_MODEL_PARAMS[best_model_name]['params']
else:
    best_params = {'q_long_extreme': 10, 'q_long_moderate': 30, 'q_short_extreme': 10, 'q_short_moderate': 30}

summary = {
    'timestamp': datetime.now().isoformat(),
    'config': {k: v for k, v in CONFIG.items() if not isinstance(v, dict)},
    'initial_capital': initial_capital,
    'test_period': {
        'start': str(dates_test[0].date()),
        'end': str(dates_test[-1].date()),
        'n_days': len(dates_test),
    },
    'benchmark': {
        'spy_final_value': spy_final,
        'spy_total_return': spy_total_return,
    },
    'best_model': {
        'name': best_model_name,
        'method': 'optimized',
        'params': best_params,
        'final_value': equity_ranking[0][1]['final_value'],
        'profit_loss': equity_ranking[0][1]['profit_loss'],
        'sharpe': best_sharpe,
        'sharpe_ci': [best_boot['sharpe_ci_lower'], best_boot['sharpe_ci_upper']],
        'prob_sharpe_positive': best_boot['sharpe_prob_positive'],
    },
    'models_beating_benchmark': sum(1 for _, d in equity_ranking if d['final_value'] > spy_final),
    'total_models': len(models_data),
}

json_path = os.path.join(RESULTS_DIR, 'strategy_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  + Guardado: {json_path}")

# 3. JSON con bootstrap (metodo optimizado)
bootstrap_json = {}
for model in models_data.keys():
    bootstrap_json[model] = {
        'optimized': {k: float(v) for k, v in bootstrap_results_optimized[model].items()},
    }
    if model in OPTIMAL_MODEL_PARAMS:
        bootstrap_json[model]['params'] = OPTIMAL_MODEL_PARAMS[model]['params']

bootstrap_path = os.path.join(RESULTS_DIR, 'bootstrap_confidence.json')
with open(bootstrap_path, 'w') as f:
    json.dump(bootstrap_json, f, indent=2)
print(f"  + Guardado: {bootstrap_path}")

# =============================================================================
# PASO 6: RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL - METODO OPTIMIZADO POR MODELO")
print("="*80)

print(f"\n  CAPITAL INICIAL: ${initial_capital:,}")
print(f"  PERIODO: {dates_test[0].date()} a {dates_test[-1].date()} ({len(dates_test)} dias)")

print(f"\n  TOP 5 MODELOS (Parametros Optimizados):")
print("  " + "-"*90)
top5 = comparison_df.head(5)
for _, row in top5.iterrows():
    params_str = f"L({row['q_long_extreme']:.0f},{row['q_long_moderate']:.0f}) S({row['q_short_extreme']:.0f},{row['q_short_moderate']:.0f})"
    print(f"  {row['model']:20s} | Sharpe: {row['opt_sharpe']:+.2f} | Return: {row['opt_return']*100:+7.1f}% | ${row['final_equity']:>10,.0f} | {params_str}")

print(f"\n  BENCHMARK (Buy & Hold SPY): ${spy_final:,.2f} ({spy_total_return*100:+.2f}%)")
print(f"  MODELOS QUE SUPERAN SPY: {sum(1 for _, d in equity_ranking if d['final_value'] > spy_final)}/{len(models_data)}")

best_model = equity_ranking[0][0]
best_equity = equity_ranking[0][1]['final_value']
best_sharpe = results_optimized[best_model]['sharpe_ratio']
best_boot = bootstrap_results_optimized[best_model]

print(f"\n  MEJOR MODELO: {best_model}")
if best_model in OPTIMAL_MODEL_PARAMS:
    p = OPTIMAL_MODEL_PARAMS[best_model]['params']
    print(f"    Parametros: Long({p['q_long_extreme']},{p['q_long_moderate']}) Short({p['q_short_extreme']},{p['q_short_moderate']})")
print(f"    Valor Final: ${best_equity:,.2f}")
print(f"    vs SPY: {'+' if best_equity > spy_final else ''}{((best_equity/spy_final)-1)*100:.1f}%")
print(f"    Sharpe Ratio: {best_sharpe:.2f}")
print(f"    Sharpe 95% CI: [{best_boot['sharpe_ci_lower']:.2f}, {best_boot['sharpe_ci_upper']:.2f}]")
print(f"    P(Sharpe > 0): {best_boot['sharpe_prob_positive']*100:.1f}%")

# Mostrar todos los modelos que superan SPY
print(f"\n  TODOS LOS MODELOS QUE SUPERAN SPY:")
print("  " + "-"*80)
for name, data in equity_ranking:
    if data['final_value'] > spy_final:
        excess_pct = ((data['final_value'] / spy_final) - 1) * 100
        sharpe = results_optimized[name]['sharpe_ratio']
        boot = bootstrap_results_optimized[name]
        print(f"    {name:20s} ${data['final_value']:>10,.0f} (+{excess_pct:5.1f}% vs SPY) | Sharpe: {sharpe:.2f} | P(>0): {boot['sharpe_prob_positive']*100:.0f}%")

print("\n" + "="*80)
print("EVALUACION COMPLETADA")
print("="*80)
print()
