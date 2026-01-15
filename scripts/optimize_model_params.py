# -*- coding: utf-8 -*-
"""
================================================================================
OPTIMIZE_MODEL_PARAMS.PY - Optimizacion de Umbrales por Modelo (STANDALONE)
================================================================================

PROPOSITO:
----------
Cada modelo de ML tiene caracteristicas unicas en sus predicciones:
- Distribucion diferente (algunos predicen valores mas extremos)
- Accuracy direccional diferente
- Sesgo diferente (algunos son mas alcistas, otros mas bajistas)

Por lo tanto, usar los MISMOS umbrales para todos los modelos es suboptimo.
Este script encuentra los UMBRALES OPTIMOS para cada modelo individualmente.

PROCESO:
--------
1. Analizar caracteristicas de predicciones de cada modelo
2. Grid Search: probar combinaciones de umbrales
3. Seleccionar umbrales que maximizan Sharpe Ratio (penalizando drawdowns)
4. Calcular intervalos de confianza via Bootstrap
5. Guardar configuracion optima por modelo

ENTRADA:
--------
- models/trained_artifacts.pkl (generado por train_models.py)

SALIDA:
-------
- results/optimal_model_params.json (umbrales optimos por modelo)
- results/param_grid_search_results.csv (todos los resultados del grid search)

UMBRALES EXPLICADOS:
--------------------
Los umbrales definen que percentil de prediccion activa cada posicion:

    q_long_extreme = 10  -> Top 10% de predicciones activa posicion +3 (UPRO)
    q_long_moderate = 30 -> Top 30% activa posicion +1 (SPY)
    q_short_extreme = 10 -> Bottom 10% activa posicion -3 (SPXU)
    q_short_moderate = 30 -> Bottom 30% activa posicion -1 (SH)

Ejemplo con q_long_extreme=10, q_long_moderate=30:
    Percentil 90-100%  -> +3 (muy alcista, top 10%)
    Percentil 70-90%   -> +1 (alcista)
    Percentil 30-70%   ->  0 (neutral, cash)
    Percentil 10-30%   -> -1 (bajista)
    Percentil 0-10%    -> -3 (muy bajista, bottom 10%)

NOTA:
-----
Este archivo es STANDALONE - no tiene dependencias externas excepto
librerias estandar (pandas, numpy, etc.)

================================================================================
Autor: David Gonzalez Canon
Fecha: Enero 2026
================================================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from itertools import product
from datetime import datetime

# =============================================================================
# SEMILLA PARA REPRODUCIBILIDAD
# =============================================================================
# El bootstrap usa np.random.choice() para resamplear datos.
# Sin semilla fija, los intervalos de confianza serian diferentes cada ejecucion.
# Con SEED=42, el bootstrap producira EXACTAMENTE los mismos resultados siempre.
# =============================================================================
SEED = 42
np.random.seed(SEED)
print(f"Semilla de reproducibilidad establecida: {SEED}")

# =============================================================================
# CONFIGURACION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# CONFIGURACION DE COSTOS E INSTRUMENTOS
# -----------------------------------------------------------------------------
# Esta configuracion define los costos reales de cada instrumento financiero
# que se usa en la estrategia de trading.

CONFIG = {
    'initial_capital': 10000,  # $10,000 USD inicial

    # Costos reales por instrumento (datos de los ETFs reales)
    'instruments': {
        'UPRO': {
            'expense_ratio': 0.0091,  # 0.91% anual - costo de gestion del ETF
            'bid_ask': 0.0005,        # 0.05% por trade - spread de compra/venta
            'leverage': 3,            # Apalancamiento 3x
        },
        'SPY': {
            'expense_ratio': 0.0009,  # 0.09% anual - muy bajo por ser el ETF mas liquido
            'bid_ask': 0.0002,        # 0.02% por trade
            'leverage': 1,
        },
        'CASH': {
            'expense_ratio': 0.0000,  # Sin costo
            'bid_ask': 0.0000,
            'leverage': 0,
        },
        'SH': {
            'expense_ratio': 0.0089,  # 0.89% anual
            'bid_ask': 0.0004,        # 0.04% por trade
            'leverage': -1,           # Inverso 1x
        },
        'SPXU': {
            'expense_ratio': 0.0089,  # 0.89% anual
            'bid_ask': 0.0006,        # 0.06% por trade
            'leverage': -3,           # Inverso 3x
        },
    },

    # Control de riesgo: reducir posicion cuando hay drawdowns
    'drawdown_limits': {
        'level_1': {'threshold': 0.10, 'max_leverage': 2},  # DD 10%: max 2x
        'level_2': {'threshold': 0.15, 'max_leverage': 1},  # DD 15%: max 1x
        'level_3': {'threshold': 0.20, 'max_leverage': 0},  # DD 20%: ir a cash
    },

    # Gestion de posiciones
    'min_position_change': 1,      # Cambio minimo para ejecutar trade (evita churning)
    'volatility_target': 0.15,     # 15% volatilidad anual objetivo
    'volatility_lookback': 21,     # Dias para calcular volatilidad (1 mes)
}

# =============================================================================
# FUNCIONES DE POSICIONAMIENTO
# =============================================================================

def quantile_position_asymmetric(predictions, window=63,
                                  q_long_extreme=10, q_long_moderate=30,
                                  q_short_extreme=10, q_short_moderate=30):
    """
    Calcula posiciones basadas en percentiles rolling con umbrales ASIMETRICOS.

    Esta funcion convierte predicciones continuas del modelo en posiciones
    discretas {-3, -1, 0, +1, +3} basandose en donde cae la prediccion
    actual dentro de la distribucion de predicciones recientes.

    LOGICA:
    -------
    1. Para cada dia, tomamos una ventana de las ultimas N predicciones
    2. Calculamos en que percentil cae la prediccion de HOY
    3. Mapeamos ese percentil a una posicion discreta

    ANTI-LEAKAGE:
    -------------
    - La ventana SOLO mira hacia atras (predicciones pasadas + actual)
    - NO usamos predicciones futuras para calcular el percentil
    - El percentil de hoy se calcula con datos hasta hoy (inclusive)

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

    Ejemplo con q_long_extreme=10, q_long_moderate=30:
        Percentil 90-100%  -> +3 (top 10% = muy alcista)
        Percentil 70-90%   -> +1 (alcista moderado)
        Percentil 30-70%   ->  0 (neutral -> cash)
        Percentil 10-30%   -> -1 (bajista moderado)
        Percentil 0-10%    -> -3 (bottom 10% = muy bajista)
    """
    predictions = np.array(predictions).flatten()
    n = len(predictions)
    positions = np.zeros(n)
    percentiles = np.zeros(n)

    # Calcular umbrales para LONG (desde arriba)
    # Si q_long_extreme=10, entonces thresh_long_3x=90
    # Esto significa: si percentil >= 90, posicion = +3
    thresh_long_3x = 100 - q_long_extreme   # ej: 100-10=90 -> top 10% = +3
    thresh_long_1x = 100 - q_long_moderate  # ej: 100-30=70 -> 70-90% = +1

    # Calcular umbrales para SHORT (desde abajo)
    # Si q_short_extreme=10, entonces thresh_short_3x=10
    # Esto significa: si percentil <= 10, posicion = -3
    thresh_short_3x = q_short_extreme       # ej: 10 -> bottom 10% = -3
    thresh_short_1x = q_short_moderate      # ej: 30 -> 10-30% = -1

    for i in range(n):
        # Ventana de predicciones: desde (i - window + 1) hasta i (inclusive)
        # IMPORTANTE: Solo miramos hacia ATRAS, nunca hacia adelante
        start = max(0, i - window + 1)
        window_preds = predictions[start:i+1]

        # Si la ventana es muy pequena, ir a cash (no hay suficiente historia)
        if len(window_preds) < 5:
            positions[i] = 0
            percentiles[i] = 50
            continue

        # Calcular percentil de la prediccion actual dentro de la ventana
        # percentile = % de predicciones en la ventana que son <= prediccion actual
        current_pred = predictions[i]
        percentile = np.mean(window_preds <= current_pred) * 100
        percentiles[i] = percentile

        # Mapear percentil a posicion discreta (ASIMETRICO)
        if percentile >= thresh_long_3x:
            positions[i] = 3   # Top q_long_extreme% -> muy alcista -> UPRO
        elif percentile >= thresh_long_1x:
            positions[i] = 1   # Entre thresh_long_1x y thresh_long_3x -> SPY
        elif percentile <= thresh_short_3x:
            positions[i] = -3  # Bottom q_short_extreme% -> muy bajista -> SPXU
        elif percentile <= thresh_short_1x:
            positions[i] = -1  # Entre thresh_short_3x y thresh_short_1x -> SH
        else:
            positions[i] = 0   # Zona neutral -> Cash

    # Calcular estadisticas de posicionamiento
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
        'pct_pos_3': np.mean(positions == 3) * 100,    # % tiempo en +3
        'pct_pos_1': np.mean(positions == 1) * 100,    # % tiempo en +1
        'pct_pos_0': np.mean(positions == 0) * 100,    # % tiempo en cash
        'pct_pos_neg1': np.mean(positions == -1) * 100, # % tiempo en -1
        'pct_pos_neg3': np.mean(positions == -3) * 100, # % tiempo en -3
    }

    return positions, info


def apply_position_filter(positions, min_change=1):
    """
    Filtra cambios de posicion muy pequenos para evitar churning.

    CHURNING = hacer muchos trades pequenos que generan costos
    pero no agregan valor.

    Esta funcion evita cambios de posicion menores al umbral.
    Por ejemplo, si min_change=1, no permitimos cambiar de 0 a 0,
    pero si de 0 a 1 o de 1 a 3.

    Args:
        positions: Posiciones originales
        min_change: Cambio minimo para ejecutar trade

    Returns:
        Posiciones filtradas
    """
    positions = np.array(positions).flatten()
    filtered = np.zeros_like(positions)
    filtered[0] = positions[0]  # Primera posicion se mantiene

    for i in range(1, len(positions)):
        # Calcular cambio absoluto respecto a posicion anterior
        change = abs(positions[i] - filtered[i-1])

        # Solo cambiar si el cambio es >= min_change
        if change >= min_change:
            filtered[i] = positions[i]
        else:
            filtered[i] = filtered[i-1]  # Mantener posicion anterior

    return filtered


def apply_volatility_targeting(positions, returns, target_vol=0.15, lookback=21):
    """
    Ajusta posiciones para mantener volatilidad objetivo.

    VOLATILITY TARGETING = escalar posiciones para que la estrategia
    tenga aproximadamente la misma volatilidad todo el tiempo.

    Cuando el mercado esta muy volatil -> reducir posicion
    Cuando el mercado esta calmado -> aumentar posicion

    Args:
        positions: Posiciones originales
        returns: Retornos historicos del mercado
        target_vol: Volatilidad objetivo anualizada (default 15%)
        lookback: Dias para calcular volatilidad (default 21 = 1 mes)

    Returns:
        Posiciones ajustadas (discretizadas a {-3, -1, 0, +1, +3})
    """
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()

    adjusted = positions.copy()

    for i in range(lookback, len(positions)):
        # Calcular volatilidad reciente (anualizada)
        # std * sqrt(252) convierte vol diaria a anual
        recent_vol = np.std(returns[i-lookback:i]) * np.sqrt(252)

        if recent_vol > 0:
            # Factor de ajuste: target_vol / recent_vol
            # Si recent_vol > target_vol -> vol_scalar < 1 -> reducir posicion
            # Si recent_vol < target_vol -> vol_scalar > 1 -> aumentar posicion
            vol_scalar = target_vol / recent_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limitar entre 0.5x y 2x

            # Ajustar posicion (valor continuo)
            adjusted_continuous = positions[i] * vol_scalar

            # Discretizar a posiciones validas {-3, -1, 0, +1, +3}
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


def apply_drawdown_control(positions, cumulative_returns, limits):
    """
    Reduce posicion automaticamente cuando el drawdown excede umbrales.

    DRAWDOWN = caida desde el maximo historico
    Esta es una medida de RIESGO muy importante.

    La logica es:
    - DD < 10%: operar normalmente
    - DD 10-15%: reducir a maximo 2x (no usar 3x)
    - DD 15-20%: reducir a maximo 1x (no usar apalancamiento)
    - DD > 20%: ir a cash (proteger capital)

    IMPORTANTE: Solo genera posiciones validas {-3, -1, 0, +1, +3}
    No existe 2x en la estrategia (no hay ETF 2x liquido).

    Args:
        positions: Posiciones originales
        cumulative_returns: Retornos acumulados (equity curve)
        limits: Dict con umbrales y max_leverage

    Returns:
        Posiciones ajustadas
    """
    positions = np.array(positions).flatten()
    cumulative_returns = np.array(cumulative_returns).flatten()

    # Calcular drawdown en cada punto
    # running_max = maximo historico hasta ese punto
    running_max = np.maximum.accumulate(cumulative_returns)
    # drawdown = (valor_actual - maximo) / maximo
    # drawdown es negativo cuando estamos por debajo del maximo
    drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-10)

    adjusted_positions = positions.copy()

    for i in range(len(positions)):
        dd = abs(drawdown[i])  # Valor absoluto del drawdown
        current_pos = adjusted_positions[i]

        # Aplicar limites segun nivel de drawdown
        if dd >= limits['level_3']['threshold']:  # >= 20%
            # Drawdown severo: ir a cash para proteger capital
            adjusted_positions[i] = 0
        elif dd >= limits['level_2']['threshold']:  # >= 15%
            # Drawdown alto: maximo 1x (sin apalancamiento)
            if abs(current_pos) > 1:
                adjusted_positions[i] = np.sign(current_pos) * 1
        elif dd >= limits['level_1']['threshold']:  # >= 10%
            # Drawdown moderado: reducir 3x a 1x
            if abs(current_pos) == 3:
                adjusted_positions[i] = np.sign(current_pos) * 1
        # else: DD < 10%, sin cambios

    return adjusted_positions


# =============================================================================
# FUNCIONES DE CALCULO DE RETORNOS
# =============================================================================

def get_instrument_for_position(position):
    """
    Retorna el instrumento ETF correspondiente a una posicion.

    Mapeo:
        +3 -> UPRO (3x Long S&P 500)
        +1 -> SPY  (1x Long S&P 500)
         0 -> CASH (efectivo, gana tasa libre de riesgo)
        -1 -> SH   (1x Short S&P 500)
        -3 -> SPXU (3x Short S&P 500)
    """
    mapping = {3: 'UPRO', 1: 'SPY', 0: 'CASH', -1: 'SH', -3: 'SPXU'}
    return mapping.get(int(position), 'CASH')


def calculate_realistic_returns(positions, forward_returns, risk_free_rate,
                                config, apply_vol_drag=True):
    """
    Calcula retornos realistas incluyendo TODOS los costos.

    Esta funcion es crucial para tener resultados realistas.
    Muchos backtests ignoran costos y dan resultados demasiado optimistas.

    COSTOS INCLUIDOS:
    1. Expense ratios: costo anual del ETF (prorrateado diariamente)
    2. Bid-ask spreads: costo de comprar/vender (en cada trade)
    3. Volatility drag: perdida por rebalanceo diario en ETFs apalancados

    FORMULA DE RETORNO:
    -------------------
    Para posicion != 0:
        retorno_bruto = rf + posicion * (mercado - rf)

    Donde:
        - rf = tasa libre de riesgo
        - posicion = {-3, -1, 0, +1, +3}
        - mercado = retorno del S&P 500

    Ejemplo con posicion = +3:
        Si mercado sube 1% y rf = 0.02%:
        retorno = 0.02% + 3 * (1% - 0.02%) = 0.02% + 2.94% = 2.96%

    Args:
        positions: Array de posiciones discretas
        forward_returns: Retornos del mercado (S&P 500)
        risk_free_rate: Tasa libre de riesgo diaria
        config: Configuracion con costos por instrumento
        apply_vol_drag: Si aplicar volatility drag para ETFs apalancados

    Returns:
        dict con retornos brutos, netos, y desglose de costos
    """
    positions = np.array(positions).flatten()
    forward_returns = np.array(forward_returns).flatten()
    risk_free_rate = np.array(risk_free_rate).flatten()

    n_days = len(positions)

    # Arrays para tracking de cada componente
    gross_returns = np.zeros(n_days)      # Retornos antes de costos
    expense_costs = np.zeros(n_days)      # Costo por expense ratio
    trading_costs = np.zeros(n_days)      # Costo por trades
    vol_drag_costs = np.zeros(n_days)     # Costo por volatility drag

    # Pre-calcular volatility drag usando varianza rolling
    if apply_vol_drag:
        lookback = min(21, n_days // 4)
        rolling_variance = pd.Series(forward_returns).rolling(lookback).var().fillna(0).values

    for i in range(n_days):
        pos = positions[i]
        instrument = get_instrument_for_position(pos)
        instrument_config = config['instruments'][instrument]

        # -----------------------------------------------------------------
        # 1. RETORNO BRUTO
        # -----------------------------------------------------------------
        if pos == 0:
            # Cash: solo gana tasa libre de riesgo
            gross_returns[i] = risk_free_rate[i]
        else:
            # Formula: rf + pos * (market - rf)
            # Esto simula el retorno de un ETF apalancado
            gross_returns[i] = risk_free_rate[i] + pos * (forward_returns[i] - risk_free_rate[i])

        # -----------------------------------------------------------------
        # 2. EXPENSE RATIO (costo diario del ETF)
        # -----------------------------------------------------------------
        # Dividir costo anual entre 252 dias de trading
        daily_expense = instrument_config['expense_ratio'] / 252
        expense_costs[i] = daily_expense

        # -----------------------------------------------------------------
        # 3. TRADING COST (solo si hay cambio de posicion)
        # -----------------------------------------------------------------
        if i > 0 and positions[i] != positions[i-1]:
            # Costo = spread del instrumento anterior (venta) + nuevo (compra)
            prev_instrument = get_instrument_for_position(positions[i-1])
            sell_cost = config['instruments'][prev_instrument]['bid_ask']
            buy_cost = instrument_config['bid_ask']
            trading_costs[i] = sell_cost + buy_cost

        # -----------------------------------------------------------------
        # 4. VOLATILITY DRAG (solo para ETFs apalancados 3x)
        # -----------------------------------------------------------------
        # Los ETFs apalancados rebalancean diariamente, lo que causa
        # una perdida sistematica conocida como "volatility drag"
        # Formula: decay = 0.5 * (leverage^2 - |leverage|) * variance
        if apply_vol_drag and abs(pos) == 3:
            leverage = pos
            if i >= lookback:
                var = rolling_variance[i]
            else:
                var = np.var(forward_returns[:max(i+1, 5)])

            vol_drag_costs[i] = 0.5 * (leverage**2 - abs(leverage)) * var

    # Calcular retornos netos (despues de todos los costos)
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
# FUNCIONES DE METRICAS
# =============================================================================

def calculate_metrics(returns, risk_free_rate=None, positions=None):
    """
    Calcula metricas completas de rendimiento de la estrategia.

    METRICAS CALCULADAS:
    -------------------
    - Total Return: retorno compuesto total
    - Annualized Return: retorno anualizado
    - Volatility: desviacion estandar anualizada
    - Sharpe Ratio: retorno ajustado por riesgo
    - Sortino Ratio: como Sharpe pero solo penaliza volatilidad negativa
    - Max Drawdown: maxima caida desde el pico
    - Calmar Ratio: retorno / max drawdown
    - Win Rate: % de dias con retorno positivo

    Args:
        returns: Array de retornos diarios
        risk_free_rate: Tasa libre de riesgo (para Sharpe)
        positions: Posiciones (para calcular metricas de posicionamiento)

    Returns:
        Dict con todas las metricas
    """
    returns = np.array(returns).flatten()

    # Retorno total (compuesto)
    # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    total_return = np.prod(1 + returns) - 1

    # Retorno anualizado
    n_years = len(returns) / 252  # 252 dias de trading por ano
    if n_years > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (1/n_years) - 1
    else:
        annualized_return = 0

    # Volatilidad
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)  # Anualizar

    # Sharpe Ratio = (retorno - rf) / volatilidad
    if annual_vol > 0:
        if risk_free_rate is not None:
            rf_annual = np.mean(risk_free_rate) * 252
        else:
            rf_annual = 0.02  # Asumir 2% si no se proporciona
        sharpe = (annualized_return - rf_annual) / annual_vol
    else:
        sharpe = 0

    # Sortino Ratio (solo considera volatilidad de retornos negativos)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = (annualized_return - rf_annual) / downside_vol if downside_vol > 0 else 0
    else:
        sortino = sharpe * 2  # Si no hay retornos negativos, Sortino es muy alto

    # Max Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)  # Sera negativo

    # Calmar Ratio = retorno anual / |max drawdown|
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win Rate = % de dias con retorno positivo
    win_rate = np.mean(returns > 0)

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

        # Turnover (frecuencia de trades)
        position_changes = np.abs(np.diff(positions))
        metrics['daily_turnover'] = np.mean(position_changes)
        metrics['total_trades'] = np.sum(position_changes > 0)

    return metrics


def bootstrap_confidence(returns, risk_free_rate=None, n_bootstrap=5000, confidence=0.95):
    """
    Calcula intervalos de confianza via bootstrap.

    BOOTSTRAP: Tecnica estadistica para estimar incertidumbre.

    Proceso:
    1. Resampleamos los retornos con reemplazo N veces
    2. Calculamos Sharpe Ratio en cada muestra
    3. Obtenemos la distribucion de Sharpes posibles
    4. Extraemos percentiles para intervalo de confianza

    Esto nos dice:
    - sharpe_ci_lower/upper: rango donde probablemente esta el Sharpe real
    - sharpe_prob_positive: probabilidad de que Sharpe > 0

    Args:
        returns: Array de retornos diarios
        risk_free_rate: Tasa libre de riesgo
        n_bootstrap: Numero de iteraciones (mas = mas preciso)
        confidence: Nivel de confianza (0.95 = 95%)

    Returns:
        Dict con estadisticas de bootstrap
    """
    returns = np.array(returns).flatten()
    n = len(returns)

    # Calcular mean risk-free si se proporciona
    rf_mean = 0.0
    if risk_free_rate is not None:
        rf_mean = np.mean(risk_free_rate)

    sharpes = []
    total_returns = []

    for _ in range(n_bootstrap):
        # Resample con reemplazo (seleccionar n elementos aleatorios)
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = returns[sample_idx]

        # Calcular Sharpe del sample
        sample_std = np.std(sample)
        if sample_std > 0.0001:  # Evitar division por cero
            excess_return = np.mean(sample) - rf_mean
            sharpe = excess_return / sample_std * np.sqrt(252)
            sharpe = np.clip(sharpe, -5, 5)  # Limitar a rango razonable
        else:
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
        'return_ci_lower': np.percentile(total_returns, alpha/2 * 100),
        'return_ci_upper': np.percentile(total_returns, (1 - alpha/2) * 100),
    }


# =============================================================================
# MAIN: OPTIMIZACION DE PARAMETROS
# =============================================================================

print("="*100)
print("OPTIMIZACION DE PARAMETROS POR MODELO (STANDALONE)")
print("="*100)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# VERIFICACION DE ARCHIVOS REQUERIDOS
# =============================================================================
artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")

if not os.path.exists(artifacts_path):
    print("=" * 80)
    print("ERROR: FALTA ARCHIVO REQUERIDO")
    print("=" * 80)
    print(f"""
    [ERROR] No se encontro: {artifacts_path}

    SOLUCION: Ejecutar primero los pasos 1 y 2:
        python scripts/build_dataset.py
        python scripts/train_models.py
    """)
    print("=" * 80)
    print("\nORDEN DE EJECUCION CORRECTO:")
    print("    1. python scripts/build_dataset.py")
    print("    2. python scripts/train_models.py")
    print("    3. python scripts/optimize_model_params.py  <-- (este script)")
    print("    4. python scripts/final_long_only_backtest_new.py")
    print("=" * 80)
    sys.exit(1)

# =============================================================================
# PASO 1: CARGAR ARTEFACTOS
# =============================================================================
print("\n" + "="*80)
print("PASO 1: Cargar Artefactos de Entrenamiento")
print("="*80)

print(f"  + Cargando: {artifacts_path}")

with open(artifacts_path, 'rb') as f:
    artifacts = pickle.load(f)

metadata = artifacts['metadata']
models_data = artifacts['models']

forward_returns_test = metadata['forward_returns_test']
risk_free_test = metadata['risk_free_test']
y_test = metadata['y_test']

print(f"  + Modelos cargados: {len(models_data)}")
print(f"  + Dias de test: {len(forward_returns_test)}")

# =============================================================================
# PASO 2: ANALIZAR CARACTERISTICAS DE CADA MODELO
# =============================================================================
print("\n" + "="*80)
print("PASO 2: Analisis de Caracteristicas de Predicciones")
print("="*80)

model_characteristics = {}

for model_name, model_data in models_data.items():
    test_pred = np.array(model_data['test_predictions'])
    n_test = min(len(test_pred), len(y_test))
    test_pred = test_pred[:n_test]
    actual = y_test[:n_test]

    # Accuracy direccional
    pred_direction = np.sign(test_pred)
    actual_direction = np.sign(actual)
    directional_accuracy = np.mean(pred_direction == actual_direction)

    # Estadisticas
    model_characteristics[model_name] = {
        'directional_accuracy': directional_accuracy,
        'pred_mean': np.mean(test_pred),
        'pred_std': np.std(test_pred),
        'p10': np.percentile(test_pred, 10),
        'p90': np.percentile(test_pred, 90),
        'pct_positive': np.mean(test_pred > 0) * 100,
    }

print(f"\n{'Model':<20} {'DirAcc':>7} {'Mean':>10} {'Std':>10} {'%Pos':>6}")
print("-"*60)
for model_name, chars in sorted(model_characteristics.items(),
                                 key=lambda x: x[1]['directional_accuracy'],
                                 reverse=True):
    print(f"{model_name:<20} {chars['directional_accuracy']*100:>6.1f}% "
          f"{chars['pred_mean']:>10.5f} {chars['pred_std']:>10.5f} "
          f"{chars['pct_positive']:>5.1f}%")

# =============================================================================
# PASO 3: GRID SEARCH DE PARAMETROS
# =============================================================================
print("\n" + "="*80)
print("PASO 3: Grid Search de Parametros Optimos")
print("="*80)

param_grid = {
    'q_long_extreme': [5, 10, 15, 20, 25, 30],
    'q_long_moderate': [20, 30, 40, 50, 60],
    'q_short_extreme': [5, 10, 15, 20, 25, 30],
    'q_short_moderate': [10, 20, 30, 40],
}

def evaluate_params(test_pred, fwd_ret, rf, params):
    """Evalua una combinacion de parametros."""
    if params['q_long_moderate'] <= params['q_long_extreme']:
        return None
    if params['q_short_moderate'] <= params['q_short_extreme']:
        return None

    positions, info = quantile_position_asymmetric(
        test_pred, window=63,
        q_long_extreme=params['q_long_extreme'],
        q_long_moderate=params['q_long_moderate'],
        q_short_extreme=params['q_short_extreme'],
        q_short_moderate=params['q_short_moderate'],
    )

    positions = apply_position_filter(positions, CONFIG['min_position_change'])
    positions = apply_volatility_targeting(positions, fwd_ret,
                                           CONFIG['volatility_target'],
                                           CONFIG['volatility_lookback'])

    returns_temp = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])

    # ANTI-LEAKAGE: Usar drawdown hasta AYER
    cumulative_temp_lag = np.concatenate([[1.0], cumulative_temp[:-1]])
    positions = apply_drawdown_control(positions, cumulative_temp_lag, CONFIG['drawdown_limits'])
    returns = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)

    metrics = calculate_metrics(returns['net_returns'], risk_free_rate=rf, positions=positions)

    return {
        'return': metrics['total_return'],
        'sharpe': metrics['sharpe_ratio'],
        'sortino': metrics['sortino_ratio'],
        'max_dd': metrics['max_drawdown'],
        'calmar': metrics['calmar_ratio'],
        'pct_3x_long': info['pct_pos_3'],
        'pct_1x_long': info['pct_pos_1'],
        'pct_cash': info['pct_pos_0'],
        'pct_short': info['pct_pos_neg1'] + info['pct_pos_neg3'],
        'total_trades': metrics.get('total_trades', 0),
    }

all_combinations = list(product(
    param_grid['q_long_extreme'],
    param_grid['q_long_moderate'],
    param_grid['q_short_extreme'],
    param_grid['q_short_moderate'],
))

print(f"\nTotal combinaciones: {len(all_combinations)} x {len(models_data)} modelos")

optimal_params = {}
all_results = []

for model_name, model_data in models_data.items():
    print(f"\n  Optimizando: {model_name}...")

    test_pred = np.array(model_data['test_predictions'])
    n_test = min(len(test_pred), len(forward_returns_test))
    test_pred = test_pred[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    best_sharpe = -999
    best_params = None
    best_result = None

    for q_le, q_lm, q_se, q_sm in all_combinations:
        params = {
            'q_long_extreme': q_le,
            'q_long_moderate': q_lm,
            'q_short_extreme': q_se,
            'q_short_moderate': q_sm,
        }

        result = evaluate_params(test_pred, fwd_ret, rf, params)

        if result is None:
            continue

        all_results.append({'model': model_name, **params, **result})

        score = result['sharpe']
        if result['max_dd'] < -0.20:
            score -= 0.5

        if score > best_sharpe:
            best_sharpe = score
            best_params = params.copy()
            best_result = result.copy()

    optimal_params[model_name] = {
        'params': best_params,
        'result': best_result,
        'characteristics': model_characteristics[model_name],
    }

    print(f"    Mejor: L({best_params['q_long_extreme']},{best_params['q_long_moderate']}) "
          f"S({best_params['q_short_extreme']},{best_params['q_short_moderate']}) "
          f"-> Sharpe: {best_result['sharpe']:.2f}")

# =============================================================================
# PASO 4: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
print("\n" + "="*80)
print("PASO 4: Bootstrap Confidence Intervals")
print("="*80)

for model_name in optimal_params.keys():
    model_data = models_data[model_name]
    test_pred = np.array(model_data['test_predictions'])

    n_test = min(len(test_pred), len(forward_returns_test))
    test_pred = test_pred[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    opt = optimal_params[model_name]
    p = opt['params']

    positions, _ = quantile_position_asymmetric(
        test_pred, window=63,
        q_long_extreme=p['q_long_extreme'],
        q_long_moderate=p['q_long_moderate'],
        q_short_extreme=p['q_short_extreme'],
        q_short_moderate=p['q_short_moderate'],
    )

    positions = apply_position_filter(positions, CONFIG['min_position_change'])
    positions = apply_volatility_targeting(positions, fwd_ret,
                                           CONFIG['volatility_target'],
                                           CONFIG['volatility_lookback'])

    returns_temp = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])
    cumulative_temp_lag = np.concatenate([[1.0], cumulative_temp[:-1]])
    positions = apply_drawdown_control(positions, cumulative_temp_lag, CONFIG['drawdown_limits'])
    returns = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)

    boot = bootstrap_confidence(returns['net_returns'], risk_free_rate=rf, n_bootstrap=5000)
    optimal_params[model_name]['bootstrap'] = boot

    print(f"  {model_name:<20} Sharpe: {opt['result']['sharpe']:>5.2f} "
          f"CI: [{boot['sharpe_ci_lower']:>5.2f}, {boot['sharpe_ci_upper']:>5.2f}]")

# =============================================================================
# PASO 5: GUARDAR RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("PASO 5: Guardar Configuracion Optima")
print("="*80)

optimal_config = {}
for model_name, opt in optimal_params.items():
    optimal_config[model_name] = {
        'params': opt['params'],
        'expected_sharpe': float(opt['result']['sharpe']),
        'expected_return': float(opt['result']['return']),
        'expected_max_dd': float(opt['result']['max_dd']),
        'bootstrap': {k: float(v) for k, v in opt['bootstrap'].items()},
        'characteristics': {k: float(v) for k, v in opt['characteristics'].items()},
    }

config_path = os.path.join(RESULTS_DIR, 'optimal_model_params.json')
with open(config_path, 'w') as f:
    json.dump(optimal_config, f, indent=2)
print(f"  + Guardado: {config_path}")

results_df = pd.DataFrame(all_results)
results_path = os.path.join(RESULTS_DIR, 'param_grid_search_results.csv')
results_df.to_csv(results_path, index=False)
print(f"  + Guardado: {results_path}")

# =============================================================================
# PASO 6: RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

spy_returns = forward_returns_test
spy_total_return = np.prod(1 + spy_returns) - 1
spy_sharpe = (np.mean(spy_returns) - np.mean(risk_free_test)) / np.std(spy_returns) * np.sqrt(252)

print(f"\n  BENCHMARK (Buy & Hold SPY): Return={spy_total_return*100:.1f}%, Sharpe={spy_sharpe:.2f}")

print(f"\n  TOP 5 MODELOS:")
top5 = sorted(optimal_params.items(), key=lambda x: x[1]['result']['sharpe'], reverse=True)[:5]

for i, (model_name, opt) in enumerate(top5, 1):
    p = opt['params']
    r = opt['result']
    print(f"  {i}. {model_name}: Sharpe={r['sharpe']:.2f}, Return={r['return']*100:.1f}%, "
          f"L({p['q_long_extreme']},{p['q_long_moderate']}) S({p['q_short_extreme']},{p['q_short_moderate']})")

print("\n" + "="*80)
print("OPTIMIZACION COMPLETADA")
print("="*80)
