# -*- coding: utf-8 -*-
"""
================================================================================
SIGNAL FILTERING - FILTROS ANTI-OVERFITTING PARA SENALES DE TRADING
================================================================================

Implementa filtros inspirados en HSBC-ML RuleFinder para evitar overfitting
en backtesting de estrategias de trading.

FILTROS IMPLEMENTADOS:
1. First-in-X-Days: Solo toma la primera senal en una ventana de X dias
2. Minimum Gap: Requiere minimo de dias entre senales
3. Confirmation Filter: Requiere confirmacion en siguientes periodos
4. Volume Filter: Filtra por volumen minimo
5. Volatility Filter: Filtra por nivel de volatilidad

PROPOSITO:
- Evitar "signal stacking" que infla metricas de backtest
- Reducir sobreajuste a ruido de mercado
- Simular condiciones reales de trading

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

print("=" * 80)
print("SIGNAL FILTERING - FILTROS ANTI-OVERFITTING")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


# =============================================================================
# FIRST-IN-X-DAYS FILTER (HSBC-ML Core)
# =============================================================================

def count_signals_in_window(signal_series, window):
    """
    Cuenta senales en una ventana deslizante.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales (1=senal, 0=no senal)
    window : int - Tamano de ventana en dias

    Returns:
    --------
    pd.Series: Conteo de senales en ventana
    """
    return signal_series.rolling(window=window, min_periods=1).sum()


def first_in_x_days(signal_series, x_days):
    """
    Filtro First-in-X-Days - Solo mantiene la primera senal en cada ventana.

    Inspirado en HSBC-ML filtraxdays() function.

    Este filtro es CRITICO para evitar overfitting porque:
    - Evita contar multiples senales cercanas como eventos independientes
    - Simula la realidad donde solo puedes actuar una vez por periodo
    - Reduce inflacion artificial de metricas de backtest

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales (1=senal, 0=no senal)
    x_days : int - Ventana minima entre senales

    Returns:
    --------
    pd.Series: Serie filtrada con solo primera senal por ventana
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    filtered = pd.Series(0, index=signal_series.index)
    last_signal_idx = -x_days - 1  # Permite primera senal

    for i, (idx, val) in enumerate(signal_series.items()):
        if val == 1:
            if i - last_signal_idx > x_days:
                filtered.loc[idx] = 1
                last_signal_idx = i

    return filtered


def first_in_x_days_vectorized(signal_series, x_days):
    """
    Version vectorizada del filtro First-in-X-Days (mas rapida).

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales (1=senal, 0=no senal)
    x_days : int - Ventana minima entre senales

    Returns:
    --------
    pd.Series: Serie filtrada con solo primera senal por ventana
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    # Contar senales en ventana (incluyendo actual)
    count_in_window = signal_series.rolling(window=x_days, min_periods=1).sum()

    # Primera senal es donde: hay senal Y es la unica en la ventana
    # O es la primera senal despues de un gap
    filtered = signal_series.copy()

    # Marcar como 0 donde hay multiples senales en ventana
    for i in range(1, len(signal_series)):
        if signal_series.iloc[i] == 1:
            # Verificar si hubo senal en los ultimos x_days
            start_idx = max(0, i - x_days)
            if signal_series.iloc[start_idx:i].sum() > 0:
                filtered.iloc[i] = 0

    return filtered


# =============================================================================
# MINIMUM GAP FILTER
# =============================================================================

def minimum_gap_filter(signal_series, min_gap):
    """
    Requiere un minimo de dias entre senales consecutivas.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales
    min_gap : int - Gap minimo requerido entre senales

    Returns:
    --------
    pd.Series: Serie filtrada
    """
    return first_in_x_days(signal_series, min_gap)


# =============================================================================
# CONFIRMATION FILTER
# =============================================================================

def confirmation_filter(signal_series, price_series, confirmation_days=1,
                        direction='same', threshold=0):
    """
    Requiere confirmacion del movimiento en dias siguientes.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales
    price_series : pd.Series - Serie de precios
    confirmation_days : int - Dias para confirmar
    direction : str - 'same' (mismo movimiento), 'opposite' (reversal)
    threshold : float - Umbral minimo de movimiento

    Returns:
    --------
    pd.Series: Serie filtrada con senales confirmadas
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    # Calcular retornos futuros
    future_returns = price_series.pct_change(confirmation_days).shift(-confirmation_days)

    # Determinar direccion de la senal original
    # Asumimos senal=1 es compra (esperamos subida)
    if direction == 'same':
        confirmed = (future_returns > threshold).astype(int)
    else:  # opposite
        confirmed = (future_returns < -threshold).astype(int)

    # Solo mantener senales confirmadas
    filtered = signal_series * confirmed

    return filtered.fillna(0).astype(int)


# =============================================================================
# VOLUME FILTER
# =============================================================================

def volume_filter(signal_series, volume_series, min_percentile=50, window=20):
    """
    Filtra senales que ocurren con volumen bajo.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales
    volume_series : pd.Series - Serie de volumen
    min_percentile : int - Percentil minimo de volumen requerido (0-100)
    window : int - Ventana para calcular percentil

    Returns:
    --------
    pd.Series: Serie filtrada
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    # Calcular percentil del volumen en ventana historica
    vol_percentile = volume_series.rolling(window=window).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )

    # Filtrar senales con volumen bajo
    volume_ok = (vol_percentile >= min_percentile).astype(int)
    filtered = signal_series * volume_ok

    return filtered.fillna(0).astype(int)


# =============================================================================
# VOLATILITY FILTER
# =============================================================================

def volatility_filter(signal_series, price_series, mode='normal',
                      vol_window=20, low_pct=20, high_pct=80):
    """
    Filtra senales basado en nivel de volatilidad.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales
    price_series : pd.Series - Serie de precios
    mode : str - 'normal' (vol media), 'low' (vol baja), 'high' (vol alta)
    vol_window : int - Ventana para calcular volatilidad
    low_pct : int - Percentil bajo de volatilidad
    high_pct : int - Percentil alto de volatilidad

    Returns:
    --------
    pd.Series: Serie filtrada
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    # Calcular volatilidad realizada
    returns = price_series.pct_change()
    volatility = returns.rolling(window=vol_window).std() * np.sqrt(252)

    # Calcular percentiles historicos
    vol_percentile = volatility.rolling(window=252).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )

    # Aplicar filtro segun modo
    if mode == 'low':
        vol_ok = (vol_percentile <= low_pct).astype(int)
    elif mode == 'high':
        vol_ok = (vol_percentile >= high_pct).astype(int)
    else:  # normal
        vol_ok = ((vol_percentile > low_pct) & (vol_percentile < high_pct)).astype(int)

    filtered = signal_series * vol_ok

    return filtered.fillna(0).astype(int)


# =============================================================================
# REGIME FILTER
# =============================================================================

def regime_filter(signal_series, regime_series, allowed_regimes=[1]):
    """
    Filtra senales que no ocurren en el regimen de mercado deseado.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales
    regime_series : pd.Series - Serie de regimen (-1=bear, 0=neutral, 1=bull)
    allowed_regimes : list - Regimenes permitidos

    Returns:
    --------
    pd.Series: Serie filtrada
    """
    if signal_series.sum() == 0:
        return signal_series.copy()

    regime_ok = regime_series.isin(allowed_regimes).astype(int)
    filtered = signal_series * regime_ok

    return filtered.fillna(0).astype(int)


# =============================================================================
# COMBINED FILTER PIPELINE
# =============================================================================

def apply_filter_pipeline(signal_series, filters_config, df=None):
    """
    Aplica una secuencia de filtros a las senales.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales original
    filters_config : list of dict - Configuracion de filtros a aplicar
    df : pd.DataFrame - DataFrame con datos adicionales (precios, volumen, etc.)

    Returns:
    --------
    pd.Series: Serie filtrada
    dict: Estadisticas de filtrado

    Example:
    --------
    filters = [
        {'type': 'first_in_x_days', 'x_days': 5},
        {'type': 'volume', 'min_percentile': 50, 'volume_col': 'SPY_VOLUME'},
        {'type': 'volatility', 'mode': 'normal', 'price_col': 'SPY_CLOSE'}
    ]
    filtered, stats = apply_filter_pipeline(signals, filters, df)
    """
    filtered = signal_series.copy()
    stats = {
        'original_count': signal_series.sum(),
        'filters_applied': []
    }

    for filter_cfg in filters_config:
        filter_type = filter_cfg.get('type')
        before_count = filtered.sum()

        if filter_type == 'first_in_x_days':
            filtered = first_in_x_days(filtered, filter_cfg.get('x_days', 5))

        elif filter_type == 'minimum_gap':
            filtered = minimum_gap_filter(filtered, filter_cfg.get('min_gap', 5))

        elif filter_type == 'confirmation':
            if df is not None:
                price_col = filter_cfg.get('price_col', 'SPY_CLOSE')
                filtered = confirmation_filter(
                    filtered,
                    df[price_col],
                    filter_cfg.get('confirmation_days', 1),
                    filter_cfg.get('direction', 'same'),
                    filter_cfg.get('threshold', 0)
                )

        elif filter_type == 'volume':
            if df is not None:
                vol_col = filter_cfg.get('volume_col', 'SPY_VOLUME')
                filtered = volume_filter(
                    filtered,
                    df[vol_col],
                    filter_cfg.get('min_percentile', 50),
                    filter_cfg.get('window', 20)
                )

        elif filter_type == 'volatility':
            if df is not None:
                price_col = filter_cfg.get('price_col', 'SPY_CLOSE')
                filtered = volatility_filter(
                    filtered,
                    df[price_col],
                    filter_cfg.get('mode', 'normal'),
                    filter_cfg.get('vol_window', 20),
                    filter_cfg.get('low_pct', 20),
                    filter_cfg.get('high_pct', 80)
                )

        elif filter_type == 'regime':
            if df is not None:
                regime_col = filter_cfg.get('regime_col', 'regime')
                if regime_col in df.columns:
                    filtered = regime_filter(
                        filtered,
                        df[regime_col],
                        filter_cfg.get('allowed_regimes', [1])
                    )

        after_count = filtered.sum()
        stats['filters_applied'].append({
            'type': filter_type,
            'config': filter_cfg,
            'before': before_count,
            'after': after_count,
            'removed': before_count - after_count,
            'removal_rate': (before_count - after_count) / before_count if before_count > 0 else 0
        })

    stats['final_count'] = filtered.sum()
    stats['total_removal_rate'] = 1 - (stats['final_count'] / stats['original_count']) if stats['original_count'] > 0 else 0

    return filtered, stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_filter_stats(stats):
    """Imprime estadisticas de filtrado de forma legible."""
    print("-" * 60)
    print("ESTADISTICAS DE FILTRADO")
    print("-" * 60)
    print(f"Senales originales: {stats['original_count']}")
    print(f"Senales finales: {stats['final_count']}")
    print(f"Tasa de remocion total: {stats['total_removal_rate']:.1%}")
    print()

    for i, f in enumerate(stats['filters_applied'], 1):
        print(f"Filtro {i}: {f['type']}")
        print(f"  Antes: {f['before']} -> Despues: {f['after']}")
        print(f"  Removidas: {f['removed']} ({f['removal_rate']:.1%})")


def compare_filtered_vs_unfiltered(df, signal_col, filtered_signal, return_col,
                                   horizons=[1, 5, 10, 21]):
    """
    Compara metricas de senales filtradas vs no filtradas.

    Returns DataFrame con comparacion de metricas.
    """
    results = []

    for horizon in horizons:
        future_ret = df[return_col].shift(-horizon) if return_col in df.columns else df['SPY_CLOSE'].pct_change(horizon).shift(-horizon)

        # Senales originales
        orig_signals = df[signal_col] == 1
        orig_mean = future_ret[orig_signals].mean()
        orig_std = future_ret[orig_signals].std()
        orig_count = orig_signals.sum()
        orig_winrate = (future_ret[orig_signals] > 0).mean()

        # Senales filtradas
        filt_signals = filtered_signal == 1
        filt_mean = future_ret[filt_signals].mean()
        filt_std = future_ret[filt_signals].std()
        filt_count = filt_signals.sum()
        filt_winrate = (future_ret[filt_signals] > 0).mean()

        results.append({
            'horizon': horizon,
            'original_count': orig_count,
            'original_mean': orig_mean,
            'original_std': orig_std,
            'original_winrate': orig_winrate,
            'filtered_count': filt_count,
            'filtered_mean': filt_mean,
            'filtered_std': filt_std,
            'filtered_winrate': filt_winrate,
            'count_reduction': 1 - filt_count/orig_count if orig_count > 0 else 0
        })

    return pd.DataFrame(results)


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso
    print("EJEMPLO DE USO DE FILTROS")
    print("-" * 60)

    # Crear datos de ejemplo
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='B')

    df_example = pd.DataFrame({
        'date': dates,
        'SPY_CLOSE': 100 * (1 + np.random.randn(n).cumsum() * 0.01),
        'SPY_VOLUME': np.random.randint(1000000, 10000000, n)
    })

    # Crear senales de ejemplo (RSI oversold simulado)
    df_example['signal'] = (np.random.rand(n) < 0.1).astype(int)

    print(f"Senales originales: {df_example['signal'].sum()}")

    # Aplicar filtro First-in-X-Days
    filtered = first_in_x_days(df_example['signal'], x_days=5)
    print(f"Senales despues de First-in-5-Days: {filtered.sum()}")

    # Aplicar pipeline completo
    filters = [
        {'type': 'first_in_x_days', 'x_days': 5},
        {'type': 'volume', 'min_percentile': 40, 'volume_col': 'SPY_VOLUME'},
    ]

    filtered_pipeline, stats = apply_filter_pipeline(
        df_example['signal'],
        filters,
        df_example
    )

    print_filter_stats(stats)

    print("\n" + "=" * 60)
    print("[OK] SIGNAL FILTERING COMPLETADO")
    print("=" * 60)
