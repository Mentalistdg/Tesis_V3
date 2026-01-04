# -*- coding: utf-8 -*-
"""
================================================================================
BACKTESTING UTILITIES - EVENT STUDY & TRADING METRICS
================================================================================

Framework de backtesting inspirado en HSBC-ML RuleFinder con metricas
profesionales de trading y analisis de eventos.

COMPONENTES:
1. Event Study Framework: Analisis t-N a t+N alrededor de senales
2. Trading Metrics: Win Rate, Profit Factor, Sharpe, Sortino, etc.
3. Performance Attribution: Descomposicion de retornos
4. Statistical Tests: Significancia de senales

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy import stats

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

print("=" * 80)
print("BACKTESTING UTILITIES - EVENT STUDY & TRADING METRICS")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


# =============================================================================
# EVENT STUDY FRAMEWORK (HSBC-ML Style)
# =============================================================================

def create_event_window(signal_series, price_series, lookback=10, lookforward=10):
    """
    Crea ventanas de eventos alrededor de cada senal.

    Inspirado en HSBC-ML event study methodology.

    Parameters:
    -----------
    signal_series : pd.Series - Serie de senales (1=evento, 0=no evento)
    price_series : pd.Series - Serie de precios
    lookback : int - Periodos hacia atras (t-N)
    lookforward : int - Periodos hacia adelante (t+N)

    Returns:
    --------
    pd.DataFrame: Matriz con retornos de cada evento en ventana temporal
    """
    events = signal_series[signal_series == 1].index

    if len(events) == 0:
        print("[WARN] No hay eventos para analizar")
        return pd.DataFrame()

    # Calcular retornos
    returns = price_series.pct_change()

    # Crear matriz de eventos
    window_range = range(-lookback, lookforward + 1)
    event_data = []

    for event_date in events:
        event_idx = signal_series.index.get_loc(event_date)

        event_returns = {}
        for offset in window_range:
            target_idx = event_idx + offset
            if 0 <= target_idx < len(returns):
                event_returns[f't{offset:+d}' if offset != 0 else 't0'] = returns.iloc[target_idx]
            else:
                event_returns[f't{offset:+d}' if offset != 0 else 't0'] = np.nan

        event_returns['event_date'] = event_date
        event_data.append(event_returns)

    df_events = pd.DataFrame(event_data)
    df_events.set_index('event_date', inplace=True)

    return df_events


def calculate_cumulative_returns(event_window_df, from_col='t0'):
    """
    Calcula retornos acumulados desde un punto de referencia.

    Parameters:
    -----------
    event_window_df : pd.DataFrame - Matriz de eventos
    from_col : str - Columna desde donde calcular acumulados

    Returns:
    --------
    pd.DataFrame: Retornos acumulados
    """
    if event_window_df.empty:
        return pd.DataFrame()

    # Ordenar columnas temporalmente
    cols = sorted([c for c in event_window_df.columns if c.startswith('t')],
                  key=lambda x: int(x.replace('t', '').replace('+', '')))

    df_sorted = event_window_df[cols].copy()

    # Encontrar indice de from_col
    from_idx = cols.index(from_col)

    # Calcular retornos acumulados desde from_col
    cumulative = df_sorted.copy()
    for i, col in enumerate(cols):
        if i >= from_idx:
            cumulative[col] = (1 + df_sorted.iloc[:, from_idx:i+1]).prod(axis=1) - 1
        else:
            cumulative[col] = np.nan

    return cumulative


def event_study_statistics(event_window_df):
    """
    Calcula estadisticas agregadas del event study.

    Inspirado en HSBC-ML retstats analysis.

    Parameters:
    -----------
    event_window_df : pd.DataFrame - Matriz de eventos

    Returns:
    --------
    pd.DataFrame: Estadisticas por periodo temporal
    """
    if event_window_df.empty:
        return pd.DataFrame()

    # Columnas temporales
    t_cols = [c for c in event_window_df.columns if c.startswith('t')]

    stats_data = []
    for col in t_cols:
        col_data = event_window_df[col].dropna()

        if len(col_data) == 0:
            continue

        pos_returns = col_data[col_data > 0]
        neg_returns = col_data[col_data < 0]

        stats_data.append({
            'period': col,
            'count': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            'q25': col_data.quantile(0.25),
            'median': col_data.median(),
            'q75': col_data.quantile(0.75),
            'max': col_data.max(),
            'win_rate': len(pos_returns) / len(col_data) if len(col_data) > 0 else 0,
            'mean_win': pos_returns.mean() if len(pos_returns) > 0 else 0,
            'mean_loss': neg_returns.mean() if len(neg_returns) > 0 else 0,
            'skewness': col_data.skew(),
            't_stat': col_data.mean() / (col_data.std() / np.sqrt(len(col_data))) if col_data.std() > 0 else 0,
            'p_value': 2 * (1 - stats.t.cdf(abs(col_data.mean() / (col_data.std() / np.sqrt(len(col_data)))), len(col_data)-1)) if col_data.std() > 0 else 1
        })

    return pd.DataFrame(stats_data)


# =============================================================================
# TRADING METRICS (Professional)
# =============================================================================

def calculate_win_rate(returns_series):
    """
    Calcula Win Rate - Porcentaje de operaciones ganadoras.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos de operaciones

    Returns:
    --------
    float: Win rate (0 a 1)
    """
    returns = returns_series.dropna()
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(returns_series):
    """
    Calcula Profit Factor - Ratio de ganancias totales / perdidas totales.

    PF > 1: Estrategia rentable
    PF > 2: Estrategia muy rentable
    PF < 1: Estrategia perdedora

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos

    Returns:
    --------
    float: Profit Factor
    """
    returns = returns_series.dropna()
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0

    return gross_profit / gross_loss


def calculate_sharpe_ratio(returns_series, risk_free_rate=0.0, periods_per_year=252):
    """
    Calcula Sharpe Ratio anualizado.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos
    risk_free_rate : float - Tasa libre de riesgo anualizada
    periods_per_year : int - Periodos por ano (252 para diario)

    Returns:
    --------
    float: Sharpe Ratio anualizado
    """
    returns = returns_series.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns_series, risk_free_rate=0.0, periods_per_year=252):
    """
    Calcula Sortino Ratio - Similar a Sharpe pero solo penaliza downside volatility.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos
    risk_free_rate : float - Tasa libre de riesgo anualizada
    periods_per_year : int - Periodos por ano

    Returns:
    --------
    float: Sortino Ratio anualizado
    """
    returns = returns_series.dropna()
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0

    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(cumulative_returns):
    """
    Calcula Maximum Drawdown.

    Parameters:
    -----------
    cumulative_returns : pd.Series - Retornos acumulados (wealth curve)

    Returns:
    --------
    float: Max Drawdown (valor negativo)
    """
    wealth = (1 + cumulative_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(returns_series, periods_per_year=252):
    """
    Calcula Calmar Ratio - Retorno anualizado / Max Drawdown.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos
    periods_per_year : int - Periodos por ano

    Returns:
    --------
    float: Calmar Ratio
    """
    returns = returns_series.dropna()
    if len(returns) == 0:
        return 0.0

    annual_return = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return np.inf if annual_return > 0 else 0

    return annual_return / max_dd


def calculate_expectancy(returns_series):
    """
    Calcula Expectancy - Ganancia esperada por operacion.

    E = (Win% * Avg Win) - (Loss% * Avg Loss)

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos

    Returns:
    --------
    float: Expectancy
    """
    returns = returns_series.dropna()
    if len(returns) == 0:
        return 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    return (win_rate * avg_win) - (loss_rate * avg_loss)


# =============================================================================
# COMPREHENSIVE METRICS REPORT
# =============================================================================

def calculate_all_metrics(returns_series, risk_free_rate=0.0, periods_per_year=252):
    """
    Calcula todas las metricas de trading de una vez.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos
    risk_free_rate : float - Tasa libre de riesgo
    periods_per_year : int - Periodos por ano

    Returns:
    --------
    dict: Diccionario con todas las metricas
    """
    returns = returns_series.dropna()

    if len(returns) == 0:
        return {
            'n_trades': 0,
            'total_return': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'avg_return': 0,
            'std_return': 0,
            'skewness': 0,
            'kurtosis': 0
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    metrics = {
        # Count metrics
        'n_trades': len(returns),
        'n_wins': len(wins),
        'n_losses': len(losses),

        # Return metrics
        'total_return': (1 + returns).prod() - 1,
        'avg_return': returns.mean(),
        'std_return': returns.std(),
        'median_return': returns.median(),

        # Win/Loss metrics
        'win_rate': calculate_win_rate(returns),
        'avg_win': wins.mean() if len(wins) > 0 else 0,
        'avg_loss': losses.mean() if len(losses) > 0 else 0,
        'max_win': returns.max(),
        'max_loss': returns.min(),

        # Ratio metrics
        'profit_factor': calculate_profit_factor(returns),
        'expectancy': calculate_expectancy(returns),
        'payoff_ratio': abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.inf,

        # Risk-adjusted metrics
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),

        # Distribution metrics
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),

        # Annualized metrics
        'annual_return': (1 + returns).prod() ** (periods_per_year / len(returns)) - 1,
        'annual_volatility': returns.std() * np.sqrt(periods_per_year)
    }

    return metrics


def print_metrics_report(metrics, title="METRICAS DE TRADING"):
    """
    Imprime reporte de metricas formateado.

    Parameters:
    -----------
    metrics : dict - Diccionario de metricas
    title : str - Titulo del reporte
    """
    print("=" * 60)
    print(title)
    print("=" * 60)

    print("\n📊 ESTADISTICAS BASICAS:")
    print(f"  Operaciones totales: {metrics.get('n_trades', 0)}")
    print(f"  Operaciones ganadoras: {metrics.get('n_wins', 0)}")
    print(f"  Operaciones perdedoras: {metrics.get('n_losses', 0)}")

    print("\n💰 RETORNOS:")
    print(f"  Retorno total: {metrics.get('total_return', 0):.2%}")
    print(f"  Retorno promedio: {metrics.get('avg_return', 0):.4%}")
    print(f"  Retorno anualizado: {metrics.get('annual_return', 0):.2%}")
    print(f"  Volatilidad anualizada: {metrics.get('annual_volatility', 0):.2%}")

    print("\n🎯 WIN/LOSS:")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  Ganancia promedio: {metrics.get('avg_win', 0):.4%}")
    print(f"  Perdida promedio: {metrics.get('avg_loss', 0):.4%}")
    print(f"  Mejor operacion: {metrics.get('max_win', 0):.4%}")
    print(f"  Peor operacion: {metrics.get('max_loss', 0):.4%}")

    print("\n📈 RATIOS:")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Expectancy: {metrics.get('expectancy', 0):.4%}")
    print(f"  Payoff Ratio: {metrics.get('payoff_ratio', 0):.2f}")

    print("\n⚖️ RISK-ADJUSTED:")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    print("\n📉 DISTRIBUCION:")
    print(f"  Skewness: {metrics.get('skewness', 0):.2f}")
    print(f"  Kurtosis: {metrics.get('kurtosis', 0):.2f}")

    print("=" * 60)


# =============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# =============================================================================

def test_signal_significance(returns_with_signal, returns_without_signal, alpha=0.05):
    """
    Test de significancia para comparar retornos con y sin senal.

    Parameters:
    -----------
    returns_with_signal : pd.Series - Retornos cuando hay senal
    returns_without_signal : pd.Series - Retornos cuando no hay senal
    alpha : float - Nivel de significancia

    Returns:
    --------
    dict: Resultados del test
    """
    r_with = returns_with_signal.dropna()
    r_without = returns_without_signal.dropna()

    if len(r_with) < 2 or len(r_without) < 2:
        return {'significant': False, 'reason': 'Insufficient data'}

    # T-test
    t_stat, p_value = stats.ttest_ind(r_with, r_without)

    # Mann-Whitney U test (no asume normalidad)
    u_stat, u_pvalue = stats.mannwhitneyu(r_with, r_without, alternative='two-sided')

    return {
        'significant_ttest': p_value < alpha,
        'significant_mannwhitney': u_pvalue < alpha,
        't_statistic': t_stat,
        't_pvalue': p_value,
        'u_statistic': u_stat,
        'u_pvalue': u_pvalue,
        'mean_with_signal': r_with.mean(),
        'mean_without_signal': r_without.mean(),
        'mean_difference': r_with.mean() - r_without.mean(),
        'n_with_signal': len(r_with),
        'n_without_signal': len(r_without)
    }


def bootstrap_confidence_interval(returns_series, metric_func, n_bootstrap=1000,
                                  confidence_level=0.95):
    """
    Calcula intervalo de confianza bootstrap para una metrica.

    Parameters:
    -----------
    returns_series : pd.Series - Serie de retornos
    metric_func : callable - Funcion que calcula la metrica
    n_bootstrap : int - Numero de muestras bootstrap
    confidence_level : float - Nivel de confianza

    Returns:
    --------
    tuple: (lower_bound, upper_bound, point_estimate)
    """
    returns = returns_series.dropna().values
    if len(returns) < 2:
        return (np.nan, np.nan, np.nan)

    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_metrics.append(metric_func(pd.Series(sample)))

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
    point_estimate = metric_func(returns_series)

    return (lower, upper, point_estimate)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("EJEMPLO DE USO - BACKTESTING UTILITIES")
    print("-" * 60)

    # Crear datos de ejemplo
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='B')

    # Simular precios
    returns = np.random.randn(n) * 0.01 + 0.0003  # Slight positive bias
    prices = 100 * (1 + pd.Series(returns)).cumprod()

    # Crear senales
    signals = (np.random.rand(n) < 0.05).astype(int)

    df_example = pd.DataFrame({
        'date': dates,
        'price': prices.values,
        'signal': signals
    })
    df_example.set_index('date', inplace=True)

    print(f"Datos: {len(df_example)} observaciones")
    print(f"Senales: {df_example['signal'].sum()}")

    # Event Study
    print("\n1. EVENT STUDY")
    event_df = create_event_window(df_example['signal'], df_example['price'],
                                   lookback=5, lookforward=10)
    print(f"Eventos encontrados: {len(event_df)}")

    event_stats = event_study_statistics(event_df)
    print("\nEstadisticas por periodo:")
    print(event_stats[['period', 'mean', 'win_rate', 't_stat', 'p_value']].head(10))

    # Trading Metrics
    print("\n2. TRADING METRICS")
    # Simular retornos de estrategia
    strategy_returns = pd.Series(np.random.randn(100) * 0.02 + 0.001)

    metrics = calculate_all_metrics(strategy_returns)
    print_metrics_report(metrics)

    # Statistical Test
    print("\n3. SIGNIFICANCE TEST")
    returns_signal = pd.Series(np.random.randn(50) * 0.02 + 0.003)
    returns_no_signal = pd.Series(np.random.randn(200) * 0.02 + 0.0001)

    test_results = test_signal_significance(returns_signal, returns_no_signal)
    print(f"T-test p-value: {test_results['t_pvalue']:.4f}")
    print(f"Significant at 5%: {test_results['significant_ttest']}")

    print("\n" + "=" * 60)
    print("[OK] BACKTESTING UTILITIES COMPLETADO")
    print("=" * 60)
