# -*- coding: utf-8 -*-
"""
================================================================================
TRADING RULES DSL - LENGUAJE PARA DEFINIR ESTRATEGIAS
================================================================================

Domain Specific Language (DSL) para definir reglas de trading de forma simple
y legible, inspirado en HSBC-ML RuleFinder.

SINTAXIS:
- "rsi14 < 30"           -> RSI menor que 30
- "close_pct > 0.60"     -> Precio en percentil 60+
- "upx == 5"             -> Exactamente 5 dias UP
- "downx >= 7"           -> Al menos 7 dias DOWN
- "macd_hist > 0"        -> MACD histogram positivo

OPERADORES SOPORTADOS:
- ==, !=, <, <=, >, >=
- AND (combinacion de reglas)
- OR (alternativa de reglas)

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

print("=" * 80)
print("TRADING RULES DSL - DEFINICION DE ESTRATEGIAS")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


# =============================================================================
# PARSER DE CONDICIONES
# =============================================================================

def extract_operator(condition_str):
    """
    Extrae el operador de una condicion.

    Inspirado en HSBC-ML operador() function.

    Parameters:
    -----------
    condition_str : str - Condicion como string (ej: "rsi14 < 30")

    Returns:
    --------
    str: Operador encontrado
    """
    operators = ['==', '!=', '<=', '>=', '<', '>']

    for op in operators:
        if op in condition_str:
            return op

    return None


def parse_condition(condition_str):
    """
    Parsea una condicion en sus componentes.

    Inspirado en HSBC-ML lee_cond() function.

    Parameters:
    -----------
    condition_str : str - Condicion como string

    Returns:
    --------
    tuple: (column_name, operator, value)
    """
    condition_str = condition_str.strip()
    operator = extract_operator(condition_str)

    if operator is None:
        raise ValueError(f"No se encontro operador valido en: {condition_str}")

    parts = condition_str.split(operator)
    if len(parts) != 2:
        raise ValueError(f"Condicion mal formada: {condition_str}")

    column = parts[0].strip()
    value_str = parts[1].strip()

    # Intentar convertir valor a numero
    try:
        value = float(value_str)
    except ValueError:
        value = value_str  # Mantener como string si no es numero

    return column, operator, value


# =============================================================================
# EVALUADOR DE CONDICIONES
# =============================================================================

def evaluate_condition(df, column, operator, value):
    """
    Evalua una condicion sobre un DataFrame.

    Inspirado en HSBC-ML fil() function.

    Parameters:
    -----------
    df : pd.DataFrame - DataFrame con datos
    column : str - Nombre de columna
    operator : str - Operador de comparacion
    value : float/str - Valor a comparar

    Returns:
    --------
    pd.Series: Serie booleana con resultado
    """
    if column not in df.columns:
        raise KeyError(f"Columna '{column}' no existe en el DataFrame")

    col_data = df[column]

    if operator == '==':
        return col_data == value
    elif operator == '!=':
        return col_data != value
    elif operator == '<':
        return col_data < value
    elif operator == '<=':
        return col_data <= value
    elif operator == '>':
        return col_data > value
    elif operator == '>=':
        return col_data >= value
    else:
        raise ValueError(f"Operador no soportado: {operator}")


def apply_single_rule(df, rule_str):
    """
    Aplica una regla simple al DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame - DataFrame con datos
    rule_str : str - Regla como string

    Returns:
    --------
    pd.Series: Serie booleana (True donde se cumple la regla)
    """
    column, operator, value = parse_condition(rule_str)
    return evaluate_condition(df, column, operator, value)


# =============================================================================
# COMBINADOR DE REGLAS
# =============================================================================

def combine_rules_and(df, rules):
    """
    Combina multiples reglas con AND (todas deben cumplirse).

    Parameters:
    -----------
    df : pd.DataFrame - DataFrame con datos
    rules : list of str - Lista de reglas

    Returns:
    --------
    pd.Series: Serie booleana
    """
    if not rules:
        return pd.Series(True, index=df.index)

    result = apply_single_rule(df, rules[0])

    for rule in rules[1:]:
        result = result & apply_single_rule(df, rule)

    return result


def combine_rules_or(df, rules):
    """
    Combina multiples reglas con OR (al menos una debe cumplirse).

    Parameters:
    -----------
    df : pd.DataFrame - DataFrame con datos
    rules : list of str - Lista de reglas

    Returns:
    --------
    pd.Series: Serie booleana
    """
    if not rules:
        return pd.Series(False, index=df.index)

    result = apply_single_rule(df, rules[0])

    for rule in rules[1:]:
        result = result | apply_single_rule(df, rule)

    return result


# =============================================================================
# ESTRATEGIA CLASS
# =============================================================================

class TradingStrategy:
    """
    Clase para definir y aplicar estrategias de trading.

    Example:
    --------
    strategy = TradingStrategy("RSI Oversold Strategy")
    strategy.add_entry_rule("SPY_RSI_14 < 30")
    strategy.add_entry_rule("D_DOWN_STREAK >= 3")
    strategy.add_exit_rule("SPY_RSI_14 > 70")

    signals = strategy.generate_signals(df)
    """

    def __init__(self, name="Strategy"):
        """
        Inicializa una estrategia.

        Parameters:
        -----------
        name : str - Nombre de la estrategia
        """
        self.name = name
        self.entry_rules = []
        self.exit_rules = []
        self.entry_logic = 'AND'  # 'AND' o 'OR'
        self.exit_logic = 'AND'

    def add_entry_rule(self, rule):
        """Agrega una regla de entrada."""
        self.entry_rules.append(rule)
        return self

    def add_exit_rule(self, rule):
        """Agrega una regla de salida."""
        self.exit_rules.append(rule)
        return self

    def set_entry_logic(self, logic):
        """Configura logica de entrada ('AND' o 'OR')."""
        self.entry_logic = logic.upper()
        return self

    def set_exit_logic(self, logic):
        """Configura logica de salida ('AND' o 'OR')."""
        self.exit_logic = logic.upper()
        return self

    def generate_entry_signals(self, df):
        """
        Genera senales de entrada.

        Returns:
        --------
        pd.Series: 1 donde hay senal de entrada, 0 en otro caso
        """
        if not self.entry_rules:
            return pd.Series(0, index=df.index)

        if self.entry_logic == 'AND':
            signals = combine_rules_and(df, self.entry_rules)
        else:
            signals = combine_rules_or(df, self.entry_rules)

        return signals.astype(int)

    def generate_exit_signals(self, df):
        """
        Genera senales de salida.

        Returns:
        --------
        pd.Series: 1 donde hay senal de salida, 0 en otro caso
        """
        if not self.exit_rules:
            return pd.Series(0, index=df.index)

        if self.exit_logic == 'AND':
            signals = combine_rules_and(df, self.exit_rules)
        else:
            signals = combine_rules_or(df, self.exit_rules)

        return signals.astype(int)

    def generate_signals(self, df):
        """
        Genera todas las senales.

        Returns:
        --------
        pd.DataFrame: DataFrame con senales de entrada y salida
        """
        signals = pd.DataFrame(index=df.index)
        signals['entry'] = self.generate_entry_signals(df)
        signals['exit'] = self.generate_exit_signals(df)

        # Senal neta: 1 = long, -1 = exit, 0 = hold
        signals['signal'] = signals['entry'] - signals['exit']

        return signals

    def describe(self):
        """Imprime descripcion de la estrategia."""
        print(f"\n{'='*60}")
        print(f"ESTRATEGIA: {self.name}")
        print(f"{'='*60}")

        print(f"\nReglas de ENTRADA ({self.entry_logic}):")
        for i, rule in enumerate(self.entry_rules, 1):
            print(f"  {i}. {rule}")

        if self.exit_rules:
            print(f"\nReglas de SALIDA ({self.exit_logic}):")
            for i, rule in enumerate(self.exit_rules, 1):
                print(f"  {i}. {rule}")

        print()


# =============================================================================
# ESTRATEGIAS PREDEFINIDAS (Inspired by Elder & HSBC-ML)
# =============================================================================

def create_rsi_oversold_strategy():
    """Estrategia RSI Oversold clasica."""
    strategy = TradingStrategy("RSI Oversold")
    strategy.add_entry_rule("SPY_RSI_14 < 30")
    strategy.add_exit_rule("SPY_RSI_14 > 70")
    return strategy


def create_momentum_streak_strategy():
    """Estrategia basada en rachas de momentum."""
    strategy = TradingStrategy("Momentum Streak")
    strategy.add_entry_rule("D_DOWN_STREAK >= 5")
    strategy.add_entry_rule("SPY_RSI_14 < 40")
    strategy.add_exit_rule("D_UP_STREAK >= 3")
    return strategy


def create_triple_screen_strategy():
    """Estrategia Triple Screen de Elder."""
    strategy = TradingStrategy("Elder Triple Screen")
    strategy.add_entry_rule("W_TREND == 1")           # Tendencia semanal alcista
    strategy.add_entry_rule("D_FORCE_OVERSOLD == 1")  # Force Index negativo (pullback)
    strategy.add_exit_rule("W_TREND == -1")
    return strategy


def create_mean_reversion_strategy():
    """Estrategia de mean reversion basada en percentiles."""
    strategy = TradingStrategy("Mean Reversion Percentile")
    strategy.add_entry_rule("SPY_CLOSE_pctrank_252 < 0.10")  # Percentil bajo
    strategy.add_exit_rule("SPY_CLOSE_pctrank_252 > 0.50")   # Recuperacion a media
    return strategy


def create_breakout_strategy():
    """Estrategia de breakout."""
    strategy = TradingStrategy("Breakout")
    strategy.add_entry_rule("SPY_CLOSE_pctrank_252 > 0.90")  # Nuevo maximo
    strategy.add_entry_rule("D_UP_STREAK >= 3")               # Confirmacion
    strategy.add_exit_rule("D_DOWN_STREAK >= 2")
    return strategy


# =============================================================================
# ESTRATEGIA BUILDER (Fluent Interface)
# =============================================================================

class StrategyBuilder:
    """
    Builder para crear estrategias con sintaxis fluida.

    Example:
    --------
    strategy = (StrategyBuilder("My Strategy")
                .entry_when("rsi14 < 30")
                .and_entry("volume_pct > 0.50")
                .exit_when("rsi14 > 70")
                .build())
    """

    def __init__(self, name="Strategy"):
        self.strategy = TradingStrategy(name)

    def entry_when(self, rule):
        """Primera regla de entrada."""
        self.strategy.add_entry_rule(rule)
        return self

    def and_entry(self, rule):
        """Regla adicional de entrada (AND)."""
        self.strategy.add_entry_rule(rule)
        return self

    def or_entry(self, rule):
        """Regla alternativa de entrada (OR)."""
        self.strategy.set_entry_logic('OR')
        self.strategy.add_entry_rule(rule)
        return self

    def exit_when(self, rule):
        """Primera regla de salida."""
        self.strategy.add_exit_rule(rule)
        return self

    def and_exit(self, rule):
        """Regla adicional de salida (AND)."""
        self.strategy.add_exit_rule(rule)
        return self

    def or_exit(self, rule):
        """Regla alternativa de salida (OR)."""
        self.strategy.set_exit_logic('OR')
        self.strategy.add_exit_rule(rule)
        return self

    def build(self):
        """Retorna la estrategia construida."""
        return self.strategy


# =============================================================================
# STRATEGY TESTER
# =============================================================================

def backtest_strategy(df, strategy, price_col='SPY_CLOSE', return_col=None):
    """
    Backtesting simple de una estrategia.

    Parameters:
    -----------
    df : pd.DataFrame - DataFrame con datos
    strategy : TradingStrategy - Estrategia a testear
    price_col : str - Columna de precios
    return_col : str - Columna de retornos (o None para calcular)

    Returns:
    --------
    dict: Resultados del backtest
    """
    signals = strategy.generate_signals(df)

    # Calcular retornos si no se proporcionan
    if return_col and return_col in df.columns:
        returns = df[return_col]
    else:
        returns = df[price_col].pct_change()

    # Retornos de la estrategia (simplificado)
    # Solo contamos retornos del dia siguiente a la senal
    strategy_returns = returns.shift(-1) * signals['entry']
    strategy_returns = strategy_returns.dropna()

    # Estadisticas
    entry_signals = signals['entry'].sum()

    if entry_signals == 0:
        return {
            'strategy_name': strategy.name,
            'n_signals': 0,
            'message': 'No se generaron senales'
        }

    wins = (strategy_returns > 0).sum()
    losses = (strategy_returns < 0).sum()

    return {
        'strategy_name': strategy.name,
        'n_signals': entry_signals,
        'n_wins': wins,
        'n_losses': losses,
        'win_rate': wins / entry_signals if entry_signals > 0 else 0,
        'total_return': (1 + strategy_returns).prod() - 1,
        'avg_return': strategy_returns.mean(),
        'std_return': strategy_returns.std(),
        'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
        'max_return': strategy_returns.max(),
        'min_return': strategy_returns.min()
    }


def print_backtest_results(results):
    """Imprime resultados del backtest."""
    print(f"\n{'='*60}")
    print(f"BACKTEST: {results['strategy_name']}")
    print(f"{'='*60}")

    if results.get('message'):
        print(f"  {results['message']}")
        return

    print(f"\n  Senales generadas: {results['n_signals']}")
    print(f"  Operaciones ganadoras: {results['n_wins']}")
    print(f"  Operaciones perdedoras: {results['n_losses']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"\n  Retorno total: {results['total_return']:.2%}")
    print(f"  Retorno promedio: {results['avg_return']:.4%}")
    print(f"  Volatilidad: {results['std_return']:.4%}")
    print(f"  Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"\n  Mejor operacion: {results['max_return']:.4%}")
    print(f"  Peor operacion: {results['min_return']:.4%}")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("EJEMPLO DE USO - TRADING RULES DSL")
    print("-" * 60)

    # Crear datos de ejemplo
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='B')

    # Simular datos
    close = 100 * (1 + np.random.randn(n).cumsum() * 0.01)

    df_example = pd.DataFrame({
        'date': dates,
        'SPY_CLOSE': close,
        'SPY_RSI_14': np.random.uniform(20, 80, n),
        'D_DOWN_STREAK': np.random.randint(0, 8, n),
        'D_UP_STREAK': np.random.randint(0, 8, n),
        'W_TREND': np.random.choice([-1, 0, 1], n),
        'D_FORCE_OVERSOLD': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    df_example.set_index('date', inplace=True)

    # Ejemplo 1: Usar DSL directamente
    print("\n1. REGLA SIMPLE:")
    rule = "SPY_RSI_14 < 30"
    column, op, value = parse_condition(rule)
    print(f"  Regla: {rule}")
    print(f"  Parseado: columna='{column}', operador='{op}', valor={value}")

    # Ejemplo 2: Estrategia predefinida
    print("\n2. ESTRATEGIA RSI OVERSOLD:")
    strategy = create_rsi_oversold_strategy()
    strategy.describe()

    signals = strategy.generate_signals(df_example)
    print(f"  Senales de entrada: {signals['entry'].sum()}")
    print(f"  Senales de salida: {signals['exit'].sum()}")

    # Ejemplo 3: Strategy Builder
    print("\n3. STRATEGY BUILDER:")
    custom_strategy = (StrategyBuilder("Custom RSI + Streak")
                       .entry_when("SPY_RSI_14 < 35")
                       .and_entry("D_DOWN_STREAK >= 3")
                       .exit_when("SPY_RSI_14 > 65")
                       .build())
    custom_strategy.describe()

    # Ejemplo 4: Backtest
    print("\n4. BACKTEST:")
    results = backtest_strategy(df_example, custom_strategy)
    print_backtest_results(results)

    print("\n" + "=" * 60)
    print("[OK] TRADING RULES DSL COMPLETADO")
    print("=" * 60)
