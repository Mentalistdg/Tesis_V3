# -*- coding: utf-8 -*-
"""
================================================================================
CALCULO CORRECTO DE RETORNOS - COMPARACION CON MERCADO
================================================================================
Recalcula los retornos de cada modelo comparando con Buy & Hold del mercado.

Estrategias evaluadas:
1. Buy & Hold (Benchmark)
2. Long/Cash basado en signo de prediccion
3. Long/Short basado en signo de prediccion
4. Sigmoid-weighted positions

================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

print("=" * 80)
print("CALCULO CORRECTO DE RETORNOS - COMPARACION CON MERCADO")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\nCargando datos...")

df = pd.read_csv(os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv"))
df['date'] = pd.to_datetime(df['date'])

target_col = 'market_forward_excess_returns'
exclude_cols = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'SPY_CLOSE', 'SPY_OPEN',
                'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Split temporal 80/20
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"  Test period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Preparar datos
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Precios para calcular retornos
test_prices = test_df['SPY_CLOSE'].values
test_dates = test_df['date'].values

# Retornos reales del mercado (diarios)
market_returns = np.diff(test_prices) / test_prices[:-1]

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# =============================================================================
# FUNCIONES DE ESTRATEGIA
# =============================================================================

def sigmoid(x):
    """Sigmoid function para convertir predicciones a probabilidades."""
    return 1 / (1 + np.exp(-x))


def calculate_strategy_returns(y_pred, market_returns, strategy='long_cash'):
    """
    Calcula retornos de diferentes estrategias.

    Strategies:
    - 'long_cash': Long si pred > 0, else cash (0)
    - 'long_short': Long si pred > 0, else short (-1)
    - 'sigmoid': Posicion proporcional a sigmoid(pred)
    - 'sigmoid_scaled': Sigmoid escalado por volatilidad de predicciones
    """
    # Alinear: prediccion[t] -> decision para retorno[t] (que es de t a t+1)
    # Necesitamos pred[:-1] para decidir sobre returns[:]
    pred = y_pred[:-1]  # Predicciones (una menos porque returns tiene una menos)

    if len(pred) != len(market_returns):
        min_len = min(len(pred), len(market_returns))
        pred = pred[:min_len]
        returns = market_returns[:min_len]
    else:
        returns = market_returns

    if strategy == 'long_cash':
        # Long si prediccion > 0, else cash
        positions = np.where(pred > 0, 1, 0)

    elif strategy == 'long_short':
        # Long si prediccion > 0, else short
        positions = np.where(pred > 0, 1, -1)

    elif strategy == 'sigmoid':
        # Posicion continua basada en sigmoid: 0 a 1
        positions = sigmoid(pred * 100)  # Escalar para que sigmoid no sea ~0.5 siempre

    elif strategy == 'sigmoid_scaled':
        # Sigmoid pero centrado en 0: -1 a 1
        positions = 2 * sigmoid(pred * 100) - 1

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Retornos de la estrategia
    strategy_returns = positions * returns

    return strategy_returns, positions


def calculate_metrics(strategy_returns, market_returns, positions):
    """Calcula metricas completas."""
    # Total returns
    strategy_total = np.prod(1 + strategy_returns) - 1
    market_total = np.prod(1 + market_returns[:len(strategy_returns)]) - 1
    excess_return = strategy_total - market_total

    # Annualized
    n_days = len(strategy_returns)
    years = n_days / 252
    strategy_annual = (1 + strategy_total) ** (1/years) - 1 if years > 0 else 0
    market_annual = (1 + market_total) ** (1/years) - 1 if years > 0 else 0

    # Sharpe ratio (annualized)
    strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
    market_sharpe = np.mean(market_returns[:len(strategy_returns)]) / np.std(market_returns[:len(strategy_returns)]) * np.sqrt(252) if np.std(market_returns) > 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Win rate
    wins = np.sum(strategy_returns > 0)
    losses = np.sum(strategy_returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Directional accuracy (cuando estamos en mercado)
    in_market = positions != 0
    if in_market.sum() > 0:
        correct_dir = np.sum((strategy_returns > 0) & in_market)
        dir_acc = correct_dir / in_market.sum()
    else:
        dir_acc = 0

    # Tiempo en mercado
    time_in_market = np.mean(np.abs(positions))

    return {
        'strategy_return': strategy_total,
        'market_return': market_total,
        'excess_return': excess_return,
        'strategy_annual': strategy_annual,
        'market_annual': market_annual,
        'strategy_sharpe': strategy_sharpe,
        'market_sharpe': market_sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'dir_accuracy': dir_acc,
        'time_in_market': time_in_market,
        'n_trades': int(np.sum(np.diff(positions) != 0))
    }


# =============================================================================
# ENTRENAR MODELOS Y CALCULAR RETORNOS
# =============================================================================
print("\n" + "=" * 80)
print("ENTRENANDO MODELOS")
print("=" * 80)

models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.01, random_state=42, verbose=-1),
    'CatBoost': cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.01, random_state=42, verbose=0)
}

strategies = ['long_cash', 'long_short', 'sigmoid']
all_results = []

for model_name, model in models.items():
    print(f"\n  {model_name}...")

    # Entrenar
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular metricas para cada estrategia
    for strategy in strategies:
        strat_returns, positions = calculate_strategy_returns(y_pred, market_returns, strategy)
        metrics = calculate_metrics(strat_returns, market_returns, positions)

        result = {
            'model': model_name,
            'strategy': strategy,
            **metrics
        }
        all_results.append(result)

    # Mostrar resultado principal (long_cash)
    main_result = [r for r in all_results if r['model'] == model_name and r['strategy'] == 'long_cash'][-1]
    print(f"    Strategy Return: {main_result['strategy_return']:.2%}")
    print(f"    Market Return:   {main_result['market_return']:.2%}")
    print(f"    Excess Return:   {main_result['excess_return']:.2%}")
    print(f"    Sharpe:          {main_result['strategy_sharpe']:.3f}")

# =============================================================================
# CALCULAR BENCHMARK (Buy & Hold)
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK: BUY & HOLD")
print("=" * 80)

market_total = np.prod(1 + market_returns) - 1
market_annual = (1 + market_total) ** (252/len(market_returns)) - 1
market_sharpe = np.mean(market_returns) / np.std(market_returns) * np.sqrt(252)

cumulative = np.cumprod(1 + market_returns)
running_max = np.maximum.accumulate(cumulative)
drawdowns = (cumulative - running_max) / running_max
market_max_dd = np.min(drawdowns)

print(f"  Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
print(f"  Total Return:     {market_total:.2%}")
print(f"  Annualized:       {market_annual:.2%}")
print(f"  Sharpe Ratio:     {market_sharpe:.3f}")
print(f"  Max Drawdown:     {market_max_dd:.2%}")

# =============================================================================
# RESULTADOS COMPARATIVOS
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACION: ESTRATEGIA LONG/CASH vs MERCADO")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Filtrar solo long_cash para comparacion principal
long_cash_results = results_df[results_df['strategy'] == 'long_cash'].copy()
long_cash_results = long_cash_results.sort_values('excess_return', ascending=False)

print(f"\n  {'Modelo':<20} {'Strategy':>12} {'Market':>12} {'Excess':>12} {'Sharpe':>10} {'MaxDD':>10}")
print("-" * 85)

for _, row in long_cash_results.iterrows():
    print(f"  {row['model']:<20} {row['strategy_return']:>11.2%} {row['market_return']:>11.2%} "
          f"{row['excess_return']:>11.2%} {row['strategy_sharpe']:>10.3f} {row['max_drawdown']:>10.2%}")

print("-" * 85)
print(f"  {'BUY & HOLD':<20} {market_total:>11.2%} {market_total:>11.2%} "
      f"{'0.00%':>12} {market_sharpe:>10.3f} {market_max_dd:>10.2%}")

# =============================================================================
# COMPARACION DE ESTRATEGIAS
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACION DE ESTRATEGIAS (Mejor modelo por estrategia)")
print("=" * 80)

for strategy in strategies:
    strat_results = results_df[results_df['strategy'] == strategy]
    best = strat_results.loc[strat_results['excess_return'].idxmax()]

    print(f"\n  {strategy.upper()}:")
    print(f"    Mejor modelo: {best['model']}")
    print(f"    Strategy Return: {best['strategy_return']:.2%}")
    print(f"    Excess Return: {best['excess_return']:.2%}")
    print(f"    Sharpe: {best['strategy_sharpe']:.3f}")

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("GUARDANDO RESULTADOS")
print("=" * 80)

# CSV completo
output_file = os.path.join(RESULTS_DIR, "model_returns_comparison.csv")
results_df.to_csv(output_file, index=False)
print(f"  CSV: {output_file}")

# Resumen JSON
summary = {
    'timestamp': datetime.now().isoformat(),
    'test_period': {
        'start': str(test_df['date'].min().date()),
        'end': str(test_df['date'].max().date()),
        'days': len(test_df)
    },
    'market_benchmark': {
        'total_return': float(market_total),
        'annualized_return': float(market_annual),
        'sharpe_ratio': float(market_sharpe),
        'max_drawdown': float(market_max_dd)
    },
    'best_model_by_strategy': {}
}

for strategy in strategies:
    strat_results = results_df[results_df['strategy'] == strategy]
    best = strat_results.loc[strat_results['excess_return'].idxmax()]
    summary['best_model_by_strategy'][strategy] = {
        'model': best['model'],
        'strategy_return': float(best['strategy_return']),
        'excess_return': float(best['excess_return']),
        'sharpe': float(best['strategy_sharpe'])
    }

summary_file = os.path.join(RESULTS_DIR, "returns_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  JSON: {summary_file}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

best_overall = long_cash_results.iloc[0]

print(f"\n  BENCHMARK (Buy & Hold SPY):")
print(f"    Return: {market_total:.2%}")
print(f"    Sharpe: {market_sharpe:.3f}")

print(f"\n  MEJOR MODELO ({best_overall['model']}):")
print(f"    Return: {best_overall['strategy_return']:.2%}")
print(f"    Excess vs Market: {best_overall['excess_return']:.2%}")
print(f"    Sharpe: {best_overall['strategy_sharpe']:.3f}")

if best_overall['excess_return'] > 0:
    print(f"\n  [OK] El modelo supera al mercado por {best_overall['excess_return']:.2%}")
else:
    print(f"\n  [INFO] El mercado supera al modelo por {-best_overall['excess_return']:.2%}")

print("\n" + "=" * 80)
print("[OK] ANALISIS COMPLETADO")
print("=" * 80)
