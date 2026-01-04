# -*- coding: utf-8 -*-
"""
================================================================================
COMPARACION UNIFICADA - TODOS LOS MODELOS CON ESTRATEGIA SIGMOID 3X
================================================================================
Compara todos los modelos usando la estrategia sigmoid con leverage 3x
para simular un ETF apalancado tipo UPRO.

Estrategia:
- Posiciones de 0 a 3 (0 = cash, 3 = 3x leverage)
- Sigmoid con escala 500 para decisiones agresivas
- Comparacion con Buy & Hold y con Buy & Hold apalancado

================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Time Series Models
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
from prophet import Prophet
from arch import arch_model

# Darts Deep Learning Models
try:
    from darts import TimeSeries
    from darts.models import DLinearModel, NBEATSModel, NHiTSModel, TFTModel, TCNModel
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    print("[!] Darts no disponible")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# CONFIGURACION
# =============================================================================
CONFIG = {
    'leverage_max': 3.0,        # Maximo leverage (3x como UPRO)
    'sigmoid_scale': 500,       # Escala del sigmoid (agresivo)
    'test_size': 0.20,
    'random_state': 42,
}

print("=" * 80)
print("COMPARACION UNIFICADA - ESTRATEGIA SIGMOID 3X (UPRO)")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print(f"Max Leverage: {CONFIG['leverage_max']}x")
print(f"Sigmoid Scale: {CONFIG['sigmoid_scale']}")

# =============================================================================
# FUNCIONES DE ESTRATEGIA
# =============================================================================

def sigmoid(x, scale=500):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-scale * x))


def prediction_to_position(predictions, max_leverage=3.0, scale=500):
    """
    Convierte predicciones a posiciones usando sigmoid.

    Args:
        predictions: Array de predicciones del modelo
        max_leverage: Leverage maximo (ej: 3 para 3x)
        scale: Escala del sigmoid (mayor = mas agresivo)

    Returns:
        positions: Array de posiciones [0, max_leverage]
    """
    # Sigmoid mapea a [0, 1], multiplicamos por max_leverage para [0, max_leverage]
    positions = max_leverage * sigmoid(predictions, scale)
    positions = np.clip(positions, 0, max_leverage)
    # Redondear a 1 decimal
    positions = np.round(positions * 10) / 10
    return positions


def calculate_strategy_returns(positions, market_returns, risk_free_daily=0.0):
    """
    Calcula retornos de la estrategia apalancada.

    Formula: r_strategy = rf * (1 - position) + position * r_market

    Si position = 0: Solo ganas risk-free (cash)
    Si position = 1: Retorno del mercado
    Si position = 3: 3x el retorno del mercado (menos 2x risk-free por el borrowing)
    """
    # Asegurar misma longitud
    min_len = min(len(positions), len(market_returns))
    positions = positions[:min_len]
    market_returns = market_returns[:min_len]

    # Retorno de la estrategia
    # Cuando position > 1, estas apalancado: ganas position * market pero pagas (position-1) * rf
    strategy_returns = risk_free_daily * (1 - positions) + positions * market_returns

    return strategy_returns


def calculate_metrics(strategy_returns, market_returns, positions, risk_free_annual=0.02):
    """Calcula todas las metricas de performance."""

    # Ajustar longitudes
    min_len = min(len(strategy_returns), len(market_returns))
    strategy_returns = strategy_returns[:min_len]
    market_returns = market_returns[:min_len]
    positions = positions[:min_len]

    # Retornos totales
    strategy_total = np.prod(1 + strategy_returns) - 1
    market_total = np.prod(1 + market_returns) - 1

    # Retorno apalancado del mercado (benchmark apalancado)
    leveraged_market_returns = CONFIG['leverage_max'] * market_returns
    leveraged_market_total = np.prod(1 + leveraged_market_returns) - 1

    # Excess returns
    excess_vs_market = strategy_total - market_total
    excess_vs_leveraged = strategy_total - leveraged_market_total

    # Sharpe Ratio
    rf_daily = risk_free_annual / 252
    excess_daily = strategy_returns - rf_daily
    sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252) if np.std(excess_daily) > 0 else 0

    market_excess = market_returns - rf_daily
    market_sharpe = np.mean(market_excess) / np.std(market_excess) * np.sqrt(252) if np.std(market_excess) > 0 else 0

    # Max Drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Directional Accuracy
    correct_direction = np.sum((strategy_returns > 0) == (market_returns > 0))
    dir_accuracy = correct_direction / len(strategy_returns)

    # Posicion promedio
    mean_position = np.mean(positions)

    # Win rate
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)

    return {
        'strategy_return': strategy_total,
        'market_return': market_total,
        'leveraged_market_return': leveraged_market_total,
        'excess_vs_market': excess_vs_market,
        'excess_vs_leveraged': excess_vs_leveraged,
        'sharpe': sharpe,
        'market_sharpe': market_sharpe,
        'max_drawdown': max_dd,
        'dir_accuracy': dir_accuracy,
        'mean_position': mean_position,
        'win_rate': win_rate,
    }


# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n" + "=" * 80)
print("CARGANDO DATOS")
print("=" * 80)

df = pd.read_csv(os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv"))
df['date'] = pd.to_datetime(df['date'])

target_col = 'market_forward_excess_returns'
exclude_cols = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'SPY_CLOSE', 'SPY_OPEN',
                'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Split temporal
train_size = int(len(df) * (1 - CONFIG['test_size']))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"  Total: {len(df):,} filas")
print(f"  Train: {len(train_df):,} ({100*(1-CONFIG['test_size']):.0f}%)")
print(f"  Test: {len(test_df):,} ({100*CONFIG['test_size']:.0f}%)")
print(f"  Test period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
print(f"  Features: {len(feature_cols)}")

# Preparar datos
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Retornos del mercado
test_prices = test_df['SPY_CLOSE'].values
market_returns = np.diff(test_prices) / test_prices[:-1]

# Risk free rate
rf_daily = 0.02 / 252  # Asumimos 2% anual

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_processed = scaler.transform(imputer.transform(X_test))

# =============================================================================
# ENTRENAR MODELOS ML
# =============================================================================
print("\n" + "=" * 80)
print("ENTRENANDO MODELOS ML")
print("=" * 80)

ml_models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=CONFIG['random_state'], n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.01,
        random_state=CONFIG['random_state']
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.01,
        random_state=CONFIG['random_state'], verbosity=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100, num_leaves=31, learning_rate=0.01,
        random_state=CONFIG['random_state'], verbose=-1
    ),
    'CatBoost': cb.CatBoostRegressor(
        iterations=100, depth=6, learning_rate=0.01,
        random_state=CONFIG['random_state'], verbose=0
    ),
}

all_results = []

for name, model in ml_models.items():
    print(f"\n  {name}...")
    start = time.time()

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    # Convertir predicciones a posiciones
    positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

    # Calcular retornos de estrategia
    # Alinear: prediccion[t] decide posicion para el dia t+1
    strat_returns = calculate_strategy_returns(positions[:-1], market_returns, rf_daily)

    # Metricas
    metrics = calculate_metrics(strat_returns, market_returns, positions[:-1])
    metrics['model'] = name
    metrics['category'] = 'ML'
    metrics['time'] = time.time() - start

    all_results.append(metrics)

    print(f"    Strategy Return: {metrics['strategy_return']:.2%}")
    print(f"    Market Return:   {metrics['market_return']:.2%}")
    print(f"    Excess Return:   {metrics['excess_vs_market']:.2%}")
    print(f"    Mean Position:   {metrics['mean_position']:.2f}x")
    print(f"    Sharpe:          {metrics['sharpe']:.3f}")

# =============================================================================
# MODELOS STATSFORECAST
# =============================================================================
print("\n" + "=" * 80)
print("MODELOS STATSFORECAST")
print("=" * 80)

# Preparar datos para StatsForecast
sf_df = pd.DataFrame({
    'unique_id': 'SPY',
    'ds': df['date'],
    'y': df[target_col]
}).dropna()

sf_train = sf_df.iloc[:train_size]
sf_test = sf_df.iloc[train_size:]

sf_models = [
    AutoARIMA(season_length=5),
    AutoETS(season_length=5),
    AutoTheta(season_length=5),
    SeasonalNaive(season_length=5)
]

sf = StatsForecast(models=sf_models, freq='B', n_jobs=1)

print("  Entrenando StatsForecast...")
start = time.time()
sf.fit(sf_train)
predictions = sf.predict(h=len(sf_test))
sf_time = time.time() - start

for model_name in ['AutoARIMA', 'AutoETS', 'AutoTheta', 'SeasonalNaive']:
    if model_name in predictions.columns:
        y_pred = predictions[model_name].values

        # Posiciones
        positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

        # Alinear longitudes
        min_len = min(len(positions), len(market_returns))
        positions = positions[:min_len]

        # Retornos
        strat_returns = calculate_strategy_returns(positions[:-1], market_returns[:min_len-1], rf_daily)

        # Metricas
        metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:-1])
        metrics['model'] = f'SF_{model_name}'
        metrics['category'] = 'StatsForecast'
        metrics['time'] = sf_time / 4

        all_results.append(metrics)

        print(f"\n  {model_name}:")
        print(f"    Strategy Return: {metrics['strategy_return']:.2%}")
        print(f"    Excess Return:   {metrics['excess_vs_market']:.2%}")

# =============================================================================
# PROPHET
# =============================================================================
print("\n" + "=" * 80)
print("PROPHET")
print("=" * 80)

# Prophet basico
print("\n  Prophet (basico)...")
start = time.time()

prophet_train = pd.DataFrame({
    'ds': train_df['date'],
    'y': train_df[target_col]
}).dropna()

prophet_test = pd.DataFrame({
    'ds': test_df['date'],
    'y': test_df[target_col]
}).dropna()

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(prophet_train)
forecast = model.predict(prophet_test[['ds']])

y_pred = forecast['yhat'].values
positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

min_len = min(len(positions), len(market_returns))
strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
metrics['model'] = 'Prophet'
metrics['category'] = 'Prophet'
metrics['time'] = time.time() - start

all_results.append(metrics)

print(f"    Strategy Return: {metrics['strategy_return']:.2%}")
print(f"    Excess Return:   {metrics['excess_vs_market']:.2%}")
print(f"    Sharpe:          {metrics['sharpe']:.3f}")

# =============================================================================
# GARCH
# =============================================================================
print("\n" + "=" * 80)
print("GARCH")
print("=" * 80)

print("\n  GARCH(1,1)...")
start = time.time()

try:
    returns_scaled = df[target_col].dropna() * 100
    train_returns = returns_scaled.iloc[:train_size]

    garch = arch_model(train_returns, vol='Garch', p=1, q=1, mean='AR', lags=1)
    result = garch.fit(disp='off')

    # Forecast
    forecasts = result.forecast(horizon=len(test_df))
    y_pred = forecasts.mean.values[-len(test_df):].flatten() / 100

    positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

    min_len = min(len(positions), len(market_returns))
    strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

    metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
    metrics['model'] = 'GARCH'
    metrics['category'] = 'GARCH'
    metrics['time'] = time.time() - start

    all_results.append(metrics)

    print(f"    Strategy Return: {metrics['strategy_return']:.2%}")
    print(f"    Excess Return:   {metrics['excess_vs_market']:.2%}")

except Exception as e:
    print(f"    [ERROR] {e}")

# =============================================================================
# DARTS DEEP LEARNING MODELS
# =============================================================================
if DARTS_AVAILABLE:
    print("\n" + "=" * 80)
    print("DARTS DEEP LEARNING MODELS")
    print("=" * 80)

    # Preparar datos para Darts
    darts_df = df[['date', target_col]].copy()
    darts_df = darts_df.dropna()
    darts_df = darts_df.set_index('date')
    darts_df = darts_df.asfreq('B')  # Business day frequency
    darts_df = darts_df.fillna(method='ffill')

    series = TimeSeries.from_dataframe(
        darts_df,
        value_cols=target_col,
        fill_missing_dates=True,
        freq='B'
    )

    train_series = series[:train_size]
    test_series = series[train_size:]
    n_test = len(test_series)

    darts_models = {
        'DLinear': DLinearModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=50,
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}
        ),
        'N-BEATS': NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=50,
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}
        ),
        'N-HiTS': NHiTSModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=50,
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}
        ),
        'TCN': TCNModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=50,
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}
        ),
        'TFT': TFTModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=50,
            random_state=42,
            add_relative_index=True,
            pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}
        ),
    }

    for model_name, model in darts_models.items():
        print(f"\n  {model_name}...")
        start = time.time()

        try:
            # Entrenar
            model.fit(train_series)

            # Predecir
            predictions = model.predict(n=n_test)
            y_pred = predictions.values().flatten()

            # Verificar NaN
            if np.isnan(y_pred).any():
                nan_count = np.isnan(y_pred).sum()
                print(f"    [!] {nan_count} NaN en predicciones, reemplazando con 0")
                y_pred = np.nan_to_num(y_pred, nan=0.0)

            positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

            # Alinear con market_returns
            min_len = min(len(positions), len(market_returns))
            strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

            metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
            metrics['model'] = model_name
            metrics['category'] = 'Darts'
            metrics['time'] = time.time() - start

            all_results.append(metrics)

            print(f"    Strategy Return: {metrics['strategy_return']:.2%}")
            print(f"    Excess Return:   {metrics['excess_vs_market']:.2%}")
            print(f"    Sharpe:          {metrics['sharpe']:.3f}")
            print(f"    Time:            {metrics['time']:.1f}s")

        except Exception as e:
            print(f"    [ERROR] {model_name}: {e}")
else:
    print("\n[!] Darts no disponible - saltando modelos deep learning")

# =============================================================================
# BENCHMARK
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARKS")
print("=" * 80)

# Buy & Hold (1x)
market_total = np.prod(1 + market_returns) - 1
market_sharpe = np.mean(market_returns - rf_daily) / np.std(market_returns) * np.sqrt(252)

# Buy & Hold apalancado (3x)
leveraged_returns = CONFIG['leverage_max'] * market_returns
leveraged_total = np.prod(1 + leveraged_returns) - 1
leveraged_sharpe = np.mean(leveraged_returns - rf_daily) / np.std(leveraged_returns) * np.sqrt(252)

# Max Drawdown del mercado apalancado
cumulative = np.cumprod(1 + leveraged_returns)
running_max = np.maximum.accumulate(cumulative)
drawdowns = (cumulative - running_max) / running_max
leveraged_max_dd = np.min(drawdowns)

print(f"\n  Buy & Hold (1x):")
print(f"    Return: {market_total:.2%}")
print(f"    Sharpe: {market_sharpe:.3f}")

print(f"\n  Buy & Hold ({CONFIG['leverage_max']:.0f}x):")
print(f"    Return: {leveraged_total:.2%}")
print(f"    Sharpe: {leveraged_sharpe:.3f}")
print(f"    Max DD: {leveraged_max_dd:.2%}")

# =============================================================================
# RESULTADOS FINALES
# =============================================================================
print("\n" + "=" * 80)
print("RESULTADOS FINALES - RANKING POR EXCESS RETURN")
print("=" * 80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('excess_vs_market', ascending=False)

print(f"\n  {'#':<3} {'Modelo':<20} {'Strategy':>12} {'Market':>10} {'Excess':>12} {'Sharpe':>8} {'Pos':>6}")
print("-" * 85)

for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"  {i:<3} {row['model']:<20} {row['strategy_return']:>11.2%} "
          f"{row['market_return']:>9.2%} {row['excess_vs_market']:>11.2%} "
          f"{row['sharpe']:>8.3f} {row['mean_position']:>5.1f}x")

print("-" * 85)
print(f"  {'':3} {'Buy & Hold (1x)':<20} {market_total:>11.2%} {market_total:>9.2%} {'0.00%':>12} {market_sharpe:>8.3f} {'1.0x':>6}")
print(f"  {'':3} {'Buy & Hold (3x)':<20} {leveraged_total:>11.2%} {market_total:>9.2%} {leveraged_total - market_total:>11.2%} {leveraged_sharpe:>8.3f} {'3.0x':>6}")

# Mejor modelo
best = results_df.iloc[0]
print(f"\n  MEJOR MODELO: {best['model']}")
print(f"    Strategy Return: {best['strategy_return']:.2%}")
print(f"    vs Market (1x):  {best['excess_vs_market']:.2%}")
print(f"    vs Market (3x):  {best['excess_vs_leveraged']:.2%}")
print(f"    Sharpe Ratio:    {best['sharpe']:.3f}")
print(f"    Max Drawdown:    {best['max_drawdown']:.2%}")

# Guardar resultados
output_file = os.path.join(RESULTS_DIR, "unified_strategy_3x_comparison.csv")
results_df.to_csv(output_file, index=False)
print(f"\n  Resultados guardados: {output_file}")

# JSON summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'test_period': {
        'start': str(test_df['date'].min().date()),
        'end': str(test_df['date'].max().date()),
        'days': len(test_df)
    },
    'benchmarks': {
        'market_1x': {'return': float(market_total), 'sharpe': float(market_sharpe)},
        'market_3x': {'return': float(leveraged_total), 'sharpe': float(leveraged_sharpe), 'max_dd': float(leveraged_max_dd)}
    },
    'best_model': {
        'name': best['model'],
        'strategy_return': float(best['strategy_return']),
        'excess_vs_market': float(best['excess_vs_market']),
        'sharpe': float(best['sharpe'])
    }
}

summary_file = os.path.join(RESULTS_DIR, "unified_strategy_3x_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("[OK] COMPARACION COMPLETADA")
print("=" * 80)
