# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE DATOS PARA LA APP DE VISUALIZACION
================================================================================
Este script genera todos los datos pre-computados que necesita la app.
Lee del pipeline existente y genera archivos JSON/CSV para el backend.

IMPORTANTE: Este script NO modifica el pipeline existente.
            Solo lee resultados y genera datos derivados.

Datos generados:
- daily_data.json: Series diarias por modelo (predicciones, posiciones, returns)
- models_summary.json: Metricas resumen de todos los modelos
- trades_log.json: Log de operaciones por modelo
- regime_data.json: Datos de regimen de mercado
- market_data.json: Datos del mercado (precios, returns)

================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Time Series Models
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
from prophet import Prophet
from arch import arch_model

# Darts
try:
    from darts import TimeSeries
    from darts.models import DLinearModel, NBEATSModel, NHiTSModel, TFTModel, TCNModel
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
APP_DATA_DIR = os.path.join(BASE_DIR, "app", "backend", "data")

os.makedirs(APP_DATA_DIR, exist_ok=True)

# Config
CONFIG = {
    'leverage_max': 3.0,
    'sigmoid_scale': 500,
    'test_size': 0.20,
    'random_state': 42,
}

print("=" * 80)
print("GENERADOR DE DATOS PARA APP DE VISUALIZACION")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print(f"Output: {APP_DATA_DIR}")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def sigmoid_to_position(predictions, scale=500):
    """
    Convierte predicciones a posiciones continuas usando sigmoid.
    Retorna posiciones continuas en [-3, +3].

    Siguiendo la logica de evaluate_strategy.py
    """
    predictions = np.array(predictions).flatten()
    continuous_pos = 6 / (1 + np.exp(-np.clip(scale * predictions, -500, 500))) - 3
    return continuous_pos


def discretize_position(continuous_pos):
    """
    Discretiza posiciones continuas a {-3, -1, 0, +1, +3}.

    Siguiendo la logica de evaluate_strategy.py:
    - -3: SPXU (3x short)
    - -1: SH (1x short)
    -  0: Cash
    - +1: SPY (1x long)
    - +3: UPRO (3x long)
    """
    continuous_pos = np.array(continuous_pos).flatten()
    positions = np.zeros_like(continuous_pos)

    positions[continuous_pos <= -2] = -3           # SPXU (3x short)
    positions[(continuous_pos > -2) & (continuous_pos <= -0.5)] = -1   # SH (1x short)
    positions[(continuous_pos > -0.5) & (continuous_pos < 0.5)] = 0    # Cash
    positions[(continuous_pos >= 0.5) & (continuous_pos < 2)] = 1      # SPY (1x long)
    positions[continuous_pos >= 2] = 3             # UPRO (3x long)

    return positions


def prediction_to_position(predictions, max_leverage=3.0, scale=500):
    """
    Convierte predicciones a posiciones discretas {-3, -1, 0, +1, +3}.

    Proceso:
    1. Aplica sigmoid para obtener posicion continua en [-3, +3]
    2. Discretiza a los 5 niveles permitidos

    Siguiendo la logica de evaluate_strategy.py
    """
    continuous_pos = sigmoid_to_position(predictions, scale)
    discrete_pos = discretize_position(continuous_pos)
    return discrete_pos


def calculate_strategy_returns(positions, market_returns, risk_free_daily=0.0):
    """Calcula retornos de la estrategia."""
    min_len = min(len(positions), len(market_returns))
    positions = positions[:min_len]
    market_returns = market_returns[:min_len]
    strategy_returns = risk_free_daily * (1 - positions) + positions * market_returns
    return strategy_returns


def calculate_metrics(strategy_returns, market_returns, positions):
    """Calcula metricas completas."""
    min_len = min(len(strategy_returns), len(market_returns))
    strategy_returns = strategy_returns[:min_len]
    market_returns = market_returns[:min_len]
    positions = positions[:min_len]

    # Total returns
    strategy_total = np.prod(1 + strategy_returns) - 1
    market_total = np.prod(1 + market_returns) - 1

    # Annualized
    n_days = len(strategy_returns)
    years = n_days / 252
    strategy_annual = (1 + strategy_total) ** (1/years) - 1 if years > 0 else 0

    # Sharpe ratio
    rf_daily = 0.0
    excess_returns = strategy_returns - rf_daily
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    # Sortino
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Calmar
    calmar = strategy_annual / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wins = np.sum(strategy_returns > 0)
    losses = np.sum(strategy_returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Directional accuracy
    correct = np.sum((strategy_returns > 0) == (market_returns > 0))
    dir_accuracy = correct / len(strategy_returns)

    # Trade count (position changes)
    n_trades = int(np.sum(np.abs(np.diff(positions)) > 0.1))

    return {
        'total_return': float(strategy_total),
        'annual_return': float(strategy_annual),
        'market_return': float(market_total),
        'excess_return': float(strategy_total - market_total),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'calmar': float(calmar),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'dir_accuracy': float(dir_accuracy),
        'mean_position': float(np.mean(positions)),
        'n_trades': n_trades,
        'n_days': n_days
    }


def detect_regime(returns, window=60):
    """Detecta el regimen de mercado."""
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


def generate_trade_log(positions, strategy_returns, market_returns, dates, segment_days=20):
    """Genera log de trades (LONG y SHORT) con retornos de mercado y estrategia.

    Un trade es cuando el modelo tiene exposicion al mercado (posicion != 0).
    Para modelos con posiciones estables, divide en segmentos de segment_days dias.
    Esto permite ver el P&L periodico incluso para estrategias buy-and-hold.

    Args:
        positions: Array de posiciones del modelo
        strategy_returns: Retornos de la estrategia (position * market_return)
        market_returns: Retornos del mercado (SPY) sin apalancamiento
        dates: Fechas correspondientes
        segment_days: Dias por segmento para modelos estables

    Posiciones:
    - +3: UPRO (3x long)
    - +1: SPY (1x long)
    -  0: Cash (sin exposicion) - NO es un trade
    - -1: SH (1x short)
    - -3: SPXU (3x short)
    """
    trades = []
    current_trade = None
    segment_count = 0

    for i in range(len(positions)):
        pos = positions[i]
        has_exposure = abs(pos) > 0.1  # Position != 0 (has market exposure)

        if current_trade is None and has_exposure:
            # Entrada: el modelo toma una posicion (long o short)
            current_trade = {
                'entry_idx': i,
                'entry_date': str(dates[i])[:10],
                'entry_position': float(pos),
                'strategy_returns': [],
                'market_returns': []
            }
            segment_count = 0
        elif current_trade is not None:
            # Acumular retornos de estrategia y mercado
            strat_ret = float(strategy_returns[i]) if i < len(strategy_returns) else 0
            mkt_ret = float(market_returns[i]) if i < len(market_returns) else 0
            current_trade['strategy_returns'].append(strat_ret)
            current_trade['market_returns'].append(mkt_ret)
            segment_count += 1

            # Cambio significativo de posicion = nuevo trade
            # (incluye: cambio de long a short, cambio de leverage, o salida a cash)
            position_changed = abs(pos - current_trade['entry_position']) > 0.5 or abs(pos) < 0.1
            segment_reached = segment_count >= segment_days
            is_last = i == len(positions) - 1

            if position_changed or segment_reached or is_last:
                current_trade['exit_idx'] = i
                current_trade['exit_date'] = str(dates[i])[:10]
                current_trade['exit_position'] = float(pos)
                current_trade['duration'] = i - current_trade['entry_idx']

                # Calcular retornos compuestos
                strat_rets = np.array(current_trade['strategy_returns'])
                mkt_rets = np.array(current_trade['market_returns'])
                current_trade['total_return'] = float(np.prod(1 + strat_rets) - 1)  # Strategy return
                current_trade['market_return'] = float(np.prod(1 + mkt_rets) - 1)   # Market return
                current_trade['avg_position'] = float(np.mean([current_trade['entry_position'], pos]))

                # Limpiar arrays temporales
                del current_trade['strategy_returns']
                del current_trade['market_returns']

                if current_trade['duration'] > 0:  # Solo agregar trades con duracion > 0
                    trades.append(current_trade)

                # Iniciar nuevo trade si aun hay exposicion (long o short)
                if abs(pos) > 0.1 and not is_last:
                    current_trade = {
                        'entry_idx': i,
                        'entry_date': str(dates[i])[:10],
                        'entry_position': float(pos),
                        'strategy_returns': [],
                        'market_returns': []
                    }
                    segment_count = 0
                else:
                    current_trade = None

    return trades


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

# Split
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].reset_index(drop=True)

print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"  Test period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Preparar features
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# Market data
test_prices = test_df['SPY_CLOSE'].values
test_dates = test_df['date'].values
market_returns = np.diff(test_prices) / test_prices[:-1]

# Regimes
regimes = detect_regime(market_returns)

# Risk-free rate
rf_daily = 0.0


# =============================================================================
# ENTRENAR MODELOS Y GUARDAR DATOS DIARIOS
# =============================================================================
print("\n" + "=" * 80)
print("ENTRENANDO MODELOS Y GENERANDO DATOS")
print("=" * 80)

all_models_data = {}
all_models_summary = []

# ML Models
ml_models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.01, random_state=42, verbose=-1),
    'CatBoost': cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.01, random_state=42, verbose=0)
}

for model_name, model in ml_models.items():
    print(f"\n  {model_name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

    min_len = min(len(positions), len(market_returns))
    strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

    metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
    metrics['model'] = model_name
    metrics['category'] = 'ML'
    all_models_summary.append(metrics)

    # Equity curve
    equity_curve = np.cumprod(1 + strat_returns)
    market_equity = np.cumprod(1 + market_returns[:min_len-1])

    # Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    # Trade log (with both strategy and market returns)
    trade_log = generate_trade_log(positions[:min_len-1], strat_returns, market_returns[:min_len-1], test_dates[:min_len-1])

    # Save daily data
    all_models_data[model_name] = {
        'category': 'ML',
        'dates': [str(d)[:10] for d in test_dates[:min_len-1]],
        'predictions': y_pred[:min_len-1].tolist(),
        'positions': positions[:min_len-1].tolist(),
        'strategy_returns': strat_returns.tolist(),
        'market_returns': market_returns[:min_len-1].tolist(),
        'equity_curve': equity_curve.tolist(),
        'market_equity': market_equity.tolist(),
        'drawdown': drawdown.tolist(),
        'regimes': regimes[:min_len-1],
        'trades': trade_log,
        'metrics': metrics
    }

    print(f"    Return: {metrics['total_return']:.2%} | Sharpe: {metrics['sharpe']:.3f}")


# StatsForecast models
print("\n  StatsForecast models...")
sf_data = df[['date', target_col]].copy()
sf_data = sf_data.rename(columns={'date': 'ds', target_col: 'y'})
sf_data['unique_id'] = 'SPY'
sf_data = sf_data.dropna()

sf_train = sf_data.iloc[:train_size]
sf_test = sf_data.iloc[train_size:]

sf_models = [
    AutoARIMA(season_length=5),
    AutoETS(season_length=5),
    AutoTheta(season_length=5),
    SeasonalNaive(season_length=5)
]

sf = StatsForecast(models=sf_models, freq='B', n_jobs=-1)
sf.fit(sf_train)
forecasts = sf.predict(h=len(sf_test))

for col in ['AutoARIMA', 'AutoETS', 'AutoTheta', 'SeasonalNaive']:
    model_name = f'SF_{col}'
    print(f"\n  {model_name}...")

    y_pred = forecasts[col].values
    positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

    min_len = min(len(positions), len(market_returns))
    strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

    metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
    metrics['model'] = model_name
    metrics['category'] = 'StatsForecast'
    all_models_summary.append(metrics)

    equity_curve = np.cumprod(1 + strat_returns)
    market_equity = np.cumprod(1 + market_returns[:min_len-1])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    trade_log = generate_trade_log(positions[:min_len-1], strat_returns, market_returns[:min_len-1], test_dates[:min_len-1])

    all_models_data[model_name] = {
        'category': 'StatsForecast',
        'dates': [str(d)[:10] for d in test_dates[:min_len-1]],
        'predictions': y_pred[:min_len-1].tolist(),
        'positions': positions[:min_len-1].tolist(),
        'strategy_returns': strat_returns.tolist(),
        'market_returns': market_returns[:min_len-1].tolist(),
        'equity_curve': equity_curve.tolist(),
        'market_equity': market_equity.tolist(),
        'drawdown': drawdown.tolist(),
        'regimes': regimes[:min_len-1],
        'trades': trade_log,
        'metrics': metrics
    }

    print(f"    Return: {metrics['total_return']:.2%} | Sharpe: {metrics['sharpe']:.3f}")


# Prophet
print("\n  Prophet...")
prophet_train = pd.DataFrame({'ds': train_df['date'], 'y': train_df[target_col]}).dropna()

model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(prophet_train)
prophet_test = pd.DataFrame({'ds': test_df['date']})
forecast = model.predict(prophet_test)

y_pred = forecast['yhat'].values
positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

min_len = min(len(positions), len(market_returns))
strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
metrics['model'] = 'Prophet'
metrics['category'] = 'Prophet'
all_models_summary.append(metrics)

equity_curve = np.cumprod(1 + strat_returns)
market_equity = np.cumprod(1 + market_returns[:min_len-1])
running_max = np.maximum.accumulate(equity_curve)
drawdown = (equity_curve - running_max) / running_max
trade_log = generate_trade_log(positions[:min_len-1], strat_returns, market_returns[:min_len-1], test_dates[:min_len-1])

all_models_data['Prophet'] = {
    'category': 'Prophet',
    'dates': [str(d)[:10] for d in test_dates[:min_len-1]],
    'predictions': y_pred[:min_len-1].tolist(),
    'positions': positions[:min_len-1].tolist(),
    'strategy_returns': strat_returns.tolist(),
    'market_returns': market_returns[:min_len-1].tolist(),
    'equity_curve': equity_curve.tolist(),
    'market_equity': market_equity.tolist(),
    'drawdown': drawdown.tolist(),
    'regimes': regimes[:min_len-1],
    'trades': trade_log,
    'metrics': metrics
}

print(f"    Return: {metrics['total_return']:.2%} | Sharpe: {metrics['sharpe']:.3f}")


# GARCH
print("\n  GARCH...")
try:
    returns_scaled = df[target_col].dropna() * 100
    train_returns = returns_scaled.iloc[:train_size]

    garch = arch_model(train_returns, vol='Garch', p=1, q=1, mean='AR', lags=1)
    result = garch.fit(disp='off')
    forecasts_garch = result.forecast(horizon=len(test_df))
    y_pred = forecasts_garch.mean.values[-len(test_df):].flatten() / 100

    positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

    min_len = min(len(positions), len(market_returns))
    strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

    metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
    metrics['model'] = 'GARCH'
    metrics['category'] = 'GARCH'
    all_models_summary.append(metrics)

    equity_curve = np.cumprod(1 + strat_returns)
    market_equity = np.cumprod(1 + market_returns[:min_len-1])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    trade_log = generate_trade_log(positions[:min_len-1], strat_returns, market_returns[:min_len-1], test_dates[:min_len-1])

    all_models_data['GARCH'] = {
        'category': 'GARCH',
        'dates': [str(d)[:10] for d in test_dates[:min_len-1]],
        'predictions': y_pred[:min_len-1].tolist(),
        'positions': positions[:min_len-1].tolist(),
        'strategy_returns': strat_returns.tolist(),
        'market_returns': market_returns[:min_len-1].tolist(),
        'equity_curve': equity_curve.tolist(),
        'market_equity': market_equity.tolist(),
        'drawdown': drawdown.tolist(),
        'regimes': regimes[:min_len-1],
        'trades': trade_log,
        'metrics': metrics
    }

    print(f"    Return: {metrics['total_return']:.2%} | Sharpe: {metrics['sharpe']:.3f}")
except Exception as e:
    print(f"    [ERROR] GARCH: {e}")


# Darts models
if DARTS_AVAILABLE:
    print("\n  Darts models...")

    darts_df = df[['date', target_col]].copy().dropna()
    darts_df = darts_df.set_index('date')
    darts_df = darts_df.asfreq('B')
    darts_df = darts_df.fillna(method='ffill')

    series = TimeSeries.from_dataframe(darts_df, value_cols=target_col, fill_missing_dates=True, freq='B')
    train_series = series[:train_size]
    test_series = series[train_size:]
    n_test = len(test_series)

    darts_models = {
        'DLinear': DLinearModel(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42,
                                pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}),
        'N-BEATS': NBEATSModel(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42,
                               pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}),
        'N-HiTS': NHiTSModel(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42,
                             pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}),
        'TCN': TCNModel(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42,
                        pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}),
        'TFT': TFTModel(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42,
                        add_relative_index=True, pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cpu"}),
    }

    for model_name, model in darts_models.items():
        print(f"\n  {model_name}...")
        try:
            model.fit(train_series)
            predictions = model.predict(n=n_test)
            y_pred = predictions.values().flatten()

            if np.isnan(y_pred).any():
                y_pred = np.nan_to_num(y_pred, nan=0.0)

            positions = prediction_to_position(y_pred, CONFIG['leverage_max'], CONFIG['sigmoid_scale'])

            min_len = min(len(positions), len(market_returns))
            strat_returns = calculate_strategy_returns(positions[:min_len-1], market_returns[:min_len-1], rf_daily)

            metrics = calculate_metrics(strat_returns, market_returns[:min_len-1], positions[:min_len-1])
            metrics['model'] = model_name
            metrics['category'] = 'Darts'
            all_models_summary.append(metrics)

            equity_curve = np.cumprod(1 + strat_returns)
            market_equity = np.cumprod(1 + market_returns[:min_len-1])
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            trade_log = generate_trade_log(positions[:min_len-1], strat_returns, market_returns[:min_len-1], test_dates[:min_len-1])

            all_models_data[model_name] = {
                'category': 'Darts',
                'dates': [str(d)[:10] for d in test_dates[:min_len-1]],
                'predictions': y_pred[:min_len-1].tolist(),
                'positions': positions[:min_len-1].tolist(),
                'strategy_returns': strat_returns.tolist(),
                'market_returns': market_returns[:min_len-1].tolist(),
                'equity_curve': equity_curve.tolist(),
                'market_equity': market_equity.tolist(),
                'drawdown': drawdown.tolist(),
                'regimes': regimes[:min_len-1],
                'trades': trade_log,
                'metrics': metrics
            }

            print(f"    Return: {metrics['total_return']:.2%} | Sharpe: {metrics['sharpe']:.3f}")
        except Exception as e:
            print(f"    [ERROR] {model_name}: {e}")


# =============================================================================
# GUARDAR DATOS
# =============================================================================
print("\n" + "=" * 80)
print("GUARDANDO DATOS PARA LA APP")
print("=" * 80)

# 1. Daily data por modelo
daily_data_file = os.path.join(APP_DATA_DIR, "daily_data.json")
with open(daily_data_file, 'w') as f:
    json.dump(all_models_data, f)
print(f"  [OK] {daily_data_file}")

# 2. Models summary
summary_file = os.path.join(APP_DATA_DIR, "models_summary.json")
summary = {
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'test_period': {
        'start': str(test_df['date'].min().date()),
        'end': str(test_df['date'].max().date()),
        'days': len(test_df)
    },
    'models': sorted(all_models_summary, key=lambda x: x['total_return'], reverse=True)
}
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  [OK] {summary_file}")

# 3. Market data
market_data_file = os.path.join(APP_DATA_DIR, "market_data.json")
market_equity_full = np.cumprod(1 + market_returns)
market_running_max = np.maximum.accumulate(market_equity_full)
market_drawdown = (market_equity_full - market_running_max) / market_running_max

market_data = {
    'dates': [str(d)[:10] for d in test_dates[:-1]],
    'prices': test_prices[:-1].tolist(),
    'returns': market_returns.tolist(),
    'equity_curve': market_equity_full.tolist(),
    'drawdown': market_drawdown.tolist(),
    'regimes': regimes,
    'metrics': {
        'total_return': float(np.prod(1 + market_returns) - 1),
        'annual_return': float((np.prod(1 + market_returns) ** (252/len(market_returns))) - 1),
        'sharpe': float(np.mean(market_returns) / np.std(market_returns) * np.sqrt(252)),
        'max_drawdown': float(np.min(market_drawdown)),
        'volatility': float(np.std(market_returns) * np.sqrt(252))
    }
}
with open(market_data_file, 'w') as f:
    json.dump(market_data, f)
print(f"  [OK] {market_data_file}")

# 4. Regime summary
regime_file = os.path.join(APP_DATA_DIR, "regime_data.json")
regime_counts = {}
for r in regimes:
    regime_counts[r] = regime_counts.get(r, 0) + 1

# Performance by regime for each model
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
print(f"  [OK] {regime_file}")

print("\n" + "=" * 80)
print("[OK] DATOS GENERADOS EXITOSAMENTE")
print("=" * 80)
print(f"\nArchivos generados en: {APP_DATA_DIR}")
print(f"  - daily_data.json ({len(all_models_data)} modelos)")
print(f"  - models_summary.json")
print(f"  - market_data.json")
print(f"  - regime_data.json")
