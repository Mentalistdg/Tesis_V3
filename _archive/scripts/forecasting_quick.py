# -*- coding: utf-8 -*-
"""
================================================================================
FORECASTING RAPIDO - NIXTLA & PROPHET (Version Simplificada)
================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("FORECASTING RAPIDO - NIXTLA & PROPHET")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# Cargar datos
data_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")
df = pd.read_csv(data_file)
df['date'] = pd.to_datetime(df['date'])

print(f"\nDatos: {len(df):,} filas")
print(f"Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

# Split 80/20
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

results = {}

# =============================================================================
# STATSFORECAST - Modelos simples
# =============================================================================
print("\n" + "=" * 80)
print("STATSFORECAST")
print("=" * 80)

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

    # Formato StatsForecast
    sf_train = pd.DataFrame({
        'unique_id': 'SPY',
        'ds': train_df['date'],
        'y': train_df['SPY_CLOSE']
    })

    sf_test = pd.DataFrame({
        'unique_id': 'SPY',
        'ds': test_df['date'],
        'y': test_df['SPY_CLOSE']
    })

    # Solo modelos rapidos
    models = [
        Naive(),
        SeasonalNaive(season_length=5),
        AutoARIMA(season_length=5, max_p=3, max_q=3, max_P=1, max_Q=1),
    ]

    sf = StatsForecast(models=models, freq='B', n_jobs=1)

    print("\n  Entrenando modelos...")
    sf.fit(sf_train)

    print("  Prediciendo...")
    # Predict step by step
    horizon = len(test_df)
    predictions = sf.predict(h=horizon)

    print("\n  Resultados StatsForecast:")
    print("-" * 60)

    y_true = test_df['SPY_CLOSE'].values

    for model_name in ['Naive', 'SeasonalNaive', 'AutoARIMA']:
        if model_name in predictions.columns:
            y_pred = predictions[model_name].values

            # Metricas
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))

            # Directional accuracy
            true_ret = np.diff(y_true)
            pred_ret = np.diff(y_pred)
            dir_acc = np.mean(np.sign(true_ret) == np.sign(pred_ret))

            # Trading return
            positions = np.sign(pred_ret)
            positions = np.where(positions >= 0, 1, 0)
            actual_ret = true_ret / y_true[:-1]
            strategy_ret = positions * actual_ret
            total_return = np.prod(1 + strategy_ret) - 1
            sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0

            results[f'SF_{model_name}'] = {
                'rmse': rmse,
                'mae': mae,
                'dir_acc': dir_acc,
                'return': total_return,
                'sharpe': sharpe
            }

            print(f"\n  {model_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Dir Acc: {dir_acc:.2%}")
            print(f"    Return: {total_return:.2%}")
            print(f"    Sharpe: {sharpe:.3f}")

except Exception as e:
    print(f"  [ERROR] StatsForecast: {e}")

# =============================================================================
# PROPHET
# =============================================================================
print("\n" + "=" * 80)
print("PROPHET")
print("=" * 80)

try:
    from prophet import Prophet

    # Formato Prophet
    prophet_train = pd.DataFrame({
        'ds': train_df['date'],
        'y': train_df['SPY_CLOSE']
    })

    prophet_test = pd.DataFrame({
        'ds': test_df['date'],
        'y': test_df['SPY_CLOSE']
    })

    print("\n  Entrenando Prophet...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_train)

    print("  Prediciendo...")
    forecast = model.predict(prophet_test[['ds']])

    y_true = prophet_test['y'].values
    y_pred = forecast['yhat'].values

    # Metricas
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # Directional accuracy
    true_ret = np.diff(y_true)
    pred_ret = np.diff(y_pred)
    dir_acc = np.mean(np.sign(true_ret) == np.sign(pred_ret))

    # Trading
    positions = np.sign(pred_ret)
    positions = np.where(positions >= 0, 1, 0)
    actual_ret = true_ret / y_true[:-1]
    strategy_ret = positions * actual_ret
    total_return = np.prod(1 + strategy_ret) - 1
    sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0

    results['Prophet'] = {
        'rmse': rmse,
        'mae': mae,
        'dir_acc': dir_acc,
        'return': total_return,
        'sharpe': sharpe
    }

    print(f"\n  Prophet:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Dir Acc: {dir_acc:.2%}")
    print(f"    Return: {total_return:.2%}")
    print(f"    Sharpe: {sharpe:.3f}")
    print(f"    Changepoints: {len(model.changepoints)}")

except Exception as e:
    print(f"  [ERROR] Prophet: {e}")

# =============================================================================
# COMPARACION CON ML
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACION CON MODELOS ML")
print("=" * 80)

ml_file = os.path.join(RESULTS_DIR, "model_comparison_bloomberg.csv")
if os.path.exists(ml_file):
    ml_df = pd.read_csv(ml_file)

    # Market return en test period
    market_ret = test_df['SPY_CLOSE'].iloc[-1] / test_df['SPY_CLOSE'].iloc[0] - 1
    print(f"\nMarket Return (Test): {market_ret:.2%}")

    print("\n  Ranking por Return:")
    print("-" * 70)
    print(f"  {'Modelo':<25} {'Return':>12} {'Sharpe':>10} {'Dir Acc':>10}")
    print("-" * 70)

    # Combinar resultados
    all_results = []

    # ML models
    for _, row in ml_df.iterrows():
        all_results.append({
            'model': row['model'] if 'model' in row else str(row.name),
            'type': 'ML',
            'return': row.get('strategy_return', 0) - 1,  # Convertir de multiplicador a %
            'sharpe': row.get('strategy_sharpe', 0),
            'dir_acc': row.get('test_dir_acc', 0)
        })

    # Forecast models
    for name, metrics in results.items():
        all_results.append({
            'model': name,
            'type': 'Forecast',
            'return': metrics['return'],
            'sharpe': metrics['sharpe'],
            'dir_acc': metrics['dir_acc']
        })

    # Ordenar por return
    all_results.sort(key=lambda x: x['return'], reverse=True)

    for r in all_results:
        marker = '*' if r['type'] == 'Forecast' else ' '
        print(f"  {marker}{r['model']:<24} {r['return']:>11.2%} {r['sharpe']:>10.3f} {r['dir_acc']:>10.2%}")

    print("-" * 70)
    print("  * = Modelo de Forecasting (Nixtla/Prophet)")

# Guardar resultados
output_file = os.path.join(RESULTS_DIR, "forecasting_quick_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Resultados guardados: {output_file}")

print("\n" + "=" * 80)
print("[OK] FORECASTING COMPLETADO")
print("=" * 80)
