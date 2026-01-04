# -*- coding: utf-8 -*-
"""
Optimizacion de parametros de posicionamiento por modelo.

Cada modelo tiene caracteristicas unicas:
- Distribucion de predicciones
- Accuracy direccional
- Magnitud de predicciones

Por lo tanto, cada uno necesita umbrales optimos diferentes.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_strategy import (
    quantile_position_asymmetric,
    apply_position_filter,
    apply_volatility_targeting,
    apply_drawdown_control,
    calculate_realistic_returns,
    calculate_metrics,
    bootstrap_confidence,
    CONFIG
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

print("="*100)
print("OPTIMIZACION DE PARAMETROS POR MODELO")
print("="*100)

# Cargar artefactos
print("\nCargando artefactos...")
with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), 'rb') as f:
    artifacts = pickle.load(f)

metadata = artifacts['metadata']
models_data = artifacts['models']

forward_returns_test = metadata['forward_returns_test']
risk_free_test = metadata['risk_free_test']
y_test = metadata['y_test']

# =============================================================================
# PASO 1: ANALIZAR CARACTERISTICAS DE CADA MODELO
# =============================================================================
print("\n" + "="*100)
print("PASO 1: Analisis de Caracteristicas de Predicciones")
print("="*100)

model_characteristics = {}

for model_name, model_data in models_data.items():
    test_pred = model_data['test_predictions']
    train_pred = model_data['train_predictions']

    n_test = min(len(test_pred), len(y_test))
    test_pred = test_pred[:n_test]
    actual = y_test[:n_test]

    # Calcular accuracy direccional
    pred_direction = np.sign(test_pred)
    actual_direction = np.sign(actual)
    directional_accuracy = np.mean(pred_direction == actual_direction)

    # Estadisticas de predicciones
    pred_mean = np.mean(test_pred)
    pred_std = np.std(test_pred)
    pred_min = np.min(test_pred)
    pred_max = np.max(test_pred)

    # Percentiles
    p10 = np.percentile(test_pred, 10)
    p25 = np.percentile(test_pred, 25)
    p50 = np.percentile(test_pred, 50)
    p75 = np.percentile(test_pred, 75)
    p90 = np.percentile(test_pred, 90)

    # Sesgo (% predicciones positivas)
    pct_positive = np.mean(test_pred > 0) * 100

    model_characteristics[model_name] = {
        'directional_accuracy': directional_accuracy,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'p10': p10,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'pct_positive': pct_positive,
    }

# Mostrar caracteristicas
print(f"\n{'Model':<20} {'DirAcc':>7} {'Mean':>10} {'Std':>10} {'%Pos':>6} {'P10':>10} {'P90':>10}")
print("-"*85)
for model_name, chars in sorted(model_characteristics.items(),
                                 key=lambda x: x[1]['directional_accuracy'],
                                 reverse=True):
    print(f"{model_name:<20} {chars['directional_accuracy']*100:>6.1f}% "
          f"{chars['pred_mean']:>10.5f} {chars['pred_std']:>10.5f} "
          f"{chars['pct_positive']:>5.1f}% {chars['p10']:>10.5f} {chars['p90']:>10.5f}")

# =============================================================================
# PASO 2: GRID SEARCH DE PARAMETROS POR MODELO
# =============================================================================
print("\n" + "="*100)
print("PASO 2: Grid Search de Parametros Optimos")
print("="*100)

# Parametros a buscar
param_grid = {
    'q_long_extreme': [10, 20, 30, 40, 50],      # Top X% para +3
    'q_long_moderate': [20, 30, 40, 50, 60, 70], # Top X% para +1
    'q_short_extreme': [5, 10, 15, 20],          # Bottom X% para -3
    'q_short_moderate': [10, 20, 30, 40],        # Bottom X% para -1
}

def evaluate_params(test_pred, fwd_ret, rf, params):
    """Evalua una combinacion de parametros."""
    # Validar que los parametros tienen sentido
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

    # Aplicar mejoras
    positions = apply_position_filter(positions, CONFIG['min_position_change'])
    positions = apply_volatility_targeting(positions, fwd_ret, CONFIG['volatility_target'], CONFIG['volatility_lookback'])

    returns_temp = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])

    positions = apply_drawdown_control(positions, cumulative_temp, CONFIG['drawdown_limits'])
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

# Generar todas las combinaciones
all_combinations = list(product(
    param_grid['q_long_extreme'],
    param_grid['q_long_moderate'],
    param_grid['q_short_extreme'],
    param_grid['q_short_moderate'],
))

print(f"\nTotal combinaciones a evaluar: {len(all_combinations)} x {len(models_data)} modelos")

optimal_params = {}
all_results = []

for model_name, model_data in models_data.items():
    print(f"\n  Optimizando: {model_name}...")

    test_pred = model_data['test_predictions']
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

        # Guardar para analisis
        all_results.append({
            'model': model_name,
            **params,
            **result,
        })

        # Criterio: maximizar Sharpe, pero penalizar si MaxDD < -20%
        score = result['sharpe']
        if result['max_dd'] < -0.20:
            score -= 0.5  # Penalizar drawdowns grandes

        if score > best_sharpe:
            best_sharpe = score
            best_params = params.copy()
            best_result = result.copy()

    optimal_params[model_name] = {
        'params': best_params,
        'result': best_result,
        'characteristics': model_characteristics[model_name],
    }

    print(f"    Mejor: q_long=({best_params['q_long_extreme']},{best_params['q_long_moderate']}) "
          f"q_short=({best_params['q_short_extreme']},{best_params['q_short_moderate']}) "
          f"-> Sharpe: {best_result['sharpe']:.2f}, Return: {best_result['return']*100:.1f}%")

# =============================================================================
# PASO 3: RESUMEN DE PARAMETROS OPTIMOS
# =============================================================================
print("\n" + "="*100)
print("PASO 3: Parametros Optimos por Modelo")
print("="*100)

print(f"\n{'Model':<20} {'LongExt':>8} {'LongMod':>8} {'ShortExt':>9} {'ShortMod':>9} "
      f"{'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'%3xL':>6}")
print("-"*105)

for model_name in sorted(optimal_params.keys(),
                         key=lambda x: optimal_params[x]['result']['sharpe'],
                         reverse=True):
    opt = optimal_params[model_name]
    p = opt['params']
    r = opt['result']
    print(f"{model_name:<20} {p['q_long_extreme']:>8} {p['q_long_moderate']:>8} "
          f"{p['q_short_extreme']:>9} {p['q_short_moderate']:>9} "
          f"{r['return']*100:>9.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}% "
          f"{r['pct_3x_long']:>5.1f}%")

# =============================================================================
# PASO 4: CALCULAR BOOTSTRAP CI PARA PARAMETROS OPTIMOS
# =============================================================================
print("\n" + "="*100)
print("PASO 4: Bootstrap Confidence Intervals")
print("="*100)

for model_name in optimal_params.keys():
    model_data = models_data[model_name]
    test_pred = model_data['test_predictions']

    n_test = min(len(test_pred), len(forward_returns_test))
    test_pred = test_pred[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    opt = optimal_params[model_name]
    p = opt['params']

    # Recalcular con parametros optimos
    positions, _ = quantile_position_asymmetric(
        test_pred, window=63,
        q_long_extreme=p['q_long_extreme'],
        q_long_moderate=p['q_long_moderate'],
        q_short_extreme=p['q_short_extreme'],
        q_short_moderate=p['q_short_moderate'],
    )

    positions = apply_position_filter(positions, CONFIG['min_position_change'])
    positions = apply_volatility_targeting(positions, fwd_ret, CONFIG['volatility_target'], CONFIG['volatility_lookback'])

    returns_temp = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)
    cumulative_temp = np.cumprod(1 + returns_temp['net_returns'])

    positions = apply_drawdown_control(positions, cumulative_temp, CONFIG['drawdown_limits'])
    returns = calculate_realistic_returns(positions, fwd_ret, rf, CONFIG, apply_vol_drag=True)

    # Bootstrap
    boot = bootstrap_confidence(returns['net_returns'], risk_free_rate=rf, n_bootstrap=5000, confidence=0.95)

    optimal_params[model_name]['bootstrap'] = boot

    print(f"  {model_name:<20} Sharpe: {opt['result']['sharpe']:>5.2f} "
          f"CI: [{boot['sharpe_ci_lower']:>5.2f}, {boot['sharpe_ci_upper']:>5.2f}] "
          f"P(>0): {boot['sharpe_prob_positive']*100:>5.1f}%")

# =============================================================================
# PASO 5: GUARDAR CONFIGURACION OPTIMA
# =============================================================================
print("\n" + "="*100)
print("PASO 5: Guardar Configuracion Optima")
print("="*100)

# Preparar para JSON (convertir numpy a python types)
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

# Guardar todos los resultados del grid search
results_df = pd.DataFrame(all_results)
results_path = os.path.join(RESULTS_DIR, 'param_grid_search_results.csv')
results_df.to_csv(results_path, index=False)
print(f"  + Guardado: {results_path}")

# =============================================================================
# PASO 6: COMPARACION CON BENCHMARK
# =============================================================================
print("\n" + "="*100)
print("PASO 6: Comparacion Final con Benchmark")
print("="*100)

# Buy & Hold SPY
spy_returns = forward_returns_test
spy_cumulative = np.cumprod(1 + spy_returns)
spy_total_return = spy_cumulative[-1] - 1
spy_sharpe = (np.mean(spy_returns) - np.mean(risk_free_test)) / np.std(spy_returns) * np.sqrt(252)

print(f"\n  BENCHMARK (Buy & Hold SPY):")
print(f"    Return: {spy_total_return*100:.1f}%")
print(f"    Sharpe: {spy_sharpe:.2f}")

print(f"\n  MODELOS CON PARAMETROS OPTIMOS QUE SUPERAN SPY:")
print(f"  {'Model':<20} {'Return':>10} {'vs SPY':>10} {'Sharpe':>8} {'P(Sharpe>0)':>12}")
print("  " + "-"*65)

models_beating_spy = []
for model_name in sorted(optimal_params.keys(),
                         key=lambda x: optimal_params[x]['result']['return'],
                         reverse=True):
    opt = optimal_params[model_name]
    r = opt['result']
    b = opt['bootstrap']

    if r['return'] > spy_total_return:
        excess = r['return'] - spy_total_return
        models_beating_spy.append(model_name)
        print(f"  {model_name:<20} {r['return']*100:>9.1f}% {excess*100:>+9.1f}% "
              f"{r['sharpe']:>8.2f} {b['sharpe_prob_positive']*100:>11.1f}%")

print(f"\n  Total modelos que superan SPY: {len(models_beating_spy)}/{len(models_data)}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*100)
print("RESUMEN FINAL")
print("="*100)

# Top 5 modelos
print("\n  TOP 5 MODELOS (parametros optimizados):")
top5 = sorted(optimal_params.items(), key=lambda x: x[1]['result']['sharpe'], reverse=True)[:5]

for i, (model_name, opt) in enumerate(top5, 1):
    p = opt['params']
    r = opt['result']
    b = opt['bootstrap']
    c = opt['characteristics']

    print(f"\n  {i}. {model_name}")
    print(f"     Direccional Accuracy: {c['directional_accuracy']*100:.1f}%")
    print(f"     Parametros: Long({p['q_long_extreme']},{p['q_long_moderate']}) Short({p['q_short_extreme']},{p['q_short_moderate']})")
    print(f"     Return: {r['return']*100:.1f}% | Sharpe: {r['sharpe']:.2f} | MaxDD: {r['max_dd']*100:.1f}%")
    print(f"     Bootstrap CI: [{b['sharpe_ci_lower']:.2f}, {b['sharpe_ci_upper']:.2f}] | P(Sharpe>0): {b['sharpe_prob_positive']*100:.1f}%")

print("\n" + "="*100)
print("OPTIMIZACION COMPLETADA")
print("="*100)
