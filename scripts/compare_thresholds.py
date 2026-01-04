# -*- coding: utf-8 -*-
"""
Comparacion rapida de diferentes configuraciones de umbrales para quantile positioning.
"""

import pandas as pd
import numpy as np
import pickle
import os

# Importar funciones del script principal
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_strategy import (
    quantile_position_asymmetric,
    apply_position_filter,
    apply_volatility_targeting,
    apply_drawdown_control,
    calculate_realistic_returns,
    calculate_metrics,
    CONFIG
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Cargar artefactos
print("Cargando artefactos...")
with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), 'rb') as f:
    artifacts = pickle.load(f)

metadata = artifacts['metadata']
models_data = artifacts['models']

forward_returns_test = metadata['forward_returns_test']
risk_free_test = metadata['risk_free_test']

# Configuraciones a probar
CONFIGS = {
    'conservador': {
        'name': 'Conservador (10/30)',
        'q_long_extreme': 10,   # Top 10% -> +3
        'q_long_moderate': 30,  # 30-70% -> neutral
        'q_short_extreme': 10,  # Bottom 10% -> -3
        'q_short_moderate': 30,
    },
    'agresivo_long': {
        'name': 'Agresivo Long (30/50)',
        'q_long_extreme': 30,   # Top 30% -> +3
        'q_long_moderate': 50,  # 50% -> neutral
        'q_short_extreme': 10,  # Bottom 10% -> -3 (conservador short)
        'q_short_moderate': 30,
    },
    'muy_agresivo_long': {
        'name': 'Muy Agresivo Long (40/60)',
        'q_long_extreme': 40,   # Top 40% -> +3
        'q_long_moderate': 60,  # 60% -> neutral
        'q_short_extreme': 10,  # Bottom 10% -> -3
        'q_short_moderate': 20,
    },
    'agresivo_simetrico': {
        'name': 'Agresivo Simetrico (30/50)',
        'q_long_extreme': 30,
        'q_long_moderate': 50,
        'q_short_extreme': 30,
        'q_short_moderate': 50,
    },
    'solo_extremos': {
        'name': 'Solo Extremos (20/20)',
        'q_long_extreme': 20,   # Top 20% -> +3
        'q_long_moderate': 20,  # Sin posicion +1
        'q_short_extreme': 20,
        'q_short_moderate': 20,
    },
}

def evaluate_config(model_name, test_pred, fwd_ret, rf, config):
    """Evalua una configuracion de umbrales."""
    positions, info = quantile_position_asymmetric(
        test_pred, window=63,
        q_long_extreme=config['q_long_extreme'],
        q_long_moderate=config['q_long_moderate'],
        q_short_extreme=config['q_short_extreme'],
        q_short_moderate=config['q_short_moderate'],
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
        'max_dd': metrics['max_drawdown'],
        'pct_3x': info['pct_pos_3'],
        'pct_1x': info['pct_pos_1'],
        'pct_cash': info['pct_pos_0'],
        'pct_short': info['pct_pos_neg1'] + info['pct_pos_neg3'],
    }

# Evaluar todos los modelos con todas las configuraciones
print("\n" + "="*100)
print("COMPARACION DE CONFIGURACIONES DE UMBRALES")
print("="*100)

results = []

# Solo evaluar los top 6 modelos para rapidez
top_models = ['GradientBoosting', 'Ridge', 'RandomForest', 'XGBoost', 'LightGBM', 'DLinear']

for model_name in top_models:
    if model_name not in models_data:
        continue

    model_data = models_data[model_name]
    test_pred = model_data['test_predictions']

    n_test = min(len(test_pred), len(forward_returns_test))
    test_pred = test_pred[:n_test]
    fwd_ret = forward_returns_test[:n_test]
    rf = risk_free_test[:n_test]

    print(f"\n{model_name}:")
    print("-" * 90)
    print(f"{'Config':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'%3x':>6} {'%1x':>6} {'%Cash':>6} {'%Short':>6}")
    print("-" * 90)

    for config_key, config in CONFIGS.items():
        result = evaluate_config(model_name, test_pred, fwd_ret, rf, config)

        print(f"{config['name']:<25} {result['return']*100:>9.1f}% {result['sharpe']:>8.2f} "
              f"{result['max_dd']*100:>7.1f}% {result['pct_3x']:>5.1f}% {result['pct_1x']:>5.1f}% "
              f"{result['pct_cash']:>5.1f}% {result['pct_short']:>5.1f}%")

        results.append({
            'model': model_name,
            'config': config_key,
            'config_name': config['name'],
            **result
        })

# Resumen
print("\n" + "="*100)
print("RESUMEN: MEJOR CONFIGURACION POR MODELO")
print("="*100)

results_df = pd.DataFrame(results)

for model_name in top_models:
    if model_name not in models_data:
        continue
    model_results = results_df[results_df['model'] == model_name]
    best = model_results.loc[model_results['sharpe'].idxmax()]
    print(f"{model_name:<20} -> {best['config_name']:<25} (Sharpe: {best['sharpe']:.2f}, Return: {best['return']*100:.1f}%)")

# Mejor configuracion global (promedio)
print("\n" + "="*100)
print("PROMEDIO POR CONFIGURACION (todos los modelos)")
print("="*100)

avg_by_config = results_df.groupby('config_name').agg({
    'return': 'mean',
    'sharpe': 'mean',
    'max_dd': 'mean',
}).sort_values('sharpe', ascending=False)

for config_name, row in avg_by_config.iterrows():
    print(f"{config_name:<25} Avg Return: {row['return']*100:>7.1f}% | Avg Sharpe: {row['sharpe']:>6.2f} | Avg MaxDD: {row['max_dd']*100:>6.1f}%")

# Guardar resultados
output_path = os.path.join(BASE_DIR, "results", "threshold_comparison.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResultados guardados en: {output_path}")
