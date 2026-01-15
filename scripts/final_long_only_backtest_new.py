# -*- coding: utf-8 -*-
"""
================================================================================
FINAL LONG-ONLY BACKTEST
================================================================================

Estrategia final optimizada:
- Posiciones: 0 (Cash), +1 (SPY), +3 (UPRO)
- SIN posiciones cortas
- Cuando el modelo no esta seguro -> Cash (no apostar contra el mercado)

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle
from datetime import datetime
from typing import Dict

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},  # 3x Long
    'SPY': {'expense_ratio': 0.0009, 'bid_ask': 0.0002},   # 1x Long
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},  # Risk-free
}


def quantile_position_long_only(predictions, warmup_predictions=None, window=63,
                                q_3x=10, q_1x=30):
    """
    Calcula posiciones LONG-ONLY basadas en percentiles rolling.

    Posiciones:
    - +3 (UPRO): prediccion en top q_3x percentil
    - +1 (SPY): prediccion en top q_1x percentil
    - 0 (CASH): resto (no seguro -> quedarse fuera)

    Args:
        predictions: Predicciones del modelo
        warmup_predictions: Predicciones de entrenamiento para warmup
        window: Ventana rolling para calcular percentiles
        q_3x: Percentil para 3x (ej: 10 = top 10%)
        q_1x: Percentil para 1x (ej: 30 = top 30%)
    """
    predictions = np.array(predictions).flatten()
    n = len(predictions)
    positions = np.zeros(n)

    # Umbrales
    thresh_3x = 100 - q_3x   # ej: 100 - 10 = 90 -> top 10%
    thresh_1x = 100 - q_1x   # ej: 100 - 30 = 70 -> top 30%

    # Preparar warmup
    if warmup_predictions is not None:
        warmup = np.array(warmup_predictions).flatten()
        warmup = warmup[-(window-1):] if len(warmup) >= window-1 else warmup
        all_preds = np.concatenate([warmup, predictions])
        warmup_len = len(warmup)
    else:
        all_preds = predictions
        warmup_len = 0

    for i in range(n):
        idx = i + warmup_len
        start = max(0, idx - window + 1)
        window_preds = all_preds[start:idx+1]

        if len(window_preds) < 5:
            positions[i] = 0
            continue

        current_pred = predictions[i]
        percentile = np.mean(window_preds <= current_pred) * 100

        if percentile >= thresh_3x:
            positions[i] = 3
        elif percentile >= thresh_1x:
            positions[i] = 1
        else:
            positions[i] = 0  # No seguro -> Cash

    return positions


def calculate_returns(positions, market_returns, risk_free):
    """Calcula retornos con costos realistas."""
    positions = np.array(positions).flatten()
    market_returns = np.array(market_returns).flatten()
    risk_free = np.array(risk_free).flatten()

    n = len(positions)
    returns = np.zeros(n)
    pos_to_inst = {3: 'UPRO', 1: 'SPY', 0: 'CASH'}

    for i in range(n):
        pos = int(positions[i])
        inst = pos_to_inst.get(pos, 'CASH')
        cfg = INSTRUMENTS[inst]

        # Retorno bruto
        if pos == 0:
            gross = risk_free[i]
        else:
            gross = risk_free[i] + pos * (market_returns[i] - risk_free[i])

        # Costos
        expense = cfg['expense_ratio'] / 252
        trading = 0
        if i > 0 and positions[i] != positions[i-1]:
            prev_inst = pos_to_inst.get(int(positions[i-1]), 'CASH')
            trading += INSTRUMENTS[prev_inst]['bid_ask'] + cfg['bid_ask']

        # Volatility drag para 3x
        vol_drag = 0
        if pos == 3:
            vol_drag = 0.5 * 6 * (0.01)**2

        returns[i] = gross - expense - trading - vol_drag

    return returns


def backtest_model(predictions, train_predictions, y_test, rf_test, params):
    """Backtest LONG-ONLY para un modelo."""

    # Alinear longitudes
    min_len = min(len(predictions), len(y_test), len(rf_test))
    predictions = predictions[:min_len]
    y_test = y_test[:min_len]
    rf_test = rf_test[:min_len]

    # Posiciones LONG-ONLY
    positions = quantile_position_long_only(
        predictions,
        warmup_predictions=train_predictions,
        q_3x=params.get('q_long_extreme', 10),
        q_1x=params.get('q_long_moderate', 30),
    )

    # Calcular retornos
    final_returns = calculate_returns(positions, y_test, rf_test)
    equity = np.cumprod(1 + final_returns)

    # Metricas
    total_ret = equity[-1] - 1
    n_years = len(final_returns) / 252
    annual_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(final_returns) * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    running_max = np.maximum.accumulate(equity)
    max_dd = np.max((running_max - equity) / running_max)

    # Direccional accuracy
    pred_dir = np.sign(predictions[:len(y_test)])
    actual_dir = np.sign(y_test[:len(predictions)])
    da = np.mean(pred_dir == actual_dir)

    # Distribucion de posiciones
    pct_3x = np.mean(positions == 3) * 100
    pct_1x = np.mean(positions == 1) * 100
    pct_cash = np.mean(positions == 0) * 100

    # Trades
    n_trades = np.sum(np.diff(positions) != 0)

    return {
        'total_return': total_ret,
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'directional_accuracy': da,
        'pct_3x': pct_3x,
        'pct_1x': pct_1x,
        'pct_cash': pct_cash,
        'n_trades': int(n_trades),
        'equity_curve': equity.tolist(),
        'positions': positions.tolist(),
    }


def main():
    print("=" * 80)
    print("FINAL LONG-ONLY BACKTEST")
    print("=" * 80)
    print("\nEstrategia: Solo posiciones LONG (0, +1, +3)")
    print("- +3 (UPRO): Muy alcista -> 3x apalancado")
    print("- +1 (SPY): Alcista -> 1x largo")
    print("-  0 (CASH): No seguro -> Quedarse fuera")

    # Load data
    print("\n[1] Cargando datos...")
    artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    params_path = os.path.join(RESULTS_DIR, "optimal_model_params.json")
    with open(params_path, 'r') as f:
        optimal_params = json.load(f)

    metadata = artifacts['metadata']
    y_test = np.array(metadata['forward_returns_test'][:-1])
    rf_test = np.array(metadata['risk_free_test'][:-1])

    print(f"    Modelos: {len(artifacts['models'])}")
    print(f"    Dias de test: {len(y_test)}")

    # Benchmark
    spy_equity = np.cumprod(1 + y_test)
    spy_return = spy_equity[-1] - 1
    spy_sharpe = np.mean(y_test) / np.std(y_test) * np.sqrt(252)
    spy_max_dd = np.max((np.maximum.accumulate(spy_equity) - spy_equity) /
                        np.maximum.accumulate(spy_equity))

    print(f"\n    SPY B&H: Return={spy_return*100:+.1f}%, Sharpe={spy_sharpe:.3f}, MaxDD={spy_max_dd*100:.1f}%")

    # Backtest
    print("\n[2] Ejecutando backtest LONG-ONLY...")
    results = {}

    for model_name, model_data in artifacts['models'].items():
        test_preds = np.array(model_data['test_predictions'])
        train_preds = np.array(model_data.get('train_predictions', []))

        if len(test_preds) == 0:
            continue

        # Parametros optimizados
        params = optimal_params.get(model_name, {}).get('params', {
            'q_long_extreme': 10, 'q_long_moderate': 30
        })

        metrics = backtest_model(test_preds, train_preds, y_test, rf_test, params)
        results[model_name] = metrics

        beat = "BATE SPY!" if metrics['total_return'] > spy_return else ""
        print(f"    {model_name:<20} Return: {metrics['total_return']*100:+7.1f}% | "
              f"Sharpe: {metrics['sharpe']:+6.3f} | "
              f"MaxDD: {metrics['max_drawdown']*100:5.1f}% | "
              f"3x:{metrics['pct_3x']:4.1f}% 1x:{metrics['pct_1x']:4.1f}% Cash:{metrics['pct_cash']:4.1f}% {beat}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES - LONG-ONLY (ordenado por Return)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)

    print(f"\n{'Rank':<5} {'Modelo':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'%3x':>7} {'%1x':>7} {'%Cash':>7} {'Trades':>7}")
    print("-" * 95)

    for i, (name, m) in enumerate(sorted_results):
        beat = "*" if m['total_return'] > spy_return else " "
        print(f"{i+1:<5} {name:<20} {m['total_return']*100:>+9.1f}%{beat} {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']*100:>7.1f}% {m['pct_3x']:>6.1f}% {m['pct_1x']:>6.1f}% "
              f"{m['pct_cash']:>6.1f}% {m['n_trades']:>7}")

    print("-" * 95)
    print(f"{'':5} {'SPY B&H':<20} {spy_return*100:>+9.1f}%  {spy_sharpe:>8.3f} {spy_max_dd*100:>7.1f}%")

    # Modelos que baten SPY
    beating_spy = [(n, m) for n, m in sorted_results if m['total_return'] > spy_return]
    print(f"\n[MODELOS QUE BATEN SPY: {len(beating_spy)}/23]")
    for name, m in beating_spy:
        excess = m['total_return'] - spy_return
        print(f"  - {name}: {m['total_return']*100:+.1f}% (exceso: {excess*100:+.1f}%)")

    # Estadisticas
    print("\n[ESTADISTICAS]")
    avg_return = np.mean([m['total_return'] for m in results.values()])
    avg_sharpe = np.mean([m['sharpe'] for m in results.values()])
    avg_dd = np.mean([m['max_drawdown'] for m in results.values()])

    print(f"  Retorno promedio: {avg_return*100:+.1f}%")
    print(f"  Sharpe promedio: {avg_sharpe:.3f}")
    print(f"  MaxDD promedio: {avg_dd*100:.1f}%")

    # Top 5 con mejor Sharpe
    top_sharpe = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:5]
    print(f"\n[TOP 5 POR SHARPE]")
    for name, m in top_sharpe:
        print(f"  - {name}: Sharpe={m['sharpe']:.3f}, Return={m['total_return']*100:+.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'LONG-ONLY (0, +1, +3)',
        'benchmark': {
            'total_return': float(spy_return),
            'sharpe': float(spy_sharpe),
            'max_drawdown': float(spy_max_dd)
        },
        'models': {
            n: {k: (float(v) if isinstance(v, (float, np.floating)) else
                   (int(v) if isinstance(v, (int, np.integer)) else v))
               for k, v in m.items() if k not in ['equity_curve', 'positions']}
            for n, m in results.items()
        },
        'summary': {
            'models_beating_spy': len(beating_spy),
            'avg_return': float(avg_return),
            'avg_sharpe': float(avg_sharpe),
            'avg_max_dd': float(avg_dd),
            'beating_spy_models': [n for n, _ in beating_spy]
        }
    }

    out_path = os.path.join(RESULTS_DIR, "final_long_only_backtest.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Resultados guardados en: {out_path}")

    # Save equity curves for visualization
    equity_data = {
        'dates': list(range(len(y_test))),
        'spy': spy_equity.tolist(),
        'models': {n: m['equity_curve'] for n, m in results.items()}
    }

    equity_path = os.path.join(RESULTS_DIR, "long_only_equity_curves.json")
    with open(equity_path, 'w') as f:
        json.dump(equity_data, f)
    print(f"[OK] Curvas de equity guardadas en: {equity_path}")


if __name__ == "__main__":
    main()
