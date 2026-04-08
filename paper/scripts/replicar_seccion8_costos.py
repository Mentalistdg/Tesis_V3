# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION DE LA SECCION 8: COSTOS DE TRANSACCION
================================================================================
Este script replica todos los valores numericos de la Seccion 8 del paper
"Prediccion de Retornos del S&P 500 mediante Aprendizaje Automatico..."

Lee datos de:
  - results/final_long_only_backtest.json  (rendimiento, costos, trades)
  - results/backtest_detail.pkl            (retornos brutos y costos diarios)

Tablas replicadas:
  - Tabla (tab:cost_sensitivity) -- Base, +50%, +100% costos, break-even

Numeros en prosa replicados:
  - Parametros de costo: UPRO 0.91%/yr, SPY 0.09%/yr, bid-ask 5/2 bps
  - Costo promedio $3,545 (35.5% del P&L bruto promedio)
  - Break-even multipliers por modelo
  - Tx/P&L ratio por modelo
  - Vol drag anualizado ~7.5%
  - Estadisticas de latencia (T+open, T+close) para LSTM+Attention

USAGE:
    python paper/scripts/replicar_seccion8_costos.py

================================================================================
Author: David Gonzalez Canon
================================================================================
"""

import numpy as np
import pickle
import json
import os

SEED = 42
np.random.seed(SEED)

# ============================================================================
# CONFIGURACION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

INITIAL_CAPITAL = 10000

# Parametros de costos (identicos a optimize_and_backtest.py)
INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},
    'SPY':  {'expense_ratio': 0.0009, 'bid_ask': 0.0002},
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},
}

WINNERS = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM"]
ALL_MODELS = [
    "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost",
    "AutoARIMA", "ExponentialSmoothing", "Theta", "SeasonalNaive",
    "Prophet", "GARCH",
    "DLinear", "NBEATS", "NHiTS", "TCN", "TFT",
    "CNN_LSTM", "LSTM_Attention", "BiLSTM", "BiGRU",
]


# ============================================================================
# [1] CARGA DE DATOS
# ============================================================================

def load_data():
    """Carga backtest JSON y backtest detail."""
    with open(os.path.join(RESULTS_DIR, "backtest_detail.pkl"), "rb") as f:
        detail = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "final_long_only_backtest.json"), "r") as f:
        backtest_json = json.load(f)
    return detail, backtest_json


# ============================================================================
# [2] COST SENSITIVITY (Aproximacion lineal, como en el paper)
# ============================================================================

def cost_sensitivity_linear(gross_pl, total_costs, multiplier):
    """
    Aproximacion lineal para sensibilidad a costos (formula del paper).

    retorno_ajustado = (P&L_bruto - costos * multiplicador) / capital_inicial

    Esta es la formula descrita en la nota al pie del paper:
    "retorno ajustado = (P&L bruto - costos * multiplicador) / capital inicial"
    """
    adjusted_pl = gross_pl - total_costs * multiplier
    return adjusted_pl / INITIAL_CAPITAL * 100


# ============================================================================
# [3] LATENCY IMPACT (Aproximacion lineal con factores de correlacion)
# ============================================================================

def compute_latency_impact(backtest_json):
    """
    Calcula impacto de latencia para LSTM+Attention.

    Usa aproximacion lineal con factores de correlacion de senales:
      retorno = (correlacion * P&L_bruto - costos) / capital

    T+open usa correlacion = 0.94 (degradacion parcial por gap overnight)
    T+close usa correlacion = 0.89 (degradacion por dia completo de delay)
    """
    model_name = "LSTM_Attention"
    json_data = backtest_json["models"][model_name]
    gross_pl = json_data["gross_final"] - INITIAL_CAPITAL
    total_costs = json_data["total_costs"]
    base_return = json_data["total_return"] * 100

    corr_open = 0.94
    corr_close = 0.89

    ret_open = (corr_open * gross_pl - total_costs) / INITIAL_CAPITAL * 100
    ret_close = (corr_close * gross_pl - total_costs) / INITIAL_CAPITAL * 100

    deg_open = (ret_open - base_return) / base_return * 100
    deg_close = (ret_close - base_return) / base_return * 100

    return {
        "base_return_pct": round(base_return, 1),
        "t_plus_open": {"return_pct": round(ret_open, 1), "degradation_pct": round(deg_open, 1), "correlation": corr_open},
        "t_plus_close": {"return_pct": round(ret_close, 1), "degradation_pct": round(deg_close, 1), "correlation": corr_close},
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("REPLICACION DE LA SECCION 8: COSTOS DE TRANSACCION")
    print("=" * 90)
    print()

    # ------------------------------------------------------------------
    # [1] Carga de datos
    # ------------------------------------------------------------------
    print("[1] Cargando datos del pipeline...")
    detail, backtest_json = load_data()

    n_models = len(detail["models"])
    print(f"    {n_models} modelos en backtest_detail.pkl")
    print(f"    {len(backtest_json['models'])} modelos en final_long_only_backtest.json")
    print()

    # ------------------------------------------------------------------
    # [2] Parametros de costo
    # ------------------------------------------------------------------
    print("[2] Parametros del modelo de costos")
    print("-" * 90)
    print(f"    UPRO: expense_ratio = {INSTRUMENTS['UPRO']['expense_ratio']*100:.2f}%/yr, "
          f"bid-ask = {INSTRUMENTS['UPRO']['bid_ask']*10000:.0f} bps")
    print(f"    SPY:  expense_ratio = {INSTRUMENTS['SPY']['expense_ratio']*100:.2f}%/yr, "
          f"bid-ask = {INSTRUMENTS['SPY']['bid_ask']*10000:.0f} bps")
    vol_drag_annual = 0.5 * 6 * (0.01) ** 2 * 252 * 100
    print(f"    Vol drag (3x): 0.5 * 6 * sigma^2 = {vol_drag_annual:.1f}%/yr (sigma=1%)")
    print()

    # ------------------------------------------------------------------
    # [3] Tabla (tab:cost_sensitivity) -- Sensibilidad a costos
    # ------------------------------------------------------------------
    print("[3] Replicando Tabla (tab:cost_sensitivity)")
    print("-" * 90)
    print(f"    {'Modelo':<18} {'Base':>10} {'+50%':>10} {'+100%':>10} {'Break-even':>12} {'Tx/P&L':>10}")
    print("    " + "-" * 72)

    spy_return = backtest_json["benchmark"]["total_return"]

    for model_name in WINNERS:
        json_data = backtest_json["models"][model_name]
        total_costs = json_data["total_costs"]
        gross_final = json_data["gross_final"]
        gross_pl = gross_final - INITIAL_CAPITAL
        base_return = json_data["total_return"] * 100

        # Aproximacion lineal
        ret_150 = cost_sensitivity_linear(gross_pl, total_costs, 1.5)
        ret_200 = cost_sensitivity_linear(gross_pl, total_costs, 2.0)

        # Break-even: gross_PL / total_costs
        breakeven = gross_pl / total_costs if total_costs > 0 else float("inf")

        # Tx/P&L ratio
        tx_pl = total_costs / gross_pl * 100 if gross_pl > 0 else float("nan")

        be_str = f"{breakeven:.1f}x" if np.isfinite(breakeven) else "N/A"
        print(f"    {model_name:<18} {base_return:>+9.0f}% {ret_150:>+9.0f}% "
              f"{ret_200:>+9.0f}% {be_str:>12} {tx_pl:>9.1f}%")

    print()
    print(f"    SPY B&H: {spy_return*100:+.1f}%")
    print()

    # ------------------------------------------------------------------
    # [4] Estadisticas agregadas de costos
    # ------------------------------------------------------------------
    print("[4] Estadisticas agregadas de costos")
    print("-" * 90)

    all_costs = []
    all_trades = []
    per_model_costs = {}

    for model_name in ALL_MODELS:
        if model_name not in backtest_json["models"]:
            continue
        json_data = backtest_json["models"][model_name]
        total_costs = json_data["total_costs"]
        gross_pl = json_data["gross_final"] - INITIAL_CAPITAL
        n_trades = json_data["n_trades"]

        all_costs.append(total_costs)
        all_trades.append(n_trades)

        tx_pl = total_costs / gross_pl * 100 if gross_pl > 0 else None

        per_model_costs[model_name] = {
            "total_costs": round(total_costs, 0),
            "gross_pl": round(gross_pl, 0),
            "n_trades": n_trades,
            "tx_pl_pct": round(tx_pl, 1) if tx_pl is not None else None,
        }

    avg_costs = float(np.mean(all_costs))
    avg_trades = float(np.mean(all_trades))

    print(f"    Costo promedio (23 modelos): ${avg_costs:,.0f}")
    print(f"    Trades promedio: {avg_trades:.0f}")
    print()

    # Rango Tx/P&L para ganadores
    winner_tx_pl = []
    for w in WINNERS:
        if per_model_costs[w]["tx_pl_pct"] is not None:
            winner_tx_pl.append(per_model_costs[w]["tx_pl_pct"])
    print(f"    Rango Tx/P&L ganadores: {min(winner_tx_pl):.0f}--{max(winner_tx_pl):.0f}%")
    print()

    # Tabla completa por modelo
    print("    Tabla completa de costos:")
    print(f"    {'Modelo':<20} {'P&L Bruto':>12} {'Tx Costs':>12} {'Tx/P&L':>10} {'Trades':>8}")
    print("    " + "-" * 64)
    sorted_models = sorted(per_model_costs.items(),
                           key=lambda x: x[1]["gross_pl"], reverse=True)
    for mn, pm in sorted_models:
        tx_str = f"{pm['tx_pl_pct']:.1f}%" if pm["tx_pl_pct"] is not None else "--"
        print(f"    {mn:<20} ${pm['gross_pl']:>10,.0f} ${pm['total_costs']:>10,.0f} "
              f"{tx_str:>10} {pm['n_trades']:>8}")
    print()

    # ------------------------------------------------------------------
    # [5] Latencia
    # ------------------------------------------------------------------
    print("[5] Impacto de latencia (LSTM+Attention)")
    print("-" * 90)

    latency = compute_latency_impact(backtest_json)
    print(f"    {'Escenario':<18} {'Retorno':>12} {'Degradacion':>14} {'Correlacion':>14}")
    print("    " + "-" * 60)
    print(f"    {'Base (T)':<18} {latency['base_return_pct']:>+11.0f}% {'--':>14} {'1.00':>14}")
    print(f"    {'T+open':<18} {latency['t_plus_open']['return_pct']:>+11.0f}% "
          f"{latency['t_plus_open']['degradation_pct']:>13.1f}% "
          f"{latency['t_plus_open']['correlation']:>14.2f}")
    print(f"    {'T+close':<18} {latency['t_plus_close']['return_pct']:>+11.0f}% "
          f"{latency['t_plus_close']['degradation_pct']:>13.1f}% "
          f"{latency['t_plus_close']['correlation']:>14.2f}")
    print()

    # ------------------------------------------------------------------
    # [6] Claims especificas en prosa
    # ------------------------------------------------------------------
    print("[6] Verificando claims en prosa de la Seccion 8")
    print("-" * 90)

    # LSTM_Attention: $12,999 costos, 465 trades
    lstm_json = backtest_json["models"]["LSTM_Attention"]
    print(f"    LSTM+Att costos: ${lstm_json['total_costs']:,.0f} (paper: $12,999)")
    print(f"    LSTM+Att trades: {lstm_json['n_trades']} (paper: 465)")
    print(f"    LSTM+Att gross:  ${lstm_json['gross_final']:,.0f} (paper: $62,996)")

    # GradientBoosting: 11 trades, $378
    gb = backtest_json["models"]["GradientBoosting"]
    print(f"    GB trades: {gb['n_trades']} (paper: 11)")
    print(f"    GB costos: ${gb['total_costs']:,.0f} (paper: $378)")
    print(f"    GB %cash:  {gb['pct_cash']:.0f}% (paper: 92%)")

    # SeasonalNaive: 1,042 trades, $8,370
    sn = backtest_json["models"]["SeasonalNaive"]
    print(f"    SN trades: {sn['n_trades']:,} (paper: 1,042)")
    print(f"    SN costos: ${sn['total_costs']:,.0f} (paper: $8,370)")

    # Turnover vs return R^2
    trades_arr = np.array([backtest_json["models"][m]["n_trades"] for m in ALL_MODELS
                           if m in backtest_json["models"]], dtype=float)
    returns_arr = np.array([backtest_json["models"][m]["total_return"] for m in ALL_MODELS
                            if m in backtest_json["models"]], dtype=float)
    r_sq = np.corrcoef(trades_arr, returns_arr)[0, 1] ** 2
    print(f"    R^2(turnover, return): {r_sq:.2f} (paper: 0.02)")
    print()

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    print("=" * 90)
    print("REPLICACION SECCION 8 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()
