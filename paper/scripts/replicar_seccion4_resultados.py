# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION DE LA SECCION 4: RESULTADOS EMPIRICOS
================================================================================
Este script replica todos los valores numericos de la Seccion 4 del paper
"Prediccion de Retornos del S&P 500 mediante Aprendizaje Automatico..."

Lee datos de:
  - results/final_long_only_backtest.json  (rendimiento, posiciones, costos)
  - results/backtest_detail.pkl            (posiciones diarias, retornos)
  - models/trained_artifacts.pkl           (retornos forward, risk-free)
  - data/BLOOMBERG_RAW_DATA.csv            (precios SPY y bonos para 60/40)

Tablas replicadas:
  - Tabla 5 (tab:benchmarks)    -- UPRO 3x B&H, 60/40 Portfolio, SPY B&H
  - Tabla 6 (tab:trading_stats) -- Win rate, avg win/loss, profit factor (Top 5)

Numeros en prosa replicados:
  - 4/23 modelos superan SPY, 16 con Sharpe positivo, rango 432pp
  - Retornos individuales de los 23 modelos
  - Distribuciones de posiciones (% UPRO, SPY, Cash)
  - N trades por modelo

USAGE:
    python paper/scripts/replicar_seccion4_resultados.py

================================================================================
Author: David Gonzalez Canon
================================================================================
"""

import numpy as np
import pandas as pd
import pickle
import json
import os

SEED = 42
np.random.seed(SEED)

# ============================================================================
# CONFIGURACION
# ============================================================================
# Rutas relativas al directorio raiz del repositorio
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Parametros de costos (identicos a optimize_and_backtest.py)
INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},
    'SPY':  {'expense_ratio': 0.0009, 'bid_ask': 0.0002},
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},
}

INITIAL_CAPITAL = 10000

# Los 5 modelos del Top 5 (para Tabla 6 / tab:trading_stats)
TOP5 = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM", "DLinear"]

# Los 4 ganadores que superan SPY B&H
WINNERS = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM"]


# ============================================================================
# [1] CARGA DE DATOS
# ============================================================================

def load_all_data():
    """Carga todos los archivos necesarios del pipeline."""
    detail_path = os.path.join(RESULTS_DIR, "backtest_detail.pkl")
    with open(detail_path, "rb") as f:
        detail = pickle.load(f)

    artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    json_path = os.path.join(RESULTS_DIR, "final_long_only_backtest.json")
    with open(json_path, "r") as f:
        backtest_json = json.load(f)

    bloomberg_path = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
    bloomberg = pd.read_csv(bloomberg_path, parse_dates=["date"])

    return detail, artifacts, backtest_json, bloomberg


# ============================================================================
# [2] METRICAS AUXILIARES
# ============================================================================

def compute_benchmark_metrics(returns, rf, name):
    """Calcula Sharpe, Sortino, Calmar, MaxDD para un benchmark."""
    n = len(returns)
    n_years = n / 252.0

    equity = np.cumprod(1 + returns)
    total_ret = float(equity[-1] - 1)
    annual_ret = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    annual_vol = float(np.std(returns) * np.sqrt(252))
    rf_annual = float(np.mean(rf) * 252)
    sharpe = float((annual_ret - rf_annual) / annual_vol) if annual_vol > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / running_max
    max_dd = float(np.max(dd))
    calmar = float(annual_ret / max_dd) if max_dd > 0 else 0.0

    # Sortino
    rf_daily = np.mean(rf)
    downside = np.minimum(returns - rf_daily, 0)
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    sortino = float((annual_ret - rf_annual) / downside_std) if downside_std > 0 else 0.0

    return {
        "name": name,
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(annual_ret * 100, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 4),
        "annual_vol_pct": round(annual_vol * 100, 2),
    }


# ============================================================================
# [3] TRADING STATISTICS (para Tabla 6 / tab:trading_stats)
# ============================================================================

def compute_trading_stats(positions, strategy_returns, rf):
    """
    Calcula estadisticas de trading sobre dias activos (posicion != 0).

    Win Rate, Avg Win, Avg Loss y Profit Factor se computan sobre exceso de
    retorno diario (strategy_return - risk_free) en dias con posicion activa.
    Esto coincide con la nota al pie de la Tabla 6: "calculados sobre dias
    activos (posicion != 0)".
    """
    active_mask = positions != 0
    n_active_days = int(np.sum(active_mask))
    n_position_changes = int(np.sum(np.diff(positions) != 0))

    if n_active_days > 0:
        active_excess = strategy_returns[active_mask] - rf[active_mask]
        daily_winners = active_excess[active_excess > 0]
        daily_losers = active_excess[active_excess <= 0]
        daily_win_rate = len(daily_winners) / n_active_days * 100.0
        daily_avg_win = float(np.mean(daily_winners) * 100) if len(daily_winners) > 0 else 0.0
        daily_avg_loss = float(np.mean(daily_losers) * 100) if len(daily_losers) > 0 else 0.0
        daily_gross_wins = float(np.sum(daily_winners))
        daily_gross_losses = float(np.abs(np.sum(daily_losers)))
        daily_pf = daily_gross_wins / daily_gross_losses if daily_gross_losses > 0 else float("inf")
    else:
        daily_win_rate = 0.0
        daily_avg_win = 0.0
        daily_avg_loss = 0.0
        daily_pf = 0.0

    return {
        "n_position_changes": n_position_changes,
        "n_active_days": n_active_days,
        "daily_win_rate_pct": float(daily_win_rate),
        "daily_avg_win_pct": float(daily_avg_win),
        "daily_avg_loss_pct": float(daily_avg_loss),
        "daily_profit_factor": float(daily_pf),
    }


# ============================================================================
# [4] UPRO BUY & HOLD BENCHMARK (para Tabla 5 / tab:benchmarks)
# ============================================================================

def compute_upro_buyhold(market_returns, rf, vol_window=21):
    """
    Calcula el benchmark UPRO 3x Buy & Hold.

    Formula diaria: rf + 3*(mkt - rf) - expense_daily - vol_drag - bid_ask_entrada.
    El vol_drag usa la volatilidad realizada rolling (21d, backward-only), igual que
    el tramo UPRO de la estrategia, para una comparacion consistente.
    Solo se cobra 1 trade de entrada (bid-ask 5bps) al inicio.
    """
    expense_daily = INSTRUMENTS['UPRO']['expense_ratio'] / 252
    bid_ask = INSTRUMENTS['UPRO']['bid_ask']

    market_returns = np.asarray(market_returns, dtype=float)
    n = len(market_returns)

    # Volatilidad realizada backward-only (mismo metodo que la estrategia)
    realized_vol = np.full(n, 0.01)
    for i in range(n):
        w = market_returns[max(0, i - vol_window):i]
        if len(w) >= 5:
            realized_vol[i] = np.std(w)
    vol_drag = 0.5 * 6 * realized_vol ** 2

    upro_gross = rf + 3.0 * (market_returns - rf)
    upro_returns = upro_gross - expense_daily - vol_drag
    upro_returns[0] -= bid_ask  # Solo 1 trade de entrada

    return compute_benchmark_metrics(upro_returns, rf, "UPRO_3x_BH")


# ============================================================================
# [5] 60/40 PORTFOLIO BENCHMARK (para Tabla 5 / tab:benchmarks)
# ============================================================================

def compute_6040_portfolio(spy_prices, bond_prices, rf, dates):
    """
    Calcula el benchmark 60/40 Portfolio con rebalanceo mensual.

    60% SPY + 40% LF98TRUU Index (Bloomberg US Aggregate Bond Total Return).
    Pesos se reajustan a 60/40 al inicio de cada mes calendario.
    """
    spy_ret = np.diff(spy_prices) / spy_prices[:-1]
    bond_ret = np.diff(bond_prices) / bond_prices[:-1]
    dt = pd.to_datetime(dates[1:])
    n = len(spy_ret)

    # Rebalanceo mensual
    eq_spy = 0.6
    eq_bond = 0.4
    port_eq = np.zeros(n)
    cur_month = dt[0].month

    for i in range(n):
        if dt[i].month != cur_month:
            tv = eq_spy + eq_bond
            eq_spy = 0.6 * tv
            eq_bond = 0.4 * tv
            cur_month = dt[i].month
        eq_spy *= (1 + spy_ret[i] - 0.0009 / 252)  # pata SPY neta de expense
        eq_bond *= (1 + bond_ret[i])
        port_eq[i] = eq_spy + eq_bond

    # Convertir curva de equity a retornos diarios
    port_returns = np.diff(np.concatenate([[1.0], port_eq])) / np.concatenate([[1.0], port_eq[:-1]])
    rf_trimmed = rf[:n] if len(rf) > n else rf

    return compute_benchmark_metrics(port_returns, rf_trimmed, "60_40_Portfolio")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("REPLICACION DE LA SECCION 4: RESULTADOS EMPIRICOS")
    print("=" * 90)
    print()

    # ------------------------------------------------------------------
    # [1] Carga de datos
    # ------------------------------------------------------------------
    print("[1] Cargando datos del pipeline...")
    detail, artifacts, backtest_json, bloomberg = load_all_data()

    metadata = artifacts["metadata"]
    # Los modelos tienen n-1 elementos (1301) vs forward_returns (1302)
    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    fwd_test_full = np.array(metadata["forward_returns_test"])
    rf_test_full = np.array(metadata["risk_free_test"])
    n_test = len(fwd_test)
    print(f"    Periodo de prueba: {n_test} dias (modelos), {len(fwd_test_full)} dias (benchmarks)")
    print()

    models = backtest_json["models"]
    benchmark = backtest_json["benchmark"]

    # ------------------------------------------------------------------
    # [2] Numeros en prosa -- verificacion general
    # ------------------------------------------------------------------
    print("[2] Verificando numeros en prosa de la Seccion 4...")
    print("-" * 90)

    # (a) SPY B&H
    spy_ret = benchmark["total_return"] * 100
    print(f"    SPY B&H retorno total: {spy_ret:+.1f}%")

    # (b) Modelos que superan SPY
    beating = [m for m, d in models.items() if d["total_return"] > benchmark["total_return"]]
    print(f"    Modelos que superan SPY: {len(beating)}/23 -> {beating}")

    # (c) Sharpe positivo
    n_positive_sharpe = sum(1 for d in models.values() if d["sharpe"] > 0)
    print(f"    Modelos con Sharpe > 0: {n_positive_sharpe}/23")

    # (d) Rango de retornos
    all_returns = [d["total_return"] * 100 for d in models.values()]
    ret_range = max(all_returns) - min(all_returns)
    print(f"    Rango de retornos: {ret_range:.0f} pp")

    # (e) Mejor y peor modelo
    best_model = max(models, key=lambda m: models[m]["total_return"])
    worst_model = min(models, key=lambda m: models[m]["total_return"])
    print(f"    Mejor: {best_model} ({models[best_model]['total_return']*100:+.0f}%)")
    print(f"    Peor:  {worst_model} ({models[worst_model]['total_return']*100:+.0f}%)")
    print()

    # (f) Retornos individuales de todos los modelos
    print("    Retornos individuales:")
    print(f"    {'Modelo':<22} {'Ret.Total':>10} {'Sharpe':>8} {'MaxDD':>8} {'%3x':>6} {'%1x':>6} {'%Cash':>6} {'Trades':>7}")
    print("    " + "-" * 80)
    for name in sorted(models.keys(), key=lambda m: models[m]["total_return"], reverse=True):
        m = models[name]
        marker = " ***" if name in WINNERS else ""
        print(f"    {name:<22} {m['total_return']*100:>+9.1f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']*100:>7.1f}% {m['pct_3x']:>5.1f}% {m['pct_1x']:>5.1f}% "
              f"{m['pct_cash']:>5.1f}% {m['n_trades']:>7}{marker}")
    print()

    # ------------------------------------------------------------------
    # [3] Tabla 6 -- Trading Statistics (Top 5)
    # ------------------------------------------------------------------
    print("[3] Replicando Tabla 6 (tab:trading_stats) -- Trading Statistics Top 5")
    print("-" * 90)
    print(f"    {'Modelo':<20} {'Trades':>8} {'Dias Actv':>10} {'Win Rate':>10} "
          f"{'Avg Win':>10} {'Avg Loss':>10} {'PF':>8}")
    print("    " + "-" * 78)

    for model_name in TOP5:
        if model_name not in detail["models"]:
            print(f"    WARNING: {model_name} no encontrado en backtest_detail.pkl")
            continue

        mdata = detail["models"][model_name]
        positions = np.array(mdata["positions"])
        strategy_returns = np.array(mdata["strategy_returns"])

        stats = compute_trading_stats(positions, strategy_returns, rf_test)

        print(f"    {model_name:<20} {stats['n_position_changes']:>8d} "
              f"{stats['n_active_days']:>10d} "
              f"{stats['daily_win_rate_pct']:>9.1f}% "
              f"{stats['daily_avg_win_pct']:>+9.1f}% "
              f"{stats['daily_avg_loss_pct']:>+9.1f}% "
              f"{stats['daily_profit_factor']:>8.2f}")
    print()

    # ------------------------------------------------------------------
    # [4] Tabla 5 -- Benchmarks (UPRO 3x B&H)
    # ------------------------------------------------------------------
    print("[4] Replicando Tabla 5 (tab:benchmarks) -- UPRO 3x Buy & Hold")
    print("-" * 90)

    upro_metrics = compute_upro_buyhold(fwd_test_full, rf_test_full)

    print(f"    UPRO 3x B&H (rf + 3*(mkt-rf) - expense - 1 trade):")
    print(f"      Retorno: {upro_metrics['total_return_pct']:+.1f}%")
    print(f"      Sharpe:  {upro_metrics['sharpe']:.4f}")
    print(f"      MaxDD:   {upro_metrics['max_drawdown_pct']:.1f}%")
    print(f"      Calmar:  {upro_metrics['calmar']:.4f}")
    print(f"      Sortino: {upro_metrics['sortino']:.4f}")
    print()

    # ------------------------------------------------------------------
    # [5] Tabla 5 -- Benchmarks (60/40 Portfolio)
    # ------------------------------------------------------------------
    print("[5] Replicando Tabla 5 (tab:benchmarks) -- 60/40 Portfolio")
    print("-" * 90)

    # Filtrar Bloomberg al periodo de prueba
    test_dates = pd.to_datetime(metadata["dates_test"])
    bb_test = bloomberg[bloomberg["date"].isin(test_dates)].sort_values("date").reset_index(drop=True)
    spy_prices = pd.Series(bb_test["SPY US Equity (CLOSE)"].values).ffill().values
    bond_prices = pd.Series(bb_test["LF98TRUU Index"].values).ffill().values
    bb_dates = bb_test["date"].values

    portfolio_6040 = compute_6040_portfolio(spy_prices, bond_prices, rf_test_full, bb_dates)

    print(f"    60/40 Portfolio (60% SPY + 40% LF98TRUU, rebalanceo mensual):")
    print(f"      Retorno: {portfolio_6040['total_return_pct']:+.1f}%")
    print(f"      Sharpe:  {portfolio_6040['sharpe']:.4f}")
    print(f"      MaxDD:   {portfolio_6040['max_drawdown_pct']:.1f}%")
    print(f"      Calmar:  {portfolio_6040['calmar']:.4f}")
    print(f"      Sortino: {portfolio_6040['sortino']:.4f}")
    print()

    # ------------------------------------------------------------------
    # [6] Tabla 5 -- Benchmarks -- SPY B&H
    # ------------------------------------------------------------------
    print("[6] SPY Buy & Hold (del JSON)")
    print("-" * 90)
    # SPY B&H neto del expense ratio (0.09%/anio), consistente con la estrategia.
    spy_bh_net_full = np.asarray(fwd_test_full, dtype=float) - 0.0009 / 252
    spy_metrics = compute_benchmark_metrics(spy_bh_net_full, rf_test_full, "SPY_BH")
    print(f"    SPY B&H:")
    print(f"      Retorno: {spy_metrics['total_return_pct']:+.1f}%")
    print(f"      Sharpe:  {spy_metrics['sharpe']:.4f}")
    print(f"      MaxDD:   {spy_metrics['max_drawdown_pct']:.1f}%")
    print(f"      Calmar:  {spy_metrics['calmar']:.4f}")
    print(f"      Sortino: {spy_metrics['sortino']:.4f}")
    print()

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    print("=" * 90)
    print("RESUMEN TABLA 5 (tab:benchmarks)")
    print("=" * 90)
    print(f"  {'Benchmark':<25} {'Retorno':>10} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>8} {'Calmar':>8}")
    print("  " + "-" * 72)
    for bm in [spy_metrics, upro_metrics, portfolio_6040]:
        print(f"  {bm['name']:<25} {bm['total_return_pct']:>+9.1f}% {bm['sharpe']:>8.4f} "
              f"{bm['sortino']:>9.4f} {bm['max_drawdown_pct']:>7.1f}% {bm['calmar']:>8.4f}")

    print()
    print("=" * 90)
    print("REPLICACION SECCION 4 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()
