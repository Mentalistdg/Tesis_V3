# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION DE LA SECCION 6: GESTION DE RIESGO
================================================================================
Este script replica todos los valores numericos de la Seccion 6 del paper
"Prediccion de Retornos del S&P 500 mediante Aprendizaje Automatico..."

Lee datos de:
  - results/backtest_detail.pkl   (posiciones diarias, retornos de estrategia)
  - models/trained_artifacts.pkl  (retornos forward del mercado, risk-free)

Tablas replicadas:
  - Tabla (tab:sharpe_comparison)  -- Sharpe: 4 ganadores + SPY
  - Tabla (tab:sortino_comparison) -- Sortino
  - Tabla (tab:calmar_comparison)  -- Calmar
  - Tabla (tab:var_cvar)           -- VaR/CVaR a 30 y 90 dias (rolling windows)
  - Tabla (tab:drawdown_detail)    -- Pico, valle, duracion, recuperacion
  - Tabla (tab:capture_ratios)     -- Upside/downside capture

Numeros en prosa replicados:
  - Retorno y volatilidad anualizados
  - VaR diario al 95% y 99%
  - 710 dias alcistas, 589 bajistas, 3 neutros

Genera:
  - results/risk_metrics_detail.json

USAGE:
    python paper/scripts/replicar_seccion6_riesgo.py

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

WINNERS = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM"]


# ============================================================================
# [1] CARGA DE DATOS
# ============================================================================

def load_data():
    """Carga backtest_detail.pkl y trained_artifacts.pkl."""
    with open(os.path.join(RESULTS_DIR, "backtest_detail.pkl"), "rb") as f:
        detail = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return detail, artifacts


# ============================================================================
# [2] RATIOS: SHARPE, SORTINO, CALMAR
# ============================================================================

def compute_ratios(returns, rf):
    """Calcula Sharpe, Sortino, Calmar a partir de retornos diarios."""
    n_years = len(returns) / 252
    equity = np.cumprod(1 + returns)
    total_ret = equity[-1] - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252

    # Sharpe
    sharpe = (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0

    # Sortino (desviacion downside respecto a rf diario)
    rf_daily = np.mean(rf)
    downside = np.minimum(returns - rf_daily, 0)
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    sortino = (annual_ret - rf_annual) / downside_std if downside_std > 0 else 0

    # Calmar
    running_max = np.maximum.accumulate(equity)
    max_dd = np.max((running_max - equity) / running_max)
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "annual_return_pct": float(annual_ret * 100),
        "annual_vol_pct": float(annual_vol * 100),
        "total_return_pct": float(total_ret * 100),
    }


# ============================================================================
# [3] VaR / CVaR DIARIO
# ============================================================================

def compute_var_cvar(returns):
    """
    Calcula Value-at-Risk y Conditional VaR (Expected Shortfall) diarios.

    VaR 95% = percentil 5 de los retornos
    CVaR 95% = media de retornos por debajo del VaR 95%
    """
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = np.mean(returns[returns <= var_95])
    cvar_99 = np.mean(returns[returns <= var_99])
    worst_day = np.min(returns)

    return {
        "var_95_daily_pct": float(var_95 * 100),
        "cvar_95_daily_pct": float(cvar_95 * 100),
        "var_99_daily_pct": float(var_99 * 100),
        "cvar_99_daily_pct": float(cvar_99 * 100),
        "worst_day_pct": float(worst_day * 100),
    }


# ============================================================================
# [4] VaR / CVaR ROLLING (30 y 90 dias)
# ============================================================================

def compute_var_cvar_rolling(returns, window):
    """
    Calcula VaR/CVaR usando retornos acumulados reales en ventanas rolling.

    Para cada posicion i desde window-1 hasta el final, calcula el retorno
    acumulado de los ultimos 'window' dias: prod(1 + r[i-window+1:i+1]) - 1.
    Luego VaR 95% = percentil 5, CVaR 95% = media por debajo del VaR.

    NO usa la aproximacion sqrt(T) -- usa ventanas reales.
    """
    n = len(returns)
    if n < window:
        return {"var_95_pct": None, "cvar_95_pct": None, "n_windows": 0}

    # Retornos acumulados rolling via log-sum (eficiente)
    log_returns = np.log(1 + returns)
    cum_log = np.cumsum(log_returns)

    rolling_cum = np.empty(n - window + 1)
    rolling_cum[0] = cum_log[window - 1]
    rolling_cum[1:] = cum_log[window:] - cum_log[:n - window]
    rolling_cum = np.exp(rolling_cum) - 1

    var_95 = np.percentile(rolling_cum, 5)
    tail = rolling_cum[rolling_cum <= var_95]
    cvar_95 = np.mean(tail) if len(tail) > 0 else var_95

    return {
        "var_95_pct": float(var_95 * 100),
        "cvar_95_pct": float(cvar_95 * 100),
        "n_windows": int(len(rolling_cum)),
    }


# ============================================================================
# [5] DRAWDOWN DETALLADO
# ============================================================================

def compute_drawdown_details(equity, dates):
    """
    Calcula detalles del drawdown maximo: fecha pico, fecha valle,
    duracion en dias de trading, y dias hasta recuperacion.
    """
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = dates[max_dd_idx] if dates is not None else None

    # Pico antes del valle
    peak_idx = np.argmax(equity[:max_dd_idx + 1])
    peak_date = dates[peak_idx] if dates is not None else None

    # Duracion: pico a valle (dias de trading)
    duration_days = max_dd_idx - peak_idx

    # Recuperacion: valle hasta nuevo maximo
    recovery_days = -1  # -1 = no recuperado
    recovery_date = None
    peak_value = equity[peak_idx]
    for i in range(max_dd_idx + 1, len(equity)):
        if equity[i] >= peak_value:
            recovery_days = i - max_dd_idx
            recovery_date = dates[i] if dates is not None else None
            break

    return {
        "max_dd_pct": float(max_dd * 100),
        "trough_date": max_dd_date,
        "peak_date": peak_date,
        "duration_trading_days": int(duration_days),
        "recovery_trading_days": int(recovery_days),
        "recovery_date": recovery_date,
    }


# ============================================================================
# [6] CAPTURE RATIOS (Upside/Downside)
# ============================================================================

def compute_capture_ratios(strategy_returns, market_returns, positions):
    """
    Calcula upside y downside capture ratios.

    Upside Capture = mean(strategy en dias alcistas) / mean(mercado en dias alcistas) * 100
    Downside Capture = mean(strategy en dias bajistas) / mean(mercado en dias bajistas) * 100
    Capture Ratio = Upside / Downside (mayor = mejor)
    """
    up_mask = market_returns > 0
    down_mask = market_returns < 0

    n_up = int(np.sum(up_mask))
    n_down = int(np.sum(down_mask))

    if n_up > 0 and np.mean(market_returns[up_mask]) != 0:
        upside = (np.mean(strategy_returns[up_mask]) / np.mean(market_returns[up_mask])) * 100
    else:
        upside = 0.0

    if n_down > 0 and np.mean(market_returns[down_mask]) != 0:
        downside = (np.mean(strategy_returns[down_mask]) / np.mean(market_returns[down_mask])) * 100
    else:
        downside = 0.0

    ratio = upside / downside if downside != 0 else float("inf")

    return {
        "upside_capture_pct": float(upside),
        "downside_capture_pct": float(downside),
        "capture_ratio": float(ratio),
        "n_up_days": n_up,
        "n_down_days": n_down,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("REPLICACION DE LA SECCION 6: GESTION DE RIESGO")
    print("=" * 90)
    print()

    # ------------------------------------------------------------------
    # [1] Carga de datos
    # ------------------------------------------------------------------
    print("[1] Cargando datos del pipeline...")
    detail, artifacts = load_data()
    metadata = artifacts["metadata"]

    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    raw_dates = metadata.get("dates_test", None)
    if raw_dates is not None:
        test_dates = [str(d)[:10] for d in raw_dates[:len(fwd_test)]]
    else:
        test_dates = None

    n_test = len(fwd_test)
    print(f"    Periodo de prueba: {n_test} dias")
    if test_dates:
        print(f"    Desde: {test_dates[0]} hasta: {test_dates[-1]}")

    # Dias alcistas/bajistas/neutros
    n_up = int(np.sum(fwd_test > 0))
    n_down = int(np.sum(fwd_test < 0))
    n_neutral = int(np.sum(fwd_test == 0))
    print(f"    Dias alcistas: {n_up}, bajistas: {n_down}, neutros: {n_neutral}")
    print()

    results = {}

    # ------------------------------------------------------------------
    # [2] SPY B&H (benchmark)
    # ------------------------------------------------------------------
    print("[2] Metricas SPY Buy & Hold...")
    print("-" * 90)

    spy_equity = np.cumprod(1 + fwd_test)
    spy_ratios = compute_ratios(fwd_test, rf_test)
    spy_var = compute_var_cvar(fwd_test)
    spy_dd = compute_drawdown_details(spy_equity, test_dates)
    spy_var_30d = compute_var_cvar_rolling(fwd_test, 30)
    spy_var_90d = compute_var_cvar_rolling(fwd_test, 90)
    spy_capture = {
        "upside_capture_pct": 100.0, "downside_capture_pct": 100.0,
        "capture_ratio": 1.0, "n_up_days": n_up, "n_down_days": n_down,
    }

    results["SPY_BH"] = {
        "ratios": spy_ratios, "var_cvar": spy_var,
        "var_cvar_30d": spy_var_30d, "var_cvar_90d": spy_var_90d,
        "drawdown": spy_dd, "capture": spy_capture,
    }

    print(f"    Sharpe: {spy_ratios['sharpe']:.3f}, Sortino: {spy_ratios['sortino']:.3f}, "
          f"Calmar: {spy_ratios['calmar']:.3f}")
    print(f"    MaxDD: {spy_dd['max_dd_pct']:.1f}%")
    print(f"    VaR95 diario: {spy_var['var_95_daily_pct']:.2f}%, "
          f"CVaR95: {spy_var['cvar_95_daily_pct']:.2f}%")
    print()

    # ------------------------------------------------------------------
    # [3] 4 Modelos ganadores
    # ------------------------------------------------------------------
    for model_name in WINNERS:
        print(f"[3] Metricas {model_name}...")
        print("-" * 90)

        model_detail = detail["models"][model_name]
        strat_returns = np.array(model_detail["strategy_returns"])
        positions = np.array(model_detail["positions"])

        n = min(len(strat_returns), len(fwd_test))
        strat_returns = strat_returns[:n]
        positions = positions[:n]
        market_returns = fwd_test[:n]
        rf = rf_test[:n]
        dates = test_dates[:n] if test_dates else None

        # Ratios
        ratios = compute_ratios(strat_returns, rf)
        print(f"    Sharpe: {ratios['sharpe']:.3f}, Sortino: {ratios['sortino']:.3f}, "
              f"Calmar: {ratios['calmar']:.3f}")
        print(f"    Retorno anual: {ratios['annual_return_pct']:.1f}%, "
              f"Volatilidad anual: {ratios['annual_vol_pct']:.1f}%")

        # VaR/CVaR diario
        var_cvar = compute_var_cvar(strat_returns)
        print(f"    VaR95 diario: {var_cvar['var_95_daily_pct']:.2f}%, "
              f"CVaR95: {var_cvar['cvar_95_daily_pct']:.2f}%, "
              f"Peor dia: {var_cvar['worst_day_pct']:.2f}%")

        # VaR/CVaR rolling 30d y 90d
        var_30d = compute_var_cvar_rolling(strat_returns, 30)
        var_90d = compute_var_cvar_rolling(strat_returns, 90)
        print(f"    VaR30d 95%: {var_30d['var_95_pct']:.2f}%, "
              f"CVaR30d: {var_30d['cvar_95_pct']:.2f}%")
        print(f"    VaR90d 95%: {var_90d['var_95_pct']:.2f}%, "
              f"CVaR90d: {var_90d['cvar_95_pct']:.2f}%")

        # Drawdown detallado
        equity = np.cumprod(1 + strat_returns)
        dd = compute_drawdown_details(equity, dates)
        print(f"    MaxDD: {dd['max_dd_pct']:.1f}%, "
              f"Duracion: {dd['duration_trading_days']}d, "
              f"Recuperacion: {dd['recovery_trading_days']}d")
        if dd["peak_date"]:
            print(f"    Pico: {dd['peak_date']}, Valle: {dd['trough_date']}")

        # Capture ratios
        capture = compute_capture_ratios(strat_returns, market_returns, positions)
        print(f"    Upside Capture: {capture['upside_capture_pct']:.1f}%, "
              f"Downside Capture: {capture['downside_capture_pct']:.1f}%, "
              f"Ratio: {capture['capture_ratio']:.2f}")

        results[model_name] = {
            "ratios": ratios, "var_cvar": var_cvar,
            "var_cvar_30d": var_30d, "var_cvar_90d": var_90d,
            "drawdown": dd, "capture": capture,
        }
        print()

    # ------------------------------------------------------------------
    # [4] Tablas de resumen
    # ------------------------------------------------------------------
    print("=" * 90)
    print("TABLA (tab:sharpe_comparison / tab:sortino_comparison / tab:calmar_comparison)")
    print("=" * 90)
    print(f"  {'Modelo':<20} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD%':>8}")
    print("  " + "-" * 56)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        print(f"  {label:<20} {r['ratios']['sharpe']:>8.3f} {r['ratios']['sortino']:>8.3f} "
              f"{r['ratios']['calmar']:>8.3f} {r['drawdown']['max_dd_pct']:>7.1f}%")

    print()
    print("=" * 90)
    print("TABLA (tab:var_cvar) -- VaR/CVaR Rolling")
    print("=" * 90)
    print(f"  {'Modelo':<20} {'VaR30d%':>9} {'CVaR30d%':>10} {'VaR90d%':>9} {'CVaR90d%':>10}")
    print("  " + "-" * 62)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        v30 = r["var_cvar_30d"]
        v90 = r["var_cvar_90d"]
        print(f"  {label:<20} {v30['var_95_pct']:>9.2f} {v30['cvar_95_pct']:>10.2f} "
              f"{v90['var_95_pct']:>9.2f} {v90['cvar_95_pct']:>10.2f}")

    print()
    print("=" * 90)
    print("TABLA (tab:drawdown_detail)")
    print("=" * 90)
    print(f"  {'Modelo':<20} {'MaxDD%':>8} {'Pico':>12} {'Valle':>12} {'Dur(d)':>8} {'Recov(d)':>9}")
    print("  " + "-" * 72)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        d = r["drawdown"]
        rec = str(d["recovery_trading_days"]) if d["recovery_trading_days"] > 0 else "N/R"
        peak = d["peak_date"] or "?"
        trough = d["trough_date"] or "?"
        print(f"  {label:<20} {d['max_dd_pct']:>7.1f}% {peak:>12} {trough:>12} "
              f"{d['duration_trading_days']:>8} {rec:>9}")

    print()
    print("=" * 90)
    print("TABLA (tab:capture_ratios)")
    print("=" * 90)
    print(f"  {'Modelo':<20} {'Up Cap%':>9} {'Dn Cap%':>9} {'Ratio':>7}")
    print("  " + "-" * 48)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        c = r["capture"]
        print(f"  {label:<20} {c['upside_capture_pct']:>9.1f} {c['downside_capture_pct']:>9.1f} "
              f"{c['capture_ratio']:>7.2f}")

    # ------------------------------------------------------------------
    # [5] Guardar JSON
    # ------------------------------------------------------------------
    output_path = os.path.join(RESULTS_DIR, "risk_metrics_detail.json")

    def clean_nan(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(clean_nan(results), f, indent=2)

    print()
    print(f"Guardado: {output_path}")
    print()
    print("=" * 90)
    print("REPLICACION SECCION 6 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()
