"""
Calculate all risk metrics from backtest data for the paper's Section 6.

Reads backtest_detail.pkl and trained_artifacts.pkl to compute:
- Sharpe, Sortino, Calmar (verified against existing)
- VaR 95%, VaR 99%, CVaR 95%, worst single day (daily)
- VaR/CVaR at 30-day and 90-day rolling horizons (actual rolling windows, no sqrt(T))
- MaxDD, date of MaxDD trough, duration, recovery time
- Beta vs SPY, Tracking Error, Information Ratio
- Capture Ratios (upside/downside)

Outputs: results/risk_metrics_detail.json
"""

import numpy as np
import pickle
import json
import os

# 3 levels up: seccion_6_riesgo -> scripts -> paper -> repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

WINNERS = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM"]


def load_data():
    """Load backtest detail and trained artifacts."""
    with open(os.path.join(RESULTS_DIR, "backtest_detail.pkl"), "rb") as f:
        detail = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    return detail, artifacts


def compute_drawdown_details(equity, dates):
    """Compute MaxDD, date, duration, and recovery time."""
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    # Max drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = dates[max_dd_idx] if dates is not None else None

    # Find peak before max drawdown trough
    peak_idx = np.argmax(equity[:max_dd_idx + 1])
    peak_date = dates[peak_idx] if dates is not None else None

    # Duration: peak to trough (trading days)
    duration_days = max_dd_idx - peak_idx

    # Recovery: trough to new high (or not recovered)
    recovery_days = None
    recovery_date = None
    peak_value = equity[peak_idx]
    for i in range(max_dd_idx + 1, len(equity)):
        if equity[i] >= peak_value:
            recovery_days = i - max_dd_idx
            recovery_date = dates[i] if dates is not None else None
            break

    if recovery_days is None:
        recovery_days = -1  # Not recovered by end of test

    return {
        "max_dd_pct": float(max_dd * 100),
        "trough_date": max_dd_date,
        "peak_date": peak_date,
        "duration_trading_days": int(duration_days),
        "recovery_trading_days": int(recovery_days),
        "recovery_date": recovery_date,
    }


def compute_var_cvar(returns):
    """Compute VaR and CVaR at multiple confidence levels (daily)."""
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


def compute_var_cvar_rolling(returns, window):
    """Compute VaR and CVaR using actual rolling cumulative returns.

    For each position i from window-1 to len-1, computes the cumulative
    return over the trailing 'window' days: prod(1 + r[i-window+1:i+1]) - 1.
    Then VaR 95% = 5th percentile and CVaR 95% = mean below VaR.

    No sqrt(T) approximation -- uses the actual rolling window returns.
    """
    n = len(returns)
    if n < window:
        return {
            "var_95_pct": None,
            "cvar_95_pct": None,
            "n_windows": 0,
        }

    # Compute rolling cumulative returns using a sliding product
    # For efficiency, use log-sum approach
    log_returns = np.log(1 + returns)
    cum_log = np.cumsum(log_returns)

    # Rolling sum of log returns over 'window' days
    # For i >= window-1: sum = cum_log[i] - cum_log[i-window] (with i-window=-1 being 0)
    rolling_cum_returns = np.empty(n - window + 1)
    rolling_cum_returns[0] = cum_log[window - 1]
    rolling_cum_returns[1:] = cum_log[window:] - cum_log[:n - window]

    # Convert back from log to actual cumulative returns
    rolling_cum_returns = np.exp(rolling_cum_returns) - 1

    # VaR 95% = 5th percentile of rolling cumulative returns
    var_95 = np.percentile(rolling_cum_returns, 5)

    # CVaR 95% = mean of returns at or below VaR
    tail = rolling_cum_returns[rolling_cum_returns <= var_95]
    cvar_95 = np.mean(tail) if len(tail) > 0 else var_95

    return {
        "var_95_pct": float(var_95 * 100),
        "cvar_95_pct": float(cvar_95 * 100),
        "n_windows": int(len(rolling_cum_returns)),
    }


def compute_capture_ratios(strategy_returns, market_returns, positions):
    """Compute upside/downside capture ratios and participation stats.

    Upside Capture = mean(strategy on up days) / mean(market on up days) * 100
    Downside Capture = mean(strategy on down days) / mean(market on down days) * 100
    Capture Ratio = Upside Capture / Downside Capture
    """
    up_mask = market_returns > 0
    down_mask = market_returns < 0

    n_up_days = int(np.sum(up_mask))
    n_down_days = int(np.sum(down_mask))

    # Upside capture
    if n_up_days > 0 and np.mean(market_returns[up_mask]) != 0:
        upside_capture = (np.mean(strategy_returns[up_mask])
                          / np.mean(market_returns[up_mask])) * 100
    else:
        upside_capture = 0.0

    # Downside capture
    if n_down_days > 0 and np.mean(market_returns[down_mask]) != 0:
        downside_capture = (np.mean(strategy_returns[down_mask])
                            / np.mean(market_returns[down_mask])) * 100
    else:
        downside_capture = 0.0

    # Capture ratio (higher is better: high upside, low downside)
    if downside_capture != 0:
        capture_ratio = upside_capture / downside_capture
    else:
        capture_ratio = float("inf")

    # Participation stats
    avg_position = float(np.mean(positions))

    # % of up days where strategy was long (position > 0)
    if n_up_days > 0:
        pct_long_on_up_days = float(np.mean(positions[up_mask] > 0) * 100)
    else:
        pct_long_on_up_days = 0.0

    # % of down days where strategy was long (position > 0)
    if n_down_days > 0:
        pct_long_on_down_days = float(np.mean(positions[down_mask] > 0) * 100)
    else:
        pct_long_on_down_days = 0.0

    return {
        "upside_capture_pct": float(upside_capture),
        "downside_capture_pct": float(downside_capture),
        "capture_ratio": float(capture_ratio),
        "avg_position": float(avg_position),
        "pct_long_on_up_days": float(pct_long_on_up_days),
        "pct_long_on_down_days": float(pct_long_on_down_days),
        "n_up_days": n_up_days,
        "n_down_days": n_down_days,
    }


def compute_beta_tracking(strategy_returns, market_returns):
    """Compute Beta, Tracking Error, and Information Ratio."""
    # Beta = Cov(Rp, Rm) / Var(Rm)
    cov = np.cov(strategy_returns, market_returns)[0, 1]
    var_m = np.var(market_returns, ddof=1)
    beta = cov / var_m if var_m > 0 else 0

    # Tracking Error = std(Rp - Rm) * sqrt(252), annualized
    excess = strategy_returns - market_returns
    te = np.std(excess, ddof=1) * np.sqrt(252)

    # Information Ratio = mean(Rp - Rm) * 252 / TE
    mean_excess_annual = np.mean(excess) * 252
    ir = mean_excess_annual / te if te > 0 else 0

    # R-squared
    correlation = np.corrcoef(strategy_returns, market_returns)[0, 1]
    r_squared = correlation ** 2

    return {
        "beta": float(beta),
        "tracking_error_annual_pct": float(te * 100),
        "information_ratio": float(ir),
        "r_squared": float(r_squared),
        "correlation": float(correlation),
    }


def compute_ratios(returns, rf):
    """Compute Sharpe, Sortino, Calmar from daily returns."""
    n_years = len(returns) / 252
    equity = np.cumprod(1 + returns)
    total_ret = equity[-1] - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252

    sharpe = (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0

    rf_daily = np.mean(rf)
    downside = np.minimum(returns - rf_daily, 0)
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    sortino = (annual_ret - rf_annual) / downside_std if downside_std > 0 else 0

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


def main():
    print("=" * 80)
    print("RISK METRICS CALCULATOR -- Section 6 of Paper")
    print("=" * 80)

    detail, artifacts = load_data()
    metadata = artifacts["metadata"]

    # Market returns and risk-free
    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    raw_dates = metadata.get("dates_test", None)
    if raw_dates is not None:
        # Convert numpy datetime64 to string YYYY-MM-DD
        test_dates = [str(d)[:10] for d in raw_dates[:len(fwd_test)]]
    else:
        test_dates = None

    n_test = len(fwd_test)
    print(f"\nTest period: {n_test} days")
    if test_dates:
        print(f"  From: {test_dates[0]} to {test_dates[-1]}")

    results = {}

    # -- SPY Benchmark --
    print(f"\n{'-'*60}")
    print("Computing SPY B&H metrics...")
    spy_equity = np.cumprod(1 + fwd_test)
    spy_ratios = compute_ratios(fwd_test, rf_test)
    spy_var = compute_var_cvar(fwd_test)
    spy_dd = compute_drawdown_details(spy_equity, test_dates)
    # Beta of SPY vs itself = 1.0 by definition
    spy_tracking = {
        "beta": 1.0,
        "tracking_error_annual_pct": 0.0,
        "information_ratio": float("nan"),
        "r_squared": 1.0,
        "correlation": 1.0,
    }

    # Rolling VaR/CVaR for SPY
    spy_var_30d = compute_var_cvar_rolling(fwd_test, 30)
    spy_var_90d = compute_var_cvar_rolling(fwd_test, 90)

    # SPY capture vs itself: 100% / 100% by definition
    spy_capture = {
        "upside_capture_pct": 100.0,
        "downside_capture_pct": 100.0,
        "capture_ratio": 1.0,
        "avg_position": 1.0,
        "pct_long_on_up_days": 100.0,
        "pct_long_on_down_days": 100.0,
        "n_up_days": int(np.sum(fwd_test > 0)),
        "n_down_days": int(np.sum(fwd_test < 0)),
    }

    results["SPY_BH"] = {
        "ratios": spy_ratios,
        "var_cvar": spy_var,
        "var_cvar_30d": spy_var_30d,
        "var_cvar_90d": spy_var_90d,
        "drawdown": spy_dd,
        "tracking": spy_tracking,
        "capture": spy_capture,
    }

    print(f"  Sharpe: {spy_ratios['sharpe']:.3f}")
    print(f"  MaxDD: {spy_dd['max_dd_pct']:.1f}%")
    print(f"  VaR30d 95%: {spy_var_30d['var_95_pct']:.2f}%, "
          f"CVaR30d 95%: {spy_var_30d['cvar_95_pct']:.2f}%")
    print(f"  VaR90d 95%: {spy_var_90d['var_95_pct']:.2f}%, "
          f"CVaR90d 95%: {spy_var_90d['cvar_95_pct']:.2f}%")

    # -- 4 Winning Models --
    for model_name in WINNERS:
        print(f"\n{'-'*60}")
        print(f"Computing {model_name} metrics...")

        model_detail = detail["models"][model_name]
        strat_returns = np.array(model_detail["strategy_returns"])
        positions = np.array(model_detail["positions"])
        equity = np.array(model_detail["equity_curve"])

        # Align lengths
        n = min(len(strat_returns), len(fwd_test))
        strat_returns = strat_returns[:n]
        positions = positions[:n]
        market_returns = fwd_test[:n]
        rf = rf_test[:n]
        dates = test_dates[:n] if test_dates else None

        # Ratios
        ratios = compute_ratios(strat_returns, rf)
        print(f"  Sharpe: {ratios['sharpe']:.3f}, Sortino: {ratios['sortino']:.3f}, "
              f"Calmar: {ratios['calmar']:.3f}")

        # VaR / CVaR (daily)
        var_cvar = compute_var_cvar(strat_returns)
        print(f"  VaR95: {var_cvar['var_95_daily_pct']:.2f}%, "
              f"CVaR95: {var_cvar['cvar_95_daily_pct']:.2f}%, "
              f"Worst: {var_cvar['worst_day_pct']:.2f}%")

        # VaR / CVaR (30-day and 90-day rolling)
        var_30d = compute_var_cvar_rolling(strat_returns, 30)
        var_90d = compute_var_cvar_rolling(strat_returns, 90)
        print(f"  VaR30d 95%: {var_30d['var_95_pct']:.2f}%, "
              f"CVaR30d 95%: {var_30d['cvar_95_pct']:.2f}%")
        print(f"  VaR90d 95%: {var_90d['var_95_pct']:.2f}%, "
              f"CVaR90d 95%: {var_90d['cvar_95_pct']:.2f}%")

        # Drawdown details
        equity_for_dd = np.cumprod(1 + strat_returns)
        dd = compute_drawdown_details(equity_for_dd, dates)
        print(f"  MaxDD: {dd['max_dd_pct']:.1f}%, "
              f"Duration: {dd['duration_trading_days']}d, "
              f"Recovery: {dd['recovery_trading_days']}d")
        if dd['peak_date']:
            print(f"  Peak: {dd['peak_date']}, Trough: {dd['trough_date']}")

        # Beta / Tracking Error / IR
        tracking = compute_beta_tracking(strat_returns, market_returns)
        print(f"  Beta: {tracking['beta']:.3f}, "
              f"TE: {tracking['tracking_error_annual_pct']:.1f}%, "
              f"IR: {tracking['information_ratio']:.3f}, "
              f"R2: {tracking['r_squared']:.3f}")

        # Capture Ratios
        capture = compute_capture_ratios(strat_returns, market_returns, positions)
        print(f"  Upside Capture: {capture['upside_capture_pct']:.1f}%, "
              f"Downside Capture: {capture['downside_capture_pct']:.1f}%, "
              f"Ratio: {capture['capture_ratio']:.2f}")
        print(f"  Avg Position: {capture['avg_position']:.2f}, "
              f"Long on Up: {capture['pct_long_on_up_days']:.1f}%, "
              f"Long on Down: {capture['pct_long_on_down_days']:.1f}%")

        results[model_name] = {
            "ratios": ratios,
            "var_cvar": var_cvar,
            "var_cvar_30d": var_30d,
            "var_cvar_90d": var_90d,
            "drawdown": dd,
            "tracking": tracking,
            "capture": capture,
        }

    # -- Save --
    output_path = os.path.join(RESULTS_DIR, "risk_metrics_detail.json")

    # Convert NaN to null for JSON
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
    print(f"\n{'='*80}")
    print(f"Saved to {output_path}")
    print(f"{'='*80}")

    # -- Summary Table: Ratios --
    print(f"\n{'Model':<20} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} "
          f"{'MaxDD%':>8} {'Beta':>6} {'TE%':>6} {'IR':>6}")
    print("-" * 80)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        ir_val = r['tracking']['information_ratio']
        ir_str = f"{ir_val:.3f}" if ir_val is not None and not np.isnan(ir_val) else "N/A"
        print(f"{label:<20} {r['ratios']['sharpe']:>8.3f} {r['ratios']['sortino']:>8.3f} "
              f"{r['ratios']['calmar']:>8.3f} {r['drawdown']['max_dd_pct']:>7.1f}% "
              f"{r['tracking']['beta']:>6.3f} "
              f"{r['tracking']['tracking_error_annual_pct']:>5.1f}% "
              f"{ir_str:>6}")

    # -- Summary Table: Daily VaR/CVaR --
    print(f"\n{'Model':<20} {'VaR95%':>8} {'CVaR95%':>9} {'VaR99%':>8} {'Worst%':>8}")
    print("-" * 60)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        v = r['var_cvar']
        print(f"{label:<20} {v['var_95_daily_pct']:>8.2f} {v['cvar_95_daily_pct']:>9.2f} "
              f"{v['var_99_daily_pct']:>8.2f} {v['worst_day_pct']:>8.2f}")

    # -- Summary Table: Rolling VaR/CVaR (30d and 90d) --
    print(f"\n{'Model':<20} {'VaR30d%':>9} {'CVaR30d%':>10} {'VaR90d%':>9} {'CVaR90d%':>10}")
    print("-" * 65)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        v30 = r['var_cvar_30d']
        v90 = r['var_cvar_90d']
        v30_var = f"{v30['var_95_pct']:.2f}" if v30['var_95_pct'] is not None else "N/A"
        v30_cvar = f"{v30['cvar_95_pct']:.2f}" if v30['cvar_95_pct'] is not None else "N/A"
        v90_var = f"{v90['var_95_pct']:.2f}" if v90['var_95_pct'] is not None else "N/A"
        v90_cvar = f"{v90['cvar_95_pct']:.2f}" if v90['cvar_95_pct'] is not None else "N/A"
        print(f"{label:<20} {v30_var:>9} {v30_cvar:>10} {v90_var:>9} {v90_cvar:>10}")

    # -- Summary Table: Drawdown Details --
    print(f"\n{'Model':<20} {'MaxDD%':>8} {'Peak':>12} {'Trough':>12} "
          f"{'Dur(d)':>8} {'Recov(d)':>9}")
    print("-" * 80)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        d = r['drawdown']
        rec = str(d['recovery_trading_days']) if d['recovery_trading_days'] > 0 else "N/R"
        peak = d['peak_date'] or "?"
        trough = d['trough_date'] or "?"
        print(f"{label:<20} {d['max_dd_pct']:>7.1f}% {peak:>12} {trough:>12} "
              f"{d['duration_trading_days']:>8} {rec:>9}")

    # -- Summary Table: Capture Ratios --
    print(f"\n{'Model':<20} {'Up Cap%':>9} {'Dn Cap%':>9} {'Ratio':>7} "
          f"{'AvgPos':>7} {'Long/Up%':>9} {'Long/Dn%':>9}")
    print("-" * 80)
    for name in WINNERS + ["SPY_BH"]:
        r = results[name]
        label = "SPY B&H" if name == "SPY_BH" else name
        c = r['capture']
        print(f"{label:<20} {c['upside_capture_pct']:>9.1f} {c['downside_capture_pct']:>9.1f} "
              f"{c['capture_ratio']:>7.2f} {c['avg_position']:>7.2f} "
              f"{c['pct_long_on_up_days']:>9.1f} {c['pct_long_on_down_days']:>9.1f}")

    print(f"\nMarket: {results['SPY_BH']['capture']['n_up_days']} up days, "
          f"{results['SPY_BH']['capture']['n_down_days']} down days")


if __name__ == "__main__":
    main()
