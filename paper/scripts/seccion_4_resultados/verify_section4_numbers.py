# -*- coding: utf-8 -*-
"""
================================================================================
VERIFY_SECTION4_NUMBERS.PY -- Verify ALL numerical claims in Section 4
================================================================================

Computes and cross-checks every number in Section 4 (Resultados Empiricos)
of the paper against the pipeline outputs:

1. Trading Statistics for Top 5 models (Win Rate, Avg Win/Loss, Profit Factor)
2. UPRO Buy & Hold benchmark (return, Sharpe, MaxDD, Calmar)
3. 60/40 Portfolio benchmark (return, Sharpe, MaxDD, Calmar)
4. Inline number verification (positive Sharpe count, 432pp range, per-model)

Data sources:
  - results/backtest_detail.pkl
  - models/trained_artifacts.pkl
  - results/final_long_only_backtest.json

Output: results/section4_verification.json
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
# PATH SETUP
# ============================================================================
# Three dirname calls from paper/scripts/seccion_4_resultados/ -> paper/
# Then one more to reach the repo root.
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Instruments (must match optimize_and_backtest.py exactly)
INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},
    'SPY':  {'expense_ratio': 0.0009, 'bid_ask': 0.0002},
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},
}

TOP5 = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM", "DLinear"]

ALL_MODELS = [
    "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost",
    "AutoARIMA", "ExponentialSmoothing", "Theta", "SeasonalNaive",
    "Prophet", "GARCH",
    "DLinear", "NBEATS", "NHiTS", "TCN", "TFT",
    "CNN_LSTM", "LSTM_Attention", "BiLSTM", "BiGRU",
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load backtest detail, trained artifacts, final backtest JSON, and Bloomberg raw data."""
    detail_path = os.path.join(RESULTS_DIR, "backtest_detail.pkl")
    with open(detail_path, "rb") as f:
        detail = pickle.load(f)

    artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    json_path = os.path.join(RESULTS_DIR, "final_long_only_backtest.json")
    with open(json_path, "r") as f:
        backtest_json = json.load(f)

    bloomberg_path = os.path.join(BASE_DIR, "data", "BLOOMBERG_RAW_DATA.csv")
    bloomberg = pd.read_csv(bloomberg_path, parse_dates=["date"])

    return detail, artifacts, backtest_json, bloomberg


# ============================================================================
# 1. TRADING STATISTICS
# ============================================================================

def compute_trading_stats(positions, strategy_returns, rf):
    """
    Compute trading statistics from daily positions and net returns.

    Two approaches are computed:

    (A) TRADE-LEVEL: A 'trade' is a contiguous period holding the same
        position. Win/Loss stats use cumulative return per trade.

    (B) DAILY-LEVEL (matches paper Table 4): Win Rate, Avg Win, Avg Loss,
        and Profit Factor are computed over individual ACTIVE DAYS
        (position != 0), using excess returns over risk-free.
        This matches the paper footnote: "calculados sobre dias activos
        (posicion != 0)".
    """
    n = len(positions)

    # --- TRADE-LEVEL STATS ---
    trades = []
    i = 0
    while i < n:
        entry_pos = int(positions[i])
        j = i
        cum_ret = 1.0
        while j < n and int(positions[j]) == entry_pos:
            cum_ret *= (1.0 + strategy_returns[j])
            j += 1
        trade_return = cum_ret - 1.0
        duration = j - i
        trades.append({
            "position": entry_pos,
            "return": trade_return,
            "duration": duration,
            "start_idx": i,
            "end_idx": j - 1,
        })
        i = j

    n_total_trades = len(trades)
    n_position_changes = int(np.sum(np.diff(positions) != 0))

    active_trades = [t for t in trades if t["position"] != 0]
    n_active_trades = len(active_trades)

    if n_active_trades > 0:
        trade_returns = np.array([t["return"] for t in active_trades])
        trade_winners = trade_returns[trade_returns > 0]
        trade_losers = trade_returns[trade_returns <= 0]
        trade_win_rate = len(trade_winners) / n_active_trades * 100.0
        trade_avg_win = float(np.mean(trade_winners) * 100) if len(trade_winners) > 0 else 0.0
        trade_avg_loss = float(np.mean(trade_losers) * 100) if len(trade_losers) > 0 else 0.0
        trade_gross_wins = float(np.sum(trade_winners))
        trade_gross_losses = float(np.abs(np.sum(trade_losers)))
        trade_pf = trade_gross_wins / trade_gross_losses if trade_gross_losses > 0 else float("inf")
    else:
        trade_win_rate = 0.0
        trade_avg_win = 0.0
        trade_avg_loss = 0.0
        trade_pf = 0.0

    # --- DAILY-LEVEL STATS (matches paper Table 4) ---
    # Active days: position != 0
    # Excess return = strategy_return - risk_free on that day
    active_mask = positions != 0
    n_active_days = int(np.sum(active_mask))

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
        "n_total_trades": n_total_trades,
        "n_position_changes": n_position_changes,
        "n_active_trades": n_active_trades,
        "n_active_days": n_active_days,
        # Trade-level
        "trade_win_rate_pct": float(trade_win_rate),
        "trade_avg_win_pct": float(trade_avg_win),
        "trade_avg_loss_pct": float(trade_avg_loss),
        "trade_profit_factor": float(trade_pf),
        # Daily-level (paper Table 4 definition)
        "daily_win_rate_pct": float(daily_win_rate),
        "daily_avg_win_pct": float(daily_avg_win),
        "daily_avg_loss_pct": float(daily_avg_loss),
        "daily_profit_factor": float(daily_pf),
    }


# ============================================================================
# 2. UPRO BUY & HOLD BENCHMARK
# ============================================================================

def compute_upro_buyhold(market_returns, rf):
    """
    Compute UPRO Buy & Hold benchmark over the test period.

    The correct UPRO B&H model uses the same gross return formula as the pipeline
    (rf + 3*(mkt - rf), which accounts for leverage funding cost), minus only the
    UPRO expense ratio (0.91%/yr) and one entry trade (bid-ask 5bps).

    NO separate vol_drag term: the volatility decay is already implicit in
    the daily 3x compounding. The expense ratio of UPRO (0.91%) covers all
    internal ETF costs including swap/futures maintenance. Adding vol_drag
    on top would be double-counting.

    For the strategy's leveraged positions, the pipeline applies vol_drag as
    a CONSERVATIVE additional cost. This makes it harder for the strategy
    to beat the UPRO B&H benchmark --- a deliberate design choice.

    Three versions computed for transparency:

    (A) Paper formula: rf + 3*(mkt - rf) - expense - 1 trade (CORRECT)
    (B) With vol drag: same + vol_drag (too conservative for B&H benchmark)
    (C) Naive 3x: 3*mkt - expense (ignores leverage funding cost)
    """
    n = len(market_returns)
    expense_daily = INSTRUMENTS['UPRO']['expense_ratio'] / 252
    bid_ask = INSTRUMENTS['UPRO']['bid_ask']

    # --- Version A (CORRECT): rf + 3*(mkt - rf) - expense - 1 trade ---
    # Gross return accounts for leverage funding cost (borrow at rf).
    # No vol_drag: decay is implicit in daily 3x compounding.
    # Only 1 trade: buy UPRO on day 0, hold forever.
    upro_gross = rf + 3.0 * (market_returns - rf)
    upro_returns_paper = upro_gross - expense_daily
    upro_returns_paper[0] -= bid_ask
    metrics_paper = _compute_benchmark_metrics(upro_returns_paper, rf,
                                               "UPRO_BH_paper_formula")

    # --- Version B: Same but WITH vol_drag (too conservative for B&H) ---
    vol_drag_fixed = 0.5 * 6 * (0.01) ** 2
    upro_returns_with_vd = upro_gross.copy() - expense_daily - vol_drag_fixed
    upro_returns_with_vd[0] -= bid_ask
    metrics_with_vd = _compute_benchmark_metrics(upro_returns_with_vd, rf,
                                                  "UPRO_BH_with_vol_drag")

    # --- Version C: Naive 3x (3*mkt - expense, ignores funding cost) ---
    upro_returns_naive = 3.0 * market_returns - expense_daily
    upro_returns_naive[0] -= bid_ask
    metrics_naive = _compute_benchmark_metrics(upro_returns_naive, rf,
                                               "UPRO_BH_naive_3x")

    return metrics_paper, metrics_with_vd, metrics_naive


# ============================================================================
# 3. 60/40 PORTFOLIO BENCHMARK
# ============================================================================

def compute_6040_portfolio(spy_prices, bond_prices, rf, dates):
    """
    Compute 60/40 Portfolio benchmark with monthly rebalancing.

    60% SPY + 40% LF98TRUU Index (Bloomberg US Aggregate Bond Total Return,
    the benchmark index tracked by BND). Monthly rebalancing resets weights
    to 60/40 at the start of each calendar month.
    """
    spy_ret = np.diff(spy_prices) / spy_prices[:-1]
    bond_ret = np.diff(bond_prices) / bond_prices[:-1]
    dt = pd.to_datetime(dates[1:])
    n = len(spy_ret)

    # Monthly rebalancing
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
        eq_spy *= (1 + spy_ret[i])
        eq_bond *= (1 + bond_ret[i])
        port_eq[i] = eq_spy + eq_bond

    # Convert equity curve to daily returns for _compute_benchmark_metrics
    port_returns = np.diff(np.concatenate([[1.0], port_eq])) / np.concatenate([[1.0], port_eq[:-1]])

    # Use rf trimmed to match length
    rf_trimmed = rf[:n] if len(rf) > n else rf

    metrics = _compute_benchmark_metrics(port_returns, rf_trimmed, "60_40_Portfolio")
    return metrics


# ============================================================================
# SHARED METRIC COMPUTATION
# ============================================================================

def _compute_benchmark_metrics(returns, rf, name):
    """Compute standard metrics for a benchmark strategy."""
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
    downside = returns - rf_daily
    downside = np.minimum(downside, 0)
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
# 4. INLINE NUMBER VERIFICATION
# ============================================================================

def verify_inline_numbers(backtest_json):
    """
    Verify all inline numerical claims in Section 4.

    Returns a dict of {claim: {paper_value, computed_value, match}}.
    """
    models = backtest_json["models"]
    benchmark = backtest_json["benchmark"]

    checks = {}

    # --- (a) SPY B&H return ---
    spy_ret = benchmark["total_return"] * 100
    checks["SPY_BH_return_pct"] = {
        "paper_value": 102.0,
        "computed_value": round(spy_ret, 1),
        "match": abs(spy_ret - 102.0) < 1.0,
        "note": "Paper says +102%",
    }

    # --- (b) 4/23 models beat SPY ---
    beating = [m for m, d in models.items() if d["total_return"] > benchmark["total_return"]]
    checks["models_beating_SPY"] = {
        "paper_value": 4,
        "computed_value": len(beating),
        "match": len(beating) == 4,
        "note": "Paper says 4 of 23",
    }

    # --- (c) Individual model returns (rounded to nearest %) ---
    paper_returns = {
        "LSTM_Attention": 400,
        "Ridge": 263,
        "RandomForest": 131,
        "CNN_LSTM": 129,
    }
    for model_name, paper_val in paper_returns.items():
        computed = round(models[model_name]["total_return"] * 100, 0)
        checks[f"{model_name}_return_pct"] = {
            "paper_value": paper_val,
            "computed_value": computed,
            "match": abs(computed - paper_val) <= 1.0,
            "note": f"Paper says +{paper_val}%",
        }

    # --- (d) 16/23 models with positive Sharpe ---
    n_positive_sharpe = sum(1 for d in models.values() if d["sharpe"] > 0)
    checks["models_positive_sharpe"] = {
        "paper_value": 16,
        "computed_value": n_positive_sharpe,
        "match": n_positive_sharpe == 16,
        "note": "Paper says 16 of 23",
    }

    # --- (e) 432 pp range (max return - min return) ---
    all_returns = [d["total_return"] * 100 for d in models.values()]
    ret_range = max(all_returns) - min(all_returns)
    checks["return_range_pp"] = {
        "paper_value": 432,
        "computed_value": round(ret_range, 0),
        "match": abs(ret_range - 432) < 2.0,
        "note": "Paper says 432 pp range",
    }

    # --- (f) Best model: LSTM_Attention +400% ---
    best_model = max(models, key=lambda m: models[m]["total_return"])
    best_ret = round(models[best_model]["total_return"] * 100, 0)
    checks["best_model"] = {
        "paper_value": "LSTM_Attention +400%",
        "computed_value": f"{best_model} +{best_ret}%",
        "match": best_model == "LSTM_Attention" and abs(best_ret - 400) <= 1,
        "note": "Paper says LSTM_Attention is best at +400%",
    }

    # --- (g) Worst model: BiGRU -32% ---
    worst_model = min(models, key=lambda m: models[m]["total_return"])
    worst_ret = round(models[worst_model]["total_return"] * 100, 0)
    checks["worst_model"] = {
        "paper_value": "BiGRU -32%",
        "computed_value": f"{worst_model} {worst_ret:+.0f}%",
        "match": worst_model == "BiGRU" and abs(worst_ret - (-32)) <= 1,
        "note": "Paper says BiGRU is worst at -32%",
    }

    # --- (h) Per-family claims ---
    # Ridge Sharpe 0.88
    ridge_sharpe = models["Ridge"]["sharpe"]
    checks["Ridge_sharpe"] = {
        "paper_value": 0.88,
        "computed_value": round(ridge_sharpe, 2),
        "match": abs(ridge_sharpe - 0.88) < 0.01,
        "note": "Paper says Ridge Sharpe 0.88",
    }

    # Lasso +39%
    lasso_ret = round(models["Lasso"]["total_return"] * 100, 0)
    checks["Lasso_return_pct"] = {
        "paper_value": 39,
        "computed_value": lasso_ret,
        "match": abs(lasso_ret - 39) <= 1,
        "note": "Paper says Lasso +39%",
    }

    # ElasticNet +10%
    enet_ret = round(models["ElasticNet"]["total_return"] * 100, 0)
    checks["ElasticNet_return_pct"] = {
        "paper_value": 10,
        "computed_value": enet_ret,
        "match": abs(enet_ret - 10) <= 1,
        "note": "Paper says ElasticNet +10%",
    }

    # GradientBoosting +67%, Sharpe 0.62, MaxDD 8.3%
    gb = models["GradientBoosting"]
    gb_ret = round(gb["total_return"] * 100, 0)
    checks["GradientBoosting_return_pct"] = {
        "paper_value": 67,
        "computed_value": gb_ret,
        "match": abs(gb_ret - 67) <= 1,
        "note": "Paper says GradientBoosting +67%",
    }
    checks["GradientBoosting_sharpe"] = {
        "paper_value": 0.62,
        "computed_value": round(gb["sharpe"], 2),
        "match": abs(gb["sharpe"] - 0.62) < 0.01,
        "note": "Paper says GradientBoosting Sharpe 0.62",
    }
    gb_dd = round(gb["max_drawdown"] * 100, 1)
    checks["GradientBoosting_maxdd_pct"] = {
        "paper_value": 8.3,
        "computed_value": gb_dd,
        "match": abs(gb_dd - 8.3) < 0.2,
        "note": "Paper says GradientBoosting MaxDD 8.3%",
    }

    # CatBoost +47%
    catboost_ret = round(models["CatBoost"]["total_return"] * 100, 0)
    checks["CatBoost_return_pct"] = {
        "paper_value": 47,
        "computed_value": catboost_ret,
        "match": abs(catboost_ret - 47) <= 1,
        "note": "Paper says CatBoost +47%",
    }

    # XGBoost +14%
    xgb_ret = round(models["XGBoost"]["total_return"] * 100, 0)
    checks["XGBoost_return_pct"] = {
        "paper_value": 14,
        "computed_value": xgb_ret,
        "match": abs(xgb_ret - 14) <= 2,
        "note": "Paper says XGBoost +14%",
    }

    # GradientBoosting: 16 unique preds, 86% filtered
    gb_unique = gb["signal_quality"]["n_unique_test_preds"]
    checks["GradientBoosting_unique_preds"] = {
        "paper_value": 16,
        "computed_value": gb_unique,
        "match": gb_unique == 16,
        "note": "Paper says 16 unique predictions",
    }
    gb_filtered = round(gb["signal_quality"]["pct_filtered_days"], 0)
    checks["GradientBoosting_filtered_pct"] = {
        "paper_value": 86,
        "computed_value": gb_filtered,
        "match": abs(gb_filtered - 86) <= 1,
        "note": "Paper says 86% filtered",
    }

    # DLinear +68%
    dlinear_ret = round(models["DLinear"]["total_return"] * 100, 0)
    checks["DLinear_return_pct"] = {
        "paper_value": 68,
        "computed_value": dlinear_ret,
        "match": abs(dlinear_ret - 68) <= 1,
        "note": "Paper says DLinear +68%",
    }

    # NBEATS +44%
    nbeats_ret = round(models["NBEATS"]["total_return"] * 100, 0)
    checks["NBEATS_return_pct"] = {
        "paper_value": 44,
        "computed_value": nbeats_ret,
        "match": abs(nbeats_ret - 44) <= 1,
        "note": "Paper says NBEATS +44%",
    }

    # NHiTS +22%
    nhits_ret = round(models["NHiTS"]["total_return"] * 100, 0)
    checks["NHiTS_return_pct"] = {
        "paper_value": 22,
        "computed_value": nhits_ret,
        "match": abs(nhits_ret - 22) <= 1,
        "note": "Paper says NHiTS +22%",
    }

    # TFT +15%
    tft_ret = round(models["TFT"]["total_return"] * 100, 0)
    checks["TFT_return_pct"] = {
        "paper_value": 15,
        "computed_value": tft_ret,
        "match": abs(tft_ret - 15) <= 1,
        "note": "Paper says TFT +15%",
    }

    # LSTM_Attention Sharpe 1.33, MaxDD 20.5%
    lstm_a = models["LSTM_Attention"]
    checks["LSTM_Attention_sharpe"] = {
        "paper_value": 1.33,
        "computed_value": round(lstm_a["sharpe"], 2),
        "match": abs(lstm_a["sharpe"] - 1.33) < 0.01,
        "note": "Paper says LSTM_Attention Sharpe 1.33",
    }
    lstm_dd = round(lstm_a["max_drawdown"] * 100, 1)
    checks["LSTM_Attention_maxdd_pct"] = {
        "paper_value": 20.5,
        "computed_value": lstm_dd,
        "match": abs(lstm_dd - 20.5) < 0.2,
        "note": "Paper says LSTM_Attention MaxDD 20.5%",
    }

    # CNN_LSTM Sharpe 0.60, MaxDD 35.6%
    cnn = models["CNN_LSTM"]
    checks["CNN_LSTM_sharpe"] = {
        "paper_value": 0.60,
        "computed_value": round(cnn["sharpe"], 2),
        "match": abs(cnn["sharpe"] - 0.60) < 0.01,
        "note": "Paper says CNN_LSTM Sharpe 0.60",
    }
    cnn_dd = round(cnn["max_drawdown"] * 100, 1)
    checks["CNN_LSTM_maxdd_pct"] = {
        "paper_value": 35.6,
        "computed_value": cnn_dd,
        "match": abs(cnn_dd - 35.6) < 0.2,
        "note": "Paper says CNN_LSTM MaxDD 35.6%",
    }

    # BiLSTM +6%, BiGRU -32%
    bilstm_ret = round(models["BiLSTM"]["total_return"] * 100, 0)
    checks["BiLSTM_return_pct"] = {
        "paper_value": 6,
        "computed_value": bilstm_ret,
        "match": abs(bilstm_ret - 6) <= 1,
        "note": "Paper says BiLSTM +6%",
    }
    bigru_ret = round(models["BiGRU"]["total_return"] * 100, 0)
    checks["BiGRU_return_pct"] = {
        "paper_value": -32,
        "computed_value": bigru_ret,
        "match": abs(bigru_ret - (-32)) <= 1,
        "note": "Paper says BiGRU -32%",
    }

    # Theta +45%, SeasonalNaive +45%
    theta_ret = round(models["Theta"]["total_return"] * 100, 0)
    checks["Theta_return_pct"] = {
        "paper_value": 45,
        "computed_value": theta_ret,
        "match": abs(theta_ret - 45) <= 1,
        "note": "Paper says Theta +45%",
    }
    snaive_ret = round(models["SeasonalNaive"]["total_return"] * 100, 0)
    checks["SeasonalNaive_return_pct"] = {
        "paper_value": 45,
        "computed_value": snaive_ret,
        "match": abs(snaive_ret - 45) <= 1,
        "note": "Paper says SeasonalNaive +45%",
    }

    # Prophet +8%, ExponentialSmoothing -15%
    prophet_ret = round(models["Prophet"]["total_return"] * 100, 0)
    checks["Prophet_return_pct"] = {
        "paper_value": 8,
        "computed_value": prophet_ret,
        "match": abs(prophet_ret - 8) <= 1,
        "note": "Paper says Prophet +8%",
    }
    es_ret = round(models["ExponentialSmoothing"]["total_return"] * 100, 0)
    checks["ExponentialSmoothing_return_pct"] = {
        "paper_value": -15,
        "computed_value": es_ret,
        "match": abs(es_ret - (-15)) <= 1,
        "note": "Paper says ExponentialSmoothing -15%",
    }

    # --- Position distribution claims ---
    # LSTM_Attention: 12.2% UPRO, 32.7% SPY, 55.1% cash
    lstm_a_pct3 = round(lstm_a["pct_3x"], 1)
    lstm_a_pct1 = round(lstm_a["pct_1x"], 1)
    lstm_a_cash = round(lstm_a["pct_cash"], 1)
    checks["LSTM_Attention_pct_UPRO"] = {
        "paper_value": 12.2,
        "computed_value": lstm_a_pct3,
        "match": abs(lstm_a_pct3 - 12.2) < 0.2,
        "note": "Paper says 12.2% in UPRO",
    }
    checks["LSTM_Attention_pct_SPY"] = {
        "paper_value": 32.7,
        "computed_value": lstm_a_pct1,
        "match": abs(lstm_a_pct1 - 32.7) < 0.2,
        "note": "Paper says 32.7% in SPY",
    }
    checks["LSTM_Attention_pct_cash"] = {
        "paper_value": 55.1,
        "computed_value": lstm_a_cash,
        "match": abs(lstm_a_cash - 55.1) < 0.2,
        "note": "Paper says 55.1% in cash",
    }

    # ExponentialSmoothing ~72% long (pct_3x + pct_1x)
    es = models["ExponentialSmoothing"]
    es_long = round(es["pct_3x"] + es["pct_1x"], 0)
    checks["ExponentialSmoothing_pct_long"] = {
        "paper_value": 72,
        "computed_value": es_long,
        "match": abs(es_long - 72) <= 1,
        "note": "Paper says ~72% of time in long position",
    }

    # TFT ~60% long
    tft = models["TFT"]
    tft_long = round(tft["pct_3x"] + tft["pct_1x"], 0)
    checks["TFT_pct_long"] = {
        "paper_value": 60,
        "computed_value": tft_long,
        "match": abs(tft_long - 60) <= 1,
        "note": "Paper says TFT ~60% long",
    }

    # AutoARIMA 100% cash
    aa_cash = round(models["AutoARIMA"]["pct_cash"], 0)
    checks["AutoARIMA_pct_cash"] = {
        "paper_value": 100,
        "computed_value": aa_cash,
        "match": aa_cash == 100,
        "note": "Paper says AutoARIMA 100% cash",
    }

    # GradientBoosting ~92% cash
    gb_cash = round(gb["pct_cash"], 0)
    checks["GradientBoosting_pct_cash"] = {
        "paper_value": 92,
        "computed_value": gb_cash,
        "match": abs(gb_cash - 92) <= 1,
        "note": "Paper says GradientBoosting ~92% cash",
    }

    # RandomForest MaxDD 14.7%
    rf_dd = round(models["RandomForest"]["max_drawdown"] * 100, 1)
    checks["RandomForest_maxdd_pct"] = {
        "paper_value": 14.7,
        "computed_value": rf_dd,
        "match": abs(rf_dd - 14.7) < 0.2,
        "note": "Paper says RandomForest MaxDD 14.7%",
    }

    # --- Benchmark table (Table 5) claims ---
    # SPY B&H: +102.2%, Sharpe 0.66, MaxDD -25.4%
    spy_sharpe = round(benchmark["sharpe"], 2)
    spy_dd = round(benchmark["max_drawdown"] * 100, 1)
    checks["SPY_BH_sharpe"] = {
        "paper_value": 0.66,
        "computed_value": spy_sharpe,
        "match": abs(spy_sharpe - 0.66) < 0.02,
        "note": "Paper says SPY Sharpe 0.66",
    }
    checks["SPY_BH_maxdd_pct"] = {
        "paper_value": 25.4,
        "computed_value": spy_dd,
        "match": abs(spy_dd - 25.4) < 0.2,
        "note": "Paper says SPY MaxDD -25.4%",
    }

    # --- N Trades verification for Top 5 ---
    paper_n_trades = {
        "LSTM_Attention": 465,
        "Ridge": 409,
        "RandomForest": 175,
        "CNN_LSTM": 633,
        "DLinear": 788,
    }
    for mname, paper_val in paper_n_trades.items():
        computed = models[mname]["n_trades"]
        checks[f"{mname}_n_trades"] = {
            "paper_value": paper_val,
            "computed_value": computed,
            "match": computed == paper_val,
            "note": f"Paper Table 4 says {paper_val} trades",
        }

    return checks


# ============================================================================
# 5. VERIFY BENCHMARK TABLE (Table 5) numbers
# ============================================================================

def verify_benchmark_table(upro_metrics, portfolio_6040_metrics, backtest_json):
    """
    Verify the UPRO 3x B&H and 60/40 numbers in Table 5.
    """
    checks = {}

    # UPRO 3x B&H: Paper says +246.4%, Sharpe 0.47, MaxDD -64.8%, Calmar 0.42
    # Now computed with correct formula: rf + 3*(mkt-rf) - expense - 1 trade
    checks["UPRO_BH_return_pct"] = {
        "paper_value": 246.4,
        "computed_value": upro_metrics["total_return_pct"],
        "match": abs(upro_metrics["total_return_pct"] - 246.4) < 1.0,
        "note": "UPRO B&H = rf + 3*(mkt-rf) - expense - 1 trade, no vol_drag",
    }
    checks["UPRO_BH_sharpe"] = {
        "paper_value": 0.47,
        "computed_value": upro_metrics["sharpe"],
        "match": abs(upro_metrics["sharpe"] - 0.47) < 0.02,
        "note": "Paper says UPRO Sharpe 0.47",
    }
    checks["UPRO_BH_maxdd_pct"] = {
        "paper_value": 64.8,
        "computed_value": upro_metrics["max_drawdown_pct"],
        "match": abs(upro_metrics["max_drawdown_pct"] - 64.8) < 1.0,
        "note": "Paper says UPRO MaxDD -64.8%",
    }
    checks["UPRO_BH_calmar"] = {
        "paper_value": 0.42,
        "computed_value": upro_metrics["calmar"],
        "match": abs(upro_metrics["calmar"] - 0.42) < 0.03,
        "note": "Paper says UPRO Calmar 0.42",
    }

    # 60/40: 60% SPY + 40% LF98TRUU (Bloomberg US Agg Bond TR), monthly rebal
    checks["6040_return_pct"] = {
        "paper_value": 71.1,
        "computed_value": portfolio_6040_metrics["total_return_pct"],
        "match": abs(portfolio_6040_metrics["total_return_pct"] - 71.1) < 1.0,
        "note": "60/40 with LF98TRUU bond index, monthly rebalancing",
    }
    checks["6040_sharpe"] = {
        "paper_value": 0.68,
        "computed_value": portfolio_6040_metrics["sharpe"],
        "match": abs(portfolio_6040_metrics["sharpe"] - 0.68) < 0.02,
        "note": "Paper says 60/40 Sharpe 0.68",
    }
    checks["6040_maxdd_pct"] = {
        "paper_value": 21.1,
        "computed_value": portfolio_6040_metrics["max_drawdown_pct"],
        "match": abs(portfolio_6040_metrics["max_drawdown_pct"] - 21.1) < 1.0,
        "note": "Paper says 60/40 MaxDD -21.1%",
    }
    checks["6040_calmar"] = {
        "paper_value": 0.52,
        "computed_value": portfolio_6040_metrics["calmar"],
        "match": abs(portfolio_6040_metrics["calmar"] - 0.52) < 0.03,
        "note": "Paper says 60/40 Calmar 0.52",
    }

    return checks


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("SECTION 4 VERIFICATION -- Resultados Empiricos")
    print("=" * 90)
    print()

    # --- Load data ---
    print("[1] Loading data...")
    detail, artifacts, backtest_json, bloomberg = load_all_data()

    metadata = artifacts["metadata"]
    # Model positions have n-1 elements (1301) vs forward_returns (1302).
    # Use trimmed arrays for model-based checks, full arrays for benchmarks.
    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    fwd_test_full = np.array(metadata["forward_returns_test"])
    rf_test_full = np.array(metadata["risk_free_test"])
    n_test = len(fwd_test)
    n_years = n_test / 252.0
    print(f"    Test period: {n_test} days for models, {len(fwd_test_full)} days for benchmarks")
    print()

    output = {}

    # ==================================================================
    # PART 1: TRADING STATISTICS (Top 5)
    # ==================================================================
    print("[2] Computing Trading Statistics (Top 5 models)...")
    print("-" * 90)
    hdr = (f"{'Model':<20} {'N Trades':>10} {'ActvDays':>10} {'Win Rate':>10} "
           f"{'Avg Win':>10} {'Avg Loss':>10} {'PF':>10}")
    print(hdr)
    print("-" * 90)

    trading_stats = {}
    for model_name in TOP5:
        if model_name not in detail["models"]:
            print(f"    WARNING: {model_name} not found in backtest_detail.pkl")
            continue

        mdata = detail["models"][model_name]
        positions = np.array(mdata["positions"])
        strategy_returns = np.array(mdata["strategy_returns"])

        stats = compute_trading_stats(positions, strategy_returns, rf_test)
        trading_stats[model_name] = stats

        print(f"{model_name:<20} {stats['n_position_changes']:>10d} "
              f"{stats['n_active_days']:>10d} "
              f"{stats['daily_win_rate_pct']:>9.1f}% "
              f"{stats['daily_avg_win_pct']:>+9.1f}% "
              f"{stats['daily_avg_loss_pct']:>+9.1f}% "
              f"{stats['daily_profit_factor']:>10.2f}")

    output["trading_stats"] = trading_stats
    print()

    # Verify against paper's Table 4
    print("    Verification against Paper Table 4:")
    paper_table4 = {
        "LSTM_Attention": {"win_rate": 53.8, "avg_win": 1.5, "avg_loss": -1.2, "pf": 1.52},
        "Ridge":          {"win_rate": 56.0, "avg_win": 1.9, "avg_loss": -1.8, "pf": 1.36},
        "RandomForest":   {"win_rate": 53.1, "avg_win": 1.6, "avg_loss": -1.2, "pf": 1.51},
        "CNN_LSTM":       {"win_rate": 51.4, "avg_win": 1.4, "avg_loss": -1.1, "pf": 1.31},
        "DLinear":        {"win_rate": 53.6, "avg_win": 0.9, "avg_loss": -0.9, "pf": 1.16},
    }

    trading_checks = {}
    for model_name, paper_vals in paper_table4.items():
        computed = trading_stats[model_name]
        wr_match = abs(computed["daily_win_rate_pct"] - paper_vals["win_rate"]) < 0.5
        aw_match = abs(computed["daily_avg_win_pct"] - paper_vals["avg_win"]) < 0.15
        al_match = abs(computed["daily_avg_loss_pct"] - paper_vals["avg_loss"]) < 0.15
        pf_match = abs(computed["daily_profit_factor"] - paper_vals["pf"]) < 0.05

        status = "OK" if all([wr_match, aw_match, al_match, pf_match]) else "MISMATCH"
        print(f"      {model_name:<20} {status}")
        if not wr_match:
            print(f"        Win Rate: paper={paper_vals['win_rate']:.1f}%, "
                  f"computed={computed['daily_win_rate_pct']:.1f}%")
        if not aw_match:
            print(f"        Avg Win:  paper={paper_vals['avg_win']:+.1f}%, "
                  f"computed={computed['daily_avg_win_pct']:+.1f}%")
        if not al_match:
            print(f"        Avg Loss: paper={paper_vals['avg_loss']:+.1f}%, "
                  f"computed={computed['daily_avg_loss_pct']:+.1f}%")
        if not pf_match:
            print(f"        PF:       paper={paper_vals['pf']:.2f}, "
                  f"computed={computed['daily_profit_factor']:.2f}")

        trading_checks[model_name] = {
            "win_rate_match": wr_match,
            "avg_win_match": aw_match,
            "avg_loss_match": al_match,
            "profit_factor_match": pf_match,
            "all_match": all([wr_match, aw_match, al_match, pf_match]),
        }

    output["trading_checks"] = trading_checks
    print()

    # ==================================================================
    # PART 2: UPRO BUY & HOLD BENCHMARK
    # ==================================================================
    print("[3] Computing UPRO Buy & Hold benchmark...")
    upro_paper, upro_with_vd, upro_naive = compute_upro_buyhold(fwd_test_full, rf_test_full)

    print(f"    UPRO B&H (paper formula: rf + 3*(mkt-rf) - expense - 1 trade):")
    print(f"      Return: {upro_paper['total_return_pct']:+.1f}%  |  "
          f"Sharpe: {upro_paper['sharpe']:.4f}  |  "
          f"MaxDD: {upro_paper['max_drawdown_pct']:.1f}%  |  "
          f"Calmar: {upro_paper['calmar']:.4f}")

    print(f"    UPRO B&H (with vol_drag -- too conservative for B&H):")
    print(f"      Return: {upro_with_vd['total_return_pct']:+.1f}%  |  "
          f"Sharpe: {upro_with_vd['sharpe']:.4f}  |  "
          f"MaxDD: {upro_with_vd['max_drawdown_pct']:.1f}%  |  "
          f"Calmar: {upro_with_vd['calmar']:.4f}")

    print(f"    UPRO B&H (naive 3x: 3*mkt - expense, ignores funding cost):")
    print(f"      Return: {upro_naive['total_return_pct']:+.1f}%  |  "
          f"Sharpe: {upro_naive['sharpe']:.4f}  |  "
          f"MaxDD: {upro_naive['max_drawdown_pct']:.1f}%  |  "
          f"Calmar: {upro_naive['calmar']:.4f}")

    print(f"    Paper claims: Return=+246.4%, Sharpe=0.47, MaxDD=-64.8%, Calmar=0.42")

    output["upro_buyhold_paper"] = upro_paper
    output["upro_buyhold_with_vol_drag"] = upro_with_vd
    output["upro_buyhold_naive_3x"] = upro_naive

    # Use the paper formula (Version A) for Table 5 verification
    upro_for_verification = upro_paper
    print(f"    (Using paper formula for Table 5 verification)")
    print()

    # ==================================================================
    # PART 3: 60/40 PORTFOLIO BENCHMARK
    # ==================================================================
    print("[4] Computing 60/40 Portfolio benchmark (LF98TRUU bond data)...")
    # Filter Bloomberg to test dates for real SPY/bond prices
    test_dates = pd.to_datetime(metadata["dates_test"])
    bb_test = bloomberg[bloomberg["date"].isin(test_dates)].sort_values("date").reset_index(drop=True)
    spy_prices = pd.Series(bb_test["SPY US Equity (CLOSE)"].values).ffill().values
    bond_prices = pd.Series(bb_test["LF98TRUU Index"].values).ffill().values
    bb_dates = bb_test["date"].values
    portfolio_6040 = compute_6040_portfolio(spy_prices, bond_prices, rf_test_full, bb_dates)

    print(f"    60/40 Portfolio:")
    print(f"      Return: {portfolio_6040['total_return_pct']:+.1f}%  |  "
          f"Sharpe: {portfolio_6040['sharpe']:.4f}  |  "
          f"MaxDD: {portfolio_6040['max_drawdown_pct']:.1f}%  |  "
          f"Calmar: {portfolio_6040['calmar']:.4f}")

    print(f"    Paper claims: Return=+71.1%, Sharpe=0.68, MaxDD=-21.1%, Calmar=0.52")
    output["portfolio_6040"] = portfolio_6040
    print()

    # ==================================================================
    # PART 4: INLINE NUMBER VERIFICATION
    # ==================================================================
    print("[5] Verifying all inline numbers from Section 4...")
    inline_checks = verify_inline_numbers(backtest_json)
    output["inline_checks"] = inline_checks

    print("-" * 90)
    print(f"{'Check':<40} {'Paper':>12} {'Computed':>12} {'Match':>8}")
    print("-" * 90)
    n_pass = 0
    n_fail = 0
    for check_name, check_data in inline_checks.items():
        paper_val = check_data["paper_value"]
        computed_val = check_data["computed_value"]
        match = check_data["match"]
        status = "OK" if match else "FAIL"
        if match:
            n_pass += 1
        else:
            n_fail += 1

        # Format values for display
        pv_str = str(paper_val)
        cv_str = str(computed_val)
        print(f"{check_name:<40} {pv_str:>12} {cv_str:>12} {status:>8}")

    print("-" * 90)
    print(f"Inline checks: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")
    print()

    # ==================================================================
    # PART 5: BENCHMARK TABLE VERIFICATION
    # ==================================================================
    print("[6] Verifying benchmark table (Table 5)...")
    bench_checks = verify_benchmark_table(upro_for_verification, portfolio_6040, backtest_json)
    output["benchmark_checks"] = bench_checks

    print("-" * 90)
    print(f"{'Check':<40} {'Paper':>12} {'Computed':>12} {'Match':>8}")
    print("-" * 90)
    n_bench_pass = 0
    n_bench_fail = 0
    for check_name, check_data in bench_checks.items():
        paper_val = check_data["paper_value"]
        computed_val = check_data["computed_value"]
        match = check_data["match"]
        status = "OK" if match else "FAIL"
        if match:
            n_bench_pass += 1
        else:
            n_bench_fail += 1
        print(f"{check_name:<40} {paper_val:>12} {computed_val:>12} {status:>8}")

    print("-" * 90)
    print(f"Benchmark checks: {n_bench_pass} passed, {n_bench_fail} failed "
          f"out of {n_bench_pass + n_bench_fail}")
    print()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    total_pass = n_pass + n_bench_pass
    total_fail = n_fail + n_bench_fail
    total_checks = total_pass + total_fail

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Trading stats (Top 5):    "
          f"{sum(1 for v in trading_checks.values() if v['all_match'])}/5 fully match paper")
    print(f"  Inline number checks:     {n_pass}/{n_pass + n_fail} match")
    print(f"  Benchmark table checks:   {n_bench_pass}/{n_bench_pass + n_bench_fail} match")
    print(f"  -----------------------------------------")
    print(f"  TOTAL:                    {total_pass}/{total_checks} checks passed")

    if total_fail > 0:
        print(f"\n  WARNING: {total_fail} checks failed. Review mismatches above.")
    else:
        print(f"\n  ALL CHECKS PASSED.")

    print("=" * 90)

    # ==================================================================
    # SAVE JSON OUTPUT
    # ==================================================================
    output_path = os.path.join(RESULTS_DIR, "section4_verification.json")

    # Convert any numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        else:
            return convert_numpy(obj)

    output_clean = deep_convert(output)

    with open(output_path, "w") as f:
        json.dump(output_clean, f, indent=2)

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
