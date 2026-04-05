# -*- coding: utf-8 -*-
"""
================================================================================
OPTIMIZE_AND_BACKTEST.PY — Meta-KNN Threshold Optimization + Final Backtest
================================================================================

Replaces the previous two-step pipeline:
  - optimize_model_params.py (oracle-based grid search over test period)
  - final_long_only_backtest_new.py (static thresholds from oracle)

NEW APPROACH: Non-oracle, fully out-of-sample threshold optimization.

Two components:

1. META-KNN THRESHOLD PREDICTION:
   - Slides 63-day windows over TRAINING predictions (step=21d)
   - For each window, computes 7 meta-features describing prediction behavior
   - Evaluates all 26 threshold combos on NEXT 63 training days -> labels optimal
   - Trains KNN(K=5, distance-weighted) on this meta-dataset
   - For each TEST day, predicts optimal (q_ext, q_mod) from recent features

2. SIGNAL QUALITY FILTER:
   - Computes rolling 63-day std of model predictions
   - If std < 0.001: predictions lack variation -> default to CASH
   - Prevents degenerate models (near-constant predictions) from being
     amplified into extreme positions by the percentile ranking system

NO FUTURE DATA USED at any point. All thresholds are determined from
training data via KNN, applied dynamically in test.

OUTPUTS (backward-compatible with previous pipeline):
  - results/final_long_only_backtest.json
  - results/long_only_equity_curves.json
  - results/optimal_model_params.json
  - results/backtest_detail.pkl (positions/percentiles for update_backend_data.py)

================================================================================
Author: David Gonzalez Canon
================================================================================
"""

import numpy as np
import pickle
import os
import json
from datetime import datetime
from collections import Counter
from scipy.stats import skew as scipy_skew

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},
    'SPY':  {'expense_ratio': 0.0009, 'bid_ask': 0.0002},
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},
}

# All valid (q_ext, q_mod) combos where q_mod > q_ext
PARAM_COMBOS = [(q_ext, q_mod)
                for q_ext in [5, 10, 15, 20, 25, 30]
                for q_mod in [20, 30, 40, 50, 60]
                if q_mod > q_ext]

COMBO_TO_IDX = {combo: i for i, combo in enumerate(PARAM_COMBOS)}
IDX_TO_COMBO = {i: combo for combo, i in COMBO_TO_IDX.items()}

PCTILE_WINDOW = 63
FEATURE_WINDOW = 63
EVAL_HORIZON = 63
SLIDE_STEP = 21

# Signal quality filter: minimum rolling std of predictions
# Calibrated: all genuine models have median rolling std > 0.001,
# all degenerate models (XGBoost, GradientBoosting, AutoARIMA, SeasonalNaive)
# have median < 0.001
MIN_SIGNAL_STD = 0.001

FEATURE_NAMES = [
    'mean_pred', 'std_pred', 'skew_pred', 'autocorr_pred',
    'trend_pred', 'mean_vol', 'recent_return'
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_rolling_percentiles(all_preds, window=63):
    """Compute rolling percentiles for the entire series (backward-looking)."""
    n = len(all_preds)
    percentiles = np.full(n, 50.0)
    for i in range(n):
        start = max(0, i - window + 1)
        w = all_preds[start:i+1]
        if len(w) < 5:
            continue
        percentiles[i] = np.mean(w <= all_preds[i]) * 100
    return percentiles


def compute_rolling_std(preds, window=63):
    """Compute rolling standard deviation of predictions (backward-looking)."""
    n = len(preds)
    rolling_std = np.full(n, 0.0)
    for i in range(n):
        start = max(0, i - window + 1)
        w = preds[start:i+1]
        if len(w) >= 5:
            rolling_std[i] = np.std(w)
    return rolling_std


def evaluate_combo(percentiles, fwd_returns, rf, q_ext, q_mod):
    """Evaluate a threshold combo over a window (penalized Sharpe)."""
    if len(percentiles) < 10:
        return -999

    thresh_3x = 100 - q_ext
    thresh_1x = 100 - q_mod

    positions = np.zeros(len(percentiles))
    positions[percentiles >= thresh_3x] = 3
    mask_1x = (percentiles >= thresh_1x) & (percentiles < thresh_3x)
    positions[mask_1x] = 1

    returns = rf + positions * (fwd_returns - rf)

    std = np.std(returns)
    if std < 0.0001:
        return -999

    sharpe = (np.mean(returns) - np.mean(rf)) / std * np.sqrt(252)
    sharpe = np.clip(sharpe, -5, 5)

    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    max_dd = np.max((running_max - equity) / running_max)

    score = sharpe
    if max_dd > 0.20:
        score -= 0.5

    return score


def compute_meta_features(preds_window, market_returns_window):
    """Compute 7 meta-features from a window of predictions and market data."""
    n = len(preds_window)

    mean_pred = np.mean(preds_window)

    std_pred = np.std(preds_window)
    if std_pred < 1e-12:
        std_pred = 1e-12

    if n >= 3 and std_pred > 1e-10:
        skew_pred = float(scipy_skew(preds_window))
    else:
        skew_pred = 0.0

    if n >= 3:
        x = preds_window[:-1]
        y = preds_window[1:]
        mx, my = np.mean(x), np.mean(y)
        num = np.sum((x - mx) * (y - my))
        den = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
        autocorr_pred = num / den if den > 1e-12 else 0.0
    else:
        autocorr_pred = 0.0

    t = np.arange(n, dtype=float)
    t_mean = np.mean(t)
    p_mean = np.mean(preds_window)
    num = np.sum((t - t_mean) * (preds_window - p_mean))
    den = np.sum((t - t_mean)**2)
    trend_pred = num / den if den > 1e-12 else 0.0

    if n >= 21:
        vols = []
        for i in range(21, n + 1):
            vols.append(np.std(market_returns_window[i-21:i]))
        mean_vol = np.mean(vols) if vols else np.std(market_returns_window)
    else:
        mean_vol = np.std(market_returns_window)

    recent_return = np.prod(1 + market_returns_window) - 1

    result = np.array([
        mean_pred, std_pred, skew_pred, autocorr_pred,
        trend_pred, mean_vol, recent_return
    ])
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def build_meta_dataset(train_preds, train_percentiles, fwd_train, rf_train):
    """
    Phase 1: Build meta-learning dataset from TRAINING data only.

    Slides 63-day windows over training predictions, computes 7 features per
    window, evaluates all 26 combos on the NEXT 63 training days, and records
    the optimal combo as the label.
    """
    n = len(train_preds)
    X_list = []
    y_list = []

    max_start = n - FEATURE_WINDOW - EVAL_HORIZON
    if max_start < 0:
        return np.array([]).reshape(0, 7), np.array([])

    for start in range(0, max_start + 1, SLIDE_STEP):
        end = start + FEATURE_WINDOW
        eval_end = end + EVAL_HORIZON

        preds_w = train_preds[start:end]
        market_w = fwd_train[start:end]
        features = compute_meta_features(preds_w, market_w)

        eval_pctiles = train_percentiles[end:eval_end]
        eval_fwd = fwd_train[end:eval_end]
        eval_rf = rf_train[end:eval_end]

        if len(eval_pctiles) < 10:
            continue

        best_score = -999
        best_combo_idx = 0
        for combo_idx, (q_ext, q_mod) in enumerate(PARAM_COMBOS):
            score = evaluate_combo(eval_pctiles, eval_fwd, eval_rf, q_ext, q_mod)
            if score > best_score:
                best_score = score
                best_combo_idx = combo_idx

        if np.all(np.isfinite(features)):
            X_list.append(features)
            y_list.append(best_combo_idx)

    if not X_list:
        return np.array([]).reshape(0, 7), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)
    valid_mask = np.all(np.isfinite(X), axis=1)
    return X[valid_mask], y[valid_mask]


def train_meta_knn(X_train, y_train):
    """Phase 2: Train KNN meta-model on training meta-dataset."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(5, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_scaled, y_train)

    train_acc = knn.score(X_scaled, y_train)

    return {
        'scaler': scaler,
        'knn': knn,
        'train_acc': train_acc,
    }


def meta_knn_filtered_backtest(test_preds, train_preds,
                                fwd_test, fwd_train,
                                rf_test, rf_train,
                                meta_model, min_signal_std):
    """
    Phase 3: Apply Meta-KNN with signal quality filter in test.

    For each test day t:
    1. Compute rolling std of predictions over last 63 days
    2. If std < min_signal_std -> CASH (no reliable signal)
    3. Otherwise: compute 7 features, predict thresholds via KNN, apply
    """
    n_tr = min(len(train_preds), len(fwd_train), len(rf_train))
    train_preds_al = train_preds[-n_tr:]
    fwd_train_al = fwd_train[-n_tr:]
    rf_train_al = rf_train[-n_tr:]

    all_preds = np.concatenate([train_preds_al, test_preds])
    all_fwd = np.concatenate([fwd_train_al, fwd_test])
    all_rf = np.concatenate([rf_train_al, rf_test])

    n_train = len(train_preds_al)
    n_test = len(test_preds)

    # Pre-compute percentiles and rolling std for entire series (backward-looking)
    all_percentiles = compute_rolling_percentiles(all_preds, window=PCTILE_WINDOW)
    all_rolling_std = compute_rolling_std(all_preds, window=PCTILE_WINDOW)

    scaler = meta_model['scaler']
    knn = meta_model['knn']

    positions = np.zeros(n_test)
    percentiles_out = np.zeros(n_test)
    chosen_params = []
    signal_valid = np.zeros(n_test, dtype=bool)

    for t in range(n_test):
        abs_t = n_train + t

        percentiles_out[t] = all_percentiles[abs_t]

        # Step 1: Check signal quality
        today_std = all_rolling_std[abs_t]
        if today_std < min_signal_std:
            positions[t] = 0
            chosen_params.append((0, 0))
            signal_valid[t] = False
            continue

        signal_valid[t] = True

        # Step 2: Compute meta-features from last FEATURE_WINDOW days
        feat_start = max(0, abs_t - FEATURE_WINDOW)
        feat_end = abs_t

        if feat_end - feat_start < 10:
            best_combo = (10, 30)
        else:
            preds_w = all_preds[feat_start:feat_end]
            market_w = all_fwd[feat_start:feat_end]
            features = compute_meta_features(preds_w, market_w)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            combo_idx = knn.predict(features_scaled)[0]
            best_combo = IDX_TO_COMBO[combo_idx]

        # Step 3: Apply thresholds to today's percentile
        today_pctile = all_percentiles[abs_t]
        thresh_3x = 100 - best_combo[0]
        thresh_1x = 100 - best_combo[1]

        if today_pctile >= thresh_3x:
            positions[t] = 3
        elif today_pctile >= thresh_1x:
            positions[t] = 1
        else:
            positions[t] = 0

        chosen_params.append(best_combo)

    return positions, percentiles_out, chosen_params, signal_valid


def calculate_returns_with_costs(positions, market_returns, risk_free):
    """Compute net returns with full cost model."""
    n = len(positions)
    returns = np.zeros(n)
    gross_returns = np.zeros(n)
    expense_costs = np.zeros(n)
    trading_costs = np.zeros(n)
    vol_drag_costs = np.zeros(n)
    pos_to_inst = {3: 'UPRO', 1: 'SPY', 0: 'CASH'}

    for i in range(n):
        pos = int(positions[i])
        inst = pos_to_inst.get(pos, 'CASH')
        cfg = INSTRUMENTS[inst]

        if pos == 0:
            gross = risk_free[i]
        else:
            gross = risk_free[i] + pos * (market_returns[i] - risk_free[i])

        gross_returns[i] = gross

        expense = cfg['expense_ratio'] / 252
        trading = 0
        if i > 0 and positions[i] != positions[i-1]:
            prev_inst = pos_to_inst.get(int(positions[i-1]), 'CASH')
            trading += INSTRUMENTS[prev_inst]['bid_ask'] + cfg['bid_ask']

        vol_drag = 0
        if pos == 3:
            vol_drag = 0.5 * 6 * (0.01)**2

        expense_costs[i] = expense
        trading_costs[i] = trading
        vol_drag_costs[i] = vol_drag

        returns[i] = gross - expense - trading - vol_drag

    return returns, gross_returns, expense_costs, trading_costs, vol_drag_costs


def compute_metrics(returns, gross_returns, rf, positions, n_test):
    """Compute all strategy metrics."""
    equity = np.cumprod(1 + returns)
    gross_equity = np.cumprod(1 + gross_returns)
    total_ret = equity[-1] - 1
    n_years = n_test / 252
    annual_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252
    sharpe = (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0

    running_max = np.maximum.accumulate(equity)
    max_dd = np.max((running_max - equity) / running_max)

    # Sortino ratio
    rf_daily = np.mean(rf)
    downside = returns - rf_daily
    downside = np.minimum(downside, 0)
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
    sortino = (annual_ret - rf_annual) / downside_std if downside_std > 0 else 0

    # Calmar ratio
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    n_trades = int(np.sum(np.diff(positions) != 0))

    pct_3x = np.mean(positions == 3) * 100
    pct_1x = np.mean(positions == 1) * 100
    pct_cash = np.mean(positions == 0) * 100

    # Cost breakdown ($10,000 initial)
    initial_capital = 10000
    final_capital = initial_capital * equity[-1]
    gross_final = initial_capital * gross_equity[-1]
    total_costs_dollar = gross_final - final_capital

    return {
        'total_return': float(total_ret),
        'annual_return': float(annual_ret),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'calmar': float(calmar),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'pct_3x': float(pct_3x),
        'pct_1x': float(pct_1x),
        'pct_cash': float(pct_cash),
        'n_trades': n_trades,
        'final_capital': float(final_capital),
        'gross_final': float(gross_final),
        'total_costs': float(total_costs_dollar),
        'equity_curve': equity.tolist(),
        'positions': positions.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 100)
    print("OPTIMIZE & BACKTEST — Meta-KNN + Signal Quality Filter")
    print("=" * 100)
    print(f"Timestamp: {datetime.now()}")
    print(f"Feature window: {FEATURE_WINDOW}d | Eval horizon: {EVAL_HORIZON}d | Slide step: {SLIDE_STEP}d")
    print(f"Combos: {len(PARAM_COMBOS)} | Percentile window: {PCTILE_WINDOW}d")
    print(f"Signal quality filter: rolling {PCTILE_WINDOW}d std >= {MIN_SIGNAL_STD}")
    print(f"KNN: K=5, distance-weighted")
    print(f"Meta-features: {FEATURE_NAMES}")
    print()

    # --- Load data ---
    print("[1] Loading trained_artifacts.pkl...")
    artifacts_path = os.path.join(MODELS_DIR, "trained_artifacts.pkl")
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    metadata = artifacts['metadata']
    fwd_test = np.array(metadata['forward_returns_test'][:-1])
    rf_test = np.array(metadata['risk_free_test'][:-1])
    fwd_train = np.array(metadata['forward_returns_train'])
    rf_train = np.array(metadata['risk_free_train'])

    n_test = len(fwd_test)
    n_train = len(fwd_train)
    n_years = n_test / 252
    print(f"    Train: {n_train} days | Test: {n_test} days ({n_years:.1f} years)")

    # Benchmark
    spy_equity = np.cumprod(1 + fwd_test)
    spy_return = float(spy_equity[-1] - 1)
    spy_cagr = float((1 + spy_return) ** (252/n_test) - 1)
    spy_rf_annual = float(np.mean(rf_test) * 252)
    spy_annual_vol = float(np.std(fwd_test) * np.sqrt(252))
    spy_sharpe = float((spy_cagr - spy_rf_annual) / spy_annual_vol) if spy_annual_vol > 0 else 0
    spy_running_max = np.maximum.accumulate(spy_equity)
    spy_max_dd = float(np.max((spy_running_max - spy_equity) / spy_running_max))
    print(f"    SPY B&H: Return={spy_return*100:+.1f}%, Sharpe={spy_sharpe:.3f}, MaxDD={spy_max_dd*100:.1f}%")

    # --- Process each model ---
    backtest_results = {}
    optimal_params = {}
    equity_curves = {'dates': list(range(n_test)), 'spy': spy_equity.tolist(), 'models': {}}
    detail_data = {}  # For update_backend_data.py

    model_count = len(artifacts['models'])
    print(f"\n[2] Processing {model_count} models...\n")

    for model_idx, (model_name, model_data) in enumerate(artifacts['models'].items()):
        test_preds = np.array(model_data['test_predictions']).flatten()
        train_preds = np.array(model_data.get('train_predictions', [])).flatten()

        if len(test_preds) == 0 or len(train_preds) == 0:
            print(f"    {model_name:<20} SKIP (no predictions)")
            continue

        nt = min(len(test_preds), len(fwd_test))
        test_preds_al = test_preds[:nt]
        fwd_al = fwd_test[:nt]
        rf_al = rf_test[:nt]

        n_tr = min(len(train_preds), len(fwd_train), len(rf_train))
        train_preds_tr = train_preds[:n_tr]
        fwd_train_tr = fwd_train[:n_tr]
        rf_train_tr = rf_train[:n_tr]

        print(f"  [{model_idx+1}/{model_count}] {model_name}")

        # Prediction quality diagnostic
        test_std = np.std(test_preds_al)
        n_unique = len(np.unique(np.round(test_preds_al, 6)))

        # Phase 1: Build meta-learning dataset from training data
        train_percentiles = compute_rolling_percentiles(train_preds_tr, window=PCTILE_WINDOW)

        X_meta, y_meta = build_meta_dataset(
            train_preds_tr, train_percentiles,
            fwd_train_tr, rf_train_tr
        )

        if len(X_meta) < 10:
            print(f"        SKIP (insufficient meta-dataset: {len(X_meta)} samples)")
            continue

        n_unique_classes = len(np.unique(y_meta))
        print(f"        Meta-dataset: {len(X_meta)} windows, {n_unique_classes} unique combos")

        # Phase 2: Train KNN on training meta-dataset
        meta_model = train_meta_knn(X_meta, y_meta)
        print(f"        KNN train accuracy: {meta_model['train_acc']:.1%}")

        # Phase 3: Backtest with signal quality filter
        positions, percentiles, chosen_params, signal_valid = meta_knn_filtered_backtest(
            test_preds_al, train_preds,
            fwd_al, fwd_train,
            rf_al, rf_train,
            meta_model, MIN_SIGNAL_STD,
        )

        # Compute returns with full cost model
        net_returns, gross_returns, expense_costs, trading_costs, vol_drag_costs = \
            calculate_returns_with_costs(positions, fwd_al, rf_al)

        metrics = compute_metrics(net_returns, gross_returns, rf_al, positions, nt)

        # Directional accuracy
        pred_dir = np.sign(test_preds_al[:len(fwd_al)])
        actual_dir = np.sign(fwd_al[:len(test_preds_al)])
        da = float(np.mean(pred_dir == actual_dir))
        metrics['directional_accuracy'] = da

        # Signal quality stats
        pct_signal_days = float(np.mean(signal_valid) * 100)
        pct_filtered_days = 100 - pct_signal_days
        metrics['signal_quality'] = {
            'pct_signal_days': pct_signal_days,
            'pct_filtered_days': pct_filtered_days,
            'n_unique_test_preds': int(n_unique),
            'test_pred_std': float(test_std),
            'min_signal_std': float(MIN_SIGNAL_STD),
        }

        # Dynamic threshold stats (only for signal-valid days)
        valid_params = [p for p, v in zip(chosen_params, signal_valid) if v]
        if valid_params:
            p_counter = Counter(valid_params)
            most_common_p = p_counter.most_common(1)[0]
            unique_combos = len(set(valid_params))
        else:
            most_common_p = ((0, 0), 0)
            unique_combos = 0

        metrics['dynamic_stats'] = {
            'unique_combos_used': unique_combos,
            'most_common_params': list(most_common_p[0]) if valid_params else [0, 0],
            'most_common_pct': float(most_common_p[1] / len(valid_params) * 100) if valid_params else 0,
            'n_meta_train_samples': len(X_meta),
            'knn_train_acc': float(meta_model['train_acc']),
        }

        # Store results
        backtest_results[model_name] = metrics
        equity_curves['models'][model_name] = metrics['equity_curve']

        # Optimal params (most common combo for display/compatibility)
        if valid_params:
            repr_q_ext, repr_q_mod = most_common_p[0]
        else:
            repr_q_ext, repr_q_mod = 10, 30
        optimal_params[model_name] = {
            'params': {
                'q_long_extreme': repr_q_ext,
                'q_long_moderate': repr_q_mod,
            },
            'method': 'meta_knn_dynamic',
            'note': 'Thresholds are DYNAMIC per day via KNN. These are the most common values for display.',
            'expected_sharpe': metrics['sharpe'],
            'expected_return': metrics['total_return'],
            'expected_max_dd': -metrics['max_drawdown'],
            'characteristics': {
                'directional_accuracy': da,
                'pred_std': float(test_std),
                'n_unique_preds': int(n_unique),
                'pct_signal_days': pct_signal_days,
            },
        }

        # Detail data for update_backend_data.py
        costs = expense_costs + trading_costs + vol_drag_costs
        equity = np.cumprod(1 + net_returns)
        drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)

        detail_data[model_name] = {
            'predictions': test_preds_al.tolist(),
            'percentiles': percentiles.tolist(),
            'positions': positions.tolist(),
            'strategy_returns': net_returns.tolist(),
            'gross_returns': gross_returns.tolist(),
            'equity_curve': equity.tolist(),
            'drawdown': drawdown.tolist(),
            'daily_costs': costs.tolist(),
            'expense_costs': expense_costs.tolist(),
            'trading_costs': trading_costs.tolist(),
            'vol_drag_costs': vol_drag_costs.tolist(),
            'signal_valid': signal_valid.tolist(),
            'chosen_params': [list(p) for p in chosen_params],
        }

        beat = " BEATS SPY!" if metrics['total_return'] > spy_return else ""
        print(f"        Return: {metrics['total_return']*100:+7.1f}% | "
              f"Sharpe: {metrics['sharpe']:+6.3f} | MaxDD: {metrics['max_drawdown']*100:5.1f}% | "
              f"Signal: {pct_signal_days:.0f}%{beat}")
        print()

    # =================================================================
    # RESULTS TABLE
    # =================================================================
    print("\n" + "=" * 110)
    print("FINAL RESULTS — Meta-KNN + Signal Quality Filter (sorted by Return)")
    print("=" * 110)

    sorted_results = sorted(backtest_results.items(),
                            key=lambda x: x[1]['total_return'], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<20} {'Return':>10} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'%3x':>7} {'%1x':>7} {'%Cash':>7} {'Trades':>7} {'Signal%':>8}")
    print("-" * 115)

    for i, (name, m) in enumerate(sorted_results):
        beat = "*" if m['total_return'] > spy_return else " "
        sq = m['signal_quality']
        print(f"{i+1:<5} {name:<20} {m['total_return']*100:>+9.1f}%{beat} {m['sharpe']:>8.3f} "
              f"{m['sortino']:>8.3f} {m['max_drawdown']*100:>7.1f}% "
              f"{m['pct_3x']:>6.1f}% {m['pct_1x']:>6.1f}% "
              f"{m['pct_cash']:>6.1f}% {m['n_trades']:>7} {sq['pct_signal_days']:>7.0f}%")

    print("-" * 115)
    print(f"{'':5} {'SPY B&H':<20} {spy_return*100:>+9.1f}%  {spy_sharpe:>8.3f} "
          f"{'':>8} {spy_max_dd*100:>7.1f}%")

    beating_spy = [(n, m) for n, m in sorted_results if m['total_return'] > spy_return]
    print(f"\n[MODELS BEATING SPY: {len(beating_spy)}/{len(backtest_results)}]")
    for name, m in beating_spy:
        excess = m['total_return'] - spy_return
        sq = m['signal_quality']
        print(f"  - {name}: {m['total_return']*100:+.1f}% (excess: {excess*100:+.1f}%, "
              f"Sharpe: {m['sharpe']:.3f}, Signal: {sq['pct_signal_days']:.0f}%)")

    # Statistics
    print(f"\n[STATISTICS]")
    avg_return = np.mean([m['total_return'] for m in backtest_results.values()])
    avg_sharpe = np.mean([m['sharpe'] for m in backtest_results.values()])
    avg_dd = np.mean([m['max_drawdown'] for m in backtest_results.values()])
    print(f"  Average return: {avg_return*100:+.1f}%")
    print(f"  Average Sharpe: {avg_sharpe:.3f}")
    print(f"  Average MaxDD: {avg_dd*100:.1f}%")

    top_sharpe = sorted(backtest_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:5]
    print(f"\n[TOP 5 BY SHARPE]")
    for name, m in top_sharpe:
        print(f"  - {name}: Sharpe={m['sharpe']:.3f}, Return={m['total_return']*100:+.1f}%")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    print("\n[3] Saving results...")

    # 1. final_long_only_backtest.json (backward-compatible)
    output_backtest = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'LONG-ONLY (0, +1, +3)',
        'methodology': 'Meta-KNN dynamic thresholds + signal quality filter (no future data)',
        'benchmark': {
            'total_return': spy_return,
            'sharpe': spy_sharpe,
            'max_drawdown': spy_max_dd,
        },
        'models': {
            n: {k: v for k, v in m.items()
                if k not in ['equity_curve', 'positions']}
            for n, m in backtest_results.items()
        },
        'summary': {
            'models_beating_spy': len(beating_spy),
            'beating_spy_models': [n for n, _ in beating_spy],
            'avg_return': float(avg_return),
            'avg_sharpe': float(avg_sharpe),
            'avg_max_dd': float(avg_dd),
        },
        'config': {
            'feature_window': FEATURE_WINDOW,
            'eval_horizon': EVAL_HORIZON,
            'slide_step': SLIDE_STEP,
            'pctile_window': PCTILE_WINDOW,
            'min_signal_std': MIN_SIGNAL_STD,
            'n_combos': len(PARAM_COMBOS),
            'feature_names': FEATURE_NAMES,
        },
    }

    path = os.path.join(RESULTS_DIR, "final_long_only_backtest.json")
    with open(path, 'w') as f:
        json.dump(output_backtest, f, indent=2)
    print(f"    [OK] {path}")

    # 2. long_only_equity_curves.json (backward-compatible)
    path = os.path.join(RESULTS_DIR, "long_only_equity_curves.json")
    with open(path, 'w') as f:
        json.dump(equity_curves, f)
    print(f"    [OK] {path}")

    # 3. optimal_model_params.json (backward-compatible structure)
    path = os.path.join(RESULTS_DIR, "optimal_model_params.json")
    with open(path, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    print(f"    [OK] {path}")

    # 4. backtest_detail.pkl (for update_backend_data.py)
    detail_output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'meta_knn_filtered',
        'test_days': n_test,
        'models': detail_data,
    }
    path = os.path.join(RESULTS_DIR, "backtest_detail.pkl")
    with open(path, 'wb') as f:
        pickle.dump(detail_output, f)
    print(f"    [OK] {path}")

    print(f"\n{'='*100}")
    print(f"PIPELINE COMPLETE — {len(beating_spy)}/{len(backtest_results)} models beat SPY B&H")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
