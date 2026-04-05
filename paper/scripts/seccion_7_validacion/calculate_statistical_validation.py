"""
Calculate all statistical validation metrics for the paper's Section 7.

Reads backtest_detail.pkl and trained_artifacts.pkl to compute:
- Bootstrap Sharpe CI (5,000 iterations) + P(S>0) + P(S>SPY)
- Lo (2002) AR(1)-corrected SE of Sharpe
- Diebold-Mariano test: strategy returns vs B&H returns (economic significance)
- Clark-West test: prediction MSE vs historical mean (predictive ability)
- Model Confidence Set (Hansen 2011) based on strategy returns
- Overfitting analysis: prediction metrics (MSE, R2, directional accuracy) train vs test
- Bonferroni correction (23 models)
- FDR Benjamini-Hochberg

Outputs: results/statistical_validation.json
"""

import numpy as np
import pickle
import json
import os
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

WINNERS = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM"]
ALL_MODELS = [
    "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost",
    "AutoARIMA", "ExponentialSmoothing", "Theta", "SeasonalNaive",
    "Prophet", "GARCH",
    "DLinear", "NBEATS", "NHiTS", "TCN", "TFT",
    "CNN_LSTM", "LSTM_Attention", "BiLSTM", "BiGRU",
]

N_BOOTSTRAP = 5000
SEED = 42


def load_data():
    """Load backtest detail and trained artifacts."""
    with open(os.path.join(RESULTS_DIR, "backtest_detail.pkl"), "rb") as f:
        detail = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return detail, artifacts


def compute_sharpe(returns, rf):
    """Compute annualized Sharpe ratio from daily returns."""
    n_years = len(returns) / 252
    equity = np.cumprod(1 + returns)
    total_ret = equity[-1] - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252
    return (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0


# =========================================================================
# 1. BOOTSTRAP SHARPE CONFIDENCE INTERVALS
# =========================================================================

def bootstrap_sharpe(returns, rf, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap CI for Sharpe ratio with P(S>0)."""
    rng = np.random.RandomState(seed)
    n = len(returns)
    sharpes = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_ret = returns[idx]
        boot_rf = rf[idx]
        sharpes[b] = compute_sharpe(boot_ret, boot_rf)

    ci_lower = float(np.percentile(sharpes, 2.5))
    ci_upper = float(np.percentile(sharpes, 97.5))
    prob_positive = float(np.mean(sharpes > 0) * 100)

    return {
        "sharpe_point": float(compute_sharpe(returns, rf)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prob_positive": prob_positive,
        "bootstrap_sharpes": sharpes,
    }


def lo2002_se(returns, sharpe_hat):
    """Lo (2002) AR(1)-corrected standard error of the Sharpe ratio."""
    T = len(returns)
    rho1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    numerator = 1 + (sharpe_hat ** 2 / 2) * (1 + rho1)
    denominator = T * (1 - rho1)
    se = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
    return float(se), float(rho1)


# =========================================================================
# 2. DIEBOLD-MARIANO TEST (on strategy returns vs B&H)
# =========================================================================

def newey_west_variance(d, max_lag=5):
    """Newey-West HAC estimator for variance of mean(d)."""
    T = len(d)
    d_demean = d - np.mean(d)
    gamma_0 = np.sum(d_demean ** 2) / T
    total = gamma_0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)
        gamma_j = np.sum(d_demean[j:] * d_demean[:-j]) / T
        total += 2 * weight * gamma_j
    return total / T


def diebold_mariano_returns(strategy_returns, benchmark_returns, max_lag=5):
    """
    Diebold-Mariano test comparing strategy returns vs benchmark returns.
    Uses loss function L = -return (so lower loss = higher return = better).
    d_t = -benchmark_return_t - (-strategy_return_t) = strategy_return_t - benchmark_return_t
    DM > 0: strategy is better. DM < 0: benchmark is better.
    """
    d = strategy_returns - benchmark_returns
    d_bar = np.mean(d)
    var_d = newey_west_variance(d, max_lag)
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0
    # One-sided p-value (alternative: strategy is better, i.e., DM > 0)
    p_value_one_sided = 1 - stats.norm.cdf(dm_stat)
    # Two-sided p-value
    p_value_two_sided = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value_one_sided": float(p_value_one_sided),
        "p_value_two_sided": float(p_value_two_sided),
        "significant_5pct": bool(p_value_one_sided < 0.05),
        "significant_1pct": bool(p_value_one_sided < 0.01),
        "mean_daily_excess": float(d_bar),
    }


# =========================================================================
# 3. CLARK-WEST TEST (on prediction accuracy)
# =========================================================================

def clark_west(predictions, actuals, benchmark_pred, max_lag=5):
    """
    Clark-West (2007) test for nested models.
    H0: restricted model (benchmark) has equal or better predictive ability.
    One-tailed test.
    """
    e1 = actuals - benchmark_pred
    e2 = actuals - predictions
    cw = e1 ** 2 - (e2 ** 2 - (e1 - e2) ** 2)
    cw_bar = np.mean(cw)
    var_cw = newey_west_variance(cw, max_lag)
    cw_stat = cw_bar / np.sqrt(var_cw) if var_cw > 0 else 0
    p_value = 1 - stats.norm.cdf(cw_stat)
    return {
        "cw_statistic": float(cw_stat),
        "p_value_one_tail": float(p_value),
        "rejects_h0_5pct": bool(p_value < 0.05),
        "rejects_h0_1pct": bool(p_value < 0.01),
    }


# =========================================================================
# 4. MODEL CONFIDENCE SET (Hansen 2011) - based on strategy returns
# =========================================================================

def model_confidence_set(model_losses, alpha=0.10, n_boot=3000, seed=SEED):
    """
    Model Confidence Set (Hansen, Lunde, Nason 2011).
    Uses T_max statistic with block bootstrap.
    model_losses: dict {model_name: array of daily losses (lower = better)}.
    """
    rng = np.random.RandomState(seed)
    names = list(model_losses.keys())
    losses = {k: np.array(v) for k, v in model_losses.items()}
    T = len(next(iter(losses.values())))
    block_len = max(1, int(T ** (1.0 / 3)))

    surviving = list(names)
    elimination_order = []

    while len(surviving) > 1:
        m = len(surviving)
        d_ij = {}
        for i in range(m):
            for j in range(i + 1, m):
                ni, nj = surviving[i], surviving[j]
                d_ij[(ni, nj)] = losses[ni] - losses[nj]

        t_stats = {}
        for (ni, nj), d in d_ij.items():
            d_bar = np.mean(d)
            se = np.std(d, ddof=1) / np.sqrt(T)
            t_stats[(ni, nj)] = d_bar / se if se > 0 else 0

        T_max_obs = max(abs(t) for t in t_stats.values())

        boot_T_max = np.empty(n_boot)
        for b in range(n_boot):
            idx = []
            while len(idx) < T:
                start = rng.randint(0, T)
                bl = rng.geometric(1.0 / block_len)
                for k in range(bl):
                    idx.append((start + k) % T)
            idx = np.array(idx[:T])

            boot_tmax = 0
            for (ni, nj), d in d_ij.items():
                d_boot = d[idx]
                d_bar_boot = np.mean(d_boot) - np.mean(d)
                se_boot = np.std(d_boot, ddof=1) / np.sqrt(T)
                t_boot = d_bar_boot / se_boot if se_boot > 0 else 0
                boot_tmax = max(boot_tmax, abs(t_boot))
            boot_T_max[b] = boot_tmax

        p_value = float(np.mean(boot_T_max >= T_max_obs))

        if p_value < alpha:
            avg_losses = {n: np.mean(losses[n]) for n in surviving}
            worst = max(avg_losses, key=avg_losses.get)
            elimination_order.append({
                "model": worst,
                "p_value": round(p_value, 4),
                "avg_loss": round(float(avg_losses[worst]), 6),
            })
            surviving.remove(worst)
        else:
            break

    return {
        "surviving_models": surviving,
        "alpha": alpha,
        "elimination_order": elimination_order,
        "n_eliminated": len(elimination_order),
        "final_p_value": round(p_value, 4),
    }


# =========================================================================
# 5. OVERFITTING ANALYSIS
# =========================================================================

def compute_prediction_metrics(predictions, actuals):
    """Compute MSE, MAE, R2, directional accuracy for raw predictions."""
    preds = np.array(predictions)
    actual = np.array(actuals)
    n = min(len(preds), len(actual))
    preds = preds[:n]
    actual = actual[:n]

    errors = actual - preds
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

    # Directional accuracy: did the model predict the correct sign?
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actual)))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": dir_acc,
    }


# =========================================================================
# 6. MULTIPLE TESTING CORRECTIONS
# =========================================================================

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni correction for multiple testing."""
    m = len(p_values)
    alpha_adj = alpha / m
    significant = {k: v < alpha_adj for k, v in p_values.items()}
    return {
        "alpha_original": alpha,
        "n_tests": m,
        "alpha_adjusted": float(alpha_adj),
        "significant_models": {k: bool(v) for k, v in significant.items()},
    }


def benjamini_hochberg(p_values, q=0.10):
    """Benjamini-Hochberg FDR procedure."""
    m = len(p_values)
    sorted_pvals = sorted(p_values.items(), key=lambda x: x[1])
    max_k = 0
    for k, (name, pval) in enumerate(sorted_pvals, 1):
        threshold = (k / m) * q
        if pval <= threshold:
            max_k = k

    significant = {}
    for k, (name, pval) in enumerate(sorted_pvals, 1):
        significant[name] = k <= max_k

    return {
        "q": q,
        "n_tests": m,
        "n_significant": max_k,
        "significant_models": {k: bool(v) for k, v in significant.items()},
        "sorted_pvalues": [(name, float(pval)) for name, pval in sorted_pvals],
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("STATISTICAL VALIDATION CALCULATOR -- Section 7 of Paper")
    print("=" * 80)

    np.random.seed(SEED)
    detail, artifacts = load_data()
    metadata = artifacts["metadata"]

    # Market returns and risk-free (test period)
    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    # Training period
    fwd_train = np.array(metadata["forward_returns_train"])
    rf_train = np.array(metadata["risk_free_train"])

    n_test = len(fwd_test)
    print(f"\nTest period: {n_test} days")
    print(f"Train period: {len(fwd_train)} days")

    results = {}

    # ==================================================================
    # 1. BOOTSTRAP SHARPE CONFIDENCE INTERVALS
    # ==================================================================
    print(f"\n{'='*60}")
    print("1. BOOTSTRAP SHARPE CONFIDENCE INTERVALS (5,000 iter)")
    print(f"{'='*60}")

    spy_boot = bootstrap_sharpe(fwd_test, rf_test)
    spy_sharpe = spy_boot["sharpe_point"]
    spy_sharpes_dist = spy_boot["bootstrap_sharpes"]

    print(f"\n  SPY B&H: Sharpe={spy_sharpe:.3f}, "
          f"CI=[{spy_boot['ci_lower']:.3f}, {spy_boot['ci_upper']:.3f}], "
          f"P(S>0)={spy_boot['prob_positive']:.1f}%")

    bootstrap_results = {
        "SPY_BH": {
            "sharpe": spy_sharpe,
            "ci_lower": spy_boot["ci_lower"],
            "ci_upper": spy_boot["ci_upper"],
            "prob_positive": spy_boot["prob_positive"],
            "prob_beat_spy": None,
        }
    }

    for model_name in WINNERS:
        model_detail = detail["models"][model_name]
        strat_returns = np.array(model_detail["strategy_returns"])
        n = min(len(strat_returns), len(rf_test))
        strat_returns = strat_returns[:n]
        rf = rf_test[:n]

        boot = bootstrap_sharpe(strat_returns, rf, seed=SEED + hash(model_name) % 10000)
        prob_beat_spy = float(np.mean(boot["bootstrap_sharpes"] > spy_sharpes_dist) * 100)
        se, rho1 = lo2002_se(strat_returns, boot["sharpe_point"])

        bootstrap_results[model_name] = {
            "sharpe": boot["sharpe_point"],
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "prob_positive": boot["prob_positive"],
            "prob_beat_spy": prob_beat_spy,
            "lo2002_se": se,
            "ar1_autocorrelation": rho1,
        }

        print(f"  {model_name}: Sharpe={boot['sharpe_point']:.3f}, "
              f"CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}], "
              f"P(S>0)={boot['prob_positive']:.1f}%, "
              f"P(S>SPY)={prob_beat_spy:.1f}%, "
              f"SE(Lo)={se:.4f}")

    results["bootstrap"] = bootstrap_results

    # ==================================================================
    # 2. DIEBOLD-MARIANO TEST (strategy returns vs B&H)
    # ==================================================================
    print(f"\n{'='*60}")
    print("2. DIEBOLD-MARIANO TEST (strategy returns vs B&H)")
    print(f"{'='*60}")

    dm_results = {}
    all_dm_pvalues = {}

    for model_name in ALL_MODELS:
        if model_name not in detail["models"]:
            continue
        model_detail = detail["models"][model_name]
        strat_returns = np.array(model_detail["strategy_returns"])
        n = min(len(strat_returns), len(fwd_test))
        strat_ret = strat_returns[:n]
        bh_ret = fwd_test[:n]

        dm = diebold_mariano_returns(strat_ret, bh_ret)
        dm_results[model_name] = dm
        all_dm_pvalues[model_name] = dm["p_value_one_sided"]

        if model_name in WINNERS:
            sig = "***" if dm["significant_1pct"] else ("**" if dm["significant_5pct"] else "")
            print(f"  {model_name}: DM={dm['dm_statistic']:+.3f}, "
                  f"p(1-tail)={dm['p_value_one_sided']:.4f} {sig}")

    results["diebold_mariano"] = {k: dm_results[k] for k in WINNERS if k in dm_results}
    results["diebold_mariano_all"] = dm_results

    # ==================================================================
    # 3. CLARK-WEST TEST (prediction accuracy vs historical mean)
    # ==================================================================
    print(f"\n{'='*60}")
    print("3. CLARK-WEST TEST (prediction MSE vs historical mean)")
    print(f"{'='*60}")

    hist_mean = np.mean(fwd_train)
    benchmark_pred = np.full(n_test, hist_mean)
    print(f"  Historical mean benchmark: {hist_mean*100:.4f}% daily")

    cw_results = {}
    all_cw_pvalues = {}

    for model_name in ALL_MODELS:
        if model_name not in artifacts["models"]:
            continue
        test_preds = np.array(artifacts["models"][model_name]["test_predictions"])
        n = min(len(test_preds), len(fwd_test))
        preds = test_preds[:n]
        actuals = fwd_test[:n]
        bench = benchmark_pred[:n]

        cw = clark_west(preds, actuals, bench)
        cw_results[model_name] = cw
        all_cw_pvalues[model_name] = cw["p_value_one_tail"]

        if model_name in WINNERS:
            sig = "***" if cw["rejects_h0_1pct"] else ("**" if cw["rejects_h0_5pct"] else "")
            print(f"  {model_name}: CW={cw['cw_statistic']:+.3f}, "
                  f"p={cw['p_value_one_tail']:.4f} {sig}")

    results["clark_west"] = {k: cw_results[k] for k in WINNERS if k in cw_results}
    results["clark_west_all"] = cw_results

    # ==================================================================
    # 4. MODEL CONFIDENCE SET (Hansen 2011) - strategy returns
    # ==================================================================
    print(f"\n{'='*60}")
    print("4. MODEL CONFIDENCE SET (strategy returns, alpha=0.10)")
    print(f"{'='*60}")

    # Loss = negative strategy return (lower loss = higher return = better)
    model_losses = {}
    for model_name in ALL_MODELS:
        if model_name not in detail["models"]:
            continue
        strat_ret = np.array(detail["models"][model_name]["strategy_returns"])
        n = min(len(strat_ret), n_test)
        model_losses[model_name] = -strat_ret[:n]

    # Also add SPY B&H as a model
    model_losses["SPY_BH"] = -fwd_test

    print(f"  Running MCS with {len(model_losses)} models "
          f"(including SPY B&H), 3,000 bootstrap iterations...")

    mcs = model_confidence_set(model_losses, alpha=0.10, n_boot=3000)
    results["mcs"] = {
        "surviving_models": mcs["surviving_models"],
        "alpha": mcs["alpha"],
        "n_eliminated": mcs["n_eliminated"],
        "elimination_order": mcs["elimination_order"],
        "final_p_value": mcs["final_p_value"],
    }

    print(f"  MCS survivors: {mcs['surviving_models']}")
    print(f"  Eliminated: {mcs['n_eliminated']} models")
    if mcs["elimination_order"]:
        print(f"  Last 5 eliminated:")
        for e in mcs["elimination_order"][-5:]:
            print(f"    {e['model']}: p={e['p_value']:.4f}")

    # ==================================================================
    # 5. OVERFITTING ANALYSIS (prediction metrics: train vs test)
    # ==================================================================
    print(f"\n{'='*60}")
    print("5. OVERFITTING ANALYSIS (prediction metrics train vs test)")
    print(f"{'='*60}")

    overfitting_models = WINNERS + ["XGBoost"]
    overfitting_results = {}

    for model_name in overfitting_models:
        if model_name not in artifacts["models"]:
            continue
        model_data = artifacts["models"][model_name]

        # Test metrics (from raw predictions vs realized returns)
        test_preds = np.array(model_data["test_predictions"])
        test_metrics = compute_prediction_metrics(test_preds, fwd_test)

        # Train metrics
        train_preds = np.array(model_data["train_predictions"])
        train_metrics = compute_prediction_metrics(train_preds, fwd_train)

        # Degradation in directional accuracy
        da_degradation = ((test_metrics["directional_accuracy"] -
                           train_metrics["directional_accuracy"]) /
                          train_metrics["directional_accuracy"] * 100
                          if train_metrics["directional_accuracy"] > 0.001 else 0)

        # MSE ratio (test/train — >1 means worse on test)
        mse_ratio = (test_metrics["mse"] / train_metrics["mse"]
                     if train_metrics["mse"] > 0 else float("inf"))

        overfitting_results[model_name] = {
            "train_mse": train_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "mse_ratio": float(mse_ratio),
            "train_directional_accuracy": train_metrics["directional_accuracy"],
            "test_directional_accuracy": test_metrics["directional_accuracy"],
            "da_degradation_pct": float(da_degradation),
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
        }

        print(f"  {model_name}:")
        print(f"    DA: Train={train_metrics['directional_accuracy']:.3f}, "
              f"Test={test_metrics['directional_accuracy']:.3f}, "
              f"Degrad={da_degradation:.1f}%")
        print(f"    MSE: Train={train_metrics['mse']:.6f}, "
              f"Test={test_metrics['mse']:.6f}, "
              f"Ratio={mse_ratio:.2f}x")
        print(f"    R2: Train={train_metrics['r2']:.4f}, "
              f"Test={test_metrics['r2']:.4f}")

    results["overfitting"] = overfitting_results

    # ==================================================================
    # 6. MULTIPLE TESTING CORRECTIONS
    # ==================================================================
    print(f"\n{'='*60}")
    print("6. MULTIPLE TESTING CORRECTIONS")
    print(f"{'='*60}")

    # Use DM one-sided p-values for all models
    bonf = bonferroni_correction(all_dm_pvalues)
    fdr = benjamini_hochberg(all_dm_pvalues, q=0.10)

    results["bonferroni"] = bonf
    results["fdr"] = fdr

    print(f"\n  Bonferroni (alpha_adj={bonf['alpha_adjusted']:.4f}):")
    for name in WINNERS:
        if name in bonf["significant_models"]:
            sig = "YES" if bonf["significant_models"][name] else "no"
            pval = all_dm_pvalues.get(name, 1)
            print(f"    {name}: {sig} (p={pval:.4f})")

    print(f"\n  FDR (q=0.10): {fdr['n_significant']} models significant")
    fdr_winners = [n for n, s in fdr["significant_models"].items() if s]
    print(f"    Significant: {fdr_winners}")

    # ==================================================================
    # SAVE
    # ==================================================================
    output_path = os.path.join(RESULTS_DIR, "statistical_validation.json")

    def clean_nan(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return clean_nan(obj.tolist())
        return obj

    save_results = clean_nan(results)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved to {output_path}")
    print(f"{'='*80}")

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    print(f"\n{'Model':<18} {'Sharpe':>7} {'CI 95%':>16} {'P(S>0)':>8} "
          f"{'P(S>SPY)':>9} {'DM':>7} {'CW':>7}")
    print("-" * 75)
    for name in WINNERS:
        b = bootstrap_results[name]
        dm = dm_results.get(name, {})
        cw = cw_results.get(name, {})
        p_spy = f"{b['prob_beat_spy']:.1f}%" if b['prob_beat_spy'] is not None else "N/A"
        dm_str = f"{dm.get('dm_statistic', 0):+.2f}" if dm else "N/A"
        cw_str = f"{cw.get('cw_statistic', 0):+.2f}" if cw else "N/A"
        print(f"{name:<18} {b['sharpe']:>7.3f} "
              f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
              f"{b['prob_positive']:>7.1f}% {p_spy:>9} "
              f"{dm_str:>7} {cw_str:>7}")

    b = bootstrap_results["SPY_BH"]
    print(f"{'SPY B&H':<18} {b['sharpe']:>7.3f} "
          f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
          f"{b['prob_positive']:>7.1f}% {'--':>9} "
          f"{'--':>7} {'--':>7}")

    # Overfitting summary
    print(f"\n{'Model':<18} {'Train DA':>10} {'Test DA':>10} {'Degrad%':>10} "
          f"{'MSE Ratio':>10}")
    print("-" * 60)
    for name in WINNERS + ["XGBoost"]:
        o = overfitting_results.get(name)
        if o:
            print(f"{name:<18} {o['train_directional_accuracy']:>10.3f} "
                  f"{o['test_directional_accuracy']:>10.3f} "
                  f"{o['da_degradation_pct']:>9.1f}% "
                  f"{o['mse_ratio']:>9.2f}x")


if __name__ == "__main__":
    main()
