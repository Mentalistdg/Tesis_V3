# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION DE LA SECCION 7: VALIDACION ESTADISTICA
================================================================================
Este script replica todos los valores numericos de la Seccion 7 del paper
"Prediccion de Retornos del S&P 500 mediante Aprendizaje Automatico..."

Lee datos de:
  - results/backtest_detail.pkl   (retornos de estrategia por modelo)
  - models/trained_artifacts.pkl  (predicciones crudas, retornos forward)

Tablas replicadas:
  - Tabla (tab:sharpe_ci)   -- Bootstrap Sharpe CI (5,000 iteraciones)
  - Tabla (tab:dm_test)     -- Test Diebold-Mariano
  - Tabla (tab:cw_test)     -- Test Clark-West
  - Tabla (tab:overfitting) -- DA/MSE train vs test, degradacion

Numeros en prosa replicados:
  - MCS: 24 sobreviven (23 modelos + SPY), 0 eliminados
  - Bonferroni: alpha_adj = 0.05/23 = 0.00217
  - FDR Benjamini-Hochberg (q=0.10)
  - P(Sharpe > 0) y P(Sharpe > SPY) para cada ganador
  - Lo (2002) AR(1)-corrected SE del Sharpe

Genera:
  - results/statistical_validation.json

NOTA: El bootstrap de 5,000 iteraciones toma ~30 segundos.

USAGE:
    python paper/scripts/replicar_seccion7_validacion.py

================================================================================
Author: David Gonzalez Canon
================================================================================
"""

import numpy as np
import pickle
import json
import os
from scipy import stats

SEED = 42
N_BOOTSTRAP = 5000

# ============================================================================
# CONFIGURACION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# [2] SHARPE RATIO
# ============================================================================

def compute_sharpe(returns, rf):
    """Calcula Sharpe anualizado a partir de retornos diarios."""
    n_years = len(returns) / 252
    equity = np.cumprod(1 + returns)
    total_ret = equity[-1] - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252
    return (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0


# ============================================================================
# [3] BOOTSTRAP SHARPE CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_sharpe(returns, rf, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap CI para el Sharpe ratio con P(S>0).

    Genera n_boot muestras con reemplazo de los retornos diarios,
    calcula el Sharpe para cada muestra, y reporta el intervalo de
    confianza al 95% (percentiles 2.5 y 97.5).
    """
    rng = np.random.RandomState(seed)
    n = len(returns)
    sharpes = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sharpes[b] = compute_sharpe(returns[idx], rf[idx])

    return {
        "sharpe_point": float(compute_sharpe(returns, rf)),
        "ci_lower": float(np.percentile(sharpes, 2.5)),
        "ci_upper": float(np.percentile(sharpes, 97.5)),
        "prob_positive": float(np.mean(sharpes > 0) * 100),
        "bootstrap_sharpes": sharpes,
    }


def lo2002_se(returns, sharpe_hat):
    """
    Lo (2002) AR(1)-corrected standard error del Sharpe ratio.

    Ajusta el error estandar por autocorrelacion de primer orden en los
    retornos, que puede inflar artificialmente la significancia del Sharpe.
    """
    T = len(returns)
    rho1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    numerator = 1 + (sharpe_hat ** 2 / 2) * (1 + rho1)
    denominator = T * (1 - rho1)
    se = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
    return float(se), float(rho1)


# ============================================================================
# [4] DIEBOLD-MARIANO TEST
# ============================================================================

def newey_west_variance(d, max_lag=5):
    """Estimador Newey-West HAC para la varianza de mean(d)."""
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
    Test Diebold-Mariano comparando retornos de la estrategia vs benchmark.

    Funcion de perdida: L = -retorno. Menor perdida = mayor retorno = mejor.
    d_t = strategy_return_t - benchmark_return_t
    DM > 0 => la estrategia es mejor.
    """
    d = strategy_returns - benchmark_returns
    d_bar = np.mean(d)
    var_d = newey_west_variance(d, max_lag)
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0

    p_value_one = 1 - stats.norm.cdf(dm_stat)
    p_value_two = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value_one_sided": float(p_value_one),
        "p_value_two_sided": float(p_value_two),
        "significant_5pct": bool(p_value_one < 0.05),
        "significant_1pct": bool(p_value_one < 0.01),
    }


# ============================================================================
# [5] CLARK-WEST TEST
# ============================================================================

def clark_west(predictions, actuals, benchmark_pred, max_lag=5):
    """
    Test Clark-West (2007) para modelos anidados.

    H0: modelo restringido (benchmark = media historica) tiene igual o
    mejor capacidad predictiva. Test de una cola.
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


# ============================================================================
# [6] MODEL CONFIDENCE SET (Hansen 2011)
# ============================================================================

def model_confidence_set(model_losses, alpha=0.10, n_boot=3000, seed=SEED):
    """
    Model Confidence Set (Hansen, Lunde, Nason 2011).

    Usa el estadistico T_max con block bootstrap. Elimina iterativamente
    el peor modelo hasta que el p-value del test > alpha, indicando que
    los modelos restantes no son estadisticamente distinguibles.

    model_losses: dict {nombre: array de perdidas diarias} (menor = mejor).
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
            })
            surviving.remove(worst)
        else:
            break

    return {
        "surviving_models": surviving,
        "n_eliminated": len(elimination_order),
        "elimination_order": elimination_order,
        "final_p_value": round(p_value, 4),
    }


# ============================================================================
# [7] OVERFITTING ANALYSIS
# ============================================================================

def compute_prediction_metrics(predictions, actuals):
    """Calcula MSE, R2, DA para predicciones crudas vs realizados."""
    n = min(len(predictions), len(actuals))
    preds = np.array(predictions)[:n]
    actual = np.array(actuals)[:n]

    errors = actual - preds
    mse = float(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actual)))

    return {"mse": mse, "r2": r2, "directional_accuracy": dir_acc}


# ============================================================================
# [8] CORRECCIONES POR TESTS MULTIPLES
# ============================================================================

def bonferroni_correction(p_values, alpha=0.05):
    """Correccion de Bonferroni: alpha_adj = alpha / m."""
    m = len(p_values)
    alpha_adj = alpha / m
    significant = {k: v < alpha_adj for k, v in p_values.items()}
    return {"alpha_adjusted": float(alpha_adj), "n_tests": m, "significant": significant}


def benjamini_hochberg(p_values, q=0.10):
    """Procedimiento FDR de Benjamini-Hochberg."""
    m = len(p_values)
    sorted_pvals = sorted(p_values.items(), key=lambda x: x[1])
    max_k = 0
    for k, (name, pval) in enumerate(sorted_pvals, 1):
        if pval <= (k / m) * q:
            max_k = k

    significant = {}
    for k, (name, pval) in enumerate(sorted_pvals, 1):
        significant[name] = k <= max_k

    return {"q": q, "n_significant": max_k, "significant": significant}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("REPLICACION DE LA SECCION 7: VALIDACION ESTADISTICA")
    print("=" * 90)
    print()

    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # [1] Carga de datos
    # ------------------------------------------------------------------
    print("[1] Cargando datos del pipeline...")
    detail, artifacts = load_data()
    metadata = artifacts["metadata"]

    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    rf_test = np.array(metadata["risk_free_test"][:-1])
    fwd_train = np.array(metadata["forward_returns_train"])
    rf_train = np.array(metadata["risk_free_train"])
    n_test = len(fwd_test)

    print(f"    Periodo test: {n_test} dias, train: {len(fwd_train)} dias")
    print()

    results = {}

    # ------------------------------------------------------------------
    # [2] Bootstrap Sharpe CI (5,000 iteraciones)
    # ------------------------------------------------------------------
    print("[2] Bootstrap Sharpe Confidence Intervals (5,000 iter)...")
    print("-" * 90)

    spy_boot = bootstrap_sharpe(fwd_test, rf_test)
    spy_sharpes_dist = spy_boot["bootstrap_sharpes"]

    print(f"    SPY B&H: Sharpe={spy_boot['sharpe_point']:.3f}, "
          f"CI=[{spy_boot['ci_lower']:.3f}, {spy_boot['ci_upper']:.3f}], "
          f"P(S>0)={spy_boot['prob_positive']:.1f}%")

    bootstrap_results = {"SPY_BH": {
        "sharpe": spy_boot["sharpe_point"],
        "ci_lower": spy_boot["ci_lower"],
        "ci_upper": spy_boot["ci_upper"],
        "prob_positive": spy_boot["prob_positive"],
    }}

    for model_name in WINNERS:
        strat_ret = np.array(detail["models"][model_name]["strategy_returns"])
        n = min(len(strat_ret), len(rf_test))
        strat_ret = strat_ret[:n]
        rf = rf_test[:n]

        boot = bootstrap_sharpe(strat_ret, rf, seed=SEED + hash(model_name) % 10000)
        prob_beat_spy = float(np.mean(boot["bootstrap_sharpes"] > spy_sharpes_dist) * 100)
        se, rho1 = lo2002_se(strat_ret, boot["sharpe_point"])

        bootstrap_results[model_name] = {
            "sharpe": boot["sharpe_point"],
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "prob_positive": boot["prob_positive"],
            "prob_beat_spy": prob_beat_spy,
            "lo2002_se": se,
            "ar1_rho": rho1,
        }

        print(f"    {model_name}: Sharpe={boot['sharpe_point']:.3f}, "
              f"CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}], "
              f"P(S>0)={boot['prob_positive']:.1f}%, "
              f"P(S>SPY)={prob_beat_spy:.1f}%, "
              f"SE(Lo)={se:.4f}")

    results["bootstrap"] = bootstrap_results
    print()

    # ------------------------------------------------------------------
    # [3] Diebold-Mariano Test
    # ------------------------------------------------------------------
    print("[3] Test Diebold-Mariano (retornos estrategia vs B&H)...")
    print("-" * 90)

    dm_results = {}
    all_dm_pvalues = {}

    for model_name in ALL_MODELS:
        if model_name not in detail["models"]:
            continue
        strat_ret = np.array(detail["models"][model_name]["strategy_returns"])
        n = min(len(strat_ret), len(fwd_test))
        dm = diebold_mariano_returns(strat_ret[:n], fwd_test[:n])
        dm_results[model_name] = dm
        all_dm_pvalues[model_name] = dm["p_value_one_sided"]

        if model_name in WINNERS:
            sig = "***" if dm["significant_1pct"] else ("**" if dm["significant_5pct"] else "")
            print(f"    {model_name}: DM={dm['dm_statistic']:+.3f}, "
                  f"p(1-tail)={dm['p_value_one_sided']:.4f} {sig}")

    results["diebold_mariano"] = {k: dm_results[k] for k in WINNERS if k in dm_results}
    print()

    # ------------------------------------------------------------------
    # [4] Clark-West Test
    # ------------------------------------------------------------------
    print("[4] Test Clark-West (prediccion MSE vs media historica)...")
    print("-" * 90)

    hist_mean = np.mean(fwd_train)
    benchmark_pred = np.full(n_test, hist_mean)
    print(f"    Media historica benchmark: {hist_mean*100:.4f}% diario")

    cw_results = {}
    all_cw_pvalues = {}

    for model_name in ALL_MODELS:
        if model_name not in artifacts["models"]:
            continue
        test_preds = np.array(artifacts["models"][model_name]["test_predictions"])
        n = min(len(test_preds), len(fwd_test))

        cw = clark_west(test_preds[:n], fwd_test[:n], benchmark_pred[:n])
        cw_results[model_name] = cw
        all_cw_pvalues[model_name] = cw["p_value_one_tail"]

        if model_name in WINNERS:
            sig = "***" if cw["rejects_h0_1pct"] else ("**" if cw["rejects_h0_5pct"] else "")
            print(f"    {model_name}: CW={cw['cw_statistic']:+.3f}, "
                  f"p={cw['p_value_one_tail']:.4f} {sig}")

    results["clark_west"] = {k: cw_results[k] for k in WINNERS if k in cw_results}
    print()

    # ------------------------------------------------------------------
    # [5] Model Confidence Set (Hansen 2011)
    # ------------------------------------------------------------------
    print("[5] Model Confidence Set (alpha=0.10, 3,000 bootstrap)...")
    print("-" * 90)

    # Perdida = -retorno (menor perdida = mayor retorno = mejor)
    model_losses = {}
    for model_name in ALL_MODELS:
        if model_name not in detail["models"]:
            continue
        strat_ret = np.array(detail["models"][model_name]["strategy_returns"])
        n = min(len(strat_ret), n_test)
        model_losses[model_name] = -strat_ret[:n]
    model_losses["SPY_BH"] = -fwd_test

    print(f"    Ejecutando MCS con {len(model_losses)} modelos...")
    mcs = model_confidence_set(model_losses, alpha=0.10, n_boot=3000)

    print(f"    Sobreviven: {len(mcs['surviving_models'])} modelos")
    print(f"    Eliminados: {mcs['n_eliminated']}")
    if mcs["elimination_order"]:
        for e in mcs["elimination_order"]:
            print(f"      - {e['model']}: p={e['p_value']:.4f}")

    results["mcs"] = mcs
    print()

    # ------------------------------------------------------------------
    # [6] Overfitting Analysis
    # ------------------------------------------------------------------
    print("[6] Analisis de overfitting (train vs test)...")
    print("-" * 90)

    overfitting_models = WINNERS + ["XGBoost"]
    overfitting_results = {}

    print(f"    {'Modelo':<18} {'Train DA':>10} {'Test DA':>10} {'Degrad%':>10} {'MSE Ratio':>10}")
    print("    " + "-" * 60)

    for model_name in overfitting_models:
        if model_name not in artifacts["models"]:
            continue

        model_data = artifacts["models"][model_name]
        test_preds = np.array(model_data["test_predictions"])
        train_preds = np.array(model_data["train_predictions"])

        test_metrics = compute_prediction_metrics(test_preds, fwd_test)
        train_metrics = compute_prediction_metrics(train_preds, fwd_train)

        da_degrad = ((test_metrics["directional_accuracy"] - train_metrics["directional_accuracy"])
                     / train_metrics["directional_accuracy"] * 100
                     if train_metrics["directional_accuracy"] > 0.001 else 0)

        mse_ratio = (test_metrics["mse"] / train_metrics["mse"]
                     if train_metrics["mse"] > 0 else float("inf"))

        overfitting_results[model_name] = {
            "train_da": train_metrics["directional_accuracy"],
            "test_da": test_metrics["directional_accuracy"],
            "da_degradation_pct": float(da_degrad),
            "train_mse": train_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "mse_ratio": float(mse_ratio),
        }

        print(f"    {model_name:<18} {train_metrics['directional_accuracy']:>10.3f} "
              f"{test_metrics['directional_accuracy']:>10.3f} "
              f"{da_degrad:>9.1f}% {mse_ratio:>9.2f}x")

    results["overfitting"] = overfitting_results
    print()

    # ------------------------------------------------------------------
    # [7] Correcciones por tests multiples
    # ------------------------------------------------------------------
    print("[7] Correcciones por tests multiples...")
    print("-" * 90)

    bonf = bonferroni_correction(all_dm_pvalues)
    fdr = benjamini_hochberg(all_dm_pvalues, q=0.10)

    print(f"    Bonferroni: alpha_adj = {bonf['alpha_adjusted']:.4f} ({bonf['n_tests']} tests)")
    for name in WINNERS:
        if name in bonf["significant"]:
            sig = "SI" if bonf["significant"][name] else "no"
            print(f"      {name}: {sig} (p={all_dm_pvalues.get(name, 1):.4f})")

    print(f"    FDR (q=0.10): {fdr['n_significant']} modelos significativos")
    fdr_winners = [n for n, s in fdr["significant"].items() if s]
    if fdr_winners:
        print(f"      Significativos: {fdr_winners}")

    results["bonferroni"] = bonf
    results["fdr"] = fdr
    print()

    # ------------------------------------------------------------------
    # [8] Tabla resumen
    # ------------------------------------------------------------------
    print("=" * 90)
    print("TABLA RESUMEN (tab:sharpe_ci + tab:dm_test + tab:cw_test)")
    print("=" * 90)
    print(f"  {'Modelo':<18} {'Sharpe':>7} {'CI 95%':>16} {'P(S>0)':>8} "
          f"{'P(S>SPY)':>9} {'DM':>7} {'CW':>7}")
    print("  " + "-" * 70)
    for name in WINNERS:
        b = bootstrap_results[name]
        dm = dm_results.get(name, {})
        cw = cw_results.get(name, {})
        p_spy = f"{b['prob_beat_spy']:.1f}%" if 'prob_beat_spy' in b else "N/A"
        dm_str = f"{dm.get('dm_statistic', 0):+.2f}"
        cw_str = f"{cw.get('cw_statistic', 0):+.2f}"
        print(f"  {name:<18} {b['sharpe']:>7.3f} "
              f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
              f"{b['prob_positive']:>7.1f}% {p_spy:>9} "
              f"{dm_str:>7} {cw_str:>7}")

    b = bootstrap_results["SPY_BH"]
    print(f"  {'SPY B&H':<18} {b['sharpe']:>7.3f} "
          f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
          f"{b['prob_positive']:>7.1f}% {'--':>9} {'--':>7} {'--':>7}")

    # ------------------------------------------------------------------
    # [9] Guardar JSON
    # ------------------------------------------------------------------
    output_path = os.path.join(RESULTS_DIR, "statistical_validation.json")

    def clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        return obj

    with open(output_path, "w") as f:
        json.dump(clean(results), f, indent=2)

    print()
    print(f"Guardado: {output_path}")
    print()
    print("=" * 90)
    print("REPLICACION SECCION 7 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()
