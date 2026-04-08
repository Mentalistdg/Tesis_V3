# -*- coding: utf-8 -*-
"""
================================================================================
REPLICACION DE LA SECCION 5: LA PARADOJA DEL DIRECTIONAL ACCURACY
================================================================================
Este script replica todos los valores numericos de la Seccion 5 del paper
"Prediccion de Retornos del S&P 500 mediante Aprendizaje Automatico..."

Lee datos de:
  - results/final_long_only_backtest.json  (retornos totales, DA almacenado)
  - results/backtest_detail.pkl            (posiciones diarias)
  - models/trained_artifacts.pkl           (predicciones crudas, retornos forward)

Tablas replicadas:
  - Tabla (tab:da_paradox)  -- DA general para 6 modelos clave
  - Tabla (tab:da_vs_da3x)  -- DA general, DA@3x, % tiempo en 3x

Numeros en prosa replicados:
  - Correlacion rho(DA, Return) = -0.37
  - DA y retornos individuales para GARCH, Theta, ExponentialSmoothing,
    LSTM+Attention, Ridge, CNN-LSTM, RandomForest
  - LSTM+Attention: 55% cash, 12.2% en 3x, DA 47.0%, DA@3x 62.3%
  - GARCH: DA@3x 63.6% (el mas alto), 1.7% tiempo en 3x

USAGE:
    python paper/scripts/replicar_seccion5_da_paradox.py

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
np.random.seed(SEED)

# ============================================================================
# CONFIGURACION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ============================================================================
# [1] CARGA DE DATOS
# ============================================================================

def load_data():
    """Carga backtest_detail, trained_artifacts y final_long_only_backtest."""
    with open(os.path.join(MODELS_DIR, "trained_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    with open(os.path.join(RESULTS_DIR, "backtest_detail.pkl"), "rb") as f:
        detail = pickle.load(f)

    with open(os.path.join(RESULTS_DIR, "final_long_only_backtest.json"), "r") as f:
        backtest_json = json.load(f)

    return artifacts, detail, backtest_json


# ============================================================================
# [2] CALCULO DE DIRECTIONAL ACCURACY
# ============================================================================

def compute_directional_accuracy(predictions, actual_returns):
    """
    DA = proporcion de dias donde sign(prediccion) == sign(retorno_real).
    Coincide con la computacion en optimize_and_backtest.py.
    """
    pred_dir = np.sign(predictions)
    actual_dir = np.sign(actual_returns)
    return float(np.mean(pred_dir == actual_dir))


def compute_da_at_3x(positions, actual_returns):
    """
    DA@3x = proporcion de dias con posicion==3 donde el mercado subio.

    DA@3x = sum(pos_t == 3 AND r_mkt > 0) / sum(pos_t == 3)

    Mide la capacidad del modelo para seleccionar los dias correctos
    para la posicion de maxima conviccion (3x apalancado).
    """
    mask_3x = (positions == 3)
    n_3x = int(np.sum(mask_3x))
    if n_3x == 0:
        return float('nan'), 0
    hits = np.sum(mask_3x & (actual_returns > 0))
    da_3x = float(hits / n_3x)
    return da_3x, n_3x


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("REPLICACION DE LA SECCION 5: LA PARADOJA DEL DIRECTIONAL ACCURACY")
    print("=" * 90)
    print()

    # ------------------------------------------------------------------
    # [1] Carga de datos
    # ------------------------------------------------------------------
    print("[1] Cargando datos del pipeline...")
    artifacts, detail, backtest_json = load_data()

    metadata = artifacts["metadata"]
    fwd_test = np.array(metadata["forward_returns_test"][:-1])
    n_test = len(fwd_test)
    print(f"    Periodo de prueba: {n_test} dias")
    print()

    # ------------------------------------------------------------------
    # [2] Tabla (tab:da_paradox) -- DA general para 6 modelos clave
    # ------------------------------------------------------------------
    print("[2] Replicando Tabla (tab:da_paradox) -- DA general")
    print("-" * 90)

    # Calcular DA independientemente para todos los modelos
    da_values = {}
    return_values = {}

    for model_name, model_data in artifacts["models"].items():
        test_preds = np.array(model_data["test_predictions"]).flatten()
        nt = min(len(test_preds), len(fwd_test))
        da = compute_directional_accuracy(test_preds[:nt], fwd_test[:nt])
        da_values[model_name] = da
        if model_name in backtest_json["models"]:
            return_values[model_name] = backtest_json["models"][model_name]["total_return"]

    # Modelos clave de la tabla
    table_models = ["GARCH", "Theta", "ExponentialSmoothing",
                    "LSTM_Attention", "Ridge", "CNN_LSTM"]

    print(f"    {'Modelo':<22} {'DA General':>10} {'Retorno':>10}")
    print("    " + "-" * 44)
    for model_name in table_models:
        da_pct = da_values[model_name] * 100
        ret_pct = return_values.get(model_name, 0) * 100
        print(f"    {model_name:<22} {da_pct:>9.1f}% {ret_pct:>+9.1f}%")
    print()

    # ------------------------------------------------------------------
    # [3] Tabla (tab:da_vs_da3x) -- DA general vs DA@3x
    # ------------------------------------------------------------------
    print("[3] Replicando Tabla (tab:da_vs_da3x) -- DA general vs DA@3x")
    print("-" * 90)

    da3x_models = ["LSTM_Attention", "Ridge", "RandomForest", "CNN_LSTM", "GARCH", "Theta"]

    print(f"    {'Modelo':<22} {'DA General':>10} {'DA@3x':>8} {'%Tiempo 3x':>11} {'N dias 3x':>10}")
    print("    " + "-" * 64)

    for model_name in da3x_models:
        if model_name not in detail["models"]:
            continue
        positions = np.array(detail["models"][model_name]["positions"])
        nt = len(positions)

        # DA general (calculado arriba)
        da_gen = da_values.get(model_name, 0) * 100

        # DA@3x
        da3x, n_3x = compute_da_at_3x(positions, fwd_test[:nt])
        pct_3x = float(np.mean(positions == 3) * 100)

        da3x_str = f"{da3x*100:.1f}%" if not np.isnan(da3x) else "N/A"

        print(f"    {model_name:<22} {da_gen:>9.1f}% {da3x_str:>8} "
              f"{pct_3x:>10.1f}% {n_3x:>10d}")
    print()

    # ------------------------------------------------------------------
    # [4] Correlacion rho(DA, Return)
    # ------------------------------------------------------------------
    print("[4] Correlacion rho(DA, Return) entre los 23 modelos")
    print("-" * 90)

    model_names_ordered = []
    da_array = []
    ret_array = []

    for model_name in da_values:
        if model_name in return_values:
            model_names_ordered.append(model_name)
            da_array.append(da_values[model_name])
            ret_array.append(return_values[model_name])

    da_array = np.array(da_array)
    ret_array = np.array(ret_array)
    rho, pvalue = stats.pearsonr(da_array, ret_array)

    print(f"    rho(DA, Return) = {rho:.4f}")
    print(f"    p-value         = {pvalue:.4f}")
    print(f"    Interpretacion: correlacion {'negativa' if rho < 0 else 'positiva'} moderada")
    print()

    # Mostrar ranking DA vs Return
    print("    Ranking DA vs Return (ordenado por DA descendente):")
    sorted_idx = np.argsort(-da_array)
    for i in sorted_idx:
        marker = " ***" if ret_array[i] * 100 > 102 else ""
        print(f"      {model_names_ordered[i]:<22} DA={da_array[i]*100:.1f}%  "
              f"Return={ret_array[i]*100:+.1f}%{marker}")
    print()

    # ------------------------------------------------------------------
    # [5] Claims especificas en prosa
    # ------------------------------------------------------------------
    print("[5] Verificando claims en prosa de la Seccion 5")
    print("-" * 90)

    # LSTM_Attention: 55% cash, 12.2% en 3x
    lstm_pos = np.array(detail["models"]["LSTM_Attention"]["positions"])
    pct_cash_lstm = float(np.mean(lstm_pos == 0) * 100)
    pct_3x_lstm = float(np.mean(lstm_pos == 3) * 100)
    print(f"    LSTM+Attention %cash:  {pct_cash_lstm:.1f}% (paper: 55%)")
    print(f"    LSTM+Attention %3x:    {pct_3x_lstm:.1f}% (paper: 12.2%)")

    # DA general LSTM_Attention: 47.0%
    lstm_da = da_values["LSTM_Attention"] * 100
    print(f"    LSTM+Attention DA:     {lstm_da:.1f}% (paper: 47.0%)")

    # DA@3x LSTM_Attention: 62.3%
    da3x_lstm, n_3x_lstm = compute_da_at_3x(lstm_pos, fwd_test[:len(lstm_pos)])
    print(f"    LSTM+Attention DA@3x:  {da3x_lstm*100:.1f}% (paper: 62.3%)")

    # GARCH: DA@3x 63.6%, 1.7% en 3x
    garch_pos = np.array(detail["models"]["GARCH"]["positions"])
    da3x_garch, n_3x_garch = compute_da_at_3x(garch_pos, fwd_test[:len(garch_pos)])
    pct_3x_garch = float(np.mean(garch_pos == 3) * 100)
    print(f"    GARCH DA@3x:           {da3x_garch*100:.1f}% (paper: 63.6%)")
    print(f"    GARCH %3x:             {pct_3x_garch:.1f}% (paper: 1.7%)")

    # Theta: DA general 54.6%, DA@3x 53.3%
    theta_da = da_values["Theta"] * 100
    theta_pos = np.array(detail["models"]["Theta"]["positions"])
    da3x_theta, _ = compute_da_at_3x(theta_pos, fwd_test[:len(theta_pos)])
    print(f"    Theta DA general:      {theta_da:.1f}% (paper: 54.6%)")
    print(f"    Theta DA@3x:           {da3x_theta*100:.1f}% (paper: 53.3%)")
    print()

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    print("=" * 90)
    print("REPLICACION SECCION 5 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()
