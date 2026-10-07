# -*- coding: utf-8 -*-
"""
================================================================================
PRODUCCION_LSTM.PY — Modelo ganador en produccion (LSTM + Attention)
================================================================================

Paquete autocontenido de produccion de la tesis "Prediccion de retornos del
S&P 500 con Triple Screen de Elder + ML" (David Gonzalez Canon, FEN U. de Chile).

Contiene UNICAMENTE el modelo ganador y estadisticamente significativo del
estudio, LSTM_Attention (Diebold-Mariano p=0.011 y Clark-West significativos
al 5%; unico modelo cuyo IC bootstrap del Sharpe excluye 0).

Que hace este script, en orden:
  1. Carga el dataset de 541 features y replica el split temporal 80/20.
  2. Carga los preprocesadores entrenados (imputer + scaler) y el checkpoint
     real del modelo (models/LSTM_Attention.pt) y RECALCULA las predicciones
     con PyTorch (inferencia genuina, no numeros guardados).
  3. Verifica las predicciones recalculadas contra las originales de la tesis.
  4. Entrena el Meta-KNN de umbrales con datos de ENTRENAMIENTO unicamente y
     ejecuta el backtest Long-Only {CASH, SPY, UPRO} con costos completos
     (expense ratio, bid-ask, volatility drag con vol realizada).
  5. Compara las metricas con las publicadas en la tesis (+395.4%, Sharpe 1.319).
  6. Emite la SEÑAL DE PRODUCCION del ultimo dia disponible del dataset.

USO:
    python produccion_lstm.py          (o ejecutar run.bat)

La logica de backtest es copia literal de scripts/optimize_and_backtest.py del
repositorio original para garantizar reproducibilidad exacta.
================================================================================
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew as scipy_skew
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# RUTAS (relativas a la carpeta del paquete; funciona desde cualquier cwd)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "bloomberg_triple_screen_core.csv")
MODEL_FILE = os.path.join(BASE_DIR, "activos", "spx", "modelo", "LSTM_Attention.pt")
PREPROC_FILE = os.path.join(BASE_DIR, "activos", "spx", "modelo", "preprocessors.joblib")
REFERENCE_FILE = os.path.join(BASE_DIR, "activos", "spx", "modelo", "resultados_esperados.json")

# =============================================================================
# CONFIG, MODELO Y BACKTEST: viven en pipeline/modelo.py (copia literal)
# =============================================================================
from pipeline.modelo import *  # noqa: E402,F401,F403


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 90)
    print("PRODUCCION — LSTM + Attention (modelo ganador y significativo de la tesis)")
    print("=" * 90)

    # ---------------------------------------------------------------- 1. DATOS
    print("\n[1/6] Cargando dataset y replicando split temporal 80/20...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    preproc = joblib.load(PREPROC_FILE)
    feature_cols = preproc['feature_cols']
    imputer = preproc['imputer']
    scaler = preproc['scaler']

    # Ensambla la matriz de entrada con los nombres que el checkpoint espera,
    # leyendo cada columna de su nombre actual via LEGACY_MAP
    def build_X(frame):
        return pd.DataFrame(
            {c: frame[LEGACY_MAP.get(c, c)].values for c in feature_cols})

    X_train = build_X(train_df)
    X_test = build_X(test_df)
    y_train = train_df[TARGET_COL]

    fwd_train = train_df['forward_returns'].values
    fwd_test_full = test_df['forward_returns'].values
    rf_train = train_df['risk_free_rate'].values
    rf_test_full = test_df['risk_free_rate'].values
    dates_test = test_df['date'].reset_index(drop=True)

    print(f"      Dataset: {len(df):,} filas | Features: {len(feature_cols)}")
    print(f"      Train: {len(train_df):,} dias ({train_df['date'].iloc[0].date()} a {train_df['date'].iloc[-1].date()})")
    print(f"      Test:  {len(test_df):,} dias ({dates_test.iloc[0].date()} a {dates_test.iloc[-1].date()})")

    # ------------------------------------------------------------ 2. INFERENCIA
    print("\n[2/6] Cargando checkpoint y recalculando predicciones con PyTorch...")
    X_train_scaled = scaler.transform(imputer.transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    # El target se escala igual que en entrenamiento (StandardScaler sobre y_train)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train.values.reshape(-1, 1))

    ckpt = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    n_features = ckpt['config']['n_features']
    assert n_features == len(feature_cols), \
        f"Inconsistencia: checkpoint espera {n_features} features, dataset tiene {len(feature_cols)}"

    device = torch.device('cpu')
    model = LSTMAttention(n_features, hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Checkpoint cargado: {n_params:,} parametros | hidden_dim={HIDDEN_DIM} | atencion Luong")

    X_train_seq = prepare_sequences(X_train_scaled, LOOKBACK)
    X_test_seq = prepare_sequences(
        np.vstack([X_train_scaled[-LOOKBACK:], X_test_scaled]), LOOKBACK)

    train_preds = y_scaler.inverse_transform(
        predict_in_batches(model, X_train_seq, device).reshape(-1, 1)).flatten()
    test_preds = y_scaler.inverse_transform(
        predict_in_batches(model, X_test_seq, device).reshape(-1, 1)).flatten()
    print(f"      Predicciones recalculadas: {len(train_preds)} train + {len(test_preds)} test")

    # -------------------------------------------------- 3. VERIFICAR PREDICCIONES
    print("\n[3/6] Verificando predicciones contra las originales de la tesis...")
    reference = None
    if os.path.exists(REFERENCE_FILE):
        with open(REFERENCE_FILE) as f:
            reference = json.load(f)
        ref_test = np.array(reference['test_predictions'])
        n_cmp = min(len(ref_test), len(test_preds))
        max_diff = float(np.max(np.abs(test_preds[:n_cmp] - ref_test[:n_cmp])))
        corr = float(np.corrcoef(test_preds[:n_cmp], ref_test[:n_cmp])[0, 1])
        if max_diff < 1e-6:
            print(f"      [OK] Predicciones identicas a las de la tesis (diff max {max_diff:.2e})")
        elif corr > 0.9999:
            print(f"      [OK] Predicciones equivalentes (corr {corr:.6f}, diff max {max_diff:.2e};")
            print(f"           diferencias de redondeo por hardware/version, sin impacto material)")
        else:
            print(f"      [ALERTA] Predicciones divergen (corr {corr:.4f}, diff max {max_diff:.2e}).")
            print(f"           Revisar versiones de torch/sklearn contra requirements.txt")
    else:
        print("      [AVISO] Sin archivo de referencia; se omite la verificacion")

    # ------------------------------------------------------------- 4. META-KNN
    print("\n[4/6] Entrenando Meta-KNN de umbrales (solo con datos de entrenamiento)...")
    n_tr = min(len(train_preds), len(fwd_train), len(rf_train))
    train_preds_tr = train_preds[:n_tr]
    fwd_train_tr = fwd_train[:n_tr]
    rf_train_tr = rf_train[:n_tr]

    train_percentiles = compute_rolling_percentiles(train_preds_tr, window=PCTILE_WINDOW)
    X_meta, y_meta = build_meta_dataset(train_preds_tr, train_percentiles,
                                        fwd_train_tr, rf_train_tr)
    meta_model = train_meta_knn(X_meta, y_meta)
    print(f"      Meta-dataset: {len(X_meta)} ventanas | KNN K=5 ponderado por distancia "
          f"| accuracy train {meta_model['train_acc']:.1%}")

    # ------------------------------------------------------ 5. BACKTEST COMPLETO
    # Se corre sobre los N dias de test completos (todo es retrospectivo, los
    # primeros N-1 dias son identicos al backtest canonico de la tesis); las
    # metricas se evaluan sobre N-1 dias porque el retorno forward del ultimo
    # dia se materializa fuera de la muestra. El dia N es la señal de produccion.
    print("\n[5/6] Ejecutando backtest Long-Only {CASH, SPY, UPRO} con costos completos...")
    positions, percentiles, chosen_params, signal_valid = meta_knn_filtered_backtest(
        test_preds, train_preds,
        fwd_test_full, fwd_train,
        rf_test_full, rf_train,
        meta_model, MIN_SIGNAL_STD,
    )

    n_eval = len(fwd_test_full) - 1
    pos_eval = positions[:n_eval]
    fwd_eval = fwd_test_full[:n_eval]
    rf_eval = rf_test_full[:n_eval]

    net_ret, gross_ret, _, _, _ = calculate_returns_with_costs(pos_eval, fwd_eval, rf_eval)
    metrics = compute_metrics(net_ret, gross_ret, rf_eval, pos_eval, n_eval)

    spy_bh_net = fwd_eval - INSTRUMENTS['SPY']['expense_ratio'] / 252
    spy_equity = np.cumprod(1 + spy_bh_net)
    spy_return = float(spy_equity[-1] - 1)
    spy_cagr = float((1 + spy_return) ** (252 / n_eval) - 1)
    spy_vol = float(np.std(spy_bh_net) * np.sqrt(252))
    spy_sharpe = float((spy_cagr - np.mean(rf_eval) * 252) / spy_vol) if spy_vol > 0 else 0

    print("\n" + "-" * 90)
    print(f"  RESULTADOS SOBRE EL PERIODO DE PRUEBA ({n_eval} dias, "
          f"{dates_test.iloc[0].date()} a {dates_test.iloc[n_eval - 1].date()})")
    print("-" * 90)
    print(f"  {'':24}{'LSTM_Attention':>18}{'SPY Buy&Hold':>18}")
    print(f"  {'Retorno total':<24}{metrics['total_return'] * 100:>+17.1f}%{spy_return * 100:>+17.1f}%")
    print(f"  {'Retorno anualizado':<24}{metrics['annual_return'] * 100:>+17.1f}%{spy_cagr * 100:>+17.1f}%")
    print(f"  {'Sharpe':<24}{metrics['sharpe']:>18.3f}{spy_sharpe:>18.3f}")
    print(f"  {'Sortino':<24}{metrics['sortino']:>18.3f}{'—':>18}")
    print(f"  {'Max drawdown':<24}{-metrics['max_drawdown'] * 100:>17.1f}%{'':>18}")
    print(f"  {'Capital final ($10k)':<24}{'$' + format(metrics['final_capital'], ',.0f'):>18}"
          f"{'$' + format(10000 * (1 + spy_return), ',.0f'):>18}")
    print(f"  {'Dias en UPRO/SPY/CASH':<24}"
          f"{format(metrics['pct_3x'], '.1f') + '% / ' + format(metrics['pct_1x'], '.1f') + '% / ' + format(metrics['pct_cash'], '.1f') + '%':>36}")
    print(f"  {'Numero de trades':<24}{metrics['n_trades']:>18}")

    if reference is not None:
        exp = reference['expected_metrics']
        checks = [
            ('Retorno total', metrics['total_return'], exp['total_return']),
            ('Sharpe', metrics['sharpe'], exp['sharpe']),
            ('Max drawdown', metrics['max_drawdown'], exp['max_drawdown']),
        ]
        print("\n  Verificacion contra la tesis:")
        all_ok = True
        for name, got, want in checks:
            rel = abs(got - want) / max(abs(want), 1e-9)
            ok = rel < 0.005
            all_ok = all_ok and ok
            tag = "[OK]  " if ok else "[DIFIERE]"
            print(f"    {tag} {name:<16} obtenido {got:+.4f} | tesis {want:+.4f} | diff rel {rel:.3%}")
        if all_ok:
            print("    => Reproduccion EXACTA de los resultados publicados")

    # ------------------------------------------------------ 6. SEÑAL DEL DIA
    print("\n[6/6] SEÑAL DE PRODUCCION (ultimo dia disponible del dataset)")
    print("=" * 90)
    t = len(positions) - 1
    sig_date = dates_test.iloc[t].date()
    pos_today = int(positions[t])
    q_ext, q_mod = chosen_params[t]
    print(f"  Fecha de la señal     : {sig_date} (se ejecuta al cierre, vigente el dia habil siguiente)")
    print(f"  Prediccion del modelo : {test_preds[t]:+.6f} (exceso de retorno forward)")
    print(f"  Percentil rolling 63d : {percentiles[t]:.1f}")
    if not signal_valid[t]:
        print(f"  Filtro de calidad     : SEÑAL DEGENERADA (std rolling < {MIN_SIGNAL_STD}) -> CASH forzado")
    else:
        print(f"  Umbrales Meta-KNN     : UPRO si percentil >= {100 - q_ext} | SPY si >= {100 - q_mod}")
    print(f"\n  >>> POSICION RECOMENDADA: {POS_TO_INST[pos_today]} <<<")
    print("=" * 90)
    print("\nNota: el dataset Bloomberg llega hasta la fecha indicada. Para operar en vivo se")
    print("debe regenerar el dataset con datos nuevos (build_dataset.py del repositorio completo).")


if __name__ == "__main__":
    main()
