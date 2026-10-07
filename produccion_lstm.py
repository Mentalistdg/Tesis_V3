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
MODEL_FILE = os.path.join(BASE_DIR, "models", "LSTM_Attention.pt")
PREPROC_FILE = os.path.join(BASE_DIR, "models", "preprocessors.joblib")
REFERENCE_FILE = os.path.join(BASE_DIR, "reference", "resultados_esperados.json")

# =============================================================================
# CONFIG (identica al pipeline original)
# =============================================================================
TEST_SIZE = 0.20
LOOKBACK = 30          # input_chunk_length de train_models.py
HIDDEN_DIM = 128
TARGET_COL = 'market_forward_excess_returns'
NON_FEATURE_COLS = ['date_id', 'date', 'forward_returns', 'risk_free_rate',
                    'market_forward_excess_returns']

INSTRUMENTS = {
    'UPRO': {'expense_ratio': 0.0091, 'bid_ask': 0.0005},
    'SPY':  {'expense_ratio': 0.0009, 'bid_ask': 0.0002},
    'CASH': {'expense_ratio': 0.0000, 'bid_ask': 0.0000},
}

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
MIN_SIGNAL_STD = 0.001

# El modelo fue entrenado en marzo 2026, antes de la renumeracion de variables de
# abril 2026 (familias P y S compactadas y E20 -> E19). Este mapa traduce cada
# nombre de columna que el checkpoint espera al nombre actual en el dataset.
# Mapeo verificado contra las estadisticas del scaler entrenado (coincidencia
# exacta de media y desviacion por columna).
LEGACY_MAP = {
    'E20': 'E19',
    'P4': 'P3', 'P5': 'P4', 'P7': 'P5', 'P8': 'P6', 'P9': 'P7',
    'P10': 'P8', 'P11': 'P9', 'P12': 'P10', 'P13': 'P11',
    'S4': 'S2', 'S5': 'S3', 'S6': 'S4', 'S7': 'S5', 'S8': 'S6',
    'S9': 'S7', 'S10': 'S8', 'S11': 'S9', 'S12': 'S10',
}

POS_TO_INST = {3: 'UPRO (3x apalancado)', 1: 'SPY (1x)', 0: 'CASH (sin exposicion)'}


# =============================================================================
# MODELO (copia literal de scripts/train_models.py)
# =============================================================================
class LuongAttention(nn.Module):
    """Mecanismo de atencion estilo Luong (dot product)."""
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, lstm_output, final_hidden):
        final_hidden = final_hidden.unsqueeze(2)
        attention_scores = torch.bmm(lstm_output, final_hidden).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attention_weights


class LSTMAttention(nn.Module):
    """LSTM bidireccional de 2 capas con atencion Luong."""
    def __init__(self, n_features, hidden_dim=64):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.attention = LuongAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        context, attn_weights = self.attention(lstm_out, final_hidden)
        context = self.dropout(context)
        out = self.fc(context)
        return out


def prepare_sequences(X, lookback):
    """Secuencias deslizantes (batch, lookback, features), igual que en training."""
    X_seq = []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
    return np.array(X_seq)


def predict_in_batches(model, X_seq, device, batch_size=1024):
    """Forward pass en eval por lotes (identico a un pase completo, sin dropout)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch = torch.FloatTensor(X_seq[i:i + batch_size]).to(device)
            outs.append(model(batch).cpu().numpy().flatten())
    return np.concatenate(outs)


# =============================================================================
# BACKTEST META-KNN (copia literal de scripts/optimize_and_backtest.py)
# =============================================================================
def compute_rolling_percentiles(all_preds, window=63):
    n = len(all_preds)
    percentiles = np.full(n, 50.0)
    for i in range(n):
        start = max(0, i - window + 1)
        w = all_preds[start:i + 1]
        if len(w) < 5:
            continue
        percentiles[i] = np.mean(w <= all_preds[i]) * 100
    return percentiles


def compute_rolling_std(preds, window=63):
    n = len(preds)
    rolling_std = np.full(n, 0.0)
    for i in range(n):
        start = max(0, i - window + 1)
        w = preds[start:i + 1]
        if len(w) >= 5:
            rolling_std[i] = np.std(w)
    return rolling_std


def evaluate_combo(percentiles, fwd_returns, rf, q_ext, q_mod):
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
        den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
        autocorr_pred = num / den if den > 1e-12 else 0.0
    else:
        autocorr_pred = 0.0

    t = np.arange(n, dtype=float)
    t_mean = np.mean(t)
    p_mean = np.mean(preds_window)
    num = np.sum((t - t_mean) * (preds_window - p_mean))
    den = np.sum((t - t_mean) ** 2)
    trend_pred = num / den if den > 1e-12 else 0.0

    if n >= 21:
        vols = []
        for i in range(21, n + 1):
            vols.append(np.std(market_returns_window[i - 21:i]))
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
    n_tr = min(len(train_preds), len(fwd_train), len(rf_train))
    train_preds_al = train_preds[-n_tr:]
    fwd_train_al = fwd_train[-n_tr:]
    rf_train_al = rf_train[-n_tr:]

    all_preds = np.concatenate([train_preds_al, test_preds])
    all_fwd = np.concatenate([fwd_train_al, fwd_test])
    all_rf = np.concatenate([rf_train_al, rf_test])

    n_train = len(train_preds_al)
    n_test = len(test_preds)

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

        today_std = all_rolling_std[abs_t]
        if today_std < min_signal_std:
            positions[t] = 0
            chosen_params.append((0, 0))
            signal_valid[t] = False
            continue

        signal_valid[t] = True

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


def calculate_returns_with_costs(positions, market_returns, risk_free, vol_window=21):
    n = len(positions)
    returns = np.zeros(n)
    gross_returns = np.zeros(n)
    expense_costs = np.zeros(n)
    trading_costs = np.zeros(n)
    vol_drag_costs = np.zeros(n)
    pos_to_inst = {3: 'UPRO', 1: 'SPY', 0: 'CASH'}

    realized_vol = np.full(n, 0.01)
    for i in range(n):
        w = market_returns[max(0, i - vol_window):i]
        if len(w) >= 5:
            realized_vol[i] = np.std(w)

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
        if i > 0 and positions[i] != positions[i - 1]:
            prev_inst = pos_to_inst.get(int(positions[i - 1]), 'CASH')
            trading += INSTRUMENTS[prev_inst]['bid_ask'] + cfg['bid_ask']

        vol_drag = 0
        if pos == 3:
            sigma = realized_vol[i]
            vol_drag = 0.5 * 6 * sigma ** 2

        expense_costs[i] = expense
        trading_costs[i] = trading
        vol_drag_costs[i] = vol_drag

        returns[i] = gross - expense - trading - vol_drag

    return returns, gross_returns, expense_costs, trading_costs, vol_drag_costs


def compute_metrics(returns, gross_returns, rf, positions, n_test):
    equity = np.cumprod(1 + returns)
    gross_equity = np.cumprod(1 + gross_returns)
    total_ret = equity[-1] - 1
    n_years = n_test / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252)
    rf_annual = np.mean(rf) * 252
    sharpe = (annual_ret - rf_annual) / annual_vol if annual_vol > 0 else 0

    running_max = np.maximum.accumulate(equity)
    max_dd = np.max((running_max - equity) / running_max)

    rf_daily = np.mean(rf)
    downside = returns - rf_daily
    downside = np.minimum(downside, 0)
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    sortino = (annual_ret - rf_annual) / downside_std if downside_std > 0 else 0

    calmar = annual_ret / max_dd if max_dd > 0 else 0

    n_trades = int(np.sum(np.diff(positions) != 0))

    pct_3x = np.mean(positions == 3) * 100
    pct_1x = np.mean(positions == 1) * 100
    pct_cash = np.mean(positions == 0) * 100

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
    }


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
