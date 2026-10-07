# -*- coding: utf-8 -*-
"""Modelo LSTM_Attention y backtest Meta-KNN de la tesis (copia literal de produccion_lstm.py,
que a su vez copia scripts/train_models.py y scripts/optimize_and_backtest.py del repo Tesis_V3).
No modificar: la reproduccion exacta de la tesis depende de este codigo."""
from collections import Counter

import numpy as np
from scipy.stats import skew as scipy_skew
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

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


