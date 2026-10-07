"""Etapa 5: predicciones y senales CASH/SPY/UPRO.

La prediccion de la fila t usa las features de t-30 ... t-1 (prepare_sequences), por lo que con datos
hasta el dia d se obtiene tambien la senal del dia habil siguiente (fecha de ejecucion, orden MOC).
El corte entrenamiento/prueba es por fecha (fin_entrenamiento), no 80/20.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from pipeline import modelo as M
from pipeline.calendario import siguiente_habil

POS_A_SENAL = {0: "CASH", 1: "SPY", 3: "UPRO"}


@dataclass
class ModeloCargado:
    red: torch.nn.Module
    feature_cols: list
    imputer: object
    scaler: object
    y_scaler: StandardScaler


def _fechas(dataset: pd.DataFrame) -> pd.Series:
    return dataset["date"].astype(str).str[:10]


def cargar_modelo(dir_modelo: Path, dataset_a: pd.DataFrame, fin_entrenamiento: str) -> ModeloCargado:
    torch.manual_seed(M.SEED if hasattr(M, "SEED") else 42)
    pre = joblib.load(Path(dir_modelo) / "preprocessors.joblib")
    ck = torch.load(Path(dir_modelo) / "LSTM_Attention.pt", map_location="cpu", weights_only=False)
    red = M.LSTMAttention(ck["config"]["n_features"], hidden_dim=M.HIDDEN_DIM)
    red.load_state_dict(ck["state_dict"])
    red.eval()
    entren = dataset_a[_fechas(dataset_a) <= fin_entrenamiento]
    y_scaler = StandardScaler().fit(entren[M.TARGET_COL].values.reshape(-1, 1))
    return ModeloCargado(red, pre["feature_cols"], pre["imputer"], pre["scaler"], y_scaler)


def _matriz(m: ModeloCargado, dataset: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame({c: dataset[M.LEGACY_MAP.get(c, c)].values for c in m.feature_cols})
    return m.scaler.transform(m.imputer.transform(X))


def predecir(m: ModeloCargado, dataset: pd.DataFrame) -> pd.Series:
    """Prediccion por fecha para las filas 30..n-1 y para el dia habil siguiente al ultimo dato."""
    X = _matriz(m, dataset)
    seq = M.prepare_sequences(np.vstack([X, X[-1:]]), M.LOOKBACK)   # la fila extra no entra en ninguna ventana
    p = m.y_scaler.inverse_transform(M.predict_in_batches(m.red, seq, "cpu").reshape(-1, 1)).ravel()
    fechas = list(_fechas(dataset).iloc[M.LOOKBACK:])
    return pd.Series(p, index=fechas + [_siguiente_fecha(fechas[-1])])


def _siguiente_fecha(fecha: str) -> str:
    """Dia habil NYSE siguiente (fecha de ejecucion de la senal)."""
    return siguiente_habil(pd.Timestamp(fecha).date()).strftime("%Y-%m-%d")


def entrenar_meta(m: ModeloCargado, dataset_a: pd.DataFrame, fin_entrenamiento: str) -> dict:
    """Meta-KNN con las predicciones de entrenamiento (desalineamiento [:n_tr] literal de la tesis)."""
    entren = dataset_a[_fechas(dataset_a) <= fin_entrenamiento]
    preds = predecir(m, dataset_a)
    train_preds = preds.loc[_fechas(entren).iloc[M.LOOKBACK:]].values
    fwd, rf = entren["forward_returns"].values, entren["risk_free_rate"].values
    n_tr = min(len(train_preds), len(fwd), len(rf))
    pct = M.compute_rolling_percentiles(train_preds[:n_tr], window=M.PCTILE_WINDOW)
    X, y = M.build_meta_dataset(train_preds[:n_tr], pct, fwd[:n_tr], rf[:n_tr])
    return M.train_meta_knn(X, y)


def _backtest(test_preds, train_preds, fwd_test, fwd_train, rf_test, rf_train, meta, realista):
    """meta_knn_filtered_backtest de la tesis; con realista=True la ventana de mercado del Meta-KNN
    termina en t-2 (el retorno t-1 -> t se conoce recien al cierre del dia de ejecucion)."""
    if not realista:
        return M.meta_knn_filtered_backtest(test_preds, train_preds, fwd_test, fwd_train, rf_test, rf_train,
                                            meta, M.MIN_SIGNAL_STD)
    n_tr = min(len(train_preds), len(fwd_train), len(rf_train))
    all_preds = np.concatenate([train_preds[-n_tr:], test_preds])
    all_fwd = np.concatenate([fwd_train[-n_tr:], fwd_test])
    all_fwd_conocido = np.concatenate([[np.nan], all_fwd[:-1]])
    n_train, n_test = n_tr, len(test_preds)
    pct = M.compute_rolling_percentiles(all_preds, window=M.PCTILE_WINDOW)
    rstd = M.compute_rolling_std(all_preds, window=M.PCTILE_WINDOW)
    positions, pct_out = np.zeros(n_test), np.zeros(n_test)
    chosen, valid = [], np.zeros(n_test, dtype=bool)
    for t in range(n_test):
        a = n_train + t
        pct_out[t] = pct[a]
        if rstd[a] < M.MIN_SIGNAL_STD:
            chosen.append((0, 0))
            continue
        valid[t] = True
        f0 = max(0, a - M.FEATURE_WINDOW)
        if a - f0 < 10:
            combo = (10, 30)
        else:
            mw = all_fwd_conocido[f0:a]
            feats = M.compute_meta_features(all_preds[f0:a], mw[~np.isnan(mw)])
            fs = np.nan_to_num(meta["scaler"].transform(feats.reshape(1, -1)), nan=0.0, posinf=0.0, neginf=0.0)
            combo = M.IDX_TO_COMBO[meta["knn"].predict(fs)[0]]
        if pct[a] >= 100 - combo[0]:
            positions[t] = 3
        elif pct[a] >= 100 - combo[1]:
            positions[t] = 1
        chosen.append(combo)
    return positions, pct_out, chosen, valid


def generar_senales(m: ModeloCargado, meta: dict, dataset: pd.DataFrame, fin_entrenamiento: str,
                    realista: bool = True) -> pd.DataFrame:
    preds = predecir(m, dataset)
    fechas = _fechas(dataset)
    entren = dataset[fechas <= fin_entrenamiento]
    train_preds = preds.loc[_fechas(entren).iloc[M.LOOKBACK:]].values
    test_fechas = [f for f in preds.index if f > fin_entrenamiento]
    test_preds = preds.loc[test_fechas].values
    ds = dataset.assign(date=fechas).set_index("date")
    fwd_test = ds["forward_returns"].reindex(test_fechas).values
    rf_test = ds["risk_free_rate"].reindex(test_fechas).ffill().values
    pos, pct, combos, valid = _backtest(test_preds, train_preds, fwd_test, entren["forward_returns"].values,
                                        rf_test, entren["risk_free_rate"].values, meta, realista)
    return pd.DataFrame({
        "date": test_fechas, "prediccion": test_preds, "percentil": pct,
        "q_ext": [c[0] for c in combos], "q_mod": [c[1] for c in combos],
        "posicion": pos.astype(int), "senal_valida": valid,
        "forward_returns": fwd_test, "risk_free_rate": rf_test,
    })


def metricas(senales: pd.DataFrame, config: dict, desde: str | None = None, hasta: str | None = None,
             retraso: int = 0) -> dict:
    s = senales.copy()
    s["posicion"] = s["posicion"].shift(retraso).fillna(0).astype(int)
    if desde:
        s = s[s.date >= desde]
    if hasta:
        s = s[s.date <= hasta]
    s = s[s.forward_returns.notna()]
    if s.empty:
        return {"dias": 0, "entradas": 0}
    pos = s.posicion.values
    net, gross, *_ = M.calculate_returns_with_costs(pos, s.forward_returns.values, s.risk_free_rate.values)
    mt = M.compute_metrics(net, gross, s.risk_free_rate.values, pos, len(s))
    spy = s.forward_returns.values - config["instrumentos"]["SPY"]["expense_ratio"] / 252
    mt.update({
        "dias": int(len(s)), "desde": s.date.iloc[0], "hasta": s.date.iloc[-1],
        "entradas": int(np.sum((pos[1:] > 0) & (pos[:-1] == 0)) + (pos[0] > 0)),
        "spy_total_return": float(np.prod(1 + spy) - 1),
        "equity": np.cumprod(1 + net).tolist(), "equity_spy": np.cumprod(1 + spy).tolist(),
        "fechas": s.date.tolist(), "retornos": net.tolist(),
    })
    return mt
