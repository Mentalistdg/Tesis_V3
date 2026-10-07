"""Etapa 6: log de senales emitidas, reporte del periodo en vivo y JSON para la app."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import modelo as M
from pipeline.io import escribir_atomico, escribir_csv_atomico
from pipeline.senal import POS_A_SENAL, metricas

COLS_EMITIDAS = ["emitida_en", "fecha", "activo", "senal_b", "senal_a", "prediccion_b", "prediccion_a",
                 "percentil_b", "umbral_upro", "umbral_spy", "cambio_vs_ayer", "estado", "advertencias"]


def _ultima(senales: pd.DataFrame) -> pd.Series:
    return senales.iloc[-1]


def registrar_emision(log: Path, activo: str, senales_b: pd.DataFrame, senales_a: pd.DataFrame, resultado) -> bool:
    """Agrega una fila solo si la fecha es nueva o la senal B cambio. Devuelve True si agrego."""
    log = Path(log)
    b, a = _ultima(senales_b), _ultima(senales_a)
    previo = pd.read_csv(log, encoding="utf-8") if log.exists() else pd.DataFrame(columns=COLS_EMITIDAS)
    propio = previo[previo.activo == activo]
    senal_b = POS_A_SENAL[int(b.posicion)]
    misma_fecha = propio[propio.fecha == b.date]
    if len(misma_fecha) and misma_fecha.iloc[-1].senal_b == senal_b:
        return False
    anteriores = propio[propio.fecha < b.date]
    ayer = anteriores.iloc[-1].senal_b if len(anteriores) else None
    if ayer is None and len(senales_b) > 1:
        ayer = POS_A_SENAL[int(senales_b.iloc[-2].posicion)]
    fila = {
        "emitida_en": datetime.now().isoformat(timespec="seconds"), "fecha": b.date, "activo": activo,
        "senal_b": senal_b, "senal_a": POS_A_SENAL[int(a.posicion)],
        "prediccion_b": float(b.prediccion), "prediccion_a": float(a.prediccion),
        "percentil_b": float(b.percentil), "umbral_upro": 100 - int(b.q_ext), "umbral_spy": 100 - int(b.q_mod),
        "cambio_vs_ayer": bool(ayer is not None and ayer != senal_b),
        "estado": resultado.estado, "advertencias": " | ".join(resultado.mensajes),
    }
    nuevo = pd.DataFrame([fila]) if previo.empty else pd.concat([previo, pd.DataFrame([fila])], ignore_index=True)
    escribir_csv_atomico(nuevo[COLS_EMITIDAS], log)
    return True


def _resumen(mt: dict) -> dict:
    claves = ["dias", "desde", "hasta", "entradas", "total_return", "spy_total_return", "sharpe",
              "max_drawdown", "pct_3x", "pct_1x", "pct_cash", "n_trades"]
    out = {}
    for k in claves:
        if k in mt:
            v = mt[k]
            out[k] = float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, np.integer) else v)
    return out


def _escenarios(senales, config, desde):
    return {"ejecucion_tesis": _resumen(metricas(senales, config, desde=desde)),
            "retraso_1_dia": _resumen(metricas(senales, config, desde=desde, retraso=1))}


def reporte_periodo(senales_a: pd.DataFrame, senales_b: pd.DataFrame, config: dict, desde: str) -> dict:
    s_b = senales_b[senales_b.date >= desde]
    pos = s_b.posicion.values
    entradas = [s_b.date.iloc[i] for i in range(len(pos)) if pos[i] > 0 and (i == 0 or pos[i - 1] == 0)]
    b = _ultima(senales_b)
    return {
        "generado": datetime.now().isoformat(timespec="seconds"), "desde": desde,
        "A": _escenarios(senales_a, config, desde), "B": _escenarios(senales_b, config, desde),
        "fechas_entrada_b": entradas,
        "dias_por_senal_b": {POS_A_SENAL[k]: int((pos == k).sum()) for k in (0, 1, 3)},
        "ultima": {"fecha": b.date, "senal_b": POS_A_SENAL[int(b.posicion)],
                   "senal_a": POS_A_SENAL[int(_ultima(senales_a).posicion)]},
        "nota": "Calculado con datos de Bloomberg ya revisados; lo visto en vivo puede diferir levemente.",
    }


def reporte_texto(rep: dict) -> str:
    def bloque(nombre, e):
        t, r = e["ejecucion_tesis"], e["retraso_1_dia"]
        if not t.get("dias"):
            return f"  {nombre}: sin dias con retorno realizado aun"
        return (f"  {nombre}: {t['dias']} dias ({t['desde']} a {t['hasta']}) | Entradas: {t['entradas']} | "
                f"retorno {t['total_return'] * 100:+.1f}% vs SPY {t['spy_total_return'] * 100:+.1f}% | "
                f"Sharpe {t['sharpe']:.2f} | maxDD {-abs(t['max_drawdown']) * 100:.1f}% | "
                f"UPRO/SPY/CASH {t['pct_3x']:.0f}/{t['pct_1x']:.0f}/{t['pct_cash']:.0f}%\n"
                f"      con 1 dia de retraso: retorno {r['total_return'] * 100:+.1f}%")
    u = rep["ultima"]
    return "\n".join([
        f"CRONOS - reporte del periodo en vivo desde {rep['desde']} (generado {rep['generado']})",
        bloque("Version B (operativa)", rep["B"]), bloque("Version A (diagnostico)", rep["A"]),
        f"  Entradas B (CASH -> SPY/UPRO): {len(rep['fechas_entrada_b'])} -> {', '.join(rep['fechas_entrada_b']) or 'ninguna'}",
        f"  Dias por senal B: {rep['dias_por_senal_b']}",
        f"  SENAL PARA {u['fecha']}: B = {u['senal_b']} | A = {u['senal_a']} (orden MOC al cierre)",
        f"  Nota: {rep['nota']}",
    ])


# extract_trades: copia literal de Tesis_V3/paper/update_backend_data.py
def extract_trades(positions, strategy_returns, market_returns, dates, predictions, percentiles,
                    expense_costs=None, trading_costs=None, vol_drag_costs=None):
    """Extract individual trades from position changes."""
    trades = []
    n = len(positions)
    pos_to_inst = {3: 'UPRO', 1: 'SPY', 0: 'CASH'}
    trade_id = 0

    i = 0
    while i < n:
        entry_idx = i
        entry_pos = int(positions[i])
        cum_return = 1.0
        cum_market = 1.0
        trade_expense = 0.0
        trade_trading = 0.0
        trade_vol_drag = 0.0

        j = i
        while j < n and positions[j] == entry_pos:
            cum_return *= (1 + strategy_returns[j])
            cum_market *= (1 + market_returns[j])
            if expense_costs is not None:
                trade_expense += expense_costs[j]
            if trading_costs is not None:
                trade_trading += trading_costs[j]
            if vol_drag_costs is not None:
                trade_vol_drag += vol_drag_costs[j]
            j += 1

        exit_idx = j - 1
        trade_id += 1
        tx_cost = trade_expense + trade_trading + trade_vol_drag
        trade_dict = {
            'trade_id': trade_id,
            'entry_date': dates[entry_idx],
            'entry_idx': entry_idx,
            'entry_position': entry_pos,
            'entry_instrument': pos_to_inst.get(entry_pos, 'CASH'),
            'entry_prediction': float(predictions[entry_idx]),
            'entry_percentile': float(percentiles[entry_idx]),
            'exit_date': dates[exit_idx],
            'exit_idx': exit_idx,
            'duration': exit_idx - entry_idx + 1,
            'total_return': float(cum_return - 1),
            'market_return': float(cum_market - 1),
            'tx_cost': float(tx_cost),
            'costs': {
                'expense': float(trade_expense),
                'trading': float(trade_trading),
                'vol_drag': float(trade_vol_drag),
            },
        }
        trades.append(trade_dict)
        i = j

    return trades


def entrada_app(nombre: str, senales: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    """Entrada con el esquema de models_summary.json / daily_data.json (fechas = dia de realizacion)."""
    s = senales.reset_index(drop=True)
    real = s[s.forward_returns.notna()]
    real = real[real.index < len(s) - 1]
    fechas = [s.date.iloc[i + 1] for i in real.index]
    pos = real.posicion.values.astype(int)
    fwd, rf = real.forward_returns.values, real.risk_free_rate.values
    n = len(pos)
    if n:
        net, gross, exp_c, trd_c, vol_c = M.calculate_returns_with_costs(pos, fwd, rf)
    else:
        net = gross = exp_c = trd_c = vol_c = np.array([])
    eq = np.cumprod(1 + net)
    spy_eq = np.cumprod(1 + fwd - config["instrumentos"]["SPY"]["expense_ratio"] / 252)
    pico = np.maximum.accumulate(eq) if n else eq
    dd = (eq - pico) / pico if n else eq
    anios = n / 252
    total = float(eq[-1] - 1) if n else 0.0
    spy_total = float(spy_eq[-1] - 1) if n else 0.0
    anual = float((1 + total) ** (1 / anios) - 1) if anios > 0 else 0.0
    vol = float(np.std(net) * np.sqrt(252)) if n else 0.0
    rf_a = float(np.mean(rf) * 252) if n else 0.0
    trades = extract_trades(pos, net, fwd, fechas, real.prediccion.values, real.percentil.values,
                            exp_c, trd_c, vol_c) if n else []
    activos = [t for t in trades if t["entry_position"] != 0]
    metricas_d = {
        "total_return": total, "annual_return": anual, "market_return": spy_total, "excess_return": total - spy_total,
        "sharpe": float((anual - rf_a) / vol) if vol > 0 else 0.0, "sortino": 0.0, "calmar": 0.0,
        "max_drawdown": float(dd.min()) if n else 0.0, "annual_volatility": vol,
        "win_rate": float(np.mean([t["total_return"] > 0 for t in activos])) if activos else 0.0,
        "dir_accuracy": float(np.mean(np.sign(real.prediccion.values) == np.sign(fwd))) if n else 0.0,
        "mean_position": float(np.mean(pos)) if n else 0.0, "n_trades": int(np.sum(np.diff(pos) != 0)) if n else 0,
        "n_days": n, "pct_long": float(np.mean(pos > 0) * 100) if n else 0.0,
        "pct_3x": float(np.mean(pos == 3) * 100) if n else 0.0, "pct_1x": float(np.mean(pos == 1) * 100) if n else 0.0,
        "pct_cash": float(np.mean(pos == 0) * 100) if n else 0.0, "transaction_costs": 0.0, "pct_signal_days": 100.0,
    }
    params = {"q_3x": int(s.q_ext.iloc[-1]), "q_1x": int(s.q_mod.iloc[-1])}
    daily = {
        "category": "Produccion", "params": params, "dates": fechas, "trades": trades,
        "predictions": real.prediccion.tolist(), "percentiles": real.percentil.tolist(), "positions": pos.tolist(),
        "strategy_returns": net.tolist(), "market_returns": fwd.tolist(), "equity_curve": eq.tolist(),
        "market_equity": spy_eq.tolist(), "drawdown": dd.tolist(), "daily_costs": (exp_c + trd_c + vol_c).tolist(),
        "risk_free": rf.tolist(), "metrics": metricas_d,
    }
    resumen = {"model": nombre, "category": "Produccion", "beat_spy": metricas_d["excess_return"] > 0,
               **metricas_d, "initial_capital": 10000, "final_capital": float(10000 * (1 + total)),
               "profit_loss": float(10000 * total), "n_years": float(anios), "params": params,
               "final_equity": float(10000 * (1 + total))}
    return resumen, daily


def exportar_app(dir_datos_app: Path, activo: str, senales_b: pd.DataFrame, senales_a: pd.DataFrame, resultado,
                 reporte: dict, config: dict) -> None:
    dir_datos_app = Path(dir_datos_app)
    b, a = _ultima(senales_b), _ultima(senales_a)
    hoy_b = POS_A_SENAL[int(b.posicion)]
    ayer = POS_A_SENAL[int(senales_b.iloc[-2].posicion)] if len(senales_b) > 1 else None
    tarjeta = {
        "activo": activo, "nombre": config.get("nombre", activo), "fecha": b.date,
        "senal_b": hoy_b, "senal_a": POS_A_SENAL[int(a.posicion)],
        "senal_anterior_b": ayer, "cambio_vs_ayer": bool(ayer is not None and ayer != hoy_b),
        "prediccion": float(b.prediccion), "percentil": float(b.percentil),
        "umbrales": {"UPRO": 100 - int(b.q_ext), "SPY": 100 - int(b.q_mod)}, "senal_valida": bool(b.senal_valida),
        "estado": resultado.estado, "advertencias": list(resultado.mensajes), "reporte": reporte,
        "actualizado": datetime.now().isoformat(timespec="seconds"),
        "historial": [{"date": r.date, "senal_b": POS_A_SENAL[int(r.posicion)],
                       "senal_a": POS_A_SENAL[int(sa)], "prediccion": float(r.prediccion), "percentil": float(r.percentil)}
                      for r, sa in zip(senales_b.tail(60).itertuples(), senales_a.posicion.tail(60))],
    }
    escribir_atomico(dir_datos_app / "senales.json", json.dumps({"activos": [tarjeta]}, ensure_ascii=False, indent=1))
    vivo = {"summary": [], "daily": {}}
    for nombre, s in (("LSTM_Attention_vivo_B", senales_b), ("LSTM_Attention_vivo_A", senales_a)):
        r, d = entrada_app(nombre, s[s.date >= config["ancla_inicial"]], config)
        r["activo"], d["activo"] = activo, activo
        vivo["summary"].append(r)
        vivo["daily"][nombre] = d
    escribir_atomico(dir_datos_app / "vivo.json", json.dumps(vivo, ensure_ascii=False))
