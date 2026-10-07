# -*- coding: utf-8 -*-
"""
================================================================================
BACKEND - CRONOS Strategy Visualizer (LONG-ONLY)
================================================================================
FastAPI backend serving pre-computed JSON data for the CRONOS dashboard.

Endpoints:
- /api/models                       - List all models with summary metrics
- /api/models/{name}                - Get detailed daily data for a model
- /api/models/{name}/metrics        - Get metrics (optionally filtered by date range)
- /api/models/{name}/trades         - Get trade log for a model
- /api/models/{name}/daily-log      - Get daily decision log for a model
- /api/market                       - Get SPY market data
- /api/regimes                      - Get regime analysis data

================================================================================
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import os
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CRONOS_APP_DATA", os.path.join(SCRIPT_DIR, "data"))
DIST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "frontend", "dist")

# Initialize FastAPI
app = FastAPI(
    title="Strategy Visualizer API",
    description="API for ML Trading Strategy Visualization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(filename: str):
    """Load JSON file from data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_vivo():
    """Modelos del periodo en vivo generados por el pipeline (pipeline/exportar.py)."""
    try:
        return load_json("vivo.json")
    except HTTPException:
        return {"summary": [], "daily": {}}


def load_daily():
    try:
        daily = load_json("daily_data.json")
    except HTTPException:
        daily = {}
    return {**daily, **load_vivo()["daily"]}


def calculate_metrics_for_period(data: dict, start_date: str, end_date: str) -> dict:
    """Recalculate metrics for a specific date range."""
    dates = data['dates']

    # Find indices
    start_idx = 0
    end_idx = len(dates) - 1

    for i, d in enumerate(dates):
        if d >= start_date and start_idx == 0:
            start_idx = i
        if d <= end_date:
            end_idx = i

    # Slice data
    strategy_returns = np.array(data['strategy_returns'][start_idx:end_idx+1])
    market_returns = np.array(data['market_returns'][start_idx:end_idx+1])
    positions = np.array(data['positions'][start_idx:end_idx+1])
    risk_free = np.array(data.get('risk_free', [0]*len(data['dates']))[start_idx:end_idx+1])

    if len(strategy_returns) == 0:
        return None

    # Calculate metrics
    strategy_total = float(np.prod(1 + strategy_returns) - 1)
    market_total = float(np.prod(1 + market_returns) - 1)

    n_days = len(strategy_returns)
    years = n_days / 252
    strategy_annual = float((1 + strategy_total) ** (1/years) - 1) if years > 0 else 0
    annual_vol = float(np.std(strategy_returns) * np.sqrt(252))
    rf_annual = float(np.mean(risk_free) * 252)

    sharpe = float((strategy_annual - rf_annual) / annual_vol) if annual_vol > 0 else 0

    # Drawdown
    equity_curve = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    win_rate = float(np.sum(strategy_returns > 0) / len(strategy_returns))

    return {
        'period': {'start': start_date, 'end': end_date, 'days': n_days},
        'total_return': strategy_total,
        'annual_return': strategy_annual,
        'market_return': market_total,
        'excess_return': strategy_total - market_total,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'mean_position': float(np.mean(positions)),
        'n_trades': int(np.sum(np.abs(np.diff(positions)) > 0.1))
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Strategy Visualizer API"}


@app.get("/api/models")
async def get_models():
    """Get list of all models with summary metrics."""
    try:
        summary = load_json("models_summary.json")
        vivo = load_vivo()["summary"]
        return {**summary, "models": vivo + summary.get("models", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}")
async def get_model_detail(model_name: str):
    """Get detailed data for a specific model."""
    try:
        daily_data = load_daily()
        if model_name not in daily_data:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        return daily_data[model_name]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(
    model_name: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get metrics for a model, optionally filtered by date range."""
    try:
        daily_data = load_daily()
        if model_name not in daily_data:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        data = daily_data[model_name]

        # If no date filter, return full metrics
        if not start_date and not end_date:
            return {
                'period': 'full',
                'metrics': data['metrics']
            }

        # Use default dates if not provided
        if not start_date:
            start_date = data['dates'][0]
        if not end_date:
            end_date = data['dates'][-1]

        # Calculate metrics for period
        metrics = calculate_metrics_for_period(data, start_date, end_date)
        if metrics is None:
            raise HTTPException(status_code=400, detail="No data for selected period")

        return {
            'period': 'filtered',
            'metrics': metrics,
            'full_period_metrics': data['metrics']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}/trades")
async def get_model_trades(model_name: str):
    """Get trade log for a specific model."""
    try:
        daily_data = load_daily()
        if model_name not in daily_data:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        return {"model": model_name, "trades": daily_data[model_name].get('trades', [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}/daily-log")
async def get_model_daily_log(model_name: str):
    """Get daily log for a specific model (one row per trading day).
    Constructs daily_log from the parallel arrays in daily_data.json."""
    try:
        daily_data = load_daily()
        if model_name not in daily_data:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        model = daily_data[model_name]

        # If daily_log is pre-computed, use it
        if model.get('daily_log'):
            return {
                "model": model_name,
                "daily_log": model['daily_log'],
                "warmup_used": model.get('warmup_used', False)
            }

        # Otherwise, construct from parallel arrays
        dates = model.get('dates', [])
        predictions = model.get('predictions', [])
        percentiles = model.get('percentiles', [])
        positions = model.get('positions', [])
        strategy_returns = model.get('strategy_returns', [])
        market_returns = model.get('market_returns', [])
        equity_curve = model.get('equity_curve', [])
        drawdown = model.get('drawdown', [])
        regimes = model.get('regimes', [])
        daily_costs = model.get('daily_costs', [])

        # Compute regimes from market returns if not present
        if not regimes and market_returns:
            mkt = np.array(market_returns, dtype=float)
            window = 60
            regimes = []
            for i in range(len(mkt)):
                if i < window:
                    regimes.append('unknown')
                    continue
                w = mkt[i-window:i]
                cum = float(np.prod(1 + w) - 1)
                vol = float(np.std(w) * np.sqrt(252))
                if cum > 0.10 and vol < 0.20:
                    regimes.append('bull')
                elif cum < -0.10:
                    regimes.append('bear')
                elif vol > 0.25:
                    regimes.append('high_vol')
                else:
                    regimes.append('sideways')

        n = len(dates)
        daily_log = []
        high_water_mark = 1.0

        for i in range(n):
            eq = equity_curve[i] if i < len(equity_curve) else 1.0
            if eq > high_water_mark:
                high_water_mark = eq

            pos = positions[i] if i < len(positions) else 0
            prev_pos = positions[i - 1] if i > 0 and i - 1 < len(positions) else pos
            mkt_ret = market_returns[i] if i < len(market_returns) else 0
            strat_ret = strategy_returns[i] if i < len(strategy_returns) else 0

            # Direction correct: strategy return > 0 when market moved, or both zero
            direction_correct = (strat_ret > 0) if (mkt_ret != 0) else True

            daily_log.append({
                "date": dates[i],
                "day_num": i + 1,
                "prediction": predictions[i] if i < len(predictions) else 0,
                "percentile": percentiles[i] if i < len(percentiles) else 0,
                "position_base": pos,
                "position_final": pos,
                "strategy_return": strat_ret,
                "market_return": mkt_ret,
                "equity": eq,
                "drawdown": drawdown[i] if i < len(drawdown) else 0,
                "high_water_mark": high_water_mark,
                "trading_cost": daily_costs[i] if i < len(daily_costs) else 0,
                "regime": regimes[i] if i < len(regimes) and regimes else "",
                "position_changed": pos != prev_pos,
                "direction_correct": direction_correct,
            })

        return {
            "model": model_name,
            "daily_log": daily_log,
            "warmup_used": False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market")
async def get_market_data():
    """Get market data (SPY)."""
    try:
        market_data = load_json("market_data.json")
        return market_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regimes")
async def get_regimes():
    """Get regime analysis data."""
    try:
        # Try to load regime data, or generate basic info from market data
        try:
            regime_data = load_json("regime_data.json")
            return regime_data
        except:
            # Generate basic regime info from market data
            market_data = load_json("market_data.json")
            returns = np.array(market_data['returns'])

            # Simple regime detection
            window = 60
            regimes = []
            for i in range(len(returns)):
                if i < window:
                    regimes.append('unknown')
                    continue
                window_returns = returns[i-window:i]
                cum_return = np.prod(1 + window_returns) - 1
                vol = np.std(window_returns) * np.sqrt(252)

                if cum_return > 0.10 and vol < 0.20:
                    regime = 'bull'
                elif cum_return < -0.10:
                    regime = 'bear'
                elif vol > 0.25:
                    regime = 'high_vol'
                else:
                    regime = 'sideways'
                regimes.append(regime)

            regime_counts = {}
            for r in regimes:
                regime_counts[r] = regime_counts.get(r, 0) + 1

            return {
                'current_regime': regimes[-1] if regimes else 'unknown',
                'regime_counts': regime_counts,
                'dates': market_data['dates'],
                'regimes': regimes
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SENALES DE PRODUCCION (pipeline CRONOS)
# =============================================================================

@app.get("/api/senales")
async def get_senales():
    """Senal del dia por activo (escrita por el pipeline en senales.json)."""
    if not os.path.exists(os.path.join(DATA_DIR, "senales.json")):
        raise HTTPException(status_code=503, detail="Aun no hay corridas del pipeline (falta senales.json)")
    from datetime import datetime
    from zoneinfo import ZoneInfo
    hoy_ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    datos = load_json("senales.json")
    for a in datos.get("activos", []):
        a["vencida"] = a.get("fecha", "") < hoy_ny      # la senal es para el cierre de 'fecha'
    try:
        datos["corrida"] = load_json("estado_corrida.json")
    except HTTPException:
        datos["corrida"] = None
    return datos


@app.get("/api/senales/{activo}/historial")
async def get_senales_historial(activo: str):
    """Periodo en vivo (versiones B y A) para un activo."""
    daily = load_vivo()["daily"]
    out = {}
    for version in ("B", "A"):
        d = daily.get(f"LSTM_Attention_vivo_{version}")
        if d and d.get("activo") == activo:
            out[version] = d
    if not out:
        raise HTTPException(status_code=404, detail=f"Activo sin historial: {activo}")
    return out


# Frontend compilado (npm run build) servido por FastAPI, con fallback de SPA
if os.path.isdir(DIST_DIR):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

    @app.get("/{ruta:path}", include_in_schema=False)
    async def spa(ruta: str):
        archivo = os.path.join(DIST_DIR, ruta)
        if ruta and os.path.isfile(archivo):
            return FileResponse(archivo)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
