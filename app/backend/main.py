# -*- coding: utf-8 -*-
"""
================================================================================
BACKEND - Strategy Visualizer App
================================================================================
FastAPI backend for the ML Strategy Visualization Application.

Endpoints:
- /api/models - List all models with summary metrics
- /api/models/{name} - Get detailed data for a specific model
- /api/models/{name}/metrics - Get metrics (optionally filtered by date range)
- /api/market - Get market data
- /api/signals - Get current signals from all models
- /api/regimes - Get regime analysis data
- /api/compare - Compare multiple models

================================================================================
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import date
import json
import os
import numpy as np
from pydantic import BaseModel

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

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
    with open(filepath, 'r') as f:
        return json.load(f)


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

    if len(strategy_returns) == 0:
        return None

    # Calculate metrics
    strategy_total = float(np.prod(1 + strategy_returns) - 1)
    market_total = float(np.prod(1 + market_returns) - 1)

    n_days = len(strategy_returns)
    years = n_days / 252
    strategy_annual = float((1 + strategy_total) ** (1/years) - 1) if years > 0 else 0

    sharpe = float(np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)) if np.std(strategy_returns) > 0 else 0

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

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Strategy Visualizer API"}


@app.get("/api/models")
async def get_models():
    """Get list of all models with summary metrics."""
    try:
        summary = load_json("models_summary.json")
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}")
async def get_model_detail(model_name: str):
    """Get detailed data for a specific model."""
    try:
        daily_data = load_json("daily_data.json")
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
        daily_data = load_json("daily_data.json")
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
        daily_data = load_json("daily_data.json")
        if model_name not in daily_data:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        return {"model": model_name, "trades": daily_data[model_name].get('trades', [])}
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


@app.get("/api/signals")
async def get_signals():
    """Get current signals from all models."""
    try:
        daily_data = load_json("daily_data.json")

        signals = []
        for model_name, data in daily_data.items():
            if len(data['positions']) > 0:
                current_position = data['positions'][-1]
                prev_position = data['positions'][-2] if len(data['positions']) > 1 else current_position
                change = current_position - prev_position

                # Handle positions from -3 to +3
                if current_position >= 2.0:
                    signal = 'STRONG_LONG'
                elif current_position >= 1.0:
                    signal = 'LONG'
                elif current_position > 0:
                    signal = 'WEAK_LONG'
                elif current_position == 0:
                    signal = 'CASH'
                elif current_position > -2.0:
                    signal = 'SHORT'
                else:
                    signal = 'STRONG_SHORT'

                signals.append({
                    'model': model_name,
                    'category': data['category'],
                    'position': current_position,
                    'signal': signal,
                    'change': change,
                    'date': data['dates'][-1]
                })

        # Sort by position descending
        signals = sorted(signals, key=lambda x: x['position'], reverse=True)

        # Calculate consensus
        positions = [s['position'] for s in signals]
        consensus_position = np.mean(positions)
        bullish_count = sum(1 for p in positions if p >= 1.0)
        neutral_count = sum(1 for p in positions if -1.0 < p < 1.0)
        bearish_count = sum(1 for p in positions if p <= -1.0)

        return {
            'date': signals[0]['date'] if signals else None,
            'consensus': {
                'position': float(consensus_position),
                'bullish_count': bullish_count,
                'neutral_count': neutral_count,
                'bearish_count': bearish_count,
                'total_models': len(signals)
            },
            'signals': signals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regimes")
async def get_regimes():
    """Get regime analysis data."""
    try:
        regime_data = load_json("regime_data.json")
        return regime_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
async def compare_models(
    models: str = Query(..., description="Comma-separated model names"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Compare multiple models side by side."""
    try:
        daily_data = load_json("daily_data.json")
        model_list = [m.strip() for m in models.split(',')]

        comparison = []
        for model_name in model_list:
            if model_name not in daily_data and model_name != 'BuyHold':
                continue

            if model_name == 'BuyHold':
                market_data = load_json("market_data.json")
                comparison.append({
                    'model': 'Buy & Hold',
                    'category': 'Benchmark',
                    'metrics': market_data['metrics'],
                    'equity_curve': market_data['equity_curve'],
                    'drawdown': market_data['drawdown']
                })
            else:
                data = daily_data[model_name]

                if start_date and end_date:
                    metrics = calculate_metrics_for_period(data, start_date, end_date)
                else:
                    metrics = data['metrics']

                comparison.append({
                    'model': model_name,
                    'category': data['category'],
                    'metrics': metrics,
                    'equity_curve': data['equity_curve'],
                    'drawdown': data['drawdown'],
                    'dates': data['dates']
                })

        return {'models': comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
