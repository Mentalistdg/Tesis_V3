# -*- coding: utf-8 -*-
"""
================================================================================
FORECASTING CON NIXTLA (StatsForecast) Y PROPHET
================================================================================
Modelos de series temporales adicionales para complementar el pipeline ML.

Modelos incluidos:
- StatsForecast: AutoARIMA, AutoETS, AutoTheta, SeasonalNaive, MSTL
- Prophet: Con soporte para regresores externos

Autor: Tesis ML Pipeline
================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# IMPORTS CONDICIONALES
# =============================================================================
try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        SeasonalNaive,
        MSTL,
        Naive,
        WindowAverage,
        SeasonalWindowAverage
    )
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    print("[WARNING] StatsForecast no disponible. Instalar con: pip install statsforecast")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet no disponible. Instalar con: pip install prophet")

# NeuralForecast (opcional, requiere PyTorch)
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS, PatchTST
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def prepare_data_for_statsforecast(df: pd.DataFrame,
                                    target_col: str = 'SPY_CLOSE',
                                    date_col: str = 'date') -> pd.DataFrame:
    """
    Prepara datos en formato requerido por StatsForecast.
    Formato: unique_id, ds, y
    """
    sf_df = pd.DataFrame({
        'unique_id': 'SPY',
        'ds': pd.to_datetime(df[date_col]),
        'y': df[target_col].values
    })
    return sf_df.dropna()


def prepare_data_for_prophet(df: pd.DataFrame,
                              target_col: str = 'SPY_CLOSE',
                              date_col: str = 'date',
                              regressor_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepara datos en formato requerido por Prophet.
    Formato: ds, y, [regressors]
    """
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df[date_col]),
        'y': df[target_col].values
    })

    if regressor_cols:
        for col in regressor_cols:
            if col in df.columns:
                prophet_df[col] = df[col].values

    return prophet_df.dropna()


def calculate_forecast_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula metricas de forecast."""
    # Asegurar arrays limpios
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (evitar division por cero)
    mape_mask = y_true != 0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    else:
        mape = np.nan

    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # Directional Accuracy (para retornos)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        dir_acc = np.mean(true_direction == pred_direction)
    else:
        dir_acc = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': dir_acc
    }


def calculate_trading_metrics(returns: np.ndarray,
                               predictions: np.ndarray,
                               risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Calcula metricas de trading basadas en predicciones."""
    # Posiciones: long si prediccion > 0, else cash
    positions = np.sign(predictions)
    positions = np.where(positions >= 0, 1, 0)  # Solo long o cash

    # Retornos de estrategia
    strategy_returns = positions[:-1] * returns[1:]

    # Metricas
    total_return = np.prod(1 + strategy_returns) - 1
    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0

    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    sortino = np.mean(strategy_returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

    # Max Drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    # Win rate
    wins = np.sum(strategy_returns > 0)
    total_trades = np.sum(positions[:-1] == 1)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_trades': int(total_trades)
    }


# =============================================================================
# STATSFORECAST MODELS
# =============================================================================
def run_statsforecast_models(df: pd.DataFrame,
                              target_col: str = 'SPY_CLOSE',
                              date_col: str = 'date',
                              train_ratio: float = 0.8,
                              horizon: int = 1,
                              season_length: int = 5) -> Dict:
    """
    Ejecuta modelos de StatsForecast con validacion temporal.

    Modelos:
    - AutoARIMA: ARIMA automatico
    - AutoETS: Exponential Smoothing automatico
    - AutoTheta: Theta automatico
    - SeasonalNaive: Baseline estacional
    - MSTL: Multiple Seasonal-Trend decomposition
    """
    if not STATSFORECAST_AVAILABLE:
        return {'error': 'StatsForecast no disponible'}

    print("\n" + "=" * 80)
    print("STATSFORECAST MODELS")
    print("=" * 80)

    # Preparar datos
    sf_df = prepare_data_for_statsforecast(df, target_col, date_col)

    # Split temporal
    n = len(sf_df)
    train_size = int(n * train_ratio)

    train_df = sf_df.iloc[:train_size].copy()
    test_df = sf_df.iloc[train_size:].copy()

    print(f"\n  Train: {len(train_df)} observaciones")
    print(f"  Test: {len(test_df)} observaciones")
    print(f"  Horizon: {horizon}")
    print(f"  Season length: {season_length}")

    # Configurar modelos
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        Naive(),
        WindowAverage(window_size=5, alias='WindowAvg5'),
        WindowAverage(window_size=21, alias='WindowAvg21'),
    ]

    # Crear objeto StatsForecast
    sf = StatsForecast(
        models=models,
        freq='B',  # Business day frequency
        n_jobs=-1
    )

    results = {}

    # Rolling forecast para evaluacion
    print("\n  Ejecutando rolling forecast...")

    try:
        # Cross-validation con ventana expandible (reducido para velocidad)
        crossvalidation_df = sf.cross_validation(
            df=sf_df,
            h=horizon,
            step_size=5,  # Cada 5 dias en lugar de cada dia
            n_windows=min(len(test_df) // 5, 100)  # Maximo 100 ventanas
        )

        # Calcular metricas por modelo
        model_cols = [col for col in crossvalidation_df.columns
                      if col not in ['unique_id', 'ds', 'cutoff', 'y']]

        print("\n  Resultados por modelo:")
        print("-" * 80)

        for model_name in model_cols:
            y_true = crossvalidation_df['y'].values
            y_pred = crossvalidation_df[model_name].values

            metrics = calculate_forecast_metrics(y_true, y_pred)

            # Calcular retornos para metricas de trading
            returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]

            if len(returns) > 10:
                trading_metrics = calculate_trading_metrics(returns, pred_returns)
            else:
                trading_metrics = {}

            results[model_name] = {
                'forecast_metrics': metrics,
                'trading_metrics': trading_metrics
            }

            print(f"\n  {model_name}:")
            print(f"    RMSE: {metrics['rmse']:.6f}")
            print(f"    MAE: {metrics['mae']:.6f}")
            print(f"    R2: {metrics['r2']:.4f}")
            print(f"    Dir. Acc: {metrics['directional_accuracy']:.2%}")
            if trading_metrics:
                print(f"    Sharpe: {trading_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"    Return: {trading_metrics.get('total_return', 0):.2%}")

        # Guardar predicciones
        results['predictions'] = crossvalidation_df

    except Exception as e:
        print(f"\n  [ERROR] {str(e)}")
        results['error'] = str(e)

    return results


# =============================================================================
# PROPHET MODEL
# =============================================================================
def run_prophet_model(df: pd.DataFrame,
                       target_col: str = 'SPY_CLOSE',
                       date_col: str = 'date',
                       train_ratio: float = 0.8,
                       regressor_cols: Optional[List[str]] = None,
                       yearly_seasonality: bool = True,
                       weekly_seasonality: bool = True,
                       daily_seasonality: bool = False) -> Dict:
    """
    Ejecuta Prophet con validacion temporal.

    Prophet es bueno para:
    - Series con estacionalidad fuerte
    - Deteccion de cambios de tendencia
    - Inclusion de regresores externos
    """
    if not PROPHET_AVAILABLE:
        return {'error': 'Prophet no disponible'}

    print("\n" + "=" * 80)
    print("PROPHET MODEL")
    print("=" * 80)

    # Preparar datos
    prophet_df = prepare_data_for_prophet(df, target_col, date_col, regressor_cols)

    # Split temporal
    n = len(prophet_df)
    train_size = int(n * train_ratio)

    train_df = prophet_df.iloc[:train_size].copy()
    test_df = prophet_df.iloc[train_size:].copy()

    print(f"\n  Train: {len(train_df)} observaciones")
    print(f"  Test: {len(test_df)} observaciones")
    if regressor_cols:
        print(f"  Regresores: {regressor_cols}")

    results = {}

    try:
        # Configurar Prophet
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=0.05,  # Flexibilidad en cambios de tendencia
            seasonality_prior_scale=10,
            interval_width=0.95
        )

        # Agregar regresores si existen
        if regressor_cols:
            for col in regressor_cols:
                if col in train_df.columns:
                    model.add_regressor(col)

        # Entrenar
        print("\n  Entrenando Prophet...")
        model.fit(train_df)

        # Predecir en test
        print("  Generando predicciones...")
        forecast = model.predict(test_df)

        # Metricas
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        metrics = calculate_forecast_metrics(y_true, y_pred)

        # Trading metrics
        returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        trading_metrics = calculate_trading_metrics(returns, pred_returns)

        results['forecast_metrics'] = metrics
        results['trading_metrics'] = trading_metrics
        results['predictions'] = forecast
        results['components'] = model.plot_components

        print("\n  Resultados Prophet:")
        print("-" * 80)
        print(f"    RMSE: {metrics['rmse']:.6f}")
        print(f"    MAE: {metrics['mae']:.6f}")
        print(f"    R2: {metrics['r2']:.4f}")
        print(f"    Dir. Acc: {metrics['directional_accuracy']:.2%}")
        print(f"    Sharpe: {trading_metrics['sharpe_ratio']:.3f}")
        print(f"    Return: {trading_metrics['total_return']:.2%}")
        print(f"    Max DD: {trading_metrics['max_drawdown']:.2%}")

        # Analisis de componentes
        print("\n  Componentes detectados:")
        if yearly_seasonality:
            print("    - Estacionalidad anual: SI")
        if weekly_seasonality:
            print("    - Estacionalidad semanal: SI")
        print(f"    - Changepoints detectados: {len(model.changepoints)}")

    except Exception as e:
        print(f"\n  [ERROR] {str(e)}")
        results['error'] = str(e)

    return results


# =============================================================================
# PROPHET CON REGRESORES FINANCIEROS
# =============================================================================
def run_prophet_with_financial_regressors(df: pd.DataFrame,
                                           target_col: str = 'SPY_CLOSE',
                                           date_col: str = 'date',
                                           train_ratio: float = 0.8) -> Dict:
    """
    Prophet con regresores financieros especificos.

    Usa indicadores como:
    - VIX (volatilidad)
    - Tasas de interes
    - Momentum
    """
    if not PROPHET_AVAILABLE:
        return {'error': 'Prophet no disponible'}

    print("\n" + "=" * 80)
    print("PROPHET CON REGRESORES FINANCIEROS")
    print("=" * 80)

    # Identificar regresores disponibles
    potential_regressors = ['V1', 'V2', 'I4', 'I6', 'M1', 'momentum_21d']
    available_regressors = [col for col in potential_regressors if col in df.columns]

    if not available_regressors:
        # Crear momentum si no existe
        if 'SPY_CLOSE' in df.columns:
            df = df.copy()
            df['momentum_21d'] = df['SPY_CLOSE'].pct_change(21)
            available_regressors = ['momentum_21d']

    print(f"\n  Regresores disponibles: {available_regressors}")

    return run_prophet_model(
        df=df,
        target_col=target_col,
        date_col=date_col,
        train_ratio=train_ratio,
        regressor_cols=available_regressors,
        yearly_seasonality=True,
        weekly_seasonality=True
    )


# =============================================================================
# ENSEMBLE FORECAST
# =============================================================================
def create_forecast_ensemble(statsforecast_results: Dict,
                              prophet_results: Dict,
                              weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Crea un ensemble de los mejores modelos de forecasting.
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE FORECAST")
    print("=" * 80)

    # Recopilar todos los modelos con sus metricas
    all_models = {}

    # StatsForecast models
    if 'error' not in statsforecast_results:
        for model_name, model_results in statsforecast_results.items():
            if isinstance(model_results, dict) and 'forecast_metrics' in model_results:
                all_models[f"SF_{model_name}"] = model_results

    # Prophet
    if 'error' not in prophet_results and 'forecast_metrics' in prophet_results:
        all_models['Prophet'] = prophet_results

    if not all_models:
        return {'error': 'No hay modelos validos para ensemble'}

    # Ordenar por RMSE
    sorted_models = sorted(
        all_models.items(),
        key=lambda x: x[1]['forecast_metrics'].get('rmse', float('inf'))
    )

    print("\n  Ranking de modelos (por RMSE):")
    print("-" * 80)
    for i, (name, results) in enumerate(sorted_models, 1):
        metrics = results['forecast_metrics']
        trading = results.get('trading_metrics', {})
        print(f"  {i}. {name}")
        print(f"     RMSE: {metrics['rmse']:.6f}, R2: {metrics['r2']:.4f}")
        print(f"     Dir Acc: {metrics['directional_accuracy']:.2%}")
        if trading:
            print(f"     Sharpe: {trading.get('sharpe_ratio', 0):.3f}, Return: {trading.get('total_return', 0):.2%}")

    # Mejor modelo
    best_model_name = sorted_models[0][0]
    best_model_results = sorted_models[0][1]

    print(f"\n  [MEJOR MODELO]: {best_model_name}")

    return {
        'best_model': best_model_name,
        'best_results': best_model_results,
        'all_models': dict(sorted_models),
        'ranking': [name for name, _ in sorted_models]
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_forecasting_pipeline(data_file: Optional[str] = None,
                              train_ratio: float = 0.8) -> Dict:
    """
    Pipeline completo de forecasting con Nixtla y Prophet.
    """
    print("=" * 80)
    print("FORECASTING PIPELINE - NIXTLA & PROPHET")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"StatsForecast disponible: {STATSFORECAST_AVAILABLE}")
    print(f"Prophet disponible: {PROPHET_AVAILABLE}")
    print(f"NeuralForecast disponible: {NEURALFORECAST_AVAILABLE}")

    # Cargar datos
    if data_file is None:
        data_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")

    print(f"\nCargando datos: {data_file}")

    if not os.path.exists(data_file):
        print(f"[ERROR] Archivo no encontrado: {data_file}")
        return {'error': 'Archivo no encontrado'}

    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Filas: {len(df):,}")
    print(f"  Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

    all_results = {}

    # 1. StatsForecast
    if STATSFORECAST_AVAILABLE:
        sf_results = run_statsforecast_models(
            df=df,
            target_col='SPY_CLOSE',
            date_col='date',
            train_ratio=train_ratio,
            horizon=1,
            season_length=5  # Semana de trading
        )
        all_results['statsforecast'] = sf_results

    # 2. Prophet basico
    if PROPHET_AVAILABLE:
        prophet_results = run_prophet_model(
            df=df,
            target_col='SPY_CLOSE',
            date_col='date',
            train_ratio=train_ratio
        )
        all_results['prophet_basic'] = prophet_results

        # 3. Prophet con regresores
        prophet_reg_results = run_prophet_with_financial_regressors(
            df=df,
            target_col='SPY_CLOSE',
            date_col='date',
            train_ratio=train_ratio
        )
        all_results['prophet_regressors'] = prophet_reg_results

    # 4. Ensemble
    sf_res = all_results.get('statsforecast', {'error': 'N/A'})
    prophet_res = all_results.get('prophet_regressors', all_results.get('prophet_basic', {'error': 'N/A'}))

    ensemble_results = create_forecast_ensemble(sf_res, prophet_res)
    all_results['ensemble'] = ensemble_results

    # Guardar resultados
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)

    # Preparar resumen para JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'train_ratio': train_ratio,
        'data_rows': len(df),
        'models_run': list(all_results.keys()),
        'best_model': ensemble_results.get('best_model', 'N/A'),
        'best_metrics': {}
    }

    if 'best_results' in ensemble_results:
        summary['best_metrics'] = {
            'forecast': ensemble_results['best_results'].get('forecast_metrics', {}),
            'trading': ensemble_results['best_results'].get('trading_metrics', {})
        }

    # Guardar JSON
    summary_file = os.path.join(RESULTS_DIR, "forecasting_results.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  + Resumen guardado: {summary_file}")

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL FORECASTING")
    print("=" * 80)

    if 'best_model' in ensemble_results:
        print(f"\n  MEJOR MODELO: {ensemble_results['best_model']}")

        if 'best_results' in ensemble_results:
            fm = ensemble_results['best_results'].get('forecast_metrics', {})
            tm = ensemble_results['best_results'].get('trading_metrics', {})

            print(f"\n  Metricas de Forecast:")
            print(f"    RMSE: {fm.get('rmse', 'N/A'):.6f}" if isinstance(fm.get('rmse'), (int, float)) else f"    RMSE: N/A")
            print(f"    R2: {fm.get('r2', 'N/A'):.4f}" if isinstance(fm.get('r2'), (int, float)) else f"    R2: N/A")
            print(f"    Dir. Accuracy: {fm.get('directional_accuracy', 'N/A'):.2%}" if isinstance(fm.get('directional_accuracy'), (int, float)) else f"    Dir. Accuracy: N/A")

            if tm:
                print(f"\n  Metricas de Trading:")
                print(f"    Return: {tm.get('total_return', 0):.2%}")
                print(f"    Sharpe: {tm.get('sharpe_ratio', 0):.3f}")
                print(f"    Max Drawdown: {tm.get('max_drawdown', 0):.2%}")

    print("\n" + "=" * 80)
    print("[OK] FORECASTING PIPELINE COMPLETADO")
    print("=" * 80)

    return all_results


# =============================================================================
# COMPARACION CON ML PIPELINE
# =============================================================================
def compare_with_ml_results(forecasting_results: Dict,
                             ml_results_file: Optional[str] = None) -> pd.DataFrame:
    """
    Compara resultados de forecasting con modelos ML.
    """
    print("\n" + "=" * 80)
    print("COMPARACION: FORECASTING vs ML")
    print("=" * 80)

    if ml_results_file is None:
        ml_results_file = os.path.join(RESULTS_DIR, "model_comparison_bloomberg.csv")

    comparison_data = []

    # Resultados de forecasting
    if 'ensemble' in forecasting_results and 'all_models' in forecasting_results['ensemble']:
        for model_name, results in forecasting_results['ensemble']['all_models'].items():
            if isinstance(results, dict):
                fm = results.get('forecast_metrics', {})
                tm = results.get('trading_metrics', {})
                comparison_data.append({
                    'model': model_name,
                    'type': 'Forecasting',
                    'rmse': fm.get('rmse', np.nan),
                    'r2': fm.get('r2', np.nan),
                    'directional_accuracy': fm.get('directional_accuracy', np.nan),
                    'strategy_return': tm.get('total_return', np.nan),
                    'sharpe_ratio': tm.get('sharpe_ratio', np.nan),
                    'max_drawdown': tm.get('max_drawdown', np.nan)
                })

    # Resultados de ML (si existen)
    if os.path.exists(ml_results_file):
        ml_df = pd.read_csv(ml_results_file)
        for _, row in ml_df.iterrows():
            comparison_data.append({
                'model': row.get('model', row.name),
                'type': 'ML',
                'rmse': row.get('test_rmse', np.nan),
                'r2': row.get('test_r2', np.nan),
                'directional_accuracy': row.get('test_dir_acc', np.nan),
                'strategy_return': row.get('strategy_return', np.nan),
                'sharpe_ratio': row.get('strategy_sharpe', np.nan),
                'max_drawdown': np.nan
            })

    if not comparison_data:
        print("  No hay datos para comparar")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_data)

    # Ordenar por strategy return
    comparison_df = comparison_df.sort_values('strategy_return', ascending=False)

    print("\n  Comparacion completa (ordenado por Strategy Return):")
    print("-" * 100)
    print(comparison_df.to_string(index=False))

    # Guardar
    output_file = os.path.join(RESULTS_DIR, "full_model_comparison.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n  + Comparacion guardada: {output_file}")

    return comparison_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Ejecutar pipeline
    results = run_forecasting_pipeline()

    # Comparar con ML
    if 'error' not in results:
        comparison = compare_with_ml_results(results)
