# -*- coding: utf-8 -*-
"""
================================================================================
PIPELINE UNIFICADO - COMPARACION DE 21 MODELOS
================================================================================
Compara todos los modelos de ML, Series Temporales y Deep Learning para
prediccion de retornos financieros.

Modelos incluidos:
1. ML Tradicional: Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting,
                   XGBoost, LightGBM, CatBoost
2. StatsForecast: AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
3. Prophet: Basico, Con Regresores
4. GARCH: Para volatilidad
5. MLForecast: ML con lag features
6. Darts Deep Learning: DLinear, TFT, TCN, N-BEATS, N-HiTS

================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Configuracion de paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# IMPORTS
# =============================================================================
print("=" * 80)
print("CARGANDO LIBRERIAS...")
print("=" * 80)

# ML Tradicional
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

print("  [OK] ML Tradicional (sklearn, xgboost, lightgbm, catboost)")

# StatsForecast
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

print("  [OK] StatsForecast")

# Prophet
from prophet import Prophet

print("  [OK] Prophet")

# GARCH
from arch import arch_model

print("  [OK] GARCH (arch)")

# MLForecast
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

print("  [OK] MLForecast")

# Darts
from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    NHiTSModel,
    DLinearModel,
    TFTModel,
    TCNModel
)
from darts.dataprocessing.transformers import Scaler

print("  [OK] Darts (N-BEATS, N-HiTS, DLinear, TFT, TCN)")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  [OK] PyTorch (device: {DEVICE})")


# =============================================================================
# FUNCIONES DE METRICAS
# =============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      prices: np.ndarray = None) -> Dict:
    """Calcula metricas de forecast y trading."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 10:
        return {
            'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'dir_acc': np.nan, 'strategy_return': np.nan,
            'sharpe': np.nan, 'max_dd': np.nan
        }

    # RMSE, MAE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # Directional Accuracy
    if len(y_true) > 1:
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
        dir_acc = np.mean(true_dir == pred_dir)
    else:
        dir_acc = np.nan

    # Trading metrics (si tenemos precios o retornos)
    if prices is not None and len(prices) == len(y_true):
        returns = np.diff(prices) / prices[:-1]
        positions = np.sign(y_pred[:-1])
        positions = np.where(positions >= 0, 1, 0)
        strategy_returns = positions * returns

        strategy_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0

        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
    else:
        # Usar predicciones como senales de trading
        positions = np.sign(y_pred[:-1])
        positions = np.where(positions >= 0, 1, 0)
        strategy_returns = positions * y_true[1:]

        strategy_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0

        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'dir_acc': dir_acc,
        'strategy_return': strategy_return,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


# =============================================================================
# CLASE UNIFICADA DE MODELOS
# =============================================================================
class UnifiedModelPipeline:
    """Pipeline unificado para comparar todos los modelos."""

    def __init__(self, data_file: str = None, train_ratio: float = 0.8):
        self.train_ratio = train_ratio
        self.results = {}

        # Cargar datos
        if data_file is None:
            data_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")

        print(f"\nCargando datos: {data_file}")
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Identificar columnas
        self.target_col = 'market_forward_excess_returns'
        self.price_col = 'SPY_CLOSE'
        self.date_col = 'date'

        # Feature columns (excluir metadata)
        exclude_cols = ['date', 'date_id', 'SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH',
                       'SPY_LOW', 'SPY_VOLUME', 'forward_returns', 'risk_free_rate',
                       'market_forward_excess_returns']
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]

        # Split temporal
        n = len(self.df)
        self.train_size = int(n * train_ratio)

        self.train_df = self.df.iloc[:self.train_size].copy()
        self.test_df = self.df.iloc[self.train_size:].copy()

        print(f"  Total: {n:,} filas")
        print(f"  Train: {len(self.train_df):,} ({train_ratio*100:.0f}%)")
        print(f"  Test: {len(self.test_df):,} ({(1-train_ratio)*100:.0f}%)")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Periodo: {self.df['date'].min().date()} a {self.df['date'].max().date()}")

    # =========================================================================
    # ML TRADICIONAL
    # =========================================================================
    def run_ml_models(self):
        """Ejecuta modelos de ML tradicional."""
        print("\n" + "=" * 80)
        print("ML TRADICIONAL")
        print("=" * 80)

        # Preparar datos
        X_train = self.train_df[self.feature_cols].values
        y_train = self.train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].values
        y_test = self.test_df[self.target_col].values

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)

        # Modelos
        ml_models = {
            'Ridge': Ridge(alpha=10),
            'Lasso': Lasso(alpha=0.001),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                                   min_samples_split=5, n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                          learning_rate=0.01, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01,
                                        random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.01,
                                          random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.01,
                                             random_state=42, verbose=0)
        }

        for name, model in ml_models.items():
            print(f"\n  Entrenando: {name}...")
            start = time.time()

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = calculate_metrics(y_test, y_pred)
                metrics['time'] = time.time() - start
                metrics['category'] = 'ML'

                self.results[name] = metrics

                print(f"    RMSE: {metrics['rmse']:.6f}")
                print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
                print(f"    Return: {metrics['strategy_return']:.2%}")
                print(f"    Sharpe: {metrics['sharpe']:.3f}")
                print(f"    Time: {metrics['time']:.1f}s")

            except Exception as e:
                print(f"    [ERROR] {e}")
                self.results[name] = {'error': str(e), 'category': 'ML'}

    # =========================================================================
    # STATSFORECAST
    # =========================================================================
    def run_statsforecast_models(self):
        """Ejecuta modelos de StatsForecast."""
        print("\n" + "=" * 80)
        print("STATSFORECAST")
        print("=" * 80)

        # Preparar datos en formato StatsForecast
        sf_df = pd.DataFrame({
            'unique_id': 'SPY',
            'ds': self.df[self.date_col],
            'y': self.df[self.target_col]
        }).dropna()

        train_size = int(len(sf_df) * self.train_ratio)
        sf_train = sf_df.iloc[:train_size]
        sf_test = sf_df.iloc[train_size:]

        models = [
            AutoARIMA(season_length=5),
            AutoETS(season_length=5),
            AutoTheta(season_length=5),
            SeasonalNaive(season_length=5)
        ]

        sf = StatsForecast(models=models, freq='B', n_jobs=1)

        print(f"\n  Entrenando {len(models)} modelos...")
        start = time.time()

        try:
            sf.fit(sf_train)
            predictions = sf.predict(h=len(sf_test))

            y_test = sf_test['y'].values

            for model_name in ['AutoARIMA', 'AutoETS', 'AutoTheta', 'SeasonalNaive']:
                if model_name in predictions.columns:
                    y_pred = predictions[model_name].values

                    metrics = calculate_metrics(y_test, y_pred)
                    metrics['time'] = (time.time() - start) / len(models)
                    metrics['category'] = 'StatsForecast'

                    self.results[f'SF_{model_name}'] = metrics

                    print(f"\n  {model_name}:")
                    print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
                    print(f"    Return: {metrics['strategy_return']:.2%}")
                    print(f"    Sharpe: {metrics['sharpe']:.3f}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # =========================================================================
    # PROPHET
    # =========================================================================
    def run_prophet_models(self):
        """Ejecuta modelos de Prophet."""
        print("\n" + "=" * 80)
        print("PROPHET")
        print("=" * 80)

        # Prophet basico
        print("\n  Entrenando: Prophet (basico)...")
        start = time.time()

        try:
            prophet_train = pd.DataFrame({
                'ds': self.train_df[self.date_col],
                'y': self.train_df[self.target_col]
            }).dropna()

            prophet_test = pd.DataFrame({
                'ds': self.test_df[self.date_col],
                'y': self.test_df[self.target_col]
            }).dropna()

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_train)

            forecast = model.predict(prophet_test[['ds']])

            y_test = prophet_test['y'].values
            y_pred = forecast['yhat'].values

            metrics = calculate_metrics(y_test, y_pred)
            metrics['time'] = time.time() - start
            metrics['category'] = 'Prophet'

            self.results['Prophet'] = metrics

            print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
            print(f"    Return: {metrics['strategy_return']:.2%}")
            print(f"    Sharpe: {metrics['sharpe']:.3f}")
            print(f"    Time: {metrics['time']:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {e}")

        # Prophet con regresores
        print("\n  Entrenando: Prophet (con regresores)...")
        start = time.time()

        try:
            regressors = ['V1', 'V2', 'I4', 'M1']
            available_regs = [r for r in regressors if r in self.df.columns]

            if available_regs:
                prophet_train = pd.DataFrame({
                    'ds': self.train_df[self.date_col],
                    'y': self.train_df[self.target_col]
                })
                for reg in available_regs:
                    prophet_train[reg] = self.train_df[reg].values
                prophet_train = prophet_train.dropna()

                prophet_test = pd.DataFrame({
                    'ds': self.test_df[self.date_col],
                    'y': self.test_df[self.target_col]
                })
                for reg in available_regs:
                    prophet_test[reg] = self.test_df[reg].values
                prophet_test = prophet_test.dropna()

                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                for reg in available_regs:
                    model.add_regressor(reg)

                model.fit(prophet_train)

                forecast = model.predict(prophet_test[['ds'] + available_regs])

                y_test = prophet_test['y'].values
                y_pred = forecast['yhat'].values

                metrics = calculate_metrics(y_test, y_pred)
                metrics['time'] = time.time() - start
                metrics['category'] = 'Prophet'

                self.results['Prophet_Regressors'] = metrics

                print(f"    Regresores: {available_regs}")
                print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
                print(f"    Return: {metrics['strategy_return']:.2%}")
                print(f"    Sharpe: {metrics['sharpe']:.3f}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # =========================================================================
    # GARCH
    # =========================================================================
    def run_garch_model(self):
        """Ejecuta modelo GARCH para volatilidad."""
        print("\n" + "=" * 80)
        print("GARCH")
        print("=" * 80)

        print("\n  Entrenando: GARCH(1,1)...")
        start = time.time()

        try:
            # Usar retornos para GARCH
            returns = self.df[self.target_col].dropna() * 100  # Escalar para estabilidad

            train_returns = returns.iloc[:self.train_size]
            test_returns = returns.iloc[self.train_size:]

            # Ajustar GARCH
            model = arch_model(train_returns, vol='Garch', p=1, q=1, mean='AR', lags=1)
            result = model.fit(disp='off')

            # Forecast
            forecasts = result.forecast(horizon=len(test_returns), start=train_returns.index[-1])

            # Prediccion de retornos (mean) y volatilidad
            y_test = test_returns.values
            y_pred_mean = forecasts.mean.values[-len(test_returns):].flatten()
            y_pred_vol = np.sqrt(forecasts.variance.values[-len(test_returns):].flatten())

            # Metricas usando prediccion de media
            metrics = calculate_metrics(y_test / 100, y_pred_mean / 100)
            metrics['time'] = time.time() - start
            metrics['category'] = 'GARCH'
            metrics['volatility_forecast'] = True

            self.results['GARCH'] = metrics

            print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
            print(f"    Return: {metrics['strategy_return']:.2%}")
            print(f"    Sharpe: {metrics['sharpe']:.3f}")
            print(f"    Time: {metrics['time']:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # =========================================================================
    # MLFORECAST
    # =========================================================================
    def run_mlforecast_model(self):
        """Ejecuta MLForecast con lag features automaticos."""
        print("\n" + "=" * 80)
        print("MLFORECAST")
        print("=" * 80)

        print("\n  Entrenando: MLForecast (LightGBM + lags)...")
        start = time.time()

        try:
            # Preparar datos
            mlf_df = pd.DataFrame({
                'unique_id': 'SPY',
                'ds': self.df[self.date_col],
                'y': self.df[self.target_col]
            }).dropna()

            train_size = int(len(mlf_df) * self.train_ratio)
            mlf_train = mlf_df.iloc[:train_size]
            mlf_test = mlf_df.iloc[train_size:]

            # Configurar MLForecast
            mlf = MLForecast(
                models=[lgb.LGBMRegressor(n_estimators=100, num_leaves=31,
                                          learning_rate=0.01, verbose=-1)],
                freq='B',
                lags=[1, 2, 3, 5, 10, 21],
                lag_transforms={
                    1: [ExpandingMean()],
                    5: [RollingMean(window_size=5)],
                    21: [RollingMean(window_size=21)]
                }
            )

            mlf.fit(mlf_train)
            predictions = mlf.predict(h=len(mlf_test))

            y_test = mlf_test['y'].values
            y_pred = predictions['LGBMRegressor'].values

            metrics = calculate_metrics(y_test, y_pred)
            metrics['time'] = time.time() - start
            metrics['category'] = 'MLForecast'

            self.results['MLForecast'] = metrics

            print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
            print(f"    Return: {metrics['strategy_return']:.2%}")
            print(f"    Sharpe: {metrics['sharpe']:.3f}")
            print(f"    Time: {metrics['time']:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # =========================================================================
    # DARTS DEEP LEARNING
    # =========================================================================
    def run_darts_models(self):
        """Ejecuta modelos de Deep Learning de Darts."""
        print("\n" + "=" * 80)
        print("DARTS DEEP LEARNING")
        print("=" * 80)

        # Preparar datos para Darts
        darts_df = self.df[[self.date_col, self.target_col]].dropna()
        darts_df = darts_df.set_index(self.date_col)

        # Usar fill_missing_dates para manejar gaps de fines de semana/feriados
        try:
            series = TimeSeries.from_dataframe(
                darts_df,
                value_cols=self.target_col,
                fill_missing_dates=True,
                freq='B'  # Business day frequency
            )
        except Exception as e:
            print(f"  [WARNING] Error creando TimeSeries con freq='B': {e}")
            # Intentar sin frecuencia especifica
            series = TimeSeries.from_dataframe(
                darts_df,
                value_cols=self.target_col,
                fill_missing_dates=True,
                freq=None
            )

        # Escalar
        scaler = Scaler()
        series_scaled = scaler.fit_transform(series)

        train_size = int(len(series_scaled) * self.train_ratio)
        train_series = series_scaled[:train_size]
        test_series = series_scaled[train_size:]

        # Modelos de Darts (configuracion ligera para CPU)
        darts_models = {
            'DLinear': DLinearModel(
                input_chunk_length=21,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=32,
                random_state=42,
                pl_trainer_kwargs={"accelerator": DEVICE, "enable_progress_bar": False}
            ),
            'TCN': TCNModel(
                input_chunk_length=21,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=32,
                num_filters=32,
                kernel_size=3,
                random_state=42,
                pl_trainer_kwargs={"accelerator": DEVICE, "enable_progress_bar": False}
            ),
            'TFT': TFTModel(
                input_chunk_length=21,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=32,
                hidden_size=32,
                lstm_layers=1,
                num_attention_heads=2,
                add_relative_index=True,  # Fix: agregar indice relativo como covariate
                random_state=42,
                pl_trainer_kwargs={"accelerator": DEVICE, "enable_progress_bar": False}
            ),
            'N-BEATS': NBEATSModel(
                input_chunk_length=21,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=32,
                num_stacks=2,
                num_blocks=1,
                num_layers=2,
                layer_widths=64,
                random_state=42,
                pl_trainer_kwargs={"accelerator": DEVICE, "enable_progress_bar": False}
            ),
            'N-HiTS': NHiTSModel(
                input_chunk_length=21,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=32,
                num_stacks=2,
                num_blocks=1,
                num_layers=2,
                layer_widths=64,
                random_state=42,
                pl_trainer_kwargs={"accelerator": DEVICE, "enable_progress_bar": False}
            )
        }

        for name, model in darts_models.items():
            print(f"\n  Entrenando: {name}...")
            start = time.time()

            try:
                model.fit(train_series)

                # Prediccion usando predict() simple
                n_predict = len(test_series)
                predictions = model.predict(n=n_predict)

                # Invertir escala
                predictions = scaler.inverse_transform(predictions)
                test_actual = scaler.inverse_transform(test_series)

                # Extraer valores
                y_pred = predictions.values().flatten()
                y_test = test_actual.values().flatten()

                # Ajustar longitudes
                min_len = min(len(y_test), len(y_pred))
                y_test = y_test[:min_len]
                y_pred = y_pred[:min_len]

                # Verificar que no hay NaN
                mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
                if mask.sum() < 10:
                    raise ValueError(f"Muy pocos datos validos: {mask.sum()}")

                y_test = y_test[mask]
                y_pred = y_pred[mask]

                metrics = calculate_metrics(y_test, y_pred)
                metrics['time'] = time.time() - start
                metrics['category'] = 'Darts'

                self.results[name] = metrics

                print(f"    Dir Acc: {metrics['dir_acc']:.2%}")
                print(f"    Return: {metrics['strategy_return']:.2%}")
                print(f"    Sharpe: {metrics['sharpe']:.3f}")
                print(f"    Time: {metrics['time']:.1f}s")

            except Exception as e:
                print(f"    [ERROR] {e}")
                self.results[name] = {'error': str(e), 'category': 'Darts'}

    # =========================================================================
    # EJECUTAR TODO
    # =========================================================================
    def run_all(self):
        """Ejecuta todos los modelos."""
        print("\n" + "=" * 80)
        print("PIPELINE UNIFICADO - 21 MODELOS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now()}")
        print(f"Device: {DEVICE}")

        total_start = time.time()

        # 1. ML Tradicional (8 modelos)
        self.run_ml_models()

        # 2. StatsForecast (4 modelos)
        self.run_statsforecast_models()

        # 3. Prophet (2 modelos)
        self.run_prophet_models()

        # 4. GARCH (1 modelo)
        self.run_garch_model()

        # 5. MLForecast (1 modelo)
        self.run_mlforecast_model()

        # 6. Darts Deep Learning (4 modelos)
        self.run_darts_models()

        total_time = time.time() - total_start

        return self.compile_results(total_time)

    def compile_results(self, total_time: float) -> pd.DataFrame:
        """Compila y guarda resultados."""
        print("\n" + "=" * 80)
        print("RESULTADOS FINALES")
        print("=" * 80)

        # Crear DataFrame de resultados
        results_list = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                results_list.append({
                    'model': model_name,
                    'category': metrics.get('category', 'Unknown'),
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'r2': metrics.get('r2', np.nan),
                    'dir_acc': metrics.get('dir_acc', np.nan),
                    'strategy_return': metrics.get('strategy_return', np.nan),
                    'sharpe': metrics.get('sharpe', np.nan),
                    'max_dd': metrics.get('max_dd', np.nan),
                    'time': metrics.get('time', np.nan)
                })

        results_df = pd.DataFrame(results_list)

        # Ordenar por strategy_return
        results_df = results_df.sort_values('strategy_return', ascending=False)

        # Mostrar ranking
        print("\n  RANKING POR STRATEGY RETURN:")
        print("-" * 100)
        print(f"  {'#':<3} {'Modelo':<25} {'Categoria':<15} {'Return':>12} {'Sharpe':>10} {'Dir Acc':>10} {'RMSE':>12}")
        print("-" * 100)

        for i, row in results_df.iterrows():
            idx = results_df.index.get_loc(i) + 1
            print(f"  {idx:<3} {row['model']:<25} {row['category']:<15} "
                  f"{row['strategy_return']:>11.2%} {row['sharpe']:>10.3f} "
                  f"{row['dir_acc']:>10.2%} {row['rmse']:>12.6f}")

        print("-" * 100)

        # Mejor modelo
        best = results_df.iloc[0]
        print(f"\n  [MEJOR MODELO]: {best['model']}")
        print(f"    Categoria: {best['category']}")
        print(f"    Strategy Return: {best['strategy_return']:.2%}")
        print(f"    Sharpe Ratio: {best['sharpe']:.3f}")
        print(f"    Directional Accuracy: {best['dir_acc']:.2%}")

        print(f"\n  Tiempo total: {total_time/60:.1f} minutos")
        print(f"  Modelos exitosos: {len(results_df)}/{len(self.results)}")

        # Guardar resultados
        output_file = os.path.join(RESULTS_DIR, "unified_model_comparison.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n  Resultados guardados: {output_file}")

        # Guardar JSON con metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.results),
            'successful_models': len(results_df),
            'total_time_minutes': total_time / 60,
            'best_model': best['model'],
            'best_return': float(best['strategy_return']),
            'best_sharpe': float(best['sharpe']),
            'train_ratio': self.train_ratio,
            'train_samples': len(self.train_df),
            'test_samples': len(self.test_df),
            'device': DEVICE
        }

        metadata_file = os.path.join(RESULTS_DIR, "unified_model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata guardado: {metadata_file}")

        print("\n" + "=" * 80)
        print("[OK] PIPELINE UNIFICADO COMPLETADO")
        print("=" * 80)

        return results_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    pipeline = UnifiedModelPipeline()
    results = pipeline.run_all()
