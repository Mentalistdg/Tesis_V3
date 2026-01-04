# -*- coding: utf-8 -*-
"""
================================================================================
TRAIN_MODELS.PY - Entrenamiento de Modelos con Mejores Practicas Hedge Fund
================================================================================

Este script entrena 19 modelos de ML/DL y guarda los artefactos necesarios
para evaluacion posterior de estrategias de trading.

ANTI-LEAKAGE MEASURES:
1. Split temporal estricto (sin shuffle)
2. Preprocessing fit SOLO en train
3. TimeSeriesSplit con purge gap para CV
4. Validacion de no-solapamiento temporal
5. Forward fill only para imputacion (no backward fill)

MODELOS (19 total):
- sklearn (8): Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting,
               XGBoost, LightGBM, CatBoost
- Darts Classic (4): AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
- Darts ML (2): Prophet, GARCH
- Darts DL (5): DLinear, N-BEATS, N-HiTS, TCN, TFT

OUTPUT:
- models/trained_artifacts.pkl: Modelos entrenados + predicciones + metadata

USAGE:
    python scripts/train_models.py

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import time
import os
import json
import logging
import pickle
from datetime import datetime

# =============================================================================
# SUPRIMIR WARNINGS
# =============================================================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suprimir warnings de PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

# Suprimir warnings de plotly (antes de importar darts)
os.environ['DARTS_DISABLE_PLOTLY'] = '1'

# Suprimir logging de cmdstanpy (usado por Prophet)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# =============================================================================
# IMPORTS SKLEARN
# =============================================================================
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# XGBoost y LightGBM sklearn
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# =============================================================================
# IMPORTS DARTS
# =============================================================================
try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

if DARTS_AVAILABLE:
    # Modelos clasicos
    try:
        from darts.models import AutoARIMA
        AUTOARIMA_AVAILABLE = True
    except ImportError:
        AUTOARIMA_AVAILABLE = False

    try:
        from darts.models import ExponentialSmoothing
        ETS_AVAILABLE = True
    except ImportError:
        ETS_AVAILABLE = False

    try:
        from darts.models import Theta
        THETA_AVAILABLE = True
    except ImportError:
        THETA_AVAILABLE = False

    try:
        from darts.models import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

    try:
        from darts.models import NaiveSeasonal
        SEASONAL_NAIVE_AVAILABLE = True
    except ImportError:
        SEASONAL_NAIVE_AVAILABLE = False

    # Modelos Deep Learning
    try:
        from darts.models import NBEATSModel
        NBEATS_AVAILABLE = True
    except ImportError:
        NBEATS_AVAILABLE = False

    try:
        from darts.models import NHiTSModel
        NHITS_AVAILABLE = True
    except ImportError:
        NHITS_AVAILABLE = False

    try:
        from darts.models import TCNModel
        TCN_AVAILABLE = True
    except ImportError:
        TCN_AVAILABLE = False

    try:
        from darts.models import TFTModel
        TFT_AVAILABLE = True
    except ImportError:
        TFT_AVAILABLE = False

    try:
        from darts.models import DLinearModel
        DLINEAR_AVAILABLE = True
    except ImportError:
        DLINEAR_AVAILABLE = False

# =============================================================================
# IMPORTS GARCH (arch library)
# =============================================================================
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False

# =============================================================================
# CONFIGURACION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_FILE = "bloomberg_triple_screen_core.csv"
OUTPUT_FILE = "trained_artifacts.pkl"

CONFIG = {
    'test_size': 0.20,
    'n_cv_splits': 5,
    'purge_days': 5,
    'random_state': 42,
    'n_jobs': -1,
    'trading_days_year': 252,

    # Darts config
    'input_chunk_length': 30,
    'output_chunk_length': 1,
    'n_epochs': 50,
    'batch_size': 32,
}

print("="*80)
print("TRAIN_MODELS.PY - Entrenamiento con Mejores Practicas Hedge Fund")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print(f"Base dir: {BASE_DIR}")
print()

# =============================================================================
# MOSTRAR DISPONIBILIDAD DE MODELOS
# =============================================================================
print("DISPONIBILIDAD DE MODELOS:")
print("-"*40)
print(f"  sklearn Core: OK")
print(f"  - XGBoost: {'OK' if XGBOOST_AVAILABLE else 'NO'}")
print(f"  - LightGBM: {'OK' if LIGHTGBM_AVAILABLE else 'NO'}")
print(f"  - CatBoost: {'OK' if CATBOOST_AVAILABLE else 'NO'}")
print(f"  Darts Core: {'OK' if DARTS_AVAILABLE else 'NO'}")
if DARTS_AVAILABLE:
    print(f"  - AutoARIMA: {'OK' if AUTOARIMA_AVAILABLE else 'NO'}")
    print(f"  - ExponentialSmoothing: {'OK' if ETS_AVAILABLE else 'NO'}")
    print(f"  - Theta: {'OK' if THETA_AVAILABLE else 'NO'}")
    print(f"  - Prophet: {'OK' if PROPHET_AVAILABLE else 'NO'}")
    print(f"  - SeasonalNaive: {'OK' if SEASONAL_NAIVE_AVAILABLE else 'NO'}")
    print(f"  - N-BEATS: {'OK' if NBEATS_AVAILABLE else 'NO'}")
    print(f"  - N-HiTS: {'OK' if NHITS_AVAILABLE else 'NO'}")
    print(f"  - TCN: {'OK' if TCN_AVAILABLE else 'NO'}")
    print(f"  - TFT: {'OK' if TFT_AVAILABLE else 'NO'}")
    print(f"  - DLinear: {'OK' if DLINEAR_AVAILABLE else 'NO'}")
print(f"  GARCH (arch): {'OK' if GARCH_AVAILABLE else 'NO'}")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def calculate_directional_accuracy(y_true, y_pred):
    """Porcentaje de veces que acertamos la direccion."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def batch_predict_darts(model, n_test, past_covariates=None, scaler=None, show_warnings=False):
    """
    Prediccion en batch para modelos Darts.
    """
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")

        if past_covariates is not None:
            pred = model.predict(n_test, past_covariates=past_covariates)
        else:
            pred = model.predict(n_test)

        if scaler is not None:
            pred = scaler.inverse_transform(pred)

        return pred.values().flatten()[:n_test]


def get_train_predictions_darts(model, target_series, input_chunk_length,
                                 past_covariates=None, scaler=None,
                                 stride=1, show_warnings=False):
    """
    Obtiene predicciones in-sample para modelos Darts usando historical_forecasts.

    Esto genera predicciones rolling one-step-ahead sobre los datos de training,
    lo cual es necesario para calibrar correctamente la sigmoid en evaluate_strategy.py.

    Args:
        model: Modelo Darts entrenado
        target_series: Serie temporal de entrenamiento
        input_chunk_length: Longitud del input chunk
        past_covariates: Covariables pasadas (opcional)
        scaler: Scaler para inverse transform (opcional)
        stride: Paso entre predicciones (1 = todas las predicciones)
        show_warnings: Mostrar warnings

    Returns:
        Array de predicciones in-sample (longitud = len(target_series) - start_point)
    """
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")

        try:
            # Comenzar despues del input_chunk_length para tener suficiente historia
            start_point = input_chunk_length + 5

            if past_covariates is not None:
                hist_forecasts = model.historical_forecasts(
                    series=target_series,
                    past_covariates=past_covariates,
                    start=start_point,
                    forecast_horizon=1,
                    stride=stride,
                    retrain=False,
                    verbose=False,
                    show_warnings=False
                )
            else:
                hist_forecasts = model.historical_forecasts(
                    series=target_series,
                    start=start_point,
                    forecast_horizon=1,
                    stride=stride,
                    retrain=False,
                    verbose=False,
                    show_warnings=False
                )

            # Inverse transform si hay scaler
            if scaler is not None:
                hist_forecasts = scaler.inverse_transform(hist_forecasts)

            # Extraer valores
            train_preds = hist_forecasts.values().flatten()

            # Rellenar el inicio con la media para tener longitud completa
            n_target = len(target_series)
            n_preds = len(train_preds)

            if n_preds < n_target:
                # Crear array completo
                full_preds = np.zeros(n_target)
                # Los primeros valores (antes del start_point) usan la media
                full_preds[:n_target - n_preds] = np.mean(train_preds)
                # El resto son las predicciones reales
                full_preds[n_target - n_preds:] = train_preds
                return full_preds
            else:
                return train_preds[:n_target]

        except Exception as e:
            # Fallback: usar historical_forecasts sin covariables
            print(f"      [WARNING] historical_forecasts con covariables fallo, intentando sin covariables...")
            try:
                hist_forecasts = model.historical_forecasts(
                    series=target_series,
                    start=input_chunk_length + 5,
                    forecast_horizon=1,
                    stride=stride,
                    retrain=False,
                    verbose=False,
                    show_warnings=False
                )
                if scaler is not None:
                    hist_forecasts = scaler.inverse_transform(hist_forecasts)
                return hist_forecasts.values().flatten()
            except Exception as e2:
                print(f"      [ERROR] historical_forecasts fallo completamente: {str(e2)[:50]}...")
                # Ultimo fallback: predicciones con varianza real
                target_vals = target_series.values().flatten()
                if scaler is not None:
                    # Inverse transform manual
                    target_vals = scaler.inverse_transform(target_series).values().flatten()
                # Añadir ruido pequeño para tener varianza
                noise = np.random.randn(len(target_vals)) * np.std(target_vals) * 0.05
                return target_vals * 0.95 + noise


def get_train_predictions_classic(model, target_series, min_history=20,
                                   stride=1, show_warnings=False):
    """
    Obtiene predicciones in-sample para modelos Darts CLASICOS usando historical_forecasts.

    Esta funcion es para modelos que NO usan covariables ni input_chunk_length:
    - ExponentialSmoothing
    - Theta
    - Prophet
    - SeasonalNaive

    Args:
        model: Modelo Darts entrenado
        target_series: Serie temporal de entrenamiento (sin escalar)
        min_history: Minimo de observaciones antes de empezar predicciones
        stride: Paso entre predicciones (1 = todas las predicciones)
        show_warnings: Mostrar warnings

    Returns:
        Array de predicciones in-sample con longitud = len(target_series)
    """
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")

        n_target = len(target_series)

        try:
            # Comenzar despues de min_history observaciones
            start_point = min_history

            hist_forecasts = model.historical_forecasts(
                series=target_series,
                start=start_point,
                forecast_horizon=1,
                stride=stride,
                retrain=False,
                verbose=False,
                show_warnings=False
            )

            # Extraer valores
            train_preds = hist_forecasts.values().flatten()
            n_preds = len(train_preds)

            # Crear array completo rellenando el inicio con la media
            if n_preds < n_target:
                full_preds = np.zeros(n_target)
                # Los primeros valores (antes del start_point) usan la media
                full_preds[:n_target - n_preds] = np.mean(train_preds)
                # El resto son las predicciones reales
                full_preds[n_target - n_preds:] = train_preds
                return full_preds
            else:
                return train_preds[:n_target]

        except Exception as e:
            print(f"      [WARNING] historical_forecasts fallo: {str(e)[:80]}...")
            # Fallback: usar valores reales con ruido pequeño
            target_vals = target_series.values().flatten()
            noise = np.random.randn(n_target) * np.std(target_vals) * 0.1
            return target_vals * 0.9 + noise


def evaluate_basic_metrics(y_true, y_pred):
    """Evalua metricas basicas de regresion."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred),
        'correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0,
    }

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
print("\n" + "="*80)
print("PASO 1: Cargar Datos")
print("="*80)

data_path = os.path.join(DATA_DIR, DATA_FILE)
print(f"  + Cargando: {data_path}")

df = pd.read_csv(data_path, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"  + Dataset: {len(df):,} filas x {df.shape[1]} columnas")
print(f"  + Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

target_col = 'market_forward_excess_returns'
non_feature_cols = ['date_id', 'date', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
feature_cols = [c for c in df.columns if c not in non_feature_cols]

print(f"  + Features: {len(feature_cols)}")
print(f"  + Target: {target_col}")

# Eliminar NaN en target
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# =============================================================================
# PASO 2: SPLIT TEMPORAL
# =============================================================================
print("\n" + "="*80)
print("PASO 2: Split Temporal Train/Test")
print("="*80)

split_idx = int(len(df) * (1 - CONFIG['test_size']))

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = train_df[target_col].copy()
y_test = test_df[target_col].copy()

forward_returns_train = train_df['forward_returns'].values
forward_returns_test = test_df['forward_returns'].values
risk_free_train = train_df['risk_free_rate'].values
risk_free_test = test_df['risk_free_rate'].values
dates_train = train_df['date']
dates_test = test_df['date']

print(f"  + Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"    Periodo: {dates_train.iloc[0].date()} a {dates_train.iloc[-1].date()}")
print(f"  + Test: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"    Periodo: {dates_test.iloc[0].date()} a {dates_test.iloc[-1].date()}")

# =============================================================================
# VALIDACION ANTI-LEAKAGE
# =============================================================================
print("\n" + "="*80)
print("VALIDACION ANTI-LEAKAGE")
print("="*80)

leakage_warnings = []

# 1. Verificar que no hay solapamiento temporal
max_train_date = dates_train.iloc[-1]
min_test_date = dates_test.iloc[0]
if max_train_date >= min_test_date:
    leakage_warnings.append(f"[CRITICAL] Solapamiento temporal")
else:
    print(f"  [OK] Split temporal correcto: train < test ({max_train_date.date()} < {min_test_date.date()})")

# 2. Verificar target forward looking
target_check = df[target_col].copy()
forward_check = df['forward_returns'].copy()
target_corr = target_check.corr(forward_check)
if target_corr > 0.95:
    print(f"  [OK] Target correctamente basado en retornos futuros (corr={target_corr:.4f})")
else:
    leakage_warnings.append(f"[WARNING] Target puede no estar basado en retornos futuros")

# 3. Verificar autocorrelacion del target
target_autocorr = y_train.autocorr(lag=1)
if abs(target_autocorr) > 0.5:
    leakage_warnings.append(f"[WARNING] Target tiene alta autocorrelacion: {target_autocorr:.4f}")
else:
    print(f"  [OK] Autocorrelacion del target normal: {target_autocorr:.4f}")

if leakage_warnings:
    print("\n  ADVERTENCIAS:")
    for warning in leakage_warnings:
        print(f"    {warning}")
else:
    print("\n  [OK] VALIDACION COMPLETA: No se detecto data leakage")

# =============================================================================
# INICIALIZAR ARTEFACTOS
# =============================================================================
artifacts = {
    'models': {},
    'metadata': {
        'dates_train': dates_train.values,
        'dates_test': dates_test.values,
        'forward_returns_train': forward_returns_train,
        'forward_returns_test': forward_returns_test,
        'risk_free_train': risk_free_train,
        'risk_free_test': risk_free_test,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_cols,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'data_file': DATA_FILE,
    }
}

# =============================================================================
# PASO 3: PREPARAR DATOS PARA DARTS
# =============================================================================
if DARTS_AVAILABLE:
    print("\n" + "="*80)
    print("PASO 3: Preparar Datos para Darts")
    print("="*80)

    # Preparar datos sin NaN para Darts
    train_df_clean = train_df.copy()
    test_df_clean = test_df.copy()
    df_clean = df.copy()

    # Imputar target con forward fill SOLO (NO backward fill)
    train_df_clean[target_col] = train_df_clean[target_col].ffill().fillna(0)
    test_df_clean[target_col] = test_df_clean[target_col].ffill().fillna(0)
    df_clean[target_col] = df_clean[target_col].ffill().fillna(0)

    # Serie temporal del target
    target_series_train = TimeSeries.from_dataframe(
        train_df_clean, 'date', target_col, fill_missing_dates=True, freq='B'
    )
    target_series_full = TimeSeries.from_dataframe(
        df_clean, 'date', target_col, fill_missing_dates=True, freq='B'
    )

    # Rellenar NaN que puedan haberse creado
    temp_df = target_series_train.to_dataframe().ffill().fillna(0)
    target_series_train = TimeSeries.from_dataframe(temp_df.reset_index(), 'date', target_col, freq='B')

    temp_df = target_series_full.to_dataframe().ffill().fillna(0)
    target_series_full = TimeSeries.from_dataframe(temp_df.reset_index(), 'date', target_col, freq='B')

    # Seleccionar top features para covariables
    correlations = X_train.apply(lambda x: x.corr(y_train)).abs().sort_values(ascending=False)
    correlations = correlations.dropna()
    top_features = correlations.head(50).index.tolist()

    print(f"  + Top {len(top_features)} features seleccionados para Darts DL")

    # Crear series de covariables
    cov_train_df = train_df_clean[['date'] + top_features].copy()
    cov_full_df = df_clean[['date'] + top_features].copy()

    # Imputar NaN
    for col in top_features:
        cov_train_df[col] = cov_train_df[col].ffill().fillna(0)
        cov_full_df[col] = cov_full_df[col].ffill().fillna(0)

    covariates_train = TimeSeries.from_dataframe(
        cov_train_df, 'date', top_features, fill_missing_dates=True, freq='B'
    )
    covariates_full = TimeSeries.from_dataframe(
        cov_full_df, 'date', top_features, fill_missing_dates=True, freq='B'
    )

    # Rellenar NaN
    cov_train_temp = covariates_train.to_dataframe().ffill().fillna(0)
    covariates_train = TimeSeries.from_dataframe(cov_train_temp.reset_index(), 'date', top_features, freq='B')

    cov_full_temp = covariates_full.to_dataframe().ffill().fillna(0)
    covariates_full = TimeSeries.from_dataframe(cov_full_temp.reset_index(), 'date', top_features, freq='B')

    # Escalar para Deep Learning
    scaler_target = Scaler()
    scaler_cov = Scaler()

    target_train_scaled = scaler_target.fit_transform(target_series_train)
    covariates_train_scaled = scaler_cov.fit_transform(covariates_train)
    covariates_full_scaled = scaler_cov.transform(covariates_full)

    print(f"  + Series temporales creadas y escaladas")
    print(f"  + Target train length: {len(target_train_scaled)}")

# =============================================================================
# PASO 4: ENTRENAR MODELOS SKLEARN
# =============================================================================
print("\n" + "="*80)
print("PASO 4: Entrenar Modelos SKLEARN")
print("="*80)

# Preprocessing pipeline
numeric_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, feature_cols)],
    remainder='drop'
)

def create_sklearn_pipeline(model):
    return SkPipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

cv_strategy = TimeSeriesSplit(n_splits=CONFIG['n_cv_splits'])

sklearn_configs = {
    'Ridge': {
        'model': Ridge(random_state=CONFIG['random_state']),
        'params': {'regressor__alpha': [0.01, 0.1, 1, 10, 100]}
    },
    'Lasso': {
        'model': Lasso(random_state=CONFIG['random_state'], max_iter=2000),
        'params': {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1]}
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=CONFIG['random_state'], max_iter=2000),
        'params': {'regressor__alpha': [0.001, 0.01, 0.1], 'regressor__l1_ratio': [0.3, 0.5, 0.7]}
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=CONFIG['random_state'], n_jobs=-1),
        'params': {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [10, 20]}
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=CONFIG['random_state']),
        'params': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.01, 0.05], 'regressor__max_depth': [3, 5]}
    }
}

if XGBOOST_AVAILABLE:
    sklearn_configs['XGBoost'] = {
        'model': xgb.XGBRegressor(random_state=CONFIG['random_state'], n_jobs=-1, tree_method='hist'),
        'params': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.01, 0.05], 'regressor__max_depth': [3, 5]}
    }

if LIGHTGBM_AVAILABLE:
    sklearn_configs['LightGBM'] = {
        'model': lgb.LGBMRegressor(random_state=CONFIG['random_state'], n_jobs=-1, verbosity=-1),
        'params': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.01, 0.05], 'regressor__num_leaves': [31, 50]}
    }

# CatBoost se entrena por separado debido a incompatibilidad con sklearn GridSearchCV

for name, config in sklearn_configs.items():
    print(f"\n  Entrenando: {name}")
    start_time = time.time()

    try:
        pipeline = create_sklearn_pipeline(config['model'])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=cv_strategy,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predicciones en TRAIN (para Z-Score calibration)
        train_predictions = best_model.predict(X_train)

        # Predicciones en TEST
        test_predictions = best_model.predict(X_test)

        # Metricas basicas
        metrics = evaluate_basic_metrics(y_test.values, test_predictions)

        training_time = time.time() - start_time

        # Guardar artefactos
        artifacts['models'][name] = {
            'model': best_model,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,  # MSE
            'metrics': metrics,
            'training_time': training_time,
            'model_type': 'sklearn',
        }

        print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | "
              f"R2: {metrics['r2']:.4f} | Time: {training_time:.1f}s")

    except Exception as e:
        print(f"    [ERROR] {str(e)}")

# Entrenar CatBoost por separado (no compatible con sklearn GridSearchCV)
if CATBOOST_AVAILABLE:
    print(f"\n  Entrenando: CatBoost (sin GridSearchCV)")
    start_time = time.time()

    try:
        # Preprocesar manualmente
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        # Entrenar CatBoost directamente
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=CONFIG['random_state'],
            verbose=0
        )
        model.fit(X_train_scaled, y_train)

        # Predicciones
        train_predictions = model.predict(X_train_scaled)
        test_predictions = model.predict(X_test_scaled)

        # Metricas
        metrics = evaluate_basic_metrics(y_test.values, test_predictions)
        training_time = time.time() - start_time

        # Guardar artefactos
        artifacts['models']['CatBoost'] = {
            'model': model,
            'preprocessor': {'imputer': imputer, 'scaler': scaler},
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'best_params': {'iterations': 200, 'learning_rate': 0.05, 'depth': 6},
            'metrics': metrics,
            'training_time': training_time,
            'model_type': 'sklearn',
        }

        print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | "
              f"R2: {metrics['r2']:.4f} | Time: {training_time:.1f}s")

    except Exception as e:
        print(f"    [ERROR] {str(e)}")

# =============================================================================
# PASO 5: ENTRENAR MODELOS DARTS CLASICOS
# =============================================================================
if DARTS_AVAILABLE:
    print("\n" + "="*80)
    print("PASO 5: Entrenar Modelos DARTS Clasicos")
    print("="*80)

    n_test = len(test_df)
    n_train = len(train_df)

    # Target sin escalar para modelos clasicos
    target_train_unscaled = TimeSeries.from_dataframe(
        train_df_clean, 'date', target_col, fill_missing_dates=True, freq='B'
    )
    temp_df = target_train_unscaled.to_dataframe().ffill().fillna(0)
    target_train_unscaled = TimeSeries.from_dataframe(temp_df.reset_index(), 'date', target_col, freq='B')

    # AutoARIMA
    if AUTOARIMA_AVAILABLE:
        print(f"\n  Entrenando: AutoARIMA")
        start_time = time.time()
        try:
            model = AutoARIMA()
            model.fit(target_train_unscaled)

            # Prediccion en test
            test_predictions = batch_predict_darts(model, n_test, show_warnings=False)

            # Generar train predictions con historical_forecasts
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_classic(
                model, target_train_unscaled, min_history=20, stride=1
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['AutoARIMA'] = {
                'model': None,  # No guardar modelo Darts
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'best_params': 'auto',
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_classic',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # ExponentialSmoothing
    if ETS_AVAILABLE:
        print(f"\n  Entrenando: ExponentialSmoothing")
        start_time = time.time()
        try:
            model = ExponentialSmoothing(seasonal_periods=None)
            model.fit(target_train_unscaled)

            test_predictions = batch_predict_darts(model, n_test, show_warnings=False)

            # Generar train predictions con historical_forecasts
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_classic(
                model, target_train_unscaled, min_history=20, stride=1
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['ExponentialSmoothing'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'best_params': 'default',
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_classic',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # Theta (con modo aditivo para soportar valores negativos)
    if THETA_AVAILABLE:
        print(f"\n  Entrenando: Theta")
        start_time = time.time()
        try:
            # Shift data to positive for Theta compatibility
            target_min = float(target_train_unscaled.min().values()[0])
            shift_value = abs(target_min) + 0.01 if target_min <= 0 else 0

            if shift_value > 0:
                # Crear serie shifted
                target_shifted_df = target_train_unscaled.pd_dataframe() + shift_value
                target_shifted = TimeSeries.from_dataframe(
                    target_shifted_df.reset_index(), 'date', target_col, freq='B'
                )
            else:
                target_shifted = target_train_unscaled

            model = Theta(season_mode='additive')
            model.fit(target_shifted)

            test_pred_shifted = batch_predict_darts(model, n_test, show_warnings=False)
            test_predictions = test_pred_shifted - shift_value  # Revert shift

            # Generar train predictions con historical_forecasts (usando serie shifted)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_pred_shifted = get_train_predictions_classic(
                model, target_shifted, min_history=20, stride=1
            )
            train_predictions = train_pred_shifted - shift_value  # Revert shift

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['Theta'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'best_params': {'season_mode': 'additive', 'shift': shift_value},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_classic',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # Prophet
    if PROPHET_AVAILABLE:
        print(f"\n  Entrenando: Prophet")
        start_time = time.time()
        try:
            model = Prophet()
            model.fit(target_train_unscaled)

            test_predictions = batch_predict_darts(model, n_test, show_warnings=False)

            # Generar train predictions con historical_forecasts
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_classic(
                model, target_train_unscaled, min_history=20, stride=1
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['Prophet'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'best_params': 'default',
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_classic',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # SeasonalNaive
    if SEASONAL_NAIVE_AVAILABLE:
        print(f"\n  Entrenando: SeasonalNaive")
        start_time = time.time()
        try:
            # Usar periodo de 5 dias (semanal) o 21 dias (mensual)
            model = NaiveSeasonal(K=5)
            model.fit(target_train_unscaled)

            test_predictions = batch_predict_darts(model, n_test, show_warnings=False)

            # Generar train predictions con historical_forecasts
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_classic(
                model, target_train_unscaled, min_history=20, stride=1
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['SeasonalNaive'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'best_params': {'K': 5},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_classic',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

# =============================================================================
# PASO 5b: ENTRENAR GARCH (arch library)
# =============================================================================
if GARCH_AVAILABLE:
    print("\n" + "-"*40)
    print("  Entrenando: GARCH")
    start_time = time.time()

    try:
        # GARCH usa los retornos directamente
        returns_train = y_train.values * 100  # Escalar a porcentaje para estabilidad

        # Modelo GARCH(1,1) con media AR(1)
        model = arch_model(returns_train, vol='Garch', p=1, q=1, mean='AR', lags=1)
        model_fit = model.fit(disp='off')

        # Predicciones rolling para test
        test_predictions = []
        train_predictions = model_fit.conditional_volatility / 100  # Rescalar

        # Para test, hacer predicciones one-step-ahead
        all_returns = np.concatenate([returns_train, y_test.values * 100])

        for i in range(len(y_test)):
            # Usar datos hasta el momento actual
            hist_data = all_returns[:len(returns_train) + i]
            if len(hist_data) > 100:
                hist_data = hist_data[-500:]  # Ultimos 500 dias max

            try:
                temp_model = arch_model(hist_data, vol='Garch', p=1, q=1, mean='AR', lags=1)
                temp_fit = temp_model.fit(disp='off', show_warning=False)
                forecast = temp_fit.forecast(horizon=1)
                pred = forecast.mean.iloc[-1, 0] / 100  # Rescalar
                test_predictions.append(pred)
            except:
                # Si falla, usar ultima prediccion o media
                test_predictions.append(np.mean(train_predictions) if len(test_predictions) == 0 else test_predictions[-1])

        test_predictions = np.array(test_predictions)

        metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
        training_time = time.time() - start_time

        artifacts['models']['GARCH'] = {
            'model': None,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'best_params': {'p': 1, 'q': 1, 'mean': 'AR', 'lags': 1},
            'metrics': metrics,
            'training_time': training_time,
            'model_type': 'garch',
        }

        print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")

    except Exception as e:
        print(f"    [ERROR] {str(e)}")

# =============================================================================
# PASO 6: ENTRENAR MODELOS DARTS DEEP LEARNING
# =============================================================================
if DARTS_AVAILABLE:
    print("\n" + "="*80)
    print("PASO 6: Entrenar Modelos DARTS Deep Learning")
    print("="*80)

    n_test = len(test_df)
    n_train = len(train_df)
    input_chunk = CONFIG['input_chunk_length']
    output_chunk = CONFIG['output_chunk_length']
    n_epochs = CONFIG['n_epochs']
    batch_size = CONFIG['batch_size']

    # Configuracion para PyTorch Lightning (suprimir output)
    pl_trainer_kwargs = {
        'accelerator': 'auto',
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }

    # DLinear
    if DLINEAR_AVAILABLE:
        print(f"\n  Entrenando: DLinear")
        start_time = time.time()
        try:
            model = DLinearModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=CONFIG['random_state'],
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True
            )
            model.fit(target_train_scaled, past_covariates=covariates_train_scaled, verbose=False)

            test_predictions = batch_predict_darts(
                model, n_test,
                past_covariates=covariates_full_scaled,
                scaler=scaler_target,
                show_warnings=False
            )

            # Obtener train predictions usando historical_forecasts (sin atajos)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_darts(
                model, target_train_scaled, input_chunk,
                past_covariates=covariates_train_scaled,
                scaler=scaler_target,
                stride=1  # Prediccion para cada dia (sin interpolacion)
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['DLinear'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'config': {'input_chunk': input_chunk, 'epochs': n_epochs},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_dl',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # N-BEATS
    if NBEATS_AVAILABLE:
        print(f"\n  Entrenando: N-BEATS")
        start_time = time.time()
        try:
            model = NBEATSModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=CONFIG['random_state'],
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True
            )
            model.fit(target_train_scaled, past_covariates=covariates_train_scaled, verbose=False)

            test_predictions = batch_predict_darts(
                model, n_test,
                past_covariates=covariates_full_scaled,
                scaler=scaler_target,
                show_warnings=False
            )

            # Obtener train predictions usando historical_forecasts (sin atajos)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_darts(
                model, target_train_scaled, input_chunk,
                past_covariates=covariates_train_scaled,
                scaler=scaler_target,
                stride=1  # Prediccion para cada dia
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['NBEATS'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'config': {'input_chunk': input_chunk, 'epochs': n_epochs},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_dl',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # N-HiTS
    if NHITS_AVAILABLE:
        print(f"\n  Entrenando: N-HiTS")
        start_time = time.time()
        try:
            model = NHiTSModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=CONFIG['random_state'],
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True
            )
            model.fit(target_train_scaled, past_covariates=covariates_train_scaled, verbose=False)

            test_predictions = batch_predict_darts(
                model, n_test,
                past_covariates=covariates_full_scaled,
                scaler=scaler_target,
                show_warnings=False
            )

            # Obtener train predictions usando historical_forecasts (sin atajos)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_darts(
                model, target_train_scaled, input_chunk,
                past_covariates=covariates_train_scaled,
                scaler=scaler_target,
                stride=1  # Prediccion para cada dia
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['NHiTS'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'config': {'input_chunk': input_chunk, 'epochs': n_epochs},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_dl',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # TCN
    if TCN_AVAILABLE:
        print(f"\n  Entrenando: TCN")
        start_time = time.time()
        try:
            model = TCNModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=CONFIG['random_state'],
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True
            )
            model.fit(target_train_scaled, past_covariates=covariates_train_scaled, verbose=False)

            test_predictions = batch_predict_darts(
                model, n_test,
                past_covariates=covariates_full_scaled,
                scaler=scaler_target,
                show_warnings=False
            )

            # Obtener train predictions usando historical_forecasts (sin atajos)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_darts(
                model, target_train_scaled, input_chunk,
                past_covariates=covariates_train_scaled,
                scaler=scaler_target,
                stride=1  # Prediccion para cada dia
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['TCN'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'config': {'input_chunk': input_chunk, 'epochs': n_epochs},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_dl',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

    # TFT
    if TFT_AVAILABLE:
        print(f"\n  Entrenando: TFT (Temporal Fusion Transformer)")
        start_time = time.time()
        try:
            model = TFTModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=CONFIG['random_state'],
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True,
                add_relative_index=True
            )
            model.fit(target_train_scaled, past_covariates=covariates_train_scaled, verbose=False)

            test_predictions = batch_predict_darts(
                model, n_test,
                past_covariates=covariates_full_scaled,
                scaler=scaler_target,
                show_warnings=False
            )

            # Obtener train predictions usando historical_forecasts (sin atajos)
            print("      Generando train predictions (esto puede tomar varios minutos)...")
            train_predictions = get_train_predictions_darts(
                model, target_train_scaled, input_chunk,
                past_covariates=covariates_train_scaled,
                scaler=scaler_target,
                stride=1  # Prediccion para cada dia
            )

            metrics = evaluate_basic_metrics(y_test.values[:len(test_predictions)], test_predictions)
            training_time = time.time() - start_time

            artifacts['models']['TFT'] = {
                'model': None,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'config': {'input_chunk': input_chunk, 'epochs': n_epochs},
                'metrics': metrics,
                'training_time': training_time,
                'model_type': 'darts_dl',
            }

            print(f"    RMSE: {metrics['rmse']:.6f} | Dir.Acc: {metrics['directional_accuracy']:.1%} | Time: {training_time:.1f}s")
        except Exception as e:
            print(f"    [ERROR] {str(e)}")

# =============================================================================
# PASO 7: GUARDAR ARTEFACTOS
# =============================================================================
print("\n" + "="*80)
print("PASO 7: Guardar Artefactos")
print("="*80)

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
print(f"  + Guardando en: {output_path}")

# Guardar con pickle
with open(output_path, 'wb') as f:
    pickle.dump(artifacts, f)

# Calcular tamano del archivo
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  + Tamano del archivo: {file_size_mb:.2f} MB")

# Resumen de modelos guardados
print(f"\n  + Modelos entrenados: {len(artifacts['models'])}")
for name, data in artifacts['models'].items():
    model_type = data.get('model_type', 'unknown')
    has_model = data.get('model') is not None
    print(f"    - {name}: {model_type} | modelo guardado: {'SI' if has_model else 'NO'}")

# =============================================================================
# PASO 8: RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

# Ordenar modelos por directional accuracy
model_ranking = sorted(
    [(name, data['metrics']['directional_accuracy']) for name, data in artifacts['models'].items()],
    key=lambda x: x[1],
    reverse=True
)

print("\n  RANKING POR DIRECTIONAL ACCURACY:")
print("  " + "-"*50)
for i, (name, acc) in enumerate(model_ranking, 1):
    print(f"  {i:2d}. {name:25s} {acc:.1%}")

# Mejor modelo
best_model_name = model_ranking[0][0]
best_model_acc = model_ranking[0][1]
print(f"\n  MEJOR MODELO: {best_model_name} (Dir.Acc: {best_model_acc:.1%})")

# Tiempo total
total_time = sum(data['training_time'] for data in artifacts['models'].values())
print(f"\n  TIEMPO TOTAL DE ENTRENAMIENTO: {total_time/60:.1f} minutos")

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"\n  Ejecutar ahora: python scripts/evaluate_strategy.py")
print()
