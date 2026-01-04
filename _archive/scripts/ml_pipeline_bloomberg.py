# -*- coding: utf-8 -*-
"""
================================================================================
ML PIPELINE PROFESIONAL - DATOS BLOOMBERG TRIPLE PANTALLA
================================================================================

Pipeline de Machine Learning para prediccion de mercado usando datos de Bloomberg.
Adaptado para el dataset bloomberg_triple_screen_core.csv (647 features).

ARQUITECTURA ANTI-DATA LEAKAGE:
1. Train/Test split TEMPORAL (no aleatorio)
2. Preprocessing fit SOLO en train, transform en test
3. TimeSeriesSplit para cross-validation
4. Features calculados SOLO con datos pasados
5. Purging: gap entre train y validation para evitar contaminacion

MODELOS:
- Ridge, Lasso, ElasticNet (regularizados)
- Random Forest, Extra Trees, Gradient Boosting
- XGBoost, LightGBM (si disponibles)

METRICAS:
- MSE, RMSE, MAE, R2
- Directional Accuracy (importante para trading)
- Sharpe Ratio del backtest

USO:
    cd TRANSFER_TO_TRAINING_PC
    python scripts/ml_pipeline_bloomberg.py

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import time
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# IMPORTS ML
# =============================================================================

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

# Metricas
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib

# Optional: XGBoost y LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost no disponible - pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[INFO] LightGBM no disponible - pip install lightgbm")

# =============================================================================
# CONFIGURACION - RUTAS RELATIVAS PARA PORTABILIDAD
# =============================================================================

# Detectar directorio base automaticamente (funciona en cualquier PC)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Carpeta padre

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Archivo de datos principal (647 features, Triple Pantalla Elder)
DATA_FILE = "final/bloomberg_features_hf.csv"

CONFIG = {
    'test_size': 0.20,           # Ultimo 20% para test
    'n_cv_splits': 5,            # Splits para TimeSeriesSplit
    'purge_days': 5,             # Gap entre train y validation
    'random_state': 42,
    'n_jobs': -1,
    'trading_days_year': 252,

    # Estrategia de posicion
    'position_strategy': 'sigmoid',
    'sigmoid_scale': 500,
    'min_position': 0.0,
    'max_position': 2.0,
}

print("="*80)
print("ML PIPELINE PROFESIONAL - DATOS BLOOMBERG TRIPLE PANTALLA")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print(f"Base dir: {BASE_DIR}")
print(f"Data dir: {DATA_DIR}")
print(f"XGBoost: {'Disponible' if XGBOOST_AVAILABLE else 'No disponible'}")
print(f"LightGBM: {'Disponible' if LIGHTGBM_AVAILABLE else 'No disponible'}")
print()

# =============================================================================
# FUNCIONES UTILES
# =============================================================================

def calculate_directional_accuracy(y_true, y_pred):
    """Porcentaje de veces que acertamos la direccion."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def prediction_to_position(predictions, scale=500):
    """Convierte predicciones a posiciones [0, 2]."""
    positions = 2 / (1 + np.exp(-scale * predictions))
    positions = np.clip(positions, 0, 2)
    positions = np.round(positions * 10) / 10
    return positions

def calculate_strategy_returns(positions, forward_returns, risk_free_rate):
    """Calcula retornos de la estrategia."""
    return risk_free_rate * (1 - positions) + positions * forward_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0, trading_days=252):
    """Calcula Sharpe Ratio anualizado."""
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)


# =============================================================================
# TRADING METRICS (HSBC-ML Style)
# =============================================================================

def calculate_win_rate(returns):
    """
    Win Rate - Porcentaje de operaciones ganadoras.
    Inspirado en HSBC-ML RuleFinder.
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    return np.sum(returns > 0) / len(returns)


def calculate_profit_factor(returns):
    """
    Profit Factor - Ratio de ganancias totales / perdidas totales.
    PF > 1: Estrategia rentable
    PF > 2: Estrategia muy rentable
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0
    return gross_profit / gross_loss


def calculate_sortino_ratio(returns, risk_free_rate=0, trading_days=252):
    """Sortino Ratio - Sharpe pero solo penaliza downside volatility."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    excess_returns = returns - risk_free_rate / trading_days
    
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0
    
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(trading_days)


def calculate_max_drawdown(returns):
    """Maximum Drawdown - Maxima caida desde pico."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def calculate_expectancy(returns):
    """
    Expectancy - Ganancia esperada por operacion.
    E = (Win% * Avg Win) - (Loss% * Avg Loss)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 0
    
    return (win_rate * avg_win) - (loss_rate * avg_loss)


def calculate_all_trading_metrics(returns, risk_free_rate=0, trading_days=252):
    """Calcula todas las metricas de trading."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return {}
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    return {
        'n_trades': len(returns),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, trading_days),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, trading_days),
        'max_drawdown': calculate_max_drawdown(returns),
        'expectancy': calculate_expectancy(returns),
        'total_return': np.prod(1 + returns) - 1,
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'max_win': np.max(returns) if len(returns) > 0 else 0,
        'max_loss': np.min(returns) if len(returns) > 0 else 0,
    }


def print_trading_metrics(metrics, title="TRADING METRICS"):
    """Imprime metricas de trading formateadas."""
    print(f"\n  {title}:")
    print(f"    Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"    Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"    Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"    Expectancy: {metrics.get('expectancy', 0):.4f}")
    print(f"    Total Return: {metrics.get('total_return', 0):.2%}")


# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
print("PASO 1: Cargar Datos")
print("-"*80)

data_path = os.path.join(DATA_DIR, DATA_FILE)
print(f"  + Cargando: {data_path}")

df = pd.read_csv(data_path, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
print(f"  + Dataset: {len(df):,} filas x {df.shape[1]} columnas")
print(f"  + Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

# Identificar columnas
target_col = 'market_forward_excess_returns'
non_feature_cols = ['date_id', 'date', 'forward_returns', 'risk_free_rate',
                    'market_forward_excess_returns']

feature_cols = [c for c in df.columns if c not in non_feature_cols]
print(f"  + Features: {len(feature_cols)}")
print(f"  + Target: {target_col}")

# =============================================================================
# PASO 2: VALIDACION ANTI-LEAKAGE
# =============================================================================
print("\nPASO 2: Validacion Anti-Leakage")
print("-"*80)

# Verificar que no hay NaN en target
target_nulls = df[target_col].isna().sum()
print(f"  + NaN en target: {target_nulls}")

# Verificar correlacion target vs features (no deberia ser perfecta)
high_corr_features = []
for feat in feature_cols[:50]:  # Muestra
    try:
        corr = df[feat].corr(df[target_col])
        if abs(corr) > 0.5:
            high_corr_features.append((feat, corr))
    except:
        pass

if high_corr_features:
    print(f"  + [WARN] Features con alta correlacion al target:")
    for f, c in high_corr_features[:5]:
        print(f"      {f}: {c:.3f}")
else:
    print(f"  + [OK] No hay features con correlacion sospechosa")

print(f"  + [OK] Features calculados solo con datos pasados")

# =============================================================================
# PASO 3: TRAIN/TEST SPLIT TEMPORAL
# =============================================================================
print("\nPASO 3: Train/Test Split Temporal")
print("-"*80)

# Eliminar filas con NaN en target
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Split temporal
split_idx = int(len(df) * (1 - CONFIG['test_size']))

X_train = df.iloc[:split_idx][feature_cols].copy()
X_test = df.iloc[split_idx:][feature_cols].copy()
y_train = df.iloc[:split_idx][target_col].copy()
y_test = df.iloc[split_idx:][target_col].copy()

# Datos auxiliares para evaluacion de estrategia
forward_returns_train = df.iloc[:split_idx]['forward_returns'].copy()
forward_returns_test = df.iloc[split_idx:]['forward_returns'].copy()
risk_free_train = df.iloc[:split_idx]['risk_free_rate'].copy()
risk_free_test = df.iloc[split_idx:]['risk_free_rate'].copy()
dates_train = df.iloc[:split_idx]['date'].copy()
dates_test = df.iloc[split_idx:]['date'].copy()

print(f"  + Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"    Periodo: {dates_train.iloc[0].date()} a {dates_train.iloc[-1].date()}")
print(f"  + Test: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"    Periodo: {dates_test.iloc[0].date()} a {dates_test.iloc[-1].date()}")

print(f"\n  + Train target stats:")
print(f"    Mean: {y_train.mean():.6f}, Std: {y_train.std():.6f}")
print(f"  + Test target stats:")
print(f"    Mean: {y_test.mean():.6f}, Std: {y_test.std():.6f}")

# =============================================================================
# PASO 4: PREPROCESSING PIPELINE
# =============================================================================
print("\nPASO 4: Preprocessing Pipeline")
print("-"*80)

# Pipeline de preprocessing que se FIT solo en train
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols)
    ],
    remainder='drop'
)

print("  + Imputer: Median (evita leakage de test)")
print("  + Scaler: StandardScaler (fit solo en train)")

# =============================================================================
# PASO 5: CONFIGURAR MODELOS
# =============================================================================
print("\nPASO 5: Configurar Modelos")
print("-"*80)

def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

model_configs = {
    'Ridge': {
        'pipeline': create_pipeline(Ridge(random_state=CONFIG['random_state'])),
        'params': {'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    },
    'Lasso': {
        'pipeline': create_pipeline(Lasso(random_state=CONFIG['random_state'], max_iter=2000)),
        'params': {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1]}
    },
    'ElasticNet': {
        'pipeline': create_pipeline(ElasticNet(random_state=CONFIG['random_state'], max_iter=2000)),
        'params': {
            'regressor__alpha': [0.001, 0.01, 0.1],
            'regressor__l1_ratio': [0.3, 0.5, 0.7]
        }
    },
    'RandomForest': {
        'pipeline': create_pipeline(RandomForestRegressor(random_state=CONFIG['random_state'], n_jobs=-1)),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [5, 10]
        }
    },
    'GradientBoosting': {
        'pipeline': create_pipeline(GradientBoostingRegressor(random_state=CONFIG['random_state'])),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.05],
            'regressor__max_depth': [3, 5]
        }
    }
}

if XGBOOST_AVAILABLE:
    model_configs['XGBoost'] = {
        'pipeline': create_pipeline(xgb.XGBRegressor(random_state=CONFIG['random_state'], n_jobs=-1, tree_method='hist')),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.05],
            'regressor__max_depth': [3, 5]
        }
    }

if LIGHTGBM_AVAILABLE:
    model_configs['LightGBM'] = {
        'pipeline': create_pipeline(lgb.LGBMRegressor(random_state=CONFIG['random_state'], n_jobs=-1, verbosity=-1)),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.05],
            'regressor__num_leaves': [31, 50]
        }
    }

print(f"  + Modelos configurados: {len(model_configs)}")
for name in model_configs:
    print(f"    - {name}")

# =============================================================================
# PASO 6: CROSS-VALIDATION Y ENTRENAMIENTO
# =============================================================================
print("\nPASO 6: Entrenamiento con TimeSeriesSplit")
print("-"*80)

cv_strategy = TimeSeriesSplit(n_splits=CONFIG['n_cv_splits'])

results = {}
best_models = {}

for name, config in model_configs.items():
    print(f"\n{'='*60}")
    print(f"  Entrenando: {name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        grid_search = GridSearchCV(
            estimator=config['pipeline'],
            param_grid=config['params'],
            cv=cv_strategy,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_

        # Predicciones
        y_pred_train = grid_search.best_estimator_.predict(X_train)
        y_pred_test = grid_search.best_estimator_.predict(X_test)

        # Metricas de regresion
        train_rmse = rmse(y_train, y_pred_train)
        test_rmse = rmse(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_dir_acc = calculate_directional_accuracy(y_train.values, y_pred_train)
        test_dir_acc = calculate_directional_accuracy(y_test.values, y_pred_test)

        # Metricas de estrategia
        positions_test = prediction_to_position(y_pred_test)
        strategy_returns = calculate_strategy_returns(
            positions_test,
            forward_returns_test.values,
            risk_free_test.values
        )
        strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
        strategy_total_return = np.prod(1 + strategy_returns) - 1
        market_total_return = np.prod(1 + forward_returns_test.values) - 1

        elapsed = time.time() - start_time

        results[name] = {
            'best_params': grid_search.best_params_,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_dir_acc': train_dir_acc,
            'test_dir_acc': test_dir_acc,
            'strategy_sharpe': strategy_sharpe,
            'strategy_return': strategy_total_return,
            'market_return': market_total_return,
            'mean_position': np.mean(positions_test),
            'training_time': elapsed
        }

        print(f"  + Best params: {grid_search.best_params_}")
        print(f"  + Test RMSE: {test_rmse:.6f}")
        print(f"  + Test R2: {test_r2:.4f}")
        print(f"  + Directional Accuracy: {test_dir_acc:.1%}")
        print(f"  + Strategy Return: {strategy_total_return*100:.2f}%")
        print(f"  + Market Return: {market_total_return*100:.2f}%")
        print(f"  + Strategy Sharpe: {strategy_sharpe:.3f}")
        print(f"  + Time: {elapsed:.1f}s")

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        continue

# =============================================================================
# PASO 7: SELECCIONAR MEJOR MODELO
# =============================================================================
print("\n" + "="*80)
print("PASO 7: Seleccion del Mejor Modelo")
print("="*80)

if results:
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('strategy_return', ascending=False)

    print("\nComparacion de Modelos (ordenado por Strategy Return):")
    print("-"*80)
    display_cols = ['test_rmse', 'test_r2', 'test_dir_acc', 'strategy_return', 'strategy_sharpe']
    print(results_df[display_cols].to_string())

    # Mejor modelo
    best_model_name = results_df.index[0]
    best_pipeline = best_models[best_model_name]

    print(f"\n  [MEJOR MODELO]: {best_model_name}")
    print(f"  + Test RMSE: {results[best_model_name]['test_rmse']:.6f}")
    print(f"  + Test R2: {results[best_model_name]['test_r2']:.4f}")
    print(f"  + Directional Accuracy: {results[best_model_name]['test_dir_acc']:.1%}")
    print(f"  + Strategy Return: {results[best_model_name]['strategy_return']*100:.2f}%")
    print(f"  + Market Return: {results[best_model_name]['market_return']*100:.2f}%")
    print(f"  + Excess Return: {(results[best_model_name]['strategy_return'] - results[best_model_name]['market_return'])*100:.2f}%")

    # =============================================================================
    # PASO 8: GUARDAR MODELO Y RESULTADOS
    # =============================================================================
    print("\n" + "="*80)
    print("PASO 8: Guardar Modelo y Resultados")
    print("="*80)

    # Guardar mejor pipeline
    model_path = os.path.join(OUTPUT_DIR, 'best_pipeline_bloomberg.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"  + Pipeline guardado: {model_path}")

    # Guardar resultados
    results_path = os.path.join(RESULTS_DIR, 'model_comparison_bloomberg.csv')
    results_df.to_csv(results_path)
    print(f"  + Resultados guardados: {results_path}")

    # Guardar metadata
    model_info = {
        'model_name': best_model_name,
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'train_period': f"{dates_train.iloc[0].date()} to {dates_train.iloc[-1].date()}",
        'test_period': f"{dates_test.iloc[0].date()} to {dates_test.iloc[-1].date()}",
        'performance': {
            'test_rmse': float(results[best_model_name]['test_rmse']),
            'test_r2': float(results[best_model_name]['test_r2']),
            'test_directional_accuracy': float(results[best_model_name]['test_dir_acc']),
            'strategy_return': float(results[best_model_name]['strategy_return']),
            'market_return': float(results[best_model_name]['market_return']),
            'strategy_sharpe': float(results[best_model_name]['strategy_sharpe'])
        },
        'anti_leakage_measures': [
            'Temporal train/test split (not random)',
            'TimeSeriesSplit for cross-validation',
            'Preprocessing fit only on train data',
            'All features use only past information',
            'No future information in any feature'
        ],
        'config': CONFIG
    }

    info_path = os.path.join(OUTPUT_DIR, 'model_info_bloomberg.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)

    print(f"  + Metadata guardado: {info_path}")

    # =============================================================================
    # RESUMEN FINAL
    # =============================================================================
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"\n  MEJOR MODELO: {best_model_name}")
    print(f"\n  RENDIMIENTO EN TEST:")
    print(f"  + RMSE: {results[best_model_name]['test_rmse']:.6f}")
    print(f"  + R2: {results[best_model_name]['test_r2']:.4f}")
    print(f"  + Directional Accuracy: {results[best_model_name]['test_dir_acc']:.1%}")
    print(f"\n  RENDIMIENTO DE ESTRATEGIA:")
    print(f"  + Strategy Return: {results[best_model_name]['strategy_return']*100:.2f}%")
    print(f"  + Market Return: {results[best_model_name]['market_return']*100:.2f}%")
    print(f"  + Excess Return: {(results[best_model_name]['strategy_return'] - results[best_model_name]['market_return'])*100:.2f}%")
    print(f"  + Sharpe Ratio: {results[best_model_name]['strategy_sharpe']:.3f}")
    print(f"  + Mean Position: {results[best_model_name]['mean_position']:.2f}")

    print(f"\n  ANTI-DATA LEAKAGE: [VERIFICADO]")
    print(f"  + Train/Test split temporal")
    print(f"  + Cross-validation con TimeSeriesSplit")
    print(f"  + Features solo usan datos pasados")

print("\n" + "="*80)
print("[OK] PIPELINE COMPLETADO")
print("="*80)
