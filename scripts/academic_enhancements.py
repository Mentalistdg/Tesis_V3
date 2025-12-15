"""
Academic Enhancements for ML Pipeline
======================================

Este script implementa mejoras basadas en las referencias academicas citadas:

1. Gu et al. (2020) - "Empirical Asset Pricing via Machine Learning"
   - SHAP values para interpretabilidad
   - Walk-forward validation
   - Feature importance analysis

2. Fama (1970) - "Efficient Capital Markets"
   - Tests estadisticos de significancia del alfa
   - Bootstrap para intervalos de confianza

3. Dong et al. (2022) - "Anomalies and the Expected Market Return"
   - Analisis de regimenes de mercado
   - Estabilidad temporal de las anomalias

4. Mejoras Practicas para Trading
   - Optimizacion con penalizacion de turnover
   - Kelly criterion para position sizing
   - Deteccion de degradacion del modelo

Autor: David Gonzalez Canon
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost y LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# SHAP para interpretabilidad
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP no instalado. Instalar con: pip install shap")

# Configuracion de paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "paper" / "figures"

# Crear directorios si no existen
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# SECCION 1: PRUEBAS ESTADISTICAS DE SIGNIFICANCIA
# Basado en: Fama (1970), Gu et al. (2020)
# =============================================================================

class StatisticalSignificanceTests:
    """
    Implementa pruebas estadisticas para validar la significancia de los resultados.

    Referencia:
    - Fama, E. F. (1970). Efficient Capital Markets.
    - Ledoit, O., & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio.
    """

    def __init__(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray,
                 predictions: np.ndarray = None, actuals: np.ndarray = None):
        """
        Args:
            strategy_returns: Retornos diarios de la estrategia
            benchmark_returns: Retornos diarios del benchmark (mercado)
            predictions: Predicciones del modelo (opcional)
            actuals: Valores reales (opcional)
        """
        self.strategy_returns = np.array(strategy_returns)
        self.benchmark_returns = np.array(benchmark_returns)
        self.predictions = predictions
        self.actuals = actuals
        self.results = {}

    def sharpe_ratio_bootstrap(self, n_bootstrap: int = 10000, confidence: float = 0.95) -> dict:
        """
        Bootstrap para intervalos de confianza del Sharpe Ratio.

        El Sharpe Ratio es una medida crucial pero su distribucion no es conocida
        en muestras finitas. El bootstrap proporciona intervalos de confianza
        robustos sin asumir normalidad.
        """
        n = len(self.strategy_returns)
        sharpe_samples = []

        for _ in range(n_bootstrap):
            # Muestreo con reemplazo
            idx = np.random.choice(n, size=n, replace=True)
            sample_returns = self.strategy_returns[idx]

            # Calcular Sharpe del bootstrap sample
            if sample_returns.std() > 0:
                sharpe = (sample_returns.mean() / sample_returns.std()) * np.sqrt(252)
                sharpe_samples.append(sharpe)

        sharpe_samples = np.array(sharpe_samples)

        # Calcular intervalos de confianza
        alpha = 1 - confidence
        ci_lower = np.percentile(sharpe_samples, alpha/2 * 100)
        ci_upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)

        # Sharpe ratio puntual
        point_estimate = (self.strategy_returns.mean() / self.strategy_returns.std()) * np.sqrt(252)

        # p-value: probabilidad de que Sharpe sea <= 0
        p_value = np.mean(sharpe_samples <= 0)

        result = {
            'sharpe_ratio': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence,
            'p_value_positive': p_value,
            'significant_at_5pct': p_value < 0.05,
            'significant_at_1pct': p_value < 0.01,
            'n_bootstrap': n_bootstrap
        }

        self.results['sharpe_bootstrap'] = result
        return result

    def alpha_significance_test(self, risk_free_rate: float = 0.0) -> dict:
        """
        Test t para la significancia del alfa (retorno anormal).

        H0: alpha = 0 (no hay retorno anormal)
        H1: alpha != 0 (existe retorno anormal)

        Basado en regresion de Jensen's alpha:
        R_strategy - R_f = alpha + beta * (R_market - R_f) + epsilon
        """
        # Excess returns
        excess_strategy = self.strategy_returns - risk_free_rate/252
        excess_market = self.benchmark_returns - risk_free_rate/252

        # Regresion OLS: R_strategy = alpha + beta * R_market
        n = len(excess_strategy)
        X = np.column_stack([np.ones(n), excess_market])
        y = excess_strategy

        # Estimacion por OLS
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha = beta_hat[0]
        beta = beta_hat[1]

        # Residuos y errores estandar
        y_pred = X @ beta_hat
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)

        # Matriz de varianza-covarianza de los coeficientes
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(var_coef[0, 0])
        se_beta = np.sqrt(var_coef[1, 1])

        # t-statistic y p-value para alpha
        t_stat_alpha = alpha / se_alpha
        p_value_alpha = 2 * (1 - stats.t.cdf(abs(t_stat_alpha), df=n-2))

        # Alpha anualizado
        alpha_annual = alpha * 252

        result = {
            'alpha_daily': alpha,
            'alpha_annual': alpha_annual,
            'alpha_annual_pct': alpha_annual * 100,
            'se_alpha': se_alpha,
            't_statistic': t_stat_alpha,
            'p_value': p_value_alpha,
            'significant_at_5pct': p_value_alpha < 0.05,
            'significant_at_1pct': p_value_alpha < 0.01,
            'beta': beta,
            'r_squared': 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
        }

        self.results['alpha_test'] = result
        return result

    def directional_accuracy_test(self) -> dict:
        """
        Test binomial para la precision direccional.

        H0: p = 0.5 (no mejor que azar)
        H1: p > 0.5 (mejor que azar)

        Importante para validar que el modelo tiene capacidad predictiva real.
        """
        if self.predictions is None or self.actuals is None:
            return {'error': 'Se requieren predictions y actuals'}

        # Direccion correcta
        pred_direction = np.sign(self.predictions)
        actual_direction = np.sign(self.actuals)

        correct = np.sum(pred_direction == actual_direction)
        total = len(pred_direction)
        accuracy = correct / total

        # Test binomial (one-sided: accuracy > 0.5)
        p_value = 1 - stats.binom.cdf(correct - 1, total, 0.5)

        # Intervalo de confianza para la proporcion (Wilson score)
        z = 1.96  # 95% CI
        denominator = 1 + z**2/total
        center = (accuracy + z**2/(2*total)) / denominator
        spread = z * np.sqrt((accuracy*(1-accuracy) + z**2/(4*total))/total) / denominator

        result = {
            'directional_accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'p_value': p_value,
            'significant_at_5pct': p_value < 0.05,
            'significant_at_1pct': p_value < 0.01,
            'ci_95_lower': center - spread,
            'ci_95_upper': center + spread
        }

        self.results['directional_test'] = result
        return result

    def sharpe_ratio_comparison_test(self) -> dict:
        """
        Test de Ledoit-Wolf para comparar Sharpe ratios.

        H0: SR_strategy = SR_benchmark
        H1: SR_strategy != SR_benchmark

        Referencia: Ledoit & Wolf (2008) - HAC inference for Sharpe ratios
        """
        n = len(self.strategy_returns)

        # Sharpe ratios
        sr_strategy = (self.strategy_returns.mean() / self.strategy_returns.std()) * np.sqrt(252)
        sr_benchmark = (self.benchmark_returns.mean() / self.benchmark_returns.std()) * np.sqrt(252)

        # Diferencia de Sharpe ratios
        sr_diff = sr_strategy - sr_benchmark

        # Estimacion de varianza usando HAC (Newey-West)
        # Simplificacion: usamos bootstrap para la varianza de la diferencia
        n_bootstrap = 5000
        sr_diff_samples = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            strat_sample = self.strategy_returns[idx]
            bench_sample = self.benchmark_returns[idx]

            if strat_sample.std() > 0 and bench_sample.std() > 0:
                sr_s = (strat_sample.mean() / strat_sample.std()) * np.sqrt(252)
                sr_b = (bench_sample.mean() / bench_sample.std()) * np.sqrt(252)
                sr_diff_samples.append(sr_s - sr_b)

        sr_diff_samples = np.array(sr_diff_samples)
        se_diff = sr_diff_samples.std()

        # z-statistic
        z_stat = sr_diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        result = {
            'sharpe_strategy': sr_strategy,
            'sharpe_benchmark': sr_benchmark,
            'sharpe_difference': sr_diff,
            'se_difference': se_diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_at_5pct': p_value < 0.05,
            'significant_at_1pct': p_value < 0.01,
            'strategy_outperforms': sr_diff > 0
        }

        self.results['sharpe_comparison'] = result
        return result

    def run_all_tests(self) -> dict:
        """Ejecuta todas las pruebas estadisticas."""
        print("=" * 60)
        print("PRUEBAS ESTADISTICAS DE SIGNIFICANCIA")
        print("=" * 60)

        # 1. Bootstrap Sharpe Ratio
        print("\n1. Bootstrap Sharpe Ratio...")
        sr_result = self.sharpe_ratio_bootstrap()
        print(f"   Sharpe Ratio: {sr_result['sharpe_ratio']:.4f}")
        print(f"   IC 95%: [{sr_result['ci_lower']:.4f}, {sr_result['ci_upper']:.4f}]")
        print(f"   p-value (SR > 0): {sr_result['p_value_positive']:.4f}")
        print(f"   Significativo al 5%: {sr_result['significant_at_5pct']}")

        # 2. Alpha Test
        print("\n2. Test de Significancia del Alpha (Jensen)...")
        alpha_result = self.alpha_significance_test()
        print(f"   Alpha anualizado: {alpha_result['alpha_annual_pct']:.2f}%")
        print(f"   t-statistic: {alpha_result['t_statistic']:.4f}")
        print(f"   p-value: {alpha_result['p_value']:.4f}")
        print(f"   Significativo al 5%: {alpha_result['significant_at_5pct']}")
        print(f"   Beta: {alpha_result['beta']:.4f}")

        # 3. Directional Accuracy Test
        if self.predictions is not None:
            print("\n3. Test de Precision Direccional...")
            dir_result = self.directional_accuracy_test()
            print(f"   Precision: {dir_result['directional_accuracy']:.2%}")
            print(f"   p-value (> 50%): {dir_result['p_value']:.4f}")
            print(f"   Significativo al 5%: {dir_result['significant_at_5pct']}")

        # 4. Sharpe Comparison
        print("\n4. Comparacion de Sharpe Ratios (Ledoit-Wolf)...")
        comp_result = self.sharpe_ratio_comparison_test()
        print(f"   SR Estrategia: {comp_result['sharpe_strategy']:.4f}")
        print(f"   SR Benchmark: {comp_result['sharpe_benchmark']:.4f}")
        print(f"   Diferencia: {comp_result['sharpe_difference']:.4f}")
        print(f"   z-statistic: {comp_result['z_statistic']:.4f}")
        print(f"   p-value: {comp_result['p_value']:.4f}")
        print(f"   Significativo al 5%: {comp_result['significant_at_5pct']}")

        return self.results


# =============================================================================
# SECCION 2: ANALISIS DE IMPORTANCIA DE FEATURES
# Basado en: Gu et al. (2020)
# =============================================================================

class FeatureImportanceAnalysis:
    """
    Analisis de importancia de features usando multiples metodos.

    Referencia: Gu, S., Kelly, B., & Xiu, D. (2020).
    Empirical Asset Pricing via Machine Learning. RFS.
    """

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray, feature_names: list = None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or list(X_train.columns)
        self.results = {}

    def get_model_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene feature importance nativa del modelo (si disponible).
        Aplica a: Random Forest, Gradient Boosting, XGBoost, LightGBM
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            self.results['native_importance'] = df
            return df
        elif hasattr(self.model, 'coef_'):
            # Para modelos lineales
            importance = np.abs(self.model.coef_)
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            self.results['native_importance'] = df
            return df
        else:
            return None

    def permutation_importance_analysis(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Permutation importance: mide la caida en performance al permutar cada feature.

        Ventaja: Model-agnostic, funciona con cualquier modelo.
        """
        print("Calculando permutation importance...")

        perm_importance = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=42, n_jobs=-1
        )

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.results['permutation_importance'] = df
        return df

    def shap_analysis(self, n_samples: int = 500) -> dict:
        """
        SHAP (SHapley Additive exPlanations) para interpretabilidad avanzada.

        Basado en teoria de juegos, proporciona:
        - Contribucion de cada feature a cada prediccion
        - Interacciones entre features
        - Explicaciones locales y globales
        """
        if not HAS_SHAP:
            print("SHAP no disponible. Instalar con: pip install shap")
            return None

        print("Calculando SHAP values...")

        # Subsample para eficiencia
        if len(self.X_test) > n_samples:
            idx = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_sample = self.X_test.iloc[idx] if hasattr(self.X_test, 'iloc') else self.X_test[idx]
        else:
            X_sample = self.X_test

        # Crear explainer segun tipo de modelo
        model_type = type(self.model).__name__

        try:
            if 'RandomForest' in model_type or 'GradientBoosting' in model_type:
                explainer = shap.TreeExplainer(self.model)
            elif 'XGB' in model_type or 'LGB' in model_type or 'LGBM' in model_type:
                explainer = shap.TreeExplainer(self.model)
            else:
                # Para modelos lineales u otros
                explainer = shap.Explainer(self.model, X_sample)

            shap_values = explainer.shap_values(X_sample)

            # SHAP summary
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)

            self.results['shap_importance'] = df
            self.results['shap_values'] = shap_values
            self.results['shap_explainer'] = explainer

            return {
                'importance': df,
                'values': shap_values,
                'explainer': explainer
            }

        except Exception as e:
            print(f"Error en SHAP: {e}")
            return None

    def get_top_features_by_category(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analiza los top features agrupados por categoria.

        Categorias basadas en el prefijo del feature:
        - M_: Mercado
        - E_: Economico
        - I_: Tasas de interes
        - P_: Commodities/Precios
        - V_: Volatilidad
        - S_: Sentimiento
        - W_: Elder Semanal
        - D_: Elder Diario
        - TS_: Triple Screen
        """
        if 'permutation_importance' in self.results:
            df = self.results['permutation_importance'].head(top_n * 3)
        elif 'native_importance' in self.results:
            df = self.results['native_importance'].head(top_n * 3)
        else:
            return None

        # Categorizar
        def get_category(feature_name):
            prefixes = {
                'M_': 'Mercado', 'E_': 'Economico', 'I_': 'Tasas',
                'P_': 'Commodities', 'V_': 'Volatilidad', 'S_': 'Sentimiento',
                'W_': 'Elder_Semanal', 'D_': 'Elder_Diario', 'TS_': 'Triple_Screen',
                'SPY_': 'SPY_Tecnico', 'vol_': 'Vol_OHLC', 'corr_': 'Correlaciones'
            }
            for prefix, category in prefixes.items():
                if feature_name.startswith(prefix):
                    return category
            return 'Otros'

        df['category'] = df['feature'].apply(get_category)

        # Agregar por categoria
        if 'importance_mean' in df.columns:
            category_summary = df.groupby('category')['importance_mean'].agg(['sum', 'mean', 'count'])
        else:
            category_summary = df.groupby('category')['importance'].agg(['sum', 'mean', 'count'])

        category_summary = category_summary.sort_values('sum', ascending=False)

        self.results['category_importance'] = category_summary
        return category_summary

    def run_full_analysis(self) -> dict:
        """Ejecuta analisis completo de importancia."""
        print("\n" + "=" * 60)
        print("ANALISIS DE IMPORTANCIA DE FEATURES")
        print("=" * 60)

        # 1. Native importance
        print("\n1. Feature Importance Nativa del Modelo...")
        native = self.get_model_feature_importance()
        if native is not None:
            print(f"   Top 10 features:")
            for i, row in native.head(10).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

        # 2. Permutation importance
        print("\n2. Permutation Importance...")
        perm = self.permutation_importance_analysis()
        print(f"   Top 10 features:")
        for i, (_, row) in enumerate(perm.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance_mean']:.4f}")

        # 3. SHAP (si disponible)
        if HAS_SHAP:
            print("\n3. SHAP Analysis...")
            shap_result = self.shap_analysis()
            if shap_result:
                print(f"   Top 10 features por SHAP:")
                for i, (_, row) in enumerate(shap_result['importance'].head(10).iterrows()):
                    print(f"   {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")

        # 4. Analisis por categoria
        print("\n4. Importancia por Categoria...")
        cat_summary = self.get_top_features_by_category()
        if cat_summary is not None:
            print(cat_summary.to_string())

        return self.results


# =============================================================================
# SECCION 3: WALK-FORWARD VALIDATION
# Basado en: Gu et al. (2020), practicas de hedge funds
# =============================================================================

class WalkForwardValidation:
    """
    Walk-Forward Validation para evaluar estabilidad temporal.

    A diferencia del split unico train/test, walk-forward:
    1. Simula trading real con reentrenamiento periodico
    2. Detecta degradacion del modelo
    3. Proporciona multiples estimaciones out-of-sample
    """

    def __init__(self, X: pd.DataFrame, y: np.ndarray,
                 model_class, model_params: dict = None):
        """
        Args:
            X: Features DataFrame
            y: Target array
            model_class: Clase del modelo (ej: Ridge, RandomForestRegressor)
            model_params: Parametros del modelo
        """
        self.X = X
        self.y = y
        self.model_class = model_class
        self.model_params = model_params or {}
        self.results = []

    def expanding_window(self, initial_train_pct: float = 0.5,
                         test_window_days: int = 252,
                         min_train_size: int = 1000) -> pd.DataFrame:
        """
        Expanding window: entrena con todos los datos historicos disponibles.

        Ventajas:
        - Maximiza datos de entrenamiento
        - Simula el proceso real de un gestor de fondos

        Args:
            initial_train_pct: Porcentaje inicial para primer entrenamiento
            test_window_days: Dias de test en cada ventana
            min_train_size: Minimo de observaciones para entrenar
        """
        print("\n--- Expanding Window Walk-Forward ---")

        n_samples = len(self.y)
        initial_train_size = int(n_samples * initial_train_pct)

        results = []
        window_num = 0

        current_train_end = initial_train_size

        while current_train_end + test_window_days <= n_samples:
            window_num += 1

            # Indices
            train_idx = range(0, current_train_end)
            test_idx = range(current_train_end, min(current_train_end + test_window_days, n_samples))

            # Datos
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y[test_idx]

            # Preprocesamiento
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()

            X_train_imp = imputer.fit_transform(X_train)
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_imp = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imp)

            # Entrenar modelo
            model = self.model_class(**self.model_params)
            model.fit(X_train_scaled, y_train)

            # Predicciones
            y_pred = model.predict(X_test_scaled)

            # Metricas
            metrics = self._calculate_window_metrics(y_test, y_pred, window_num,
                                                     len(train_idx), len(test_idx))
            metrics['window_type'] = 'expanding'
            metrics['train_start'] = self.X.index[train_idx[0]] if hasattr(self.X, 'index') else train_idx[0]
            metrics['train_end'] = self.X.index[train_idx[-1]] if hasattr(self.X, 'index') else train_idx[-1]
            metrics['test_start'] = self.X.index[test_idx[0]] if hasattr(self.X, 'index') else test_idx[0]
            metrics['test_end'] = self.X.index[test_idx[-1]] if hasattr(self.X, 'index') else test_idx[-1]

            results.append(metrics)

            print(f"   Window {window_num}: Train={len(train_idx)}, Test={len(test_idx)}, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, DirAcc={metrics['directional_accuracy']:.2%}")

            # Avanzar ventana
            current_train_end += test_window_days

        self.results = pd.DataFrame(results)
        return self.results

    def rolling_window(self, train_window_days: int = 1260,
                       test_window_days: int = 252,
                       step_days: int = 63) -> pd.DataFrame:
        """
        Rolling window: entrena solo con ventana fija de datos recientes.

        Ventajas:
        - Modelos mas adaptados a condiciones recientes
        - Detecta cambios de regimen

        Args:
            train_window_days: Dias de entrenamiento (ej: 1260 = 5 anos)
            test_window_days: Dias de test
            step_days: Paso entre ventanas (ej: 63 = trimestral)
        """
        print("\n--- Rolling Window Walk-Forward ---")

        n_samples = len(self.y)
        results = []
        window_num = 0

        current_start = 0

        while current_start + train_window_days + test_window_days <= n_samples:
            window_num += 1

            # Indices
            train_start = current_start
            train_end = current_start + train_window_days
            test_start = train_end
            test_end = min(train_end + test_window_days, n_samples)

            train_idx = range(train_start, train_end)
            test_idx = range(test_start, test_end)

            # Datos
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y[test_idx]

            # Preprocesamiento
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()

            X_train_imp = imputer.fit_transform(X_train)
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_imp = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imp)

            # Entrenar modelo
            model = self.model_class(**self.model_params)
            model.fit(X_train_scaled, y_train)

            # Predicciones
            y_pred = model.predict(X_test_scaled)

            # Metricas
            metrics = self._calculate_window_metrics(y_test, y_pred, window_num,
                                                     len(train_idx), len(test_idx))
            metrics['window_type'] = 'rolling'
            metrics['train_start'] = self.X.index[train_idx[0]] if hasattr(self.X, 'index') else train_idx[0]
            metrics['train_end'] = self.X.index[train_idx[-1]] if hasattr(self.X, 'index') else train_idx[-1]
            metrics['test_start'] = self.X.index[test_idx[0]] if hasattr(self.X, 'index') else test_idx[0]
            metrics['test_end'] = self.X.index[test_idx[-1]] if hasattr(self.X, 'index') else test_idx[-1]

            results.append(metrics)

            print(f"   Window {window_num}: Train=[{train_start}:{train_end}], "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, DirAcc={metrics['directional_accuracy']:.2%}")

            # Avanzar ventana
            current_start += step_days

        self.results = pd.DataFrame(results)
        return self.results

    def _calculate_window_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  window_num: int, train_size: int, test_size: int) -> dict:
        """Calcula metricas para una ventana."""
        # Metricas de regresion
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Precision direccional
        dir_correct = np.sum(np.sign(y_pred) == np.sign(y_true))
        dir_acc = dir_correct / len(y_true)

        # Simular retornos de estrategia
        # Position = sigmoid(pred * 500) * 4 - 2, mapea a [-2, 2]
        positions = 4 / (1 + np.exp(-500 * y_pred)) - 2
        strategy_returns = positions * y_true

        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Retorno acumulado
        cumulative_return = np.prod(1 + strategy_returns) - 1

        return {
            'window': window_num,
            'train_size': train_size,
            'test_size': test_size,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': dir_acc,
            'sharpe_ratio': sharpe,
            'cumulative_return': cumulative_return,
            'mean_position': positions.mean()
        }

    def get_stability_metrics(self) -> dict:
        """Calcula metricas de estabilidad a traves de ventanas."""
        if self.results is None or len(self.results) == 0:
            return None

        df = self.results

        stability = {
            'n_windows': len(df),
            'sharpe_mean': df['sharpe_ratio'].mean(),
            'sharpe_std': df['sharpe_ratio'].std(),
            'sharpe_min': df['sharpe_ratio'].min(),
            'sharpe_max': df['sharpe_ratio'].max(),
            'pct_positive_sharpe': (df['sharpe_ratio'] > 0).mean(),
            'dir_acc_mean': df['directional_accuracy'].mean(),
            'dir_acc_std': df['directional_accuracy'].std(),
            'pct_above_50_acc': (df['directional_accuracy'] > 0.5).mean(),
            'cumulative_return_mean': df['cumulative_return'].mean(),
            'cumulative_return_std': df['cumulative_return'].std()
        }

        return stability


# =============================================================================
# SECCION 4: ANALISIS DE REGIMENES DE MERCADO
# Basado en: Dong et al. (2022)
# =============================================================================

class MarketRegimeAnalysis:
    """
    Analiza el performance de la estrategia en diferentes regimenes de mercado.

    Referencia: Dong, X., Li, Y., Rapach, D., & Zhou, G. (2022).
    Anomalies and the Expected Market Return. JF.
    """

    def __init__(self, dates: pd.DatetimeIndex, strategy_returns: np.ndarray,
                 market_returns: np.ndarray, vix: np.ndarray = None):
        """
        Args:
            dates: Index de fechas
            strategy_returns: Retornos de la estrategia
            market_returns: Retornos del mercado
            vix: Serie del VIX (opcional)
        """
        self.dates = dates
        self.strategy_returns = strategy_returns
        self.market_returns = market_returns
        self.vix = vix
        self.results = {}

    def analyze_volatility_regimes(self, low_vol_threshold: float = 20,
                                   high_vol_threshold: float = 25) -> pd.DataFrame:
        """
        Analiza performance por regimen de volatilidad.

        Regimenes tipicos:
        - Baja volatilidad: VIX < 20
        - Volatilidad normal: 20 <= VIX < 25
        - Alta volatilidad: VIX >= 25
        """
        if self.vix is None:
            print("VIX no disponible para analisis de regimenes")
            return None

        # Clasificar regimenes
        regimes = pd.Series(index=range(len(self.vix)), dtype=str)
        regimes[self.vix < low_vol_threshold] = 'Low_Vol'
        regimes[(self.vix >= low_vol_threshold) & (self.vix < high_vol_threshold)] = 'Normal_Vol'
        regimes[self.vix >= high_vol_threshold] = 'High_Vol'

        results = []
        for regime in ['Low_Vol', 'Normal_Vol', 'High_Vol']:
            mask = regimes == regime
            if mask.sum() == 0:
                continue

            strat_ret = self.strategy_returns[mask]
            mkt_ret = self.market_returns[mask]

            if len(strat_ret) > 0 and strat_ret.std() > 0:
                sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            else:
                sharpe = 0

            results.append({
                'regime': regime,
                'n_days': mask.sum(),
                'pct_time': mask.mean() * 100,
                'strategy_return_ann': strat_ret.mean() * 252 * 100,
                'market_return_ann': mkt_ret.mean() * 252 * 100,
                'excess_return_ann': (strat_ret.mean() - mkt_ret.mean()) * 252 * 100,
                'strategy_vol_ann': strat_ret.std() * np.sqrt(252) * 100,
                'sharpe_ratio': sharpe,
                'win_rate': (strat_ret > 0).mean() * 100
            })

        df = pd.DataFrame(results)
        self.results['volatility_regimes'] = df
        return df

    def analyze_trend_regimes(self, lookback: int = 63) -> pd.DataFrame:
        """
        Analiza performance por regimen de tendencia.

        Regimenes basados en SMA del mercado:
        - Bull: Precio > SMA
        - Bear: Precio < SMA
        """
        # Calcular SMA del mercado (usando retornos acumulados como proxy)
        cumulative = np.cumprod(1 + self.market_returns)
        sma = pd.Series(cumulative).rolling(lookback).mean().values

        # Clasificar regimenes (empezando despues del lookback)
        regimes = pd.Series(index=range(len(self.market_returns)), dtype=str)
        regimes[:lookback] = 'Warmup'
        regimes[lookback:][cumulative[lookback:] > sma[lookback:]] = 'Bull'
        regimes[lookback:][cumulative[lookback:] <= sma[lookback:]] = 'Bear'

        results = []
        for regime in ['Bull', 'Bear']:
            mask = regimes == regime
            if mask.sum() == 0:
                continue

            strat_ret = self.strategy_returns[mask]
            mkt_ret = self.market_returns[mask]

            if len(strat_ret) > 0 and strat_ret.std() > 0:
                sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            else:
                sharpe = 0

            results.append({
                'regime': regime,
                'n_days': mask.sum(),
                'pct_time': mask.mean() * 100,
                'strategy_return_ann': strat_ret.mean() * 252 * 100,
                'market_return_ann': mkt_ret.mean() * 252 * 100,
                'excess_return_ann': (strat_ret.mean() - mkt_ret.mean()) * 252 * 100,
                'strategy_vol_ann': strat_ret.std() * np.sqrt(252) * 100,
                'sharpe_ratio': sharpe,
                'win_rate': (strat_ret > 0).mean() * 100
            })

        df = pd.DataFrame(results)
        self.results['trend_regimes'] = df
        return df

    def analyze_yearly_performance(self) -> pd.DataFrame:
        """Analiza performance por ano para detectar estabilidad temporal."""
        if not isinstance(self.dates, pd.DatetimeIndex):
            print("Se requiere DatetimeIndex para analisis anual")
            return None

        years = self.dates.year
        unique_years = sorted(years.unique())

        results = []
        for year in unique_years:
            mask = years == year
            if mask.sum() < 20:  # Minimo 20 dias
                continue

            strat_ret = self.strategy_returns[mask]
            mkt_ret = self.market_returns[mask]

            if len(strat_ret) > 0 and strat_ret.std() > 0:
                sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            else:
                sharpe = 0

            results.append({
                'year': year,
                'n_days': mask.sum(),
                'strategy_return': (np.prod(1 + strat_ret) - 1) * 100,
                'market_return': (np.prod(1 + mkt_ret) - 1) * 100,
                'excess_return': ((np.prod(1 + strat_ret) - 1) - (np.prod(1 + mkt_ret) - 1)) * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': self._calculate_max_drawdown(strat_ret) * 100,
                'win_rate': (strat_ret > 0).mean() * 100
            })

        df = pd.DataFrame(results)
        self.results['yearly_performance'] = df
        return df

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcula el maximo drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def run_full_analysis(self) -> dict:
        """Ejecuta analisis completo de regimenes."""
        print("\n" + "=" * 60)
        print("ANALISIS DE REGIMENES DE MERCADO")
        print("=" * 60)

        # 1. Regimenes de volatilidad
        if self.vix is not None:
            print("\n1. Performance por Regimen de Volatilidad:")
            vol_df = self.analyze_volatility_regimes()
            if vol_df is not None:
                print(vol_df.to_string(index=False))

        # 2. Regimenes de tendencia
        print("\n2. Performance por Regimen de Tendencia:")
        trend_df = self.analyze_trend_regimes()
        if trend_df is not None:
            print(trend_df.to_string(index=False))

        # 3. Performance anual
        print("\n3. Performance Anual:")
        yearly_df = self.analyze_yearly_performance()
        if yearly_df is not None:
            print(yearly_df.to_string(index=False))

        return self.results


# =============================================================================
# SECCION 5: MEJORAS PRACTICAS PARA TRADING
# =============================================================================

class TradingEnhancements:
    """
    Mejoras practicas para implementacion real de trading.
    """

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion para sizing optimo de posiciones.

        f* = (p * b - q) / b

        donde:
        - p = probabilidad de ganar
        - q = probabilidad de perder (1 - p)
        - b = ratio win/loss promedio

        En la practica, se usa una fraccion de Kelly (ej: Kelly/2)
        para reducir volatilidad.
        """
        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1

        kelly = (p * b - q) / b

        # Limitar a rango razonable
        kelly = max(0, min(kelly, 2.0))

        return kelly

    @staticmethod
    def optimize_with_turnover_penalty(returns: np.ndarray, positions: np.ndarray,
                                       turnover_cost: float = 0.0015,
                                       target_sharpe: float = 1.0) -> dict:
        """
        Optimiza posiciones penalizando turnover.

        Objetivo: max Sharpe - lambda * Turnover

        Util para encontrar el balance optimo entre
        seguir senales del modelo y minimizar costos.
        """
        # Calcular turnover actual
        position_changes = np.abs(np.diff(positions))
        total_turnover = position_changes.sum()
        turnover_cost_total = total_turnover * turnover_cost

        # Retornos netos
        gross_returns = positions[:-1] * returns[1:]
        net_returns = gross_returns - position_changes * turnover_cost

        # Sharpe neto
        if net_returns.std() > 0:
            net_sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
        else:
            net_sharpe = 0

        return {
            'gross_sharpe': (gross_returns.mean() / gross_returns.std()) * np.sqrt(252) if gross_returns.std() > 0 else 0,
            'net_sharpe': net_sharpe,
            'total_turnover': total_turnover,
            'turnover_cost_pct': turnover_cost_total / len(returns) * 252 * 100,
            'annualized_turnover': total_turnover / len(returns) * 252
        }

    @staticmethod
    def smooth_positions(raw_positions: np.ndarray,
                        smoothing_factor: float = 0.3) -> np.ndarray:
        """
        Suaviza posiciones para reducir turnover.

        position_t = alpha * raw_position_t + (1 - alpha) * position_{t-1}

        Args:
            raw_positions: Posiciones sin suavizar del modelo
            smoothing_factor: Alpha para el EMA (0.3 = 30% peso a nueva senal)
        """
        smoothed = np.zeros_like(raw_positions)
        smoothed[0] = raw_positions[0]

        for t in range(1, len(raw_positions)):
            smoothed[t] = smoothing_factor * raw_positions[t] + (1 - smoothing_factor) * smoothed[t-1]

        return smoothed

    @staticmethod
    def detect_model_degradation(rolling_sharpe: np.ndarray,
                                 threshold_percentile: float = 10) -> dict:
        """
        Detecta degradacion del modelo usando rolling Sharpe.

        Alerta cuando el Sharpe cae por debajo del percentil historico.
        """
        threshold = np.percentile(rolling_sharpe[~np.isnan(rolling_sharpe)], threshold_percentile)
        current_sharpe = rolling_sharpe[-1] if not np.isnan(rolling_sharpe[-1]) else rolling_sharpe[~np.isnan(rolling_sharpe)][-1]

        is_degraded = current_sharpe < threshold

        return {
            'current_rolling_sharpe': current_sharpe,
            'threshold': threshold,
            'is_degraded': is_degraded,
            'threshold_percentile': threshold_percentile,
            'historical_mean': np.nanmean(rolling_sharpe),
            'historical_std': np.nanstd(rolling_sharpe)
        }


# =============================================================================
# MAIN: EJECUTAR TODOS LOS ANALISIS
# =============================================================================

def main():
    """Ejecuta el pipeline completo de mejoras academicas."""

    print("=" * 70)
    print("MEJORAS ACADEMICAS Y PRACTICAS PARA EL PIPELINE DE ML")
    print("Basado en: Gu et al. (2020), Fama (1970), Dong et al. (2022)")
    print("=" * 70)

    # Cargar datos
    data_path = DATA_DIR / "bloomberg_triple_screen_core.csv"

    if not data_path.exists():
        print(f"ERROR: No se encontro {data_path}")
        print("Ejecuta primero el pipeline principal para generar los datos.")
        return

    print(f"\nCargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Shape: {df.shape}")

    # Identificar target y features
    target_col = 'market_forward_excess_returns'
    if target_col not in df.columns:
        target_col = [c for c in df.columns if 'forward' in c.lower() and 'return' in c.lower()]
        if target_col:
            target_col = target_col[0]
        else:
            print("ERROR: No se encontro columna de target")
            return

    # Separar features y target
    # IMPORTANTE: Excluir todas las columnas que son target o contienen info futura
    exclude_cols = [target_col, 'Date', 'date', 'date_id', 'forward_returns',
                    'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not c.startswith('Unnamed')
                    and 'forward' not in c.lower()]

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Eliminar NaN del target
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    dates = df.index[valid_idx]

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples validos: {len(y)}")

    # Split temporal
    test_size = 0.20
    split_idx = int(len(y) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    print(f"\nTrain: {len(y_train)} samples")
    print(f"Test: {len(y_test)} samples")

    # Preprocesamiento
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)

    # Ajustar feature_cols si hay diferencia en dimensiones
    n_features_scaled = X_train_scaled.shape[1]
    if n_features_scaled != len(feature_cols):
        print(f"Ajustando feature_cols: {len(feature_cols)} -> {n_features_scaled}")
        feature_cols = feature_cols[:n_features_scaled]

    # Entrenar modelo (Random Forest como ejemplo)
    print("\n" + "=" * 70)
    print("ENTRENANDO MODELO (Random Forest)")
    print("=" * 70)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Calcular retornos de estrategia
    positions = 4 / (1 + np.exp(-500 * y_pred)) - 2
    strategy_returns = positions * y_test
    market_returns = y_test  # Simplificacion: market return = excess return + rf

    # =========================================================================
    # 1. PRUEBAS ESTADISTICAS DE SIGNIFICANCIA
    # =========================================================================
    stats_tests = StatisticalSignificanceTests(
        strategy_returns=strategy_returns,
        benchmark_returns=market_returns,
        predictions=y_pred,
        actuals=y_test
    )
    stats_results = stats_tests.run_all_tests()

    # =========================================================================
    # 2. ANALISIS DE IMPORTANCIA DE FEATURES
    # =========================================================================
    feature_analysis = FeatureImportanceAnalysis(
        model=model,
        X_train=pd.DataFrame(X_train_scaled, columns=feature_cols),
        X_test=pd.DataFrame(X_test_scaled, columns=feature_cols),
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols
    )
    feature_results = feature_analysis.run_full_analysis()

    # =========================================================================
    # 3. WALK-FORWARD VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    wf_validator = WalkForwardValidation(
        X=X,
        y=y,
        model_class=RandomForestRegressor,
        model_params={'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 20,
                      'random_state': 42, 'n_jobs': -1}
    )

    # Expanding window
    expanding_results = wf_validator.expanding_window(
        initial_train_pct=0.5,
        test_window_days=252
    )

    stability = wf_validator.get_stability_metrics()
    print("\nMetricas de Estabilidad (Expanding Window):")
    for key, value in stability.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # =========================================================================
    # 4. ANALISIS DE REGIMENES
    # =========================================================================
    # Intentar obtener VIX de los datos
    vix_cols = [c for c in df.columns if 'VIX' in c.upper() and 'return' not in c.lower()]
    vix = None
    if vix_cols:
        vix_col = vix_cols[0]
        vix = df[vix_col].values[valid_idx][split_idx:]

    regime_analysis = MarketRegimeAnalysis(
        dates=dates_test,
        strategy_returns=strategy_returns,
        market_returns=market_returns,
        vix=vix
    )
    regime_results = regime_analysis.run_full_analysis()

    # =========================================================================
    # 5. MEJORAS PRACTICAS
    # =========================================================================
    print("\n" + "=" * 70)
    print("MEJORAS PRACTICAS DE TRADING")
    print("=" * 70)

    # Kelly Criterion
    win_rate = (strategy_returns > 0).mean()
    avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0
    avg_loss = strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).any() else -0.01

    kelly = TradingEnhancements.kelly_criterion(win_rate, avg_win, avg_loss)
    print(f"\n1. Kelly Criterion:")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg Win: {avg_win:.4%}")
    print(f"   Avg Loss: {avg_loss:.4%}")
    print(f"   Full Kelly: {kelly:.2f}")
    print(f"   Half Kelly (recomendado): {kelly/2:.2f}")

    # Analisis de turnover
    print(f"\n2. Analisis de Turnover:")
    turnover_analysis = TradingEnhancements.optimize_with_turnover_penalty(
        returns=y_test,
        positions=positions,
        turnover_cost=0.0015
    )
    for key, value in turnover_analysis.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # Posiciones suavizadas
    print(f"\n3. Efecto del Suavizado de Posiciones:")
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        smoothed_pos = TradingEnhancements.smooth_positions(positions, alpha)
        smooth_analysis = TradingEnhancements.optimize_with_turnover_penalty(
            returns=y_test,
            positions=smoothed_pos,
            turnover_cost=0.0015
        )
        print(f"   Alpha={alpha}: Net Sharpe={smooth_analysis['net_sharpe']:.3f}, "
              f"Turnover Anual={smooth_analysis['annualized_turnover']:.1f}")

    # =========================================================================
    # GUARDAR RESULTADOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("GUARDANDO RESULTADOS")
    print("=" * 70)

    # Crear resumen JSON
    summary = {
        'statistical_tests': {
            'sharpe_bootstrap': stats_results.get('sharpe_bootstrap', {}),
            'alpha_test': stats_results.get('alpha_test', {}),
            'directional_test': stats_results.get('directional_test', {}),
            'sharpe_comparison': stats_results.get('sharpe_comparison', {})
        },
        'walk_forward': stability,
        'kelly_criterion': {
            'full_kelly': kelly,
            'half_kelly': kelly / 2,
            'win_rate': win_rate
        },
        'turnover_analysis': turnover_analysis
    }

    # Convertir numpy a Python nativo para JSON
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    summary = convert_to_native(summary)

    # Guardar JSON
    output_path = RESULTS_DIR / "academic_enhancements_results.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Resultados guardados en: {output_path}")

    # Guardar CSVs
    if 'permutation_importance' in feature_results:
        feature_results['permutation_importance'].to_csv(
            RESULTS_DIR / "feature_importance_permutation.csv", index=False
        )
        print(f"Feature importance guardado en: {RESULTS_DIR / 'feature_importance_permutation.csv'}")

    if expanding_results is not None:
        expanding_results.to_csv(
            RESULTS_DIR / "walk_forward_expanding.csv", index=False
        )
        print(f"Walk-forward results guardado en: {RESULTS_DIR / 'walk_forward_expanding.csv'}")

    if 'yearly_performance' in regime_results:
        regime_results['yearly_performance'].to_csv(
            RESULTS_DIR / "yearly_performance.csv", index=False
        )
        print(f"Yearly performance guardado en: {RESULTS_DIR / 'yearly_performance.csv'}")

    print("\n" + "=" * 70)
    print("ANALISIS COMPLETADO")
    print("=" * 70)

    return {
        'statistical_tests': stats_results,
        'feature_importance': feature_results,
        'walk_forward': expanding_results,
        'regime_analysis': regime_results,
        'summary': summary
    }


if __name__ == "__main__":
    results = main()
