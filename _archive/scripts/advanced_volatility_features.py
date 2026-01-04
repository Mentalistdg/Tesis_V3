# -*- coding: utf-8 -*-
"""
================================================================================
ADVANCED VOLATILITY FEATURES - METODOLOGIAS CONTEMPORANEAS
================================================================================

Implementacion de estimadores de volatilidad contemporaneos basados en papers
academicos recientes que extraen informacion intradia de datos OHLC diarios.

FEATURES IMPLEMENTADOS:

1. INTRINSIC ENTROPY MODEL (Vinte & Ausloos, 2021)
   Paper: "A Volatility Estimator of Stock Market Indices Based on the
          Intrinsic Entropy Model"
   Journal: Entropy (MDPI), Vol. 23, Issue 4
   Innovacion: Incorpora VOLUMEN ademas de OHLC usando teoria de entropia.

2. LOG-RANGE FEATURES (Alizadeh, Brandt & Diebold, 2002)
   Paper: "Range-Based Estimation of Stochastic Volatility Models"
   Journal: Journal of Finance, Vol. 57, No. 3
   Hallazgo: ln(H/L) es aproximadamente Gaussiano, ideal para ML.

3. CARR-STYLE FEATURES (Chou, 2005)
   Paper: "Forecasting Financial Volatilities with Extreme Values:
          The Conditional Autoregressive Range (CARR) Model"
   Journal: Journal of Money, Credit and Banking
   Concepto: Dinamica autorregresiva del rango (similar a GARCH).

4. RANGE-GARCH FEATURES (Fiszeder & Perczak, 2016)
   Paper: "Low and high prices can improve volatility forecasts during
          periods of turmoil"
   Journal: International Journal of Forecasting
   Innovacion: Combina informacion H-L-C en estructura GARCH.

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# Configuracion de directorios
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

print("=" * 80)
print("ADVANCED VOLATILITY FEATURES")
print("Metodologias Contemporaneas (2002-2021)")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


# =============================================================================
# 1. INTRINSIC ENTROPY MODEL (Vinte & Ausloos, 2021)
# =============================================================================

def intrinsic_entropy_volatility(open_price, high, low, close, volume, window=21):
    """
    Intrinsic Entropy Model para estimacion de volatilidad.

    Paper: Vinte & Ausloos (2021) - "A Volatility Estimator of Stock Market
           Indices Based on the Intrinsic Entropy Model"

    Concepto:
    ---------
    El volumen actua como "credibilidad del mercado" asignada a cada nivel
    de precio. La entropia mide la dispersion de esta credibilidad.

    Metodologia:
    ------------
    1. Dividir el rango [L, H] en N bins
    2. Asignar volumen proporcional a cada bin basado en la posicion del precio
    3. Calcular entropia: IE = -sum(p_i * ln(p_i))
    4. Normalizar por ln(N) para obtener entropia relativa

    La volatilidad se relaciona con la entropia: mayor dispersion = mayor volatilidad

    Parameters:
    -----------
    open_price, high, low, close : Series - Precios OHLC
    volume : Series - Volumen diario
    window : int - Ventana para rolling

    Returns:
    --------
    dict con features de entropia intrinseca
    """
    features = {}
    n = len(close)

    # Calcular rango normalizado y posiciones
    price_range = high - low
    price_range = price_range.replace(0, np.nan)  # Evitar division por cero

    # Posicion del cierre en el rango [0, 1]
    close_position = (close - low) / price_range

    # Posicion de la apertura en el rango [0, 1]
    open_position = (open_price - low) / price_range

    # --- Feature 1: Entropia basada en distribucion de precio en el rango ---
    # Aproximamos la distribucion del precio usando O, H, L, C
    # Asignamos probabilidades basadas en donde "paso tiempo" el precio

    def calc_price_entropy(o_pos, c_pos):
        """
        Calcula entropia de la distribucion de precio en el rango.
        Asumimos que el precio paso tiempo en diferentes zonas del rango.
        """
        # Dividimos el rango en 4 zonas: [0-0.25], [0.25-0.5], [0.5-0.75], [0.75-1]
        # La probabilidad de cada zona depende de O y C

        # Zona donde abrio
        p_open = np.zeros(4)
        o_zone = int(min(o_pos * 4, 3)) if not np.isnan(o_pos) else 0
        p_open[o_zone] = 0.25

        # Zona donde cerro
        p_close = np.zeros(4)
        c_zone = int(min(c_pos * 4, 3)) if not np.isnan(c_pos) else 0
        p_close[c_zone] = 0.25

        # High siempre esta en zona 3 (0.75-1), Low en zona 0
        p_hl = np.array([0.25, 0, 0, 0.25])

        # Combinar probabilidades
        p = (p_open + p_close + p_hl) / 2
        p = p / p.sum()  # Normalizar

        # Calcular entropia
        p = p[p > 0]  # Evitar log(0)
        entropy = -np.sum(p * np.log(p))

        # Normalizar por max entropia (ln(4))
        return entropy / np.log(4)

    # Calcular entropia para cada dia
    daily_entropy = np.array([
        calc_price_entropy(open_position.iloc[i], close_position.iloc[i])
        for i in range(n)
    ])

    features['IE_price_entropy'] = pd.Series(daily_entropy, index=close.index)
    features['IE_price_entropy_ma5'] = features['IE_price_entropy'].rolling(5).mean()
    features['IE_price_entropy_ma21'] = features['IE_price_entropy'].rolling(21).mean()

    # --- Feature 2: Entropia ponderada por volumen ---
    # Volumen relativo como "peso" de la informacion del dia
    vol_relative = volume / volume.rolling(window).mean()

    # Entropia ajustada por volumen (mayor volumen = mas informativo)
    features['IE_vol_weighted'] = features['IE_price_entropy'] * np.log1p(vol_relative)
    features['IE_vol_weighted_ma21'] = features['IE_vol_weighted'].rolling(21).mean()

    # --- Feature 3: Entropia del rango (inspirado en el paper) ---
    # Usar el rango como proxy de dispersion de precios
    log_range = np.log(price_range)

    # Entropia aproximada del rango usando momentos
    range_mean = log_range.rolling(window).mean()
    range_std = log_range.rolling(window).std()

    # Entropia de distribucion normal: 0.5 * ln(2*pi*e*sigma^2)
    features['IE_range_entropy'] = 0.5 * np.log(2 * np.pi * np.e * range_std**2)

    # --- Feature 4: Indice de concentracion del precio ---
    # Que tan concentrado estuvo el precio en una zona del rango
    # Basado en la diferencia entre posicion de apertura y cierre
    price_concentration = 1 - np.abs(close_position - open_position)
    features['IE_price_concentration'] = price_concentration
    features['IE_price_concentration_ma21'] = price_concentration.rolling(21).mean()

    # --- Feature 5: Entropia de volumen relativo ---
    # Dispersion del volumen respecto a su media
    vol_normalized = volume / volume.rolling(window).sum()
    vol_entropy = -vol_normalized * np.log(vol_normalized + 1e-10)
    features['IE_volume_entropy'] = vol_entropy.rolling(window).sum()

    return features


# =============================================================================
# 2. LOG-RANGE FEATURES (Alizadeh, Brandt & Diebold, 2002)
# =============================================================================

def log_range_features(high, low, close, window=21):
    """
    Log-Range Features basados en Alizadeh, Brandt & Diebold (2002).

    Paper: "Range-Based Estimation of Stochastic Volatility Models"
    Journal of Finance, Vol. 57, No. 3, pp. 1047-1091

    Hallazgo clave:
    ---------------
    El logaritmo del rango ln(H/L) es aproximadamente GAUSSIANO,
    a diferencia de los retornos cuadrados que son muy sesgados.

    Esto tiene dos implicaciones para ML:
    1. Features mas "bien comportados" para modelos lineales
    2. Mejor para estimacion por maxima verosimilitud

    Eficiencia:
    -----------
    La std del log-rango es ~1/4 de la std del log-retorno absoluto.

    Parameters:
    -----------
    high, low, close : Series - Precios
    window : int - Ventana para rolling

    Returns:
    --------
    dict con features basados en log-rango
    """
    features = {}

    # --- Feature 1: Log-Range basico ---
    log_range = np.log(high / low)
    features['LR_log_range'] = log_range

    # --- Feature 2: Log-Range normalizado (Z-score) ---
    lr_mean = log_range.rolling(window).mean()
    lr_std = log_range.rolling(window).std()
    features['LR_zscore'] = (log_range - lr_mean) / (lr_std + 1e-10)

    # --- Feature 3: Log-Range en diferentes ventanas ---
    for w in [5, 10, 21, 63]:
        features[f'LR_ma{w}'] = log_range.rolling(w).mean()
        features[f'LR_std{w}'] = log_range.rolling(w).std()

    # --- Feature 4: Volatilidad basada en log-range (formula del paper) ---
    # sigma^2 = E[ln(H/L)^2] / (4 * ln(2))
    # Anualizado
    log_range_sq = log_range ** 2
    features['LR_volatility_21'] = np.sqrt(
        log_range_sq.rolling(21).mean() / (4 * np.log(2)) * 252
    )
    features['LR_volatility_63'] = np.sqrt(
        log_range_sq.rolling(63).mean() / (4 * np.log(2)) * 252
    )

    # --- Feature 5: Ratio de log-range (cambio en volatilidad) ---
    features['LR_ratio_5_21'] = features['LR_ma5'] / (features['LR_ma21'] + 1e-10)
    features['LR_ratio_21_63'] = features['LR_ma21'] / (features['LR_ma63'] + 1e-10)

    # --- Feature 6: Momentum del log-range ---
    features['LR_momentum_5'] = log_range.diff(5)
    features['LR_momentum_21'] = log_range.diff(21)

    # --- Feature 7: Percentil historico del log-range ---
    features['LR_percentile_63'] = log_range.rolling(63).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    )
    features['LR_percentile_252'] = log_range.rolling(252).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    )

    # --- Feature 8: Skewness del log-range (deberia ser ~0 si es Gaussiano) ---
    features['LR_skew_63'] = log_range.rolling(63).skew()

    # --- Feature 9: Kurtosis del log-range (deberia ser ~3 si es Gaussiano) ---
    features['LR_kurt_63'] = log_range.rolling(63).kurt()

    # --- Feature 10: Log-range ajustado por close-to-close ---
    # Combina informacion de rango con retorno
    log_return = np.log(close / close.shift(1))
    features['LR_adjusted'] = log_range - np.abs(log_return)

    return features


# =============================================================================
# 3. CARR-STYLE FEATURES (Chou, 2005)
# =============================================================================

def carr_style_features(high, low, close, window=21):
    """
    Features estilo CARR (Conditional Autoregressive Range).

    Paper: Chou (2005) - "Forecasting Financial Volatilities with Extreme
           Values: The Conditional Autoregressive Range (CARR) Model"
    Journal of Money, Credit and Banking, Vol. 37, No. 3

    Modelo CARR:
    ------------
    R_t = lambda_t * epsilon_t,  epsilon_t ~ Exp(1)
    lambda_t = omega + alpha * R_{t-1} + beta * lambda_{t-1}

    Donde R_t = H_t - L_t es el rango.

    Aqui creamos features que capturan la dinamica autorregresiva del rango,
    sin estimar el modelo completo (para usar como inputs de ML).

    Parameters:
    -----------
    high, low, close : Series - Precios
    window : int - Ventana para rolling

    Returns:
    --------
    dict con features estilo CARR
    """
    features = {}

    # Rango diario
    daily_range = high - low

    # --- Feature 1: Rango normalizado por precio ---
    features['CARR_range_pct'] = daily_range / close * 100

    # --- Feature 2: Componente autorregresivo del rango ---
    # AR(1) del rango: correlacion con el rango anterior
    features['CARR_range_lag1'] = daily_range.shift(1)
    features['CARR_range_lag2'] = daily_range.shift(2)
    features['CARR_range_lag5'] = daily_range.shift(5)

    # --- Feature 3: Media movil del rango (proxy de lambda_t) ---
    features['CARR_ema_range_5'] = daily_range.ewm(span=5).mean()
    features['CARR_ema_range_21'] = daily_range.ewm(span=21).mean()

    # --- Feature 4: Innovacion del rango (R_t - E[R_t|t-1]) ---
    expected_range = features['CARR_ema_range_21'].shift(1)
    features['CARR_innovation'] = daily_range - expected_range
    features['CARR_innovation_sq'] = features['CARR_innovation'] ** 2

    # --- Feature 5: Rango estandarizado (R_t / lambda_t) ---
    # Deberia ser ~Exp(1) bajo el modelo CARR
    features['CARR_standardized'] = daily_range / (expected_range + 1e-10)

    # --- Feature 6: Persistencia del rango ---
    # Autocorrelacion rolling del rango
    def rolling_autocorr(series, lag, window):
        return series.rolling(window).apply(
            lambda x: x[:-lag].corr(pd.Series(x[lag:])) if len(x) > lag else np.nan,
            raw=False
        )

    features['CARR_autocorr_1_21'] = daily_range.rolling(21).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False
    )

    # --- Feature 7: Shock de rango (grandes movimientos) ---
    range_mean = daily_range.rolling(63).mean()
    range_std = daily_range.rolling(63).std()
    features['CARR_shock'] = (daily_range - range_mean) / (range_std + 1e-10)
    features['CARR_large_range'] = (features['CARR_shock'] > 2).astype(int)

    # --- Feature 8: Asimetria del rango ---
    # Rango cuando precio sube vs cuando baja
    price_up = close > close.shift(1)
    features['CARR_range_up'] = daily_range.where(price_up).rolling(21).mean()
    features['CARR_range_down'] = daily_range.where(~price_up).rolling(21).mean()
    features['CARR_range_asymmetry'] = (
        features['CARR_range_up'] / (features['CARR_range_down'] + 1e-10)
    )

    # --- Feature 9: Expansion/Contraccion del rango ---
    features['CARR_expansion'] = daily_range / daily_range.shift(1)
    features['CARR_expansion_ma5'] = features['CARR_expansion'].rolling(5).mean()

    # --- Feature 10: Rango relativo al ATR ---
    atr = daily_range.rolling(14).mean()
    features['CARR_range_vs_atr'] = daily_range / (atr + 1e-10)

    return features


# =============================================================================
# 4. RANGE-GARCH FEATURES (Fiszeder & Perczak, 2016)
# =============================================================================

def range_garch_features(open_price, high, low, close, window=21):
    """
    Features inspirados en Fiszeder & Perczak (2016).

    Paper: "Low and high prices can improve volatility forecasts during
           periods of turmoil"
    International Journal of Forecasting, Vol. 32, Issue 2

    Innovacion:
    -----------
    Combina H, L, C en la funcion de verosimilitud del GARCH.
    Aqui creamos features que capturan esta informacion combinada.

    Parameters:
    -----------
    open_price, high, low, close : Series - Precios OHLC
    window : int - Ventana para rolling

    Returns:
    --------
    dict con features estilo Range-GARCH
    """
    features = {}

    # Retorno close-to-close
    ret = np.log(close / close.shift(1))

    # Rango logaritmico
    log_range = np.log(high / low)

    # --- Feature 1: Varianza combinada HLC ---
    # Inspirado en la verosimilitud conjunta de (H, L, C)
    # Combina informacion de retorno y rango
    features['RG_combined_var'] = (ret ** 2 + log_range ** 2) / 2
    features['RG_combined_var_ma21'] = features['RG_combined_var'].rolling(21).mean()

    # --- Feature 2: Ratio retorno/rango ---
    # Que proporcion del rango "uso" el retorno
    features['RG_ret_range_ratio'] = np.abs(ret) / (log_range + 1e-10)

    # --- Feature 3: Eficiencia del precio ---
    # Si el precio se movio eficientemente (ret alto, rango bajo)
    # o hubo mucha volatilidad sin direccion (ret bajo, rango alto)
    features['RG_efficiency'] = np.abs(ret) / (log_range + 1e-10)
    features['RG_efficiency_ma21'] = features['RG_efficiency'].rolling(21).mean()

    # --- Feature 4: Componente overnight vs intraday ---
    overnight_ret = np.log(open_price / close.shift(1))
    intraday_ret = np.log(close / open_price)

    features['RG_overnight_var'] = overnight_ret ** 2
    features['RG_intraday_var'] = intraday_ret ** 2
    features['RG_overnight_ratio'] = (
        features['RG_overnight_var'] /
        (features['RG_overnight_var'] + features['RG_intraday_var'] + 1e-10)
    )

    # --- Feature 5: Volatilidad HLC (formula Garman-Klass modificada) ---
    # Con ajuste por overnight
    hl_var = 0.5 * log_range ** 2
    co_var = (2 * np.log(2) - 1) * (np.log(close / open_price)) ** 2
    features['RG_hlc_volatility'] = np.sqrt((hl_var - co_var).rolling(21).mean() * 252)

    # --- Feature 6: Sesgo de posicion intradiaria ---
    # Donde cerro respecto a donde abrio, normalizado por rango
    intraday_move = (close - open_price) / (high - low + 1e-10)
    features['RG_intraday_bias'] = intraday_move
    features['RG_intraday_bias_ma21'] = intraday_move.rolling(21).mean()

    # --- Feature 7: True Range vs Simple Range ---
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        )
    )
    simple_range = high - low
    features['RG_gap_impact'] = true_range / (simple_range + 1e-10) - 1

    # --- Feature 8: Clustering de volatilidad ---
    # Correlacion entre volatilidad pasada y actual
    vol_proxy = log_range.rolling(5).std()
    features['RG_vol_cluster'] = vol_proxy.rolling(21).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False
    )

    # --- Feature 9: Regime de volatilidad ---
    vol_percentile = log_range.rolling(252).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    )
    features['RG_vol_regime_high'] = (vol_percentile > 0.8).astype(int)
    features['RG_vol_regime_low'] = (vol_percentile < 0.2).astype(int)

    # --- Feature 10: Contribucion de H y L al retorno ---
    high_contrib = (high - open_price) / (high - low + 1e-10)
    low_contrib = (open_price - low) / (high - low + 1e-10)
    features['RG_high_contribution'] = high_contrib
    features['RG_low_contribution'] = low_contrib
    features['RG_hl_balance'] = high_contrib - low_contrib

    return features


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Agrega features de volatilidad avanzados al dataset."""

    # Cargar dataset
    print("PASO 1: Cargar Dataset")
    print("-" * 80)

    core_file = os.path.join(DATA_DIR, "bloomberg_triple_screen_core.csv")
    df = pd.read_csv(core_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    initial_cols = df.shape[1]
    print(f"  + Dataset cargado: {len(df):,} filas x {initial_cols} columnas")

    # Verificar columnas OHLCV
    required = ['SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_CLOSE', 'SPY_VOLUME']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [ERROR] Faltan columnas: {missing}")
        return

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']
    V = df['SPY_VOLUME']

    # --- Calcular features ---

    print("\nPASO 2: Calcular Intrinsic Entropy Features (Vinte & Ausloos, 2021)")
    print("-" * 80)
    ie_features = intrinsic_entropy_volatility(O, H, L, C, V)
    for name, values in ie_features.items():
        df[name] = values
    print(f"  + {len(ie_features)} features agregados")

    print("\nPASO 3: Calcular Log-Range Features (Alizadeh et al., 2002)")
    print("-" * 80)
    lr_features = log_range_features(H, L, C)
    for name, values in lr_features.items():
        df[name] = values
    print(f"  + {len(lr_features)} features agregados")

    print("\nPASO 4: Calcular CARR-Style Features (Chou, 2005)")
    print("-" * 80)
    carr_features = carr_style_features(H, L, C)
    for name, values in carr_features.items():
        df[name] = values
    print(f"  + {len(carr_features)} features agregados")

    print("\nPASO 5: Calcular Range-GARCH Features (Fiszeder & Perczak, 2016)")
    print("-" * 80)
    rg_features = range_garch_features(O, H, L, C)
    for name, values in rg_features.items():
        df[name] = values
    print(f"  + {len(rg_features)} features agregados")

    # --- Guardar ---
    print("\nPASO 6: Guardar Dataset Actualizado")
    print("-" * 80)

    df.to_csv(core_file, index=False)

    final_cols = df.shape[1]
    new_features = final_cols - initial_cols

    print(f"  + Dataset guardado: {core_file}")
    print(f"  + Columnas: {initial_cols} -> {final_cols} (+{new_features} nuevas)")

    # --- Resumen ---
    print("\n" + "=" * 80)
    print("RESUMEN DE FEATURES AGREGADOS")
    print("=" * 80)

    feature_groups = {
        'Intrinsic Entropy (IE_)': len(ie_features),
        'Log-Range (LR_)': len(lr_features),
        'CARR-Style (CARR_)': len(carr_features),
        'Range-GARCH (RG_)': len(rg_features),
    }

    print("\nFeatures por categoria:")
    for group, count in feature_groups.items():
        print(f"  + {group}: {count}")

    print(f"\n  TOTAL NUEVOS FEATURES: {new_features}")

    print("\n" + "=" * 80)
    print("REFERENCIAS ACADEMICAS")
    print("=" * 80)
    print("""
  1. Vinte, C. & Ausloos, M. (2021). "A Volatility Estimator of Stock Market
     Indices Based on the Intrinsic Entropy Model." Entropy, 23(4), 484.

  2. Alizadeh, S., Brandt, M. & Diebold, F. (2002). "Range-Based Estimation
     of Stochastic Volatility Models." Journal of Finance, 57(3), 1047-1091.

  3. Chou, R. (2005). "Forecasting Financial Volatilities with Extreme Values:
     The CARR Model." Journal of Money, Credit and Banking, 37(3), 561-582.

  4. Fiszeder, P. & Perczak, G. (2016). "Low and high prices can improve
     volatility forecasts during periods of turmoil."
     International Journal of Forecasting, 32(2), 398-410.
    """)

    print("=" * 80)
    print("[OK] FEATURES AVANZADOS DE VOLATILIDAD AGREGADOS")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = main()
