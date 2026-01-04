    # -*- coding: utf-8 -*-
"""
================================================================================
BUILD DATASET - PIPELINE UNIFICADO PROFESIONAL
================================================================================

Pipeline de construccion de dataset para prediccion del S&P 500.
Combina metodologias de hedge funds cuantitativos de Wall Street.

FLUJO:
------
BLOOMBERG_RAW_DATA.csv (98 cols) --> build_dataset.py --> bloomberg_triple_screen_core.csv (~700 cols)

CATEGORIAS DE FEATURES:
-----------------------
1. BASE: Retornos, lags, rolling statistics
2. TECHNICAL: RSI, MACD, ADX, Bollinger, Stochastic (TA-Lib)
3. ELDER TRIPLE SCREEN: Weekly MACD, Force Index, Elder R, Impulse System
4. VOLATILITY CLASSIC: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
5. VOLATILITY CONTEMPORARY: Intrinsic Entropy, Log-Range, CARR, Range-GARCH
6. CROSS-ASSET: Correlaciones rolling con bonds, gold, oil, currencies
7. MACRO: Momentum de indicadores economicos
8. REGIME: Bull/Bear detection, VIX term structure
9. NORMALIZATION: Z-scores, rank percentiles (HSBC-ML style)

ANTI-DATA LEAKAGE:
------------------
- Todas las features usan SOLO informacion pasada (t-k, k>=1)
- Rolling windows miran hacia atras
- Target tiene shift(-1) aplicado correctamente
- Risk-free rate alineado temporalmente con forward returns
- Warmup period de 252 dias eliminado

REFERENCIAS ACADEMICAS:
-----------------------
- Elder, A. (1993). Trading for a Living
- Gu, Kelly & Xiu (2020). Empirical Asset Pricing via ML
- Vinte & Ausloos (2021). Intrinsic Entropy Volatility
- Alizadeh, Brandt & Diebold (2002). Range-Based SV Models
- Chou (2005). CARR Model
- Fiszeder & Perczak (2016). Range-GARCH

================================================================================
Autor: David Gonzalez Canon
Fecha: Diciembre 2025
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import json
import talib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

CONFIG = {
    # Periodos para lags
    'lag_periods': [1, 2, 3, 5, 10, 21, 63],

    # Ventanas para rolling statistics
    'rolling_windows': [5, 10, 21, 63, 126, 252],

    # Ventanas para correlaciones cross-asset
    'correlation_windows': [21, 63],

    # Periodo de warmup (dias a eliminar al inicio)
    'warmup_period': 252,

    # Umbral de NaN permitido por fila
    'max_nan_pct': 0.5,

    # Costos de transaccion (basis points)
    'transaction_cost_bps': 15,
}

# Mapeo de columnas Bloomberg a codigos internos
COLUMN_MAPPING = {
    # SPY OHLCV
    'SPY US Equity (CLOSE)': 'SPY_CLOSE',
    'SPY US Equity (OPEN)': 'SPY_OPEN',
    'SPY US Equity (HIGH)': 'SPY_HIGH',
    'SPY US Equity (LOW)': 'SPY_LOW',
    'SPY US Equity (VOLUME)': 'SPY_VOLUME',
    'BIL US Equity (Risk Free)': 'risk_free_rate_raw',

    # Market ETFs (M1-M18)
    'SPY US Equity': 'M1',
    'QQQ US Equity': 'M2',
    'IWM US Equity': 'M3',
    'EFA US Equity': 'M4',
    'EEM US Equity': 'M5',
    'VGK US Equity': 'M6',
    'EWJ US Equity': 'M7',
    'FXI US Equity': 'M8',
    'DXY Curncy': 'M9',
    'EURUSD Curncy': 'M10',
    'USDJPY Curncy': 'M11',
    'FNERTR Index': 'M12',
    'XLF US Equity': 'M13',
    'XLK US Equity': 'M14',
    'XLE US Equity': 'M15',
    'XLV US Equity': 'M16',
    'XLI US Equity': 'M17',
    'XLU US Equity': 'M18',

    # Economic Indicators (E1-E18)
    'GDP CQOQ Index': 'E1',
    'NAPMPMI Index': 'E2',
    'CONSSENT Index': 'E3',
    'NFP TCH Index': 'E4',
    'USURTOT Index': 'E5',
    'CPI YOY Index': 'E6',
    'PCE CRCH Index': 'E7',
    'RSTAMOM Index': 'E8',
    'LEI TOTL Index': 'E9',
    'INJCJC Index': 'E10',
    'NHSPSTOT Index': 'E11',
    'IP CHNG Index': 'E12',
    'CONCCONF Index': 'E13',
    'NAPMNMI Index': 'E14',
    'Extra E15': 'E15',
    'ETSLTOTL Index': 'E16',
    'GDP CYOY Index': 'E17',
    'CPTICHNG Index': 'E18',
    'PITLCHNG Index': 'E20',

    # Interest Rates (I1-I20)
    'FDTR Index': 'I1',
    'GB3 Govt': 'I2',
    'GB6 Govt': 'I3',
    'GB12 Govt': 'I4',
    'GT2 Govt': 'I5',
    'GT5 Govt': 'I6',
    'GT10 Govt': 'I7',
    'GT30 Govt': 'I8',
    'USGG10YR Index': 'I9',
    'USGG2YR Index': 'I10',
    'USYC2Y10 Index': 'I11',
    'USYC3M10 Index': 'I12',
    'US0003M Index': 'I13',
    'USSWAP10 Index': 'I14',
    'CDX IG CDSI GEN 5Y': 'I15',
    'CDX HY CDSI GEN 5Y': 'I16',
    'LF98TRUU Index': 'I17',
    'LF98OAS Index': 'I18',
    'LUACTRUU Index': 'I19',
    'LQD US Equity': 'I20',

    # Commodities (P1-P13)
    'CL1 Comdty': 'P1',
    'HO1 Comdty': 'P2',
    'GC1 Comdty': 'P4',
    'SI1 Comdty': 'P5',
    'GLD US Equity': 'P7',
    'SLV US Equity': 'P8',
    'USO US Equity': 'P9',
    'DBC US Equity': 'P10',
    'DBA US Equity': 'P11',
    'BCOMTR Index': 'P12',
    'GSCITR Index': 'P13',

    # Volatility (V1-V13)
    'VIX Index': 'V1',
    'VIX3M Index': 'V2',
    'VIX1M Index': 'V3',
    'VXV Index': 'V4',
    'VXEEM Index': 'V5',
    'VXEFA Index': 'V6',
    'GVZ Index': 'V7',
    'OVX Index': 'V8',
    'TYVIX Index': 'V9',
    'MOVE Index': 'V10',
    'CVIX Index': 'V11',
    'VIY1 Index': 'V12',
    'V2X Index': 'V13',

    # Sentiment (S1-S12)
    'AAII BULLISH Index': 'S1',
    'PUT Index': 'S4',
    'NYHL Index': 'S5',
    'TICK Index': 'S6',
    'TRIN Index': 'S7',
    'ADD Index': 'S8',
    'MCCL Index': 'S9',
    'MCSU Index': 'S10',
    'SRVOL Index': 'S11',
    'PCUSEQUI Index': 'S12',
}


# =============================================================================
# SECCION 1: FUNCIONES DE CARGA Y LIMPIEZA
# =============================================================================

def load_raw_data(filepath):
    """
    Carga datos raw de Bloomberg y renombra columnas.

    Parameters:
    -----------
    filepath : str
        Ruta al archivo BLOOMBERG_RAW_DATA.csv

    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas renombradas y fecha parseada
    """
    print("  Cargando datos raw...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Renombrar columnas
    for old_col, new_col in COLUMN_MAPPING.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    print(f"  + Filas: {len(df):,}")
    print(f"  + Columnas: {df.shape[1]}")
    print(f"  + Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

    return df


def calculate_risk_free_rate(df):
    """
    Calcula la tasa libre de riesgo diaria.

    IMPORTANTE: Los datos de Bloomberg contienen la TASA ANUALIZADA (%)
    directamente (no precios de ETF). Por ejemplo: 5.75 significa 5.75% anual.

    Conversion a tasa diaria:
    -------------------------
    tasa_diaria = tasa_anual / 100 / 252

    Ejemplo: 5.75% anual -> 5.75 / 100 / 252 = 0.0228% diario
    """
    if 'risk_free_rate_raw' in df.columns:
        # Convertir tasa anualizada (%) a tasa diaria (decimal)
        # risk_free_rate_raw esta en porcentaje (ej: 5.75 = 5.75%)
        df['risk_free_rate'] = df['risk_free_rate_raw'] / 100 / 252
    else:
        # Fallback: usar tasa de 3 meses anualizada / 252
        if 'I2' in df.columns:
            df['risk_free_rate'] = df['I2'] / 100 / 252
        else:
            df['risk_free_rate'] = 0.0

    return df


def calculate_target_variable(df):
    """
    Calcula la variable target: excess return del dia siguiente.

    ALINEACION TEMPORAL CORRECTA:
    -----------------------------
    En el dia t, el inversor DECIDE entre:
    - Invertir en SPY: obtendra forward_returns[t] = retorno de t a t+1
    - Invertir en T-Bills: obtendra risk_free_rate[t] = tasa conocida en t

    Por lo tanto:
    - forward_returns[t] = SPY[t+1]/SPY[t] - 1 (shift -1 del pct_change)
    - risk_free_rate[t] = tasa publicada en t (SIN shift, ya conocida en t)
    - target[t] = forward_returns[t] - risk_free_rate[t]

    ANTI-LEAKAGE:
    -------------
    - Las features en t usan datos hasta t (inclusive)
    - El target en t es el retorno FUTURO menos la tasa CONOCIDA en t
    - NO hay informacion del futuro en las features
    """
    # Forward returns: retorno del dia siguiente
    # En t, esto es el retorno de SPY de t a t+1
    df['forward_returns'] = df['SPY_CLOSE'].pct_change().shift(-1)

    # Risk-free rate: la tasa conocida en t (NO necesita shift)
    # La tasa publicada en t es la que puedes obtener si inviertes en t
    # Ya fue calculada en calculate_risk_free_rate(), no modificar aqui

    # Target: excess return sobre la tasa libre de riesgo
    df['market_forward_excess_returns'] = df['forward_returns'] - df['risk_free_rate']

    return df


# =============================================================================
# SECCION 2: FEATURES TECNICOS (TA-LIB)
# =============================================================================

def calculate_technical_features(df):
    """
    Calcula indicadores tecnicos profesionales usando TA-Lib.

    Indicadores:
    - RSI (7, 14, 21 dias)
    - MACD (line, signal, histogram)
    - ADX, +DI, -DI
    - Stochastic %K, %D
    - Williams %R
    - CCI
    - ATR
    - OBV
    - Bollinger Bands position
    """
    features = {}

    O = df['SPY_OPEN'].values
    H = df['SPY_HIGH'].values
    L = df['SPY_LOW'].values
    C = df['SPY_CLOSE'].values
    V = df['SPY_VOLUME'].values

    # RSI multiples periodos
    for period in [7, 14, 21]:
        features[f'SPY_RSI_{period}'] = talib.RSI(C, timeperiod=period)

    # MACD
    macd, signal, hist = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
    features['SPY_MACD'] = macd
    features['SPY_MACD_signal'] = signal
    features['SPY_MACD_hist'] = hist

    # ADX y Directional Indicators
    features['SPY_ADX_14'] = talib.ADX(H, L, C, timeperiod=14)
    features['SPY_PLUS_DI'] = talib.PLUS_DI(H, L, C, timeperiod=14)
    features['SPY_MINUS_DI'] = talib.MINUS_DI(H, L, C, timeperiod=14)
    features['SPY_DI_DIFF'] = features['SPY_PLUS_DI'] - features['SPY_MINUS_DI']

    # Stochastic
    slowk, slowd = talib.STOCH(H, L, C, fastk_period=14, slowk_period=3,
                                slowk_matype=0, slowd_period=3, slowd_matype=0)
    features['SPY_STOCH_K'] = slowk
    features['SPY_STOCH_D'] = slowd

    # Williams %R
    features['SPY_WILLIAMS_R'] = talib.WILLR(H, L, C, timeperiod=14)

    # CCI
    features['SPY_CCI_20'] = talib.CCI(H, L, C, timeperiod=20)

    # ATR
    features['SPY_ATR_14'] = talib.ATR(H, L, C, timeperiod=14)
    features['SPY_ATR_pct'] = features['SPY_ATR_14'] / C * 100

    # OBV
    features['SPY_OBV'] = talib.OBV(C, V.astype(float))

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(C, timeperiod=20, nbdevup=2, nbdevdn=2)
    band_width = upper - lower
    features['SPY_BB_position'] = (C - lower) / (band_width + 1e-10)
    features['SPY_BB_width'] = band_width / middle * 100

    # Convertir a Series con indice correcto
    for key in features:
        features[key] = pd.Series(features[key], index=df.index)

    return features


def calculate_momentum_features(df):
    """
    Calcula features de momentum y retornos.
    """
    features = {}
    close = df['SPY_CLOSE']

    # Retornos multiples horizontes
    for period in [1, 5, 21, 63, 126, 252]:
        features[f'SPY_return_{period}d'] = close.pct_change(period)

    # Momentum (ROC)
    for period in [5, 10, 21, 63]:
        features[f'SPY_momentum_{period}'] = close.pct_change(period)
        features[f'SPY_ROC_{period}'] = (close / close.shift(period) - 1) * 100

    # Distancia a medias moviles
    for window in [10, 21, 50, 100, 200]:
        ma = close.rolling(window=window).mean()
        features[f'SPY_dist_MA_{window}'] = (close - ma) / (ma + 1e-10) * 100

    # MA Crossovers
    for fast, slow in [(5, 10), (10, 21), (20, 50), (50, 200)]:
        ma_fast = close.rolling(window=fast).mean()
        ma_slow = close.rolling(window=slow).mean()
        features[f'SPY_MA_cross_{fast}_{slow}'] = np.sign(ma_fast - ma_slow)

    # Volatilidad rolling
    returns_1d = close.pct_change()
    for window in [5, 10, 21, 63, 126]:
        features[f'SPY_volatility_{window}'] = returns_1d.rolling(window=window).std() * np.sqrt(252)

    # Volatility ratios
    features['SPY_vol_ratio_5_21'] = features['SPY_volatility_5'] / (features['SPY_volatility_21'] + 1e-10)
    features['SPY_vol_ratio_21_63'] = features['SPY_volatility_21'] / (features['SPY_volatility_63'] + 1e-10)

    return features


def calculate_volume_features(df):
    """
    Features basados en volumen.
    """
    features = {}
    volume = df['SPY_VOLUME']

    # Volumen relativo
    for window in [20, 63]:
        vol_ma = volume.rolling(window=window).mean()
        features[f'SPY_volume_ratio_{window}'] = volume / (vol_ma + 1e-10)

    # Z-score del volumen
    vol_mean = volume.rolling(21).mean()
    vol_std = volume.rolling(21).std()
    features['SPY_volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-10)

    return features


# =============================================================================
# SECCION 3: ELDER TRIPLE SCREEN
# =============================================================================

def calculate_elder_triple_screen(df):
    """
    Sistema Triple Screen de Alexander Elder.

    Pantalla 1 (Semanal): Tendencia del mercado
    - MACD semanal para identificar tendencia

    Pantalla 2 (Diario): Osciladores contra tendencia
    - Force Index
    - Elder Ray (Bull/Bear Power)

    Pantalla 3: Entry signals
    - Impulse System
    """
    features = {}

    close = df['SPY_CLOSE']
    high = df['SPY_HIGH']
    low = df['SPY_LOW']
    volume = df['SPY_VOLUME']

    # === WEEKLY FEATURES (simulados con ventanas de 5 dias) ===

    # MACD semanal (usando datos diarios con periodos x5)
    weekly_close = close.rolling(5).mean()  # Proxy de cierre semanal

    ema_fast_w = weekly_close.ewm(span=12*5, adjust=False).mean()
    ema_slow_w = weekly_close.ewm(span=26*5, adjust=False).mean()
    features['W_MACD'] = ema_fast_w - ema_slow_w
    features['W_MACD_SIGNAL'] = features['W_MACD'].ewm(span=9*5, adjust=False).mean()
    features['W_MACD_HIST'] = features['W_MACD'] - features['W_MACD_SIGNAL']

    # Tendencia semanal
    features['W_TREND'] = np.sign(features['W_MACD_HIST'])
    features['W_TREND_STRENGTH'] = features['W_MACD_HIST'].abs()

    # EMA semanal
    features['W_EMA_13'] = close.ewm(span=13*5, adjust=False).mean()
    features['W_ABOVE_EMA'] = (close > features['W_EMA_13']).astype(int)

    # === DAILY FEATURES ===

    # Force Index (2 y 13 periodos)
    price_change = close.diff()
    features['D_FORCE_INDEX_2'] = (price_change * volume).ewm(span=2, adjust=False).mean()
    features['D_FORCE_INDEX_13'] = (price_change * volume).ewm(span=13, adjust=False).mean()

    # Normalizar Force Index
    fi_std = features['D_FORCE_INDEX_13'].rolling(63).std()
    features['D_FORCE_INDEX_2_NORM'] = features['D_FORCE_INDEX_2'] / (fi_std + 1e-10)
    features['D_FORCE_INDEX_13_NORM'] = features['D_FORCE_INDEX_13'] / (fi_std + 1e-10)

    # Elder Ray (Bull and Bear Power)
    ema_13 = close.ewm(span=13, adjust=False).mean()
    features['D_BULL_POWER'] = high - ema_13
    features['D_BEAR_POWER'] = low - ema_13
    features['D_ELDER_RAY'] = features['D_BULL_POWER'] + features['D_BEAR_POWER']

    # Impulse System
    ema_today = close.ewm(span=13, adjust=False).mean()
    ema_yesterday = ema_today.shift(1)
    macd_hist_today = pd.Series(talib.MACD(close.values)[2], index=close.index)
    macd_hist_yesterday = macd_hist_today.shift(1)

    # Impulse: +1 (green), -1 (red), 0 (blue)
    ema_rising = ema_today > ema_yesterday
    macd_rising = macd_hist_today > macd_hist_yesterday
    ema_falling = ema_today < ema_yesterday
    macd_falling = macd_hist_today < macd_hist_yesterday

    features['D_IMPULSE'] = np.where(
        ema_rising & macd_rising, 1,
        np.where(ema_falling & macd_falling, -1, 0)
    )

    # === TRIPLE SCREEN SIGNALS ===

    # Setup: Tendencia semanal alineada con oscilador diario
    weekly_bullish = features['W_TREND'] > 0
    weekly_bearish = features['W_TREND'] < 0

    daily_oversold = features['D_FORCE_INDEX_2_NORM'] < -1
    daily_overbought = features['D_FORCE_INDEX_2_NORM'] > 1

    features['TS_BUY_SETUP'] = (weekly_bullish & daily_oversold).astype(int)
    features['TS_SELL_SETUP'] = (weekly_bearish & daily_overbought).astype(int)

    # Signals mas estrictos
    features['TS_BUY_SIGNAL'] = (
        (features['W_TREND'] > 0) &
        (features['D_FORCE_INDEX_2_NORM'] < -0.5) &
        (features['D_BULL_POWER'] > 0)
    ).astype(int)

    features['TS_SELL_SIGNAL'] = (
        (features['W_TREND'] < 0) &
        (features['D_FORCE_INDEX_2_NORM'] > 0.5) &
        (features['D_BEAR_POWER'] < 0)
    ).astype(int)

    # Neutral
    features['TS_NEUTRAL'] = (
        (features['TS_BUY_SIGNAL'] == 0) &
        (features['TS_SELL_SIGNAL'] == 0)
    ).astype(int)

    # Strength score
    features['TS_STRENGTH'] = (
        features['W_TREND'] * 0.5 +
        features['D_IMPULSE'] * 0.3 +
        np.sign(features['D_ELDER_RAY']) * 0.2
    )

    # === ELDER PURE SIGNALS (siguiendo "Trading for a Living") ===
    # Referencia: Elder, A. (1993). Trading for a Living, Capítulos 7-8
    #
    # La lógica original de Elder para señales es:
    # - Comprar cuando Bear Power es NEGATIVO pero SUBIENDO (anticipación del giro)
    # - Vender cuando Bull Power es POSITIVO pero CAYENDO (anticipación del giro)
    # Esto es diferente de esperar confirmación (Bull Power > 0 para comprar)

    # Bear Power subiendo (divergencia alcista en pullback)
    bear_power_rising = features['D_BEAR_POWER'] > features['D_BEAR_POWER'].shift(1)
    # Bull Power cayendo (divergencia bajista en rally)
    bull_power_falling = features['D_BULL_POWER'] < features['D_BULL_POWER'].shift(1)

    # Señal Elder Pura de Compra:
    # - Tendencia semanal alcista (marea subiendo)
    # - Bear Power negativo (pullback - precio bajo EMA)
    # - Bear Power subiendo (el pullback está terminando)
    features['TS_BUY_SIGNAL_ELDER'] = (
        (features['W_TREND'] > 0) &           # Marea alcista
        (features['D_BEAR_POWER'] < 0) &      # En pullback
        bear_power_rising                      # Recuperándose
    ).astype(int)

    # Señal Elder Pura de Venta:
    # - Tendencia semanal bajista (marea bajando)
    # - Bull Power positivo (rally - precio sobre EMA)
    # - Bull Power cayendo (el rally está terminando)
    features['TS_SELL_SIGNAL_ELDER'] = (
        (features['W_TREND'] < 0) &           # Marea bajista
        (features['D_BULL_POWER'] > 0) &      # En rally
        bull_power_falling                     # Debilitándose
    ).astype(int)

    # Strength score Elder (pondera dirección de Bear/Bull Power)
    features['TS_STRENGTH_ELDER'] = (
        features['W_TREND'] * 0.4 +
        np.where(bear_power_rising, 0.3, -0.3) +
        np.where(bull_power_falling, -0.3, 0.3)
    )

    return features


# =============================================================================
# SECCION 4: VOLATILIDAD OHLC (CLASICA)
# =============================================================================

def calculate_classic_volatility(df):
    """
    Estimadores clasicos de volatilidad usando OHLC.

    Referencias:
    - Parkinson (1980): Journal of Business
    - Garman-Klass (1980): Journal of Business
    - Rogers-Satchell (1991): Annals of Applied Probability
    - Yang-Zhang (2000): Journal of Business
    """
    features = {}

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']

    log_hl = np.log(H / L)
    log_co = np.log(C / O)
    log_oc = np.log(O / C.shift(1))
    log_ho = np.log(H / O)
    log_lo = np.log(L / O)
    log_hc = np.log(H / C)
    log_lc = np.log(L / C)

    for window in [21, 63]:
        # Parkinson (1980)
        # sigma^2 = (1/4*ln2) * ln(H/L)^2
        parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)
        features[f'VOL_PARKINSON_{window}'] = np.sqrt(parkinson_var.rolling(window).mean() * 252)

        # Garman-Klass (1980)
        # sigma^2 = 0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        features[f'VOL_GARMAN_KLASS_{window}'] = np.sqrt(gk_var.rolling(window).mean() * 252)

        # Rogers-Satchell (1991) - robusto a drift
        # sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        rs_var = log_hc * log_ho + log_lc * log_lo
        features[f'VOL_ROGERS_SATCHELL_{window}'] = np.sqrt(rs_var.rolling(window).mean() * 252)

        # Yang-Zhang (2000) - combina overnight + intraday
        # Componente overnight
        overnight_var = log_oc ** 2
        overnight_mean = overnight_var.rolling(window).mean()

        # Componente open-close
        open_var = log_co ** 2
        open_mean = open_var.rolling(window).mean()

        # Componente Rogers-Satchell
        rs_mean = rs_var.rolling(window).mean()

        # Combinacion Yang-Zhang (k=0.34 es optimo)
        k = 0.34
        yz_var = overnight_mean + k * open_mean + (1 - k) * rs_mean
        features[f'VOL_YANG_ZHANG_{window}'] = np.sqrt(yz_var * 252)

    return features


# =============================================================================
# SECCION 5: VOLATILIDAD CONTEMPORANEA (PAPERS 2002-2021)
# =============================================================================

def calculate_intrinsic_entropy_features(df):
    """
    Intrinsic Entropy Model - Vinte & Ausloos (2021)
    Paper: Entropy (MDPI), Vol. 23, Issue 4

    Usa OHLC + Volumen con teoria de entropia.
    """
    features = {}

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']
    V = df['SPY_VOLUME']

    # Rango y posiciones
    price_range = H - L
    price_range = price_range.replace(0, np.nan)

    close_position = (C - L) / price_range
    open_position = (O - L) / price_range

    # Entropia de precio (aproximacion simplificada)
    # Basada en dispersion de O y C dentro del rango
    position_diff = np.abs(close_position - open_position)
    features['IE_price_entropy'] = 1 - position_diff  # Mayor cuando O y C cercanos
    features['IE_price_entropy_ma21'] = features['IE_price_entropy'].rolling(21).mean()

    # Entropia ponderada por volumen
    vol_relative = V / V.rolling(21).mean()
    features['IE_vol_weighted'] = features['IE_price_entropy'] * np.log1p(vol_relative)

    # Entropia del rango
    log_range = np.log(price_range)
    range_std = log_range.rolling(21).std()
    features['IE_range_entropy'] = 0.5 * np.log(2 * np.pi * np.e * range_std**2)

    # Concentracion de precio
    features['IE_price_concentration'] = 1 - np.abs(close_position - open_position)

    return features


def calculate_log_range_features(df):
    """
    Log-Range Features - Alizadeh, Brandt & Diebold (2002)
    Paper: Journal of Finance, Vol. 57, No. 3

    Hallazgo clave: ln(H/L) es aproximadamente Gaussiano.
    """
    features = {}

    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']

    log_range = np.log(H / L)
    features['LR_log_range'] = log_range

    # Z-score del log-range
    lr_mean = log_range.rolling(21).mean()
    lr_std = log_range.rolling(21).std()
    features['LR_zscore'] = (log_range - lr_mean) / (lr_std + 1e-10)

    # Medias moviles en diferentes ventanas
    for w in [5, 10, 21, 63]:
        features[f'LR_ma{w}'] = log_range.rolling(w).mean()
        features[f'LR_std{w}'] = log_range.rolling(w).std()

    # Volatilidad basada en log-range (formula del paper)
    log_range_sq = log_range ** 2
    features['LR_volatility_21'] = np.sqrt(log_range_sq.rolling(21).mean() / (4 * np.log(2)) * 252)
    features['LR_volatility_63'] = np.sqrt(log_range_sq.rolling(63).mean() / (4 * np.log(2)) * 252)

    # Ratios de log-range
    features['LR_ratio_5_21'] = features['LR_ma5'] / (features['LR_ma21'] + 1e-10)
    features['LR_ratio_21_63'] = features['LR_ma21'] / (features['LR_ma63'] + 1e-10)

    # Momentum del log-range
    features['LR_momentum_5'] = log_range.diff(5)
    features['LR_momentum_21'] = log_range.diff(21)

    # Percentiles historicos
    features['LR_percentile_63'] = log_range.rolling(63).rank(pct=True)
    features['LR_percentile_252'] = log_range.rolling(252).rank(pct=True)

    # Propiedades estadisticas (deberian ser ~Gaussianas)
    features['LR_skew_63'] = log_range.rolling(63).skew()
    features['LR_kurt_63'] = log_range.rolling(63).kurt()

    # Log-range ajustado por retorno
    log_return = np.log(C / C.shift(1))
    features['LR_adjusted'] = log_range - np.abs(log_return)

    return features


def calculate_carr_features(df):
    """
    CARR-Style Features - Chou (2005)
    Paper: Journal of Money, Credit and Banking, Vol. 37, No. 3

    Captura dinamica autorregresiva del rango.
    """
    features = {}

    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']

    daily_range = H - L

    # Rango normalizado por precio
    features['CARR_range_pct'] = daily_range / C * 100

    # Componente autorregresivo
    features['CARR_range_lag1'] = daily_range.shift(1)
    features['CARR_range_lag2'] = daily_range.shift(2)
    features['CARR_range_lag5'] = daily_range.shift(5)

    # EMA del rango (proxy de lambda_t en CARR)
    features['CARR_ema_range_5'] = daily_range.ewm(span=5).mean()
    features['CARR_ema_range_21'] = daily_range.ewm(span=21).mean()

    # Innovacion (sorpresa)
    expected_range = features['CARR_ema_range_21'].shift(1)
    features['CARR_innovation'] = daily_range - expected_range
    features['CARR_innovation_sq'] = features['CARR_innovation'] ** 2

    # Rango estandarizado
    features['CARR_standardized'] = daily_range / (expected_range + 1e-10)

    # Shock de rango
    range_mean = daily_range.rolling(63).mean()
    range_std = daily_range.rolling(63).std()
    features['CARR_shock'] = (daily_range - range_mean) / (range_std + 1e-10)
    features['CARR_large_range'] = (features['CARR_shock'] > 2).astype(int)

    # Asimetria up/down
    price_up = C > C.shift(1)
    features['CARR_range_up'] = daily_range.where(price_up).rolling(21).mean()
    features['CARR_range_down'] = daily_range.where(~price_up).rolling(21).mean()
    features['CARR_range_asymmetry'] = features['CARR_range_up'] / (features['CARR_range_down'] + 1e-10)

    # Expansion/Contraccion
    features['CARR_expansion'] = daily_range / daily_range.shift(1)

    # Rango vs ATR
    atr = daily_range.rolling(14).mean()
    features['CARR_range_vs_atr'] = daily_range / (atr + 1e-10)

    return features


def calculate_range_garch_features(df):
    """
    Range-GARCH Features - Fiszeder & Perczak (2016)
    Paper: International Journal of Forecasting, Vol. 32, Issue 2

    Combina H, L, C para mejorar forecasting en periodos de turbulencia.
    """
    features = {}

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']

    ret = np.log(C / C.shift(1))
    log_range = np.log(H / L)

    # Varianza combinada
    features['RG_combined_var'] = (ret ** 2 + log_range ** 2) / 2
    features['RG_combined_var_ma21'] = features['RG_combined_var'].rolling(21).mean()

    # Eficiencia del precio
    features['RG_efficiency'] = np.abs(ret) / (log_range + 1e-10)
    features['RG_efficiency_ma21'] = features['RG_efficiency'].rolling(21).mean()

    # Overnight vs Intraday
    overnight_ret = np.log(O / C.shift(1))
    intraday_ret = np.log(C / O)

    features['RG_overnight_var'] = overnight_ret ** 2
    features['RG_intraday_var'] = intraday_ret ** 2
    features['RG_overnight_ratio'] = features['RG_overnight_var'] / (
        features['RG_overnight_var'] + features['RG_intraday_var'] + 1e-10
    )

    # Volatilidad HLC
    hl_var = 0.5 * log_range ** 2
    co_var = (2 * np.log(2) - 1) * (np.log(C / O)) ** 2
    features['RG_hlc_volatility'] = np.sqrt((hl_var - co_var).rolling(21).mean() * 252)

    # Sesgo intradiario
    features['RG_intraday_bias'] = (C - O) / (H - L + 1e-10)
    features['RG_intraday_bias_ma21'] = features['RG_intraday_bias'].rolling(21).mean()

    # Gap impact
    true_range = np.maximum(H - L, np.maximum(np.abs(H - C.shift(1)), np.abs(L - C.shift(1))))
    features['RG_gap_impact'] = true_range / (H - L + 1e-10) - 1

    # Regimen de volatilidad
    vol_pct = log_range.rolling(252).rank(pct=True)
    features['RG_vol_regime_high'] = (vol_pct > 0.8).astype(int)
    features['RG_vol_regime_low'] = (vol_pct < 0.2).astype(int)

    # Balance H-L
    features['RG_high_contribution'] = (H - O) / (H - L + 1e-10)
    features['RG_low_contribution'] = (O - L) / (H - L + 1e-10)
    features['RG_hl_balance'] = features['RG_high_contribution'] - features['RG_low_contribution']

    return features


def calculate_microstructure_features(df):
    """
    Features de microestructura adicionales.
    """
    features = {}

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']
    V = df['SPY_VOLUME']

    # Rango intradiario
    daily_range = H - L
    features['INTRA_RANGE'] = daily_range
    features['INTRA_RANGE_PCT'] = daily_range / C * 100

    # True Range
    true_range = np.maximum(H - L, np.maximum(np.abs(H - C.shift(1)), np.abs(L - C.shift(1))))
    features['INTRA_TRUE_RANGE'] = true_range

    # Posicion del cierre en el rango
    features['INTRA_CLOSE_POSITION'] = (C - L) / (H - L + 1e-10)

    # Gap overnight
    features['GAP_OVERNIGHT'] = (O - C.shift(1)) / C.shift(1) * 100
    features['GAP_OVERNIGHT_ABS'] = np.abs(features['GAP_OVERNIGHT'])
    features['GAP_VS_ATR'] = np.abs(O - C.shift(1)) / (daily_range.rolling(14).mean() + 1e-10)
    features['GAP_DIRECTION'] = np.sign(O - C.shift(1))
    features['GAP_FILLED'] = (
        ((O > C.shift(1)) & (L <= C.shift(1))) |
        ((O < C.shift(1)) & (H >= C.shift(1)))
    ).astype(int)

    # Patrones de velas
    body = C - O
    features['CANDLE_BODY_PCT'] = np.abs(body) / (H - L + 1e-10)
    features['CANDLE_UPPER_SHADOW_PCT'] = (H - np.maximum(O, C)) / (H - L + 1e-10)
    features['CANDLE_LOWER_SHADOW_PCT'] = (np.minimum(O, C) - L) / (H - L + 1e-10)
    features['CANDLE_DIRECTION'] = np.sign(body)
    features['CANDLE_DOJI'] = (features['CANDLE_BODY_PCT'] < 0.1).astype(int)

    # Volumen relativo
    features['VOLUME_RELATIVE_21'] = V / V.rolling(21).mean()
    features['VOLUME_RELATIVE_63'] = V / V.rolling(63).mean()

    # Producto rango-volumen (proxy de actividad)
    features['VOLUME_RANGE_PRODUCT'] = features['INTRA_RANGE_PCT'] * features['VOLUME_RELATIVE_21']

    # Rango normalizado
    range_ma = features['INTRA_RANGE_PCT'].rolling(21).mean()
    range_std = features['INTRA_RANGE_PCT'].rolling(21).std()
    features['RANGE_PCT_MA_21'] = range_ma
    features['RANGE_PCT_STD_21'] = range_std
    features['RANGE_PCT_ZSCORE'] = (features['INTRA_RANGE_PCT'] - range_ma) / (range_std + 1e-10)

    return features


# =============================================================================
# SECCION 6: CROSS-ASSET FEATURES
# =============================================================================

def calculate_cross_asset_features(df):
    """
    Correlaciones rolling y spreads entre activos.
    """
    features = {}

    spy_ret = df['SPY_CLOSE'].pct_change()

    # Correlaciones con otros activos
    cross_assets = {
        'TLT': 'I7',    # Bonds 10Y
        'GLD': 'P4',    # Gold
        'OIL': 'P1',    # Oil
        'DXY': 'M9',    # Dollar
        'VIX': 'V1',    # VIX
    }

    for name, col in cross_assets.items():
        if col in df.columns:
            asset_ret = df[col].pct_change()
            for window in CONFIG['correlation_windows']:
                corr = spy_ret.rolling(window).corr(asset_ret)
                features[f'corr_SPY_{name}_{window}d'] = corr

    return features


def calculate_vix_features(df):
    """
    VIX Term Structure y features relacionados.
    """
    features = {}

    if 'V1' not in df.columns:
        return features

    vix = df['V1']

    # VIX basicos
    features['VIX_zscore_21'] = (vix - vix.rolling(21).mean()) / (vix.rolling(21).std() + 1e-10)
    features['VIX_zscore_63'] = (vix - vix.rolling(63).mean()) / (vix.rolling(63).std() + 1e-10)

    # Regimen
    features['VIX_regime_high'] = (vix > vix.rolling(252).quantile(0.75)).astype(int)
    features['VIX_regime_low'] = (vix < vix.rolling(252).quantile(0.25)).astype(int)

    # Cambios
    features['VIX_change_1d'] = vix.diff(1)
    features['VIX_change_5d'] = vix.diff(5)
    features['VIX_change_21d'] = vix.diff(21)

    # Term structure (si hay futuros disponibles)
    if 'V2' in df.columns:  # VIX3M
        features['vix_3m_spot_ratio'] = df['V2'] / (vix + 1e-10)
        features['vix_contango'] = (df['V2'] > vix).astype(int)
        features['vix_roll_yield'] = (df['V2'] / vix - 1) * 12

    if 'V12' in df.columns:  # VIX Futures
        features['VIX_F1'] = df['V12']
        features['vix_f1_spot_ratio'] = df['V12'] / (vix + 1e-10)

    return features


def calculate_yield_curve_features(df):
    """
    Features de curva de rendimiento.
    """
    features = {}

    # Spread 10Y - 2Y
    if 'I7' in df.columns and 'I5' in df.columns:
        features['yield_spread_10_2'] = df['I7'] - df['I5']
        features['yield_spread_10_2_change_5d'] = features['yield_spread_10_2'].diff(5)
        features['yield_spread_10_2_change_21d'] = features['yield_spread_10_2'].diff(21)
        features['yield_curve_inverted'] = (features['yield_spread_10_2'] < 0).astype(int)

    # Spread 10Y - 3M
    if 'I7' in df.columns and 'I2' in df.columns:
        features['yield_spread_10_3m'] = df['I7'] - df['I2']
        features['yield_spread_10_3m_inverted'] = (features['yield_spread_10_3m'] < 0).astype(int)

    # Butterfly (2-5-10)
    if all(col in df.columns for col in ['I5', 'I6', 'I7']):
        features['yield_butterfly'] = df['I5'] + df['I7'] - 2 * df['I6']

    return features


def calculate_credit_features(df):
    """
    Features de spreads de credito.
    """
    features = {}

    # High Yield spread
    if 'I16' in df.columns:
        hy = df['I16']
        features['hy_spread_chg_5d'] = hy.diff(5)
        features['hy_spread_chg_21d'] = hy.diff(21)
        features['hy_spread_zscore'] = (hy - hy.rolling(63).mean()) / (hy.rolling(63).std() + 1e-10)
        features['hy_spread_regime_high'] = (hy > hy.rolling(252).quantile(0.8)).astype(int)

    # Investment Grade spread
    if 'I15' in df.columns:
        ig = df['I15']
        features['ig_spread_chg_5d'] = ig.diff(5)
        features['ig_spread_chg_21d'] = ig.diff(21)

    # Ratio HY/IG
    if 'I16' in df.columns and 'I15' in df.columns:
        features['spread_ratio_hy_ig'] = df['I16'] / (df['I15'] + 1e-10)

    return features


def calculate_sector_features(df):
    """
    Features de dispersion sectorial.
    """
    features = {}

    # Sector columns (M13-M18)
    sector_cols = [f'M{i}' for i in range(13, 19) if f'M{i}' in df.columns]

    if len(sector_cols) >= 3:
        sector_rets = pd.DataFrame()
        for col in sector_cols:
            sector_rets[col] = df[col].pct_change()

        # Dispersion
        features['sector_dispersion_1d'] = sector_rets.std(axis=1)
        features['sector_dispersion_5d'] = sector_rets.rolling(5).mean().std(axis=1)
        features['sector_dispersion_21d'] = sector_rets.rolling(21).mean().std(axis=1)

        # Rotation
        features['sector_rotation_range'] = sector_rets.max(axis=1) - sector_rets.min(axis=1)

        # Breadth
        features['sectors_positive'] = (sector_rets > 0).sum(axis=1) / len(sector_cols)

    return features


# =============================================================================
# SECCION 7: MACRO Y ECONOMIC FEATURES
# =============================================================================

def calculate_economic_features(df):
    """
    Features basados en indicadores economicos.
    """
    features = {}

    econ_indicators = {
        'E1': 'gdp',
        'E2': 'pmi',
        'E4': 'nfp',
        'E5': 'unemployment',
        'E6': 'cpi',
        'E12': 'ip',
        'E13': 'consumer_conf',
    }

    for col, name in econ_indicators.items():
        if col in df.columns:
            series = df[col]

            # Momentum
            features[f'{name}_momentum_21d'] = series.diff(21)
            features[f'{name}_momentum_63d'] = series.diff(63)

            # Trend
            ma_short = series.rolling(5).mean()
            ma_long = series.rolling(21).mean()
            features[f'{name}_trend'] = np.sign(ma_short - ma_long)

    return features


# =============================================================================
# SECCION 8: REGIME Y INTERACTION FEATURES
# =============================================================================

def calculate_regime_features(df, all_features):
    """
    Indicadores de regimen de mercado.
    """
    features = {}

    signals = []

    # 1. Trend signal (precio vs MA200)
    if 'SPY_dist_MA_200' in all_features:
        signals.append((all_features['SPY_dist_MA_200'] > 0).astype(int))

    # 2. Momentum signal
    if 'SPY_momentum_21' in all_features:
        signals.append((all_features['SPY_momentum_21'] > 0).astype(int))

    # 3. VIX regime
    if 'V1' in df.columns:
        signals.append((df['V1'] < 20).astype(int))

    # 4. Breadth
    if 'sectors_positive' in all_features:
        signals.append((all_features['sectors_positive'] > 0.6).astype(int))

    if signals:
        signals_df = pd.concat(signals, axis=1)
        features['regime_score'] = signals_df.mean(axis=1)
        features['regime_bull'] = (features['regime_score'] > 0.7).astype(int)
        features['regime_bear'] = (features['regime_score'] < 0.3).astype(int)
        features['regime_neutral'] = (
            (features['regime_score'] >= 0.3) &
            (features['regime_score'] <= 0.7)
        ).astype(int)

    return features


def calculate_interaction_features(df, all_features):
    """
    Interaction terms entre features.
    """
    features = {}

    # VIX x Yield Curve
    if 'V1' in df.columns and 'yield_spread_10_2' in all_features:
        features['vix_x_curve'] = df['V1'] * all_features['yield_spread_10_2']

    # Momentum x Volatility Regime
    if 'SPY_momentum_21' in all_features and 'V1' in df.columns:
        vix_high = (df['V1'] > 20).astype(int)
        features['momentum_low_vol'] = all_features['SPY_momentum_21'] * (1 - vix_high)
        features['momentum_high_vol'] = all_features['SPY_momentum_21'] * vix_high

    return features


# =============================================================================
# SECCION 9: NORMALIZATION (Z-SCORE, RANK PERCENTILE)
# =============================================================================

def calculate_zscore_features(df, features_dict):
    """
    Z-score normalization para indicadores tecnicos.
    Estilo HSBC-ML/Hedge Fund.
    """
    zscore_features = {}

    indicators_to_normalize = [
        'SPY_RSI_14', 'SPY_MACD', 'SPY_ADX_14',
        'SPY_CCI_20', 'SPY_ATR_pct', 'SPY_momentum_21'
    ]

    for col in indicators_to_normalize:
        if col in features_dict:
            series = features_dict[col]
            for window in [21, 63]:
                mean = series.rolling(window).mean()
                std = series.rolling(window).std()
                zscore_features[f'{col}_zscore_{window}d'] = (series - mean) / (std + 1e-10)

    return zscore_features


def calculate_lagged_features(df, important_cols):
    """
    Lagged features para variables importantes.
    """
    features = {}

    for col in important_cols:
        if col in df.columns:
            for lag in CONFIG['lag_periods']:
                features[f'{col}_lag{lag}'] = df[col].shift(lag)

    return features


def calculate_rolling_stats(df, important_cols):
    """
    Rolling statistics para variables importantes.
    """
    features = {}

    for col in important_cols:
        if col in df.columns:
            series = df[col]
            for window in [5, 10, 21, 63]:
                features[f'{col}_roll_mean_{window}'] = series.rolling(window).mean()
                features[f'{col}_roll_std_{window}'] = series.rolling(window).std()
                features[f'{col}_roll_min_{window}'] = series.rolling(window).min()
                features[f'{col}_roll_max_{window}'] = series.rolling(window).max()

    return features


def calculate_variable_transformations(df, cols):
    """
    Transformaciones estandar para variables: pct_change, zscore, sma20_ratio, momentum.
    Aplicado a todas las variables de interes.
    """
    features = {}

    for col in cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Pct change
        features[f'{col}_pct_change'] = series.pct_change()

        # Z-score (21 dias)
        mean = series.rolling(21).mean()
        std = series.rolling(21).std()
        features[f'{col}_zscore'] = (series - mean) / (std + 1e-10)

        # Ratio vs SMA20
        sma20 = series.rolling(20).mean()
        features[f'{col}_sma20_ratio'] = series / (sma20 + 1e-10)

        # Momentum 5 dias
        features[f'{col}_momentum_5'] = series.diff(5)

    return features


def calculate_factor_features(df):
    """
    Factor features: Value vs Growth, Size, Tech.
    """
    features = {}

    # Value vs Growth (XLK=Tech/Growth vs XLF=Financials/Value proxy)
    if 'M14' in df.columns and 'M13' in df.columns:
        # M14 = XLK (Tech), M13 = XLF (Financials)
        value_ret = df['M13'].pct_change()
        growth_ret = df['M14'].pct_change()

        features['value_growth_spread_1d'] = value_ret - growth_ret
        features['value_growth_spread_21d'] = df['M13'].pct_change(21) - df['M14'].pct_change(21)
        features['value_momentum_21d'] = df['M13'].pct_change(21)
        features['growth_momentum_21d'] = df['M14'].pct_change(21)

    # Small vs Large (IWM vs SPY)
    if 'M3' in df.columns and 'SPY_CLOSE' in df.columns:
        features['small_large_spread_21d'] = df['M3'].pct_change(21) - df['SPY_CLOSE'].pct_change(21)

    # Tech momentum
    if 'M14' in df.columns:
        features['tech_momentum_21d'] = df['M14'].pct_change(21)
        if 'SPY_CLOSE' in df.columns:
            features['tech_vs_spy_21d'] = df['M14'].pct_change(21) - df['SPY_CLOSE'].pct_change(21)

    return features


def calculate_additional_elder_features(df):
    """
    Features adicionales del sistema Elder que faltaban.
    """
    features = {}

    close = df['SPY_CLOSE']
    high = df['SPY_HIGH']
    low = df['SPY_LOW']

    # Daily EMAs
    features['D_EMA13'] = close.ewm(span=13, adjust=False).mean()
    features['D_EMA26'] = close.ewm(span=26, adjust=False).mean()

    # Price vs EMA
    features['D1'] = (close > features['D_EMA13']).astype(int)
    features['D2'] = (close > features['D_EMA26']).astype(int)

    # Daily indicators adicionales
    daily_range = high - low
    features['D3'] = daily_range / close * 100  # Range %
    features['D4'] = (close - low) / (high - low + 1e-10)  # Close position
    features['D5'] = close.pct_change()  # Daily return

    # Force Index overbought/oversold
    price_change = close.diff()
    volume = df['SPY_VOLUME']
    force_index = (price_change * volume).ewm(span=2, adjust=False).mean()
    fi_ma = force_index.rolling(63).mean()
    fi_std = force_index.rolling(63).std()

    features['D_FORCE_OVERBOUGHT'] = (force_index > fi_ma + 2 * fi_std).astype(int)
    features['D_FORCE_OVERSOLD'] = (force_index < fi_ma - 2 * fi_std).astype(int)

    # Weekly features adicionales
    weekly_close = close.rolling(5).mean()
    features['W_ATR_PCT'] = daily_range.rolling(5).mean() / close * 100
    features['W_PRICE_VS_EMA13'] = close / features['D_EMA13'].rolling(5).mean()

    macd_hist = pd.Series(talib.MACD(close.values)[2], index=close.index)
    features['W_MACD_HIST_DIRECTION'] = np.sign(macd_hist.diff(5))

    # Triple Screen signals adicionales
    features['TS_ALIGNMENT'] = (
        (features['D1'] == 1) &
        (features['D2'] == 1) &
        (macd_hist > 0)
    ).astype(int)

    # Combined strength
    features['TS_COMBINED_STRENGTH'] = (
        features['D1'] * 0.3 +
        features['D2'] * 0.3 +
        np.sign(macd_hist) * 0.4
    )

    # TS_SIGNAL legacy
    features['TS_SIGNAL'] = np.where(
        features['TS_COMBINED_STRENGTH'] > 0.5, 1,
        np.where(features['TS_COMBINED_STRENGTH'] < -0.5, -1, 0)
    )

    return features


def calculate_additional_vix_features(df):
    """
    VIX features adicionales que faltaban.
    """
    features = {}

    if 'V1' not in df.columns:
        return features

    vix = df['V1']

    # Z-scores adicionales
    for window in [5, 10, 21, 63]:
        mean = vix.rolling(window).mean()
        std = vix.rolling(window).std()
        features[f'V1_zscore_{window}'] = (vix - mean) / (std + 1e-10)

    # VIX Futures (si existen)
    for i, col in enumerate(['V12', 'V2', 'V3', 'V4'], start=1):
        if col in df.columns:
            features[f'VIX_F{i}'] = df[col]

    # Term structure ratios
    if 'V2' in df.columns:
        features['vix_term_ratio_VIX_F1'] = df['V2'] / (vix + 1e-10)
        features['vix_contango_VIX_F1'] = (df['V2'] > vix).astype(int)
        features['vix_roll_yield_VIX_F1'] = (df['V2'] / vix - 1) * 12

        if 'V3' in df.columns:
            features['vix_term_slope_VIX_F1'] = df['V3'] - df['V2']

    # VIX x curve inverted
    if 'yield_curve_inverted' in df.columns:
        features['vix_x_curve_inverted'] = vix * df['yield_curve_inverted']

    # Volatility surface features
    if 'V5' in df.columns:  # VVIX
        features['vvix_vix_ratio'] = df['V5'] / (vix + 1e-10)
        vvix_mean = df['V5'].rolling(63).mean()
        vvix_std = df['V5'].rolling(63).std()
        features['vvix_zscore'] = (df['V5'] - vvix_mean) / (vvix_std + 1e-10)

    # SKEW features
    if 'V6' in df.columns:
        skew = df['V6']
        skew_mean = skew.rolling(63).mean()
        skew_std = skew.rolling(63).std()
        features['skew_zscore'] = (skew - skew_mean) / (skew_std + 1e-10)
        features['skew_high'] = (skew > 130).astype(int)

    # Vol risk premium
    if 'SPY_volatility_21' in df.columns or 'SPY_CLOSE' in df.columns:
        realized_vol = df['SPY_CLOSE'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        features['vol_risk_premium'] = vix - realized_vol
        vrp_mean = features['vol_risk_premium'].rolling(63).mean()
        vrp_std = features['vol_risk_premium'].rolling(63).std()
        features['vol_premium_zscore'] = (features['vol_risk_premium'] - vrp_mean) / (vrp_std + 1e-10)

    return features


def calculate_additional_yield_features(df):
    """
    Yield curve features adicionales.
    """
    features = {}

    # Yield spread changes adicionales
    if 'I7' in df.columns and 'I5' in df.columns:
        spread = df['I7'] - df['I5']
        features['yield_spread_10_2_change_1d'] = spread.diff(1)

        spread_mean = spread.rolling(63).mean()
        spread_std = spread.rolling(63).std()
        features['yield_spread_10_2_zscore'] = (spread - spread_mean) / (spread_std + 1e-10)

    return features


def calculate_additional_credit_features(df):
    """
    Credit features adicionales.
    """
    features = {}

    # Credit x VIX interaction
    if 'I16' in df.columns and 'V1' in df.columns:
        features['credit_x_vix'] = df['I16'] * df['V1']

    # Spread ratio change
    if 'I16' in df.columns and 'I15' in df.columns:
        ratio = df['I16'] / (df['I15'] + 1e-10)
        features['spread_ratio_chg'] = ratio.pct_change(21)

    return features


def calculate_macro_momentum_features(df):
    """
    Macro momentum features adicionales.
    """
    features = {}

    # Claims momentum
    if 'E10' in df.columns:
        features['claims_momentum_4w'] = -df['E10'].diff(4)

    # Inflation momentum
    if 'E6' in df.columns:
        features['inflation_momentum'] = df['E6'].diff(21)

    if 'E7' in df.columns:
        features['core_inflation_momentum'] = df['E7'].diff(21)

    return features


def calculate_microstructure_additional(df):
    """
    Microstructure features adicionales.
    """
    features = {}

    O = df['SPY_OPEN']
    H = df['SPY_HIGH']
    L = df['SPY_LOW']
    C = df['SPY_CLOSE']
    V = df['SPY_VOLUME']

    # Close location value (CLV)
    features['CLV'] = ((C - L) - (H - C)) / (H - L + 1e-10)
    features['CLV_MA_21'] = features['CLV'].rolling(21).mean()

    # Gap features adicionales
    gap = O - C.shift(1)
    atr = (H - L).rolling(14).mean()
    features['GAP_LARGE_FLAG'] = (np.abs(gap) > atr).astype(int)
    features['GAP_LARGE_FREQ_21'] = features['GAP_LARGE_FLAG'].rolling(21).sum()

    # Close position MA
    close_pos = (C - L) / (H - L + 1e-10)
    features['CLOSE_POSITION_MA_21'] = close_pos.rolling(21).mean()

    # Price efficiency
    daily_ret = np.abs(C.pct_change())
    intraday_range = (H - L) / C
    features['PRICE_EFFICIENCY'] = daily_ret / (intraday_range + 1e-10)

    # Buying/Selling pressure
    features['BUYING_PRESSURE_21'] = (close_pos > 0.6).rolling(21).mean()
    features['SELLING_PRESSURE_21'] = (close_pos < 0.4).rolling(21).mean()

    # ATR adicionales
    features['INTRA_ATR_5'] = (H - L).rolling(5).mean()
    features['INTRA_ATR_21'] = (H - L).rolling(21).mean()

    # Open position
    features['INTRA_OPEN_POSITION'] = (O - L) / (H - L + 1e-10)

    return features


def calculate_other_market_features(df):
    """
    Otros features de mercado que faltaban.
    """
    features = {}

    # DXY alternativo
    if 'M9' in df.columns:
        features['DXY_ALT'] = df['M9']

    # MOVE Index
    if 'V10' in df.columns:
        features['MOVE_INDEX'] = df['V10']

    # CVIX
    if 'V11' in df.columns:
        features['CVIX'] = df['V11']

    # Commodities con nombres alternativos
    commodity_mapping = {
        'P4': 'P9_GOLD',
        'P5': 'P10_SILVER',
        'P1': 'P12_OIL',
        'P12': 'P11_CMDTY_IDX',
    }

    for orig, alt in commodity_mapping.items():
        if orig in df.columns:
            features[alt] = df[orig]

    # REITs y otros
    if 'M12' in df.columns:
        features['M12_REIT'] = df['M12']

    # China
    if 'M8' in df.columns:
        features['S11_CHINA'] = df['M8']

    # HY Index
    if 'I16' in df.columns:
        features['S4_HY_INDEX'] = df['I16']

    # Volatility ratio
    if 'VOL_YANG_ZHANG_21' in df.columns and 'VOL_YANG_ZHANG_63' in df.columns:
        pass  # Ya calculado
    else:
        # Calcular ratio de volatilidad
        O, H, L, C = df['SPY_OPEN'], df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE']
        log_hl = np.log(H / L)
        vol_21 = np.sqrt((log_hl ** 2).rolling(21).mean() / (4 * np.log(2)) * 252)
        vol_63 = np.sqrt((log_hl ** 2).rolling(63).mean() / (4 * np.log(2)) * 252)
        features['VOL_RATIO_21_63'] = vol_21 / (vol_63 + 1e-10)

    return features


# =============================================================================
# SECCION 10: VALIDACION ANTI-LEAKAGE
# =============================================================================

def validate_no_leakage(df):
    """
    Valida que no hay data leakage en el dataset.
    """
    print("\n  Validando anti-leakage...")

    issues = []

    # 1. Verificar que forward_returns tiene shift(-1)
    spy_ret_check = df['SPY_CLOSE'].pct_change().shift(-1)
    corr = df['forward_returns'].corr(spy_ret_check)
    if corr < 0.99:
        issues.append(f"forward_returns correlation: {corr:.4f} (esperado >0.99)")

    # 2. Verificar que target es forward_returns - risk_free_rate
    target_check = df['forward_returns'] - df['risk_free_rate']
    target_diff = (df['market_forward_excess_returns'] - target_check).abs().mean()
    if target_diff > 1e-10:
        issues.append(f"Target inconsistency: {target_diff:.2e}")

    # 3. Verificar que no hay features con informacion futura
    # (esto requiere inspeccion manual, pero verificamos NaN patterns)
    last_row_nan = df.iloc[-1].isna().sum()
    expected_nan = 3  # forward_returns, risk_free_rate, target
    if last_row_nan < expected_nan:
        issues.append(f"Ultima fila tiene {last_row_nan} NaN (esperado >={expected_nan})")

    if issues:
        print("  [WARN] Posibles problemas de leakage:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] No se detectaron problemas de leakage")

    return len(issues) == 0


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def build_dataset():
    """
    Pipeline principal de construccion del dataset.
    """
    print("=" * 80)
    print("BUILD DATASET - PIPELINE UNIFICADO PROFESIONAL")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()

    # =========================================================================
    # PASO 1: Cargar datos raw
    # =========================================================================
    print("PASO 1: Cargar Datos Raw de Bloomberg")
    print("-" * 80)

    raw_file = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
    df = load_raw_data(raw_file)

    # =========================================================================
    # PASO 2: Calcular target variable
    # =========================================================================
    print("\nPASO 2: Calcular Target Variable")
    print("-" * 80)

    df = calculate_risk_free_rate(df)
    df = calculate_target_variable(df)
    print(f"  + forward_returns calculado con shift(-1)")
    print(f"  + risk_free_rate alineado temporalmente")
    print(f"  + Target: market_forward_excess_returns")

    # =========================================================================
    # PASO 3: Features tecnicos
    # =========================================================================
    print("\nPASO 3: Features Tecnicos (TA-Lib)")
    print("-" * 80)

    all_features = {}

    tech_features = calculate_technical_features(df)
    all_features.update(tech_features)
    print(f"  + Technical indicators: {len(tech_features)}")

    momentum_features = calculate_momentum_features(df)
    all_features.update(momentum_features)
    print(f"  + Momentum features: {len(momentum_features)}")

    volume_features = calculate_volume_features(df)
    all_features.update(volume_features)
    print(f"  + Volume features: {len(volume_features)}")

    # =========================================================================
    # PASO 4: Elder Triple Screen
    # =========================================================================
    print("\nPASO 4: Elder Triple Screen")
    print("-" * 80)

    elder_features = calculate_elder_triple_screen(df)
    all_features.update(elder_features)
    print(f"  + Elder Triple Screen: {len(elder_features)}")

    # =========================================================================
    # PASO 5: Volatilidad OHLC Clasica
    # =========================================================================
    print("\nPASO 5: Volatilidad OHLC Clasica")
    print("-" * 80)

    classic_vol = calculate_classic_volatility(df)
    all_features.update(classic_vol)
    print(f"  + Classic volatility: {len(classic_vol)}")

    # =========================================================================
    # PASO 6: Volatilidad Contemporanea (Papers 2002-2021)
    # =========================================================================
    print("\nPASO 6: Volatilidad Contemporanea")
    print("-" * 80)

    ie_features = calculate_intrinsic_entropy_features(df)
    all_features.update(ie_features)
    print(f"  + Intrinsic Entropy (2021): {len(ie_features)}")

    lr_features = calculate_log_range_features(df)
    all_features.update(lr_features)
    print(f"  + Log-Range (2002): {len(lr_features)}")

    carr_features = calculate_carr_features(df)
    all_features.update(carr_features)
    print(f"  + CARR-Style (2005): {len(carr_features)}")

    rg_features = calculate_range_garch_features(df)
    all_features.update(rg_features)
    print(f"  + Range-GARCH (2016): {len(rg_features)}")

    micro_features = calculate_microstructure_features(df)
    all_features.update(micro_features)
    print(f"  + Microstructure: {len(micro_features)}")

    # =========================================================================
    # PASO 7: Cross-Asset Features
    # =========================================================================
    print("\nPASO 7: Cross-Asset Features")
    print("-" * 80)

    cross_features = calculate_cross_asset_features(df)
    all_features.update(cross_features)
    print(f"  + Cross-asset correlations: {len(cross_features)}")

    vix_features = calculate_vix_features(df)
    all_features.update(vix_features)
    print(f"  + VIX features: {len(vix_features)}")

    yield_features = calculate_yield_curve_features(df)
    all_features.update(yield_features)
    print(f"  + Yield curve: {len(yield_features)}")

    credit_features = calculate_credit_features(df)
    all_features.update(credit_features)
    print(f"  + Credit: {len(credit_features)}")

    sector_features = calculate_sector_features(df)
    all_features.update(sector_features)
    print(f"  + Sector: {len(sector_features)}")

    # =========================================================================
    # PASO 8: Macro Features
    # =========================================================================
    print("\nPASO 8: Macro Features")
    print("-" * 80)

    econ_features = calculate_economic_features(df)
    all_features.update(econ_features)
    print(f"  + Economic: {len(econ_features)}")

    # =========================================================================
    # PASO 9: Regime & Interaction
    # =========================================================================
    print("\nPASO 9: Regime & Interaction Features")
    print("-" * 80)

    regime_features = calculate_regime_features(df, all_features)
    all_features.update(regime_features)
    print(f"  + Regime: {len(regime_features)}")

    interaction_features = calculate_interaction_features(df, all_features)
    all_features.update(interaction_features)
    print(f"  + Interaction: {len(interaction_features)}")

    # =========================================================================
    # PASO 10: Normalization
    # =========================================================================
    print("\nPASO 10: Normalization (Z-Score)")
    print("-" * 80)

    zscore_features = calculate_zscore_features(df, all_features)
    all_features.update(zscore_features)
    print(f"  + Z-score normalized: {len(zscore_features)}")

    # =========================================================================
    # PASO 11: Features Adicionales
    # =========================================================================
    print("\nPASO 11: Features Adicionales")
    print("-" * 80)

    # Factor features
    factor_features = calculate_factor_features(df)
    all_features.update(factor_features)
    print(f"  + Factor (Value/Growth/Size): {len(factor_features)}")

    # Elder adicionales
    elder_add = calculate_additional_elder_features(df)
    all_features.update(elder_add)
    print(f"  + Elder adicionales: {len(elder_add)}")

    # VIX adicionales
    vix_add = calculate_additional_vix_features(df)
    all_features.update(vix_add)
    print(f"  + VIX adicionales: {len(vix_add)}")

    # Yield adicionales
    yield_add = calculate_additional_yield_features(df)
    all_features.update(yield_add)
    print(f"  + Yield adicionales: {len(yield_add)}")

    # Credit adicionales
    credit_add = calculate_additional_credit_features(df)
    all_features.update(credit_add)
    print(f"  + Credit adicionales: {len(credit_add)}")

    # Macro adicionales
    macro_add = calculate_macro_momentum_features(df)
    all_features.update(macro_add)
    print(f"  + Macro adicionales: {len(macro_add)}")

    # Microstructure adicionales
    micro_add = calculate_microstructure_additional(df)
    all_features.update(micro_add)
    print(f"  + Microstructure adicionales: {len(micro_add)}")

    # Otros mercado
    other_mkt = calculate_other_market_features(df)
    all_features.update(other_mkt)
    print(f"  + Otros mercado: {len(other_mkt)}")

    # =========================================================================
    # PASO 12: Transformaciones de Variables
    # =========================================================================
    print("\nPASO 12: Transformaciones de Variables")
    print("-" * 80)

    # Variables para transformar (pct_change, zscore, sma20_ratio, momentum)
    transform_cols = [f'I{i}' for i in range(10, 21)]  # I10-I20
    transform_cols += ['I1', 'I2']
    transform_cols = [c for c in transform_cols if c in df.columns]

    var_transforms = calculate_variable_transformations(df, transform_cols)
    all_features.update(var_transforms)
    print(f"  + Variable transformations: {len(var_transforms)}")

    # =========================================================================
    # PASO 13: Lagged & Rolling Features (EXPANDIDO)
    # =========================================================================
    print("\nPASO 13: Lagged & Rolling Features")
    print("-" * 80)

    # Lista expandida de variables importantes para lags
    important_cols_lag = [
        'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
        'V1', 'M3', 'M4', 'M5', 'M13', 'M18',
        'E1', 'E4', 'E8'
    ]
    important_cols_lag = [c for c in important_cols_lag if c in df.columns]

    lagged_features = calculate_lagged_features(df, important_cols_lag)
    all_features.update(lagged_features)
    print(f"  + Lagged: {len(lagged_features)}")

    # Rolling stats para variables principales
    important_cols_roll = ['I1', 'I2', 'I4', 'I6', 'I8', 'V1']
    important_cols_roll = [c for c in important_cols_roll if c in df.columns]

    rolling_features = calculate_rolling_stats(df, important_cols_roll)
    all_features.update(rolling_features)
    print(f"  + Rolling stats: {len(rolling_features)}")

    # =========================================================================
    # PASO 14: Merge all features
    # =========================================================================
    print("\nPASO 12: Consolidar Features")
    print("-" * 80)

    # Agregar todas las features al dataframe
    for name, values in all_features.items():
        if isinstance(values, pd.Series):
            df[name] = values.values
        else:
            df[name] = values

    print(f"  + Total features agregados: {len(all_features)}")

    # =========================================================================
    # PASO 13: Limpieza de Datos (Wall Street Standards)
    # =========================================================================
    print("\nPASO 13: Limpieza de Datos")
    print("-" * 80)

    initial_cols = len(df.columns)
    initial_rows = len(df)

    # 13.1: Eliminar columnas con 100% NaN (no aportan nada)
    full_nan_cols = df.columns[df.isna().all()].tolist()
    if full_nan_cols:
        df = df.drop(columns=full_nan_cols)
        print(f"  + Columnas 100% NaN eliminadas: {len(full_nan_cols)}")

    # 13.2: Reemplazar infinitos con NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        print(f"  + Infinitos reemplazados por NaN: {inf_count}")

    # 13.3: Eliminar columnas con >50% NaN (poca informacion)
    meta_cols = ['date', 'date_id', 'SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME',
                 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    high_nan_cols = [c for c in feature_cols if df[c].isna().sum() / len(df) > 0.5]
    if high_nan_cols:
        df = df.drop(columns=high_nan_cols)
        print(f"  + Columnas >50% NaN eliminadas: {len(high_nan_cols)}")

    # 13.4: Eliminar filas con demasiados NaN (>30%)
    feature_cols = [c for c in df.columns if c not in meta_cols]
    null_pct = df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    df = df[null_pct < CONFIG['max_nan_pct']].reset_index(drop=True)

    # 13.5: Eliminar filas sin target
    df = df.dropna(subset=['market_forward_excess_returns']).reset_index(drop=True)

    # 13.6: Crear date_id
    df['date_id'] = range(len(df))

    final_cols = len(df.columns)
    final_rows = len(df)

    print(f"  + Columnas: {initial_cols} -> {final_cols} (eliminadas: {initial_cols - final_cols})")
    print(f"  + Filas: {initial_rows:,} -> {final_rows:,} (eliminadas: {initial_rows - final_rows})")

    # =========================================================================
    # PASO 14: Validacion
    # =========================================================================
    print("\nPASO 14: Validacion Anti-Leakage")
    print("-" * 80)

    validate_no_leakage(df)

    # =========================================================================
    # PASO 15: Guardar
    # =========================================================================
    print("\nPASO 15: Guardar Dataset")
    print("-" * 80)

    # Ordenar columnas
    meta_cols = ['date_id', 'date', 'SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME',
                 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    other_cols = sorted([c for c in df.columns if c not in meta_cols])
    ordered_cols = meta_cols + other_cols
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    df_final = df[ordered_cols].copy()

    # Guardar
    output_path = os.path.join(DATA_DIR, "bloomberg_triple_screen_core.csv")
    df_final.to_csv(output_path, index=False)
    print(f"  + Dataset guardado: {output_path}")

    # Contar features
    feature_count = len([c for c in df_final.columns if c not in meta_cols])

    # Guardar metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'script': 'build_dataset.py',
        'n_observations': len(df_final),
        'n_features': feature_count,
        'date_range': {
            'start': str(df_final['date'].min()),
            'end': str(df_final['date'].max())
        },
        'target_column': 'market_forward_excess_returns',
        'feature_categories': {
            'technical': len(tech_features),
            'momentum': len(momentum_features),
            'volume': len(volume_features),
            'elder_triple_screen': len(elder_features),
            'volatility_classic': len(classic_vol),
            'volatility_entropy': len(ie_features),
            'volatility_log_range': len(lr_features),
            'volatility_carr': len(carr_features),
            'volatility_range_garch': len(rg_features),
            'microstructure': len(micro_features),
            'cross_asset': len(cross_features),
            'vix': len(vix_features),
            'yield_curve': len(yield_features),
            'credit': len(credit_features),
            'sector': len(sector_features),
            'economic': len(econ_features),
            'regime': len(regime_features),
            'interaction': len(interaction_features),
            'zscore': len(zscore_features),
            'lagged': len(lagged_features),
            'rolling': len(rolling_features),
        },
        'anti_leakage_measures': [
            'All features use only past information (t-k, k>=1)',
            'Rolling windows look backward only',
            'Forward returns calculated with shift(-1)',
            'Risk-free rate aligned with forward returns period',
            'Warmup period rows removed',
        ],
        'references': [
            'Elder, A. (1993). Trading for a Living',
            'Parkinson (1980). Extreme Value Method',
            'Garman-Klass (1980). OHLC Volatility',
            'Rogers-Satchell (1991). Drift-Independent',
            'Yang-Zhang (2000). Overnight + Intraday',
            'Vinte & Ausloos (2021). Intrinsic Entropy',
            'Alizadeh, Brandt & Diebold (2002). Log-Range SV',
            'Chou (2005). CARR Model',
            'Fiszeder & Perczak (2016). Range-GARCH',
        ],
        'config': CONFIG,
    }

    metadata_path = os.path.join(DATA_DIR, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN DEL PIPELINE")
    print("=" * 80)

    print(f"""
  DATASET GENERADO:
  -----------------
  Archivo: {output_path}
  Observaciones: {len(df_final):,}
  Features: {feature_count}
  Periodo: {df_final['date'].min().date()} a {df_final['date'].max().date()}

  TARGET:
  -------
  Columna: market_forward_excess_returns
  Media diaria: {df_final['market_forward_excess_returns'].mean()*100:.4f}%
  Std diaria: {df_final['market_forward_excess_returns'].std()*100:.4f}%
  % Positivos: {(df_final['market_forward_excess_returns'] > 0).mean()*100:.1f}%

  CATEGORIAS DE FEATURES:
  -----------------------""")

    for cat, count in metadata['feature_categories'].items():
        print(f"  + {cat}: {count}")

    print(f"""
  FLUJO DEL PIPELINE:
  -------------------
  BLOOMBERG_RAW_DATA.csv (98 cols)
           |
           v
     build_dataset.py
           |
           v
  bloomberg_triple_screen_core.csv ({feature_count} features)
           |
           v
  ml_pipeline_darts_extended.py (entrenamiento)
""")

    print("=" * 80)
    print("[OK] PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)

    return df_final


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = build_dataset()
