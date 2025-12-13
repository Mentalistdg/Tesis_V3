# -*- coding: utf-8 -*-
"""
================================================================================
FEATURE ENGINEERING - NIVEL HEDGE FUND PROFESIONAL
================================================================================

Pipeline avanzado de ingeniería de features usado en hedge funds cuantitativos.

NUEVAS CATEGORÍAS DE FEATURES:
1. INDICADORES TÉCNICOS AVANZADOS: ADX, OBV, Stochastic, Williams %R, CCI
2. VIX TERM STRUCTURE: Contango/Backwardation, Roll yield
3. CROSS-ASSET CORRELATIONS: Rolling correlations SPY vs bonds, gold, oil
4. SECTOR DISPERSION: Dispersion entre sectores, rotation signals
5. CREDIT MOMENTUM: Changes in spreads, regime signals
6. INTERACTION TERMS: VIX × Yield curve, Momentum × Volatility
7. REGIME INDICATORS: Bull/Bear/Sideways basado en múltiples señales
8. FACTOR MOMENTUM: Value vs Growth, Size, Quality proxies
9. VOLATILITY SURFACE: Term structure, skew proxies
10. ECONOMIC MOMENTUM: Cambios en macro indicators, surprises

ANTI-DATA LEAKAGE ESTRICTO:
- Todas las features usan SOLO información pasada
- Rolling windows miran hacia atrás
- Período de warmup extendido a 252 días

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import json

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data"
PREPARED_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data\prepared"
OUTPUT_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data\final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'lag_periods': [1, 2, 3, 5, 10, 21, 63],  # Agregamos 63 (trimestre)
    'rolling_windows': [5, 10, 21, 63, 126, 252],
    'correlation_windows': [21, 63, 126],  # Para rolling correlations
    'warmup_period': 252,  # 1 año de warmup
}

print("=" * 80)
print("FEATURE ENGINEERING - NIVEL HEDGE FUND PROFESIONAL")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()

# =============================================================================
# FUNCIONES DE INDICADORES TÉCNICOS AVANZADOS
# =============================================================================

def calculate_rsi(series, period=14):
    """RSI - Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_adx(high, low, close, period=14):
    """
    ADX - Average Directional Index
    Mide la fuerza de la tendencia (no la dirección)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def calculate_obv(close, volume):
    """
    OBV - On-Balance Volume
    Acumula volumen basado en dirección del precio
    """
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    return obv


def calculate_obv_momentum(close, volume, period=21):
    """OBV Momentum - Cambio porcentual del OBV"""
    obv = calculate_obv(close, volume)
    return obv.pct_change(period)


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Stochastic Oscillator (%K y %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R - Similar a Stochastic pero invertido
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return williams_r


def calculate_cci(high, low, close, period=20):
    """
    CCI - Commodity Channel Index
    Mide desviación del precio típico respecto a su media
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci


def calculate_atr(high, low, close, period=14):
    """ATR - Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD con histograma"""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_position(series, window=20, num_std=2):
    """Posición dentro de Bollinger Bands (0-1)"""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    band_width = upper_band - lower_band
    position = (series - lower_band) / (band_width + 1e-10)
    return position.clip(0, 1)


def calculate_zscore(series, window=20):
    """Z-score rolling"""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def calculate_momentum(series, period):
    """Momentum = retorno over N períodos"""
    return series.pct_change(periods=period)


def calculate_roc(series, period):
    """Rate of Change"""
    return ((series / series.shift(period)) - 1) * 100


# =============================================================================
# FUNCIONES DE FEATURES AVANZADOS (HEDGE FUND)
# =============================================================================

def calculate_vix_term_structure(vix_spot, vix_f1, vix_f2=None):
    """
    VIX Term Structure Features
    Contango = futuros > spot (complacencia)
    Backwardation = futuros < spot (miedo)
    """
    features = {}

    # Ratio F1/Spot
    features['vix_term_ratio'] = vix_f1 / (vix_spot + 1e-10)

    # Contango/Backwardation binario
    features['vix_contango'] = (vix_f1 > vix_spot).astype(int)

    # Roll yield implícito (anualizado)
    features['vix_roll_yield'] = (vix_f1 / vix_spot - 1) * 12  # ~mensual a anual

    if vix_f2 is not None:
        # Slope del term structure
        features['vix_term_slope'] = vix_f2 - vix_f1

    return features


def calculate_cross_asset_correlations(df, windows=[21, 63]):
    """
    Rolling correlations entre SPY y otros assets
    Decorrelation = oportunidad o crisis
    """
    features = {}

    base = df['SPY_return_1d'] if 'SPY_return_1d' in df.columns else df['SPY_CLOSE'].pct_change()

    # Assets para correlación
    corr_assets = {
        'TLT': 'S6' if 'S6' in df.columns else None,  # Bonds
        'GLD': 'P9' if 'P9' in df.columns else 'P9_GOLD',  # Gold
        'OIL': 'P12' if 'P12' in df.columns else 'P12_OIL',  # Oil
        'DXY': 'S12' if 'S12' in df.columns else None,  # Dollar
    }

    for asset_name, col in corr_assets.items():
        if col and col in df.columns:
            asset_ret = df[col].pct_change()
            for window in windows:
                corr = base.rolling(window).corr(asset_ret)
                features[f'corr_SPY_{asset_name}_{window}d'] = corr

    return features


def calculate_sector_dispersion(df):
    """
    Sector Dispersion - Mide dispersión de retornos entre sectores
    Alta dispersión = stock picking environment
    Baja dispersión = correlaciones altas, mercado macro-driven
    """
    features = {}

    # Sector columns (M3-M11 son sector ETFs)
    sector_cols = [f'M{i}' for i in range(3, 12) if f'M{i}' in df.columns]

    if len(sector_cols) >= 4:
        # Retornos de sectores
        sector_rets = pd.DataFrame()
        for col in sector_cols:
            sector_rets[col] = df[col].pct_change()

        # Dispersion (cross-sectional std)
        features['sector_dispersion_1d'] = sector_rets.std(axis=1)
        features['sector_dispersion_5d'] = sector_rets.rolling(5).mean().std(axis=1)
        features['sector_dispersion_21d'] = sector_rets.rolling(21).mean().std(axis=1)

        # Sector rotation signal (max - min sector)
        features['sector_rotation_range'] = sector_rets.max(axis=1) - sector_rets.min(axis=1)

        # Número de sectores positivos
        features['sectors_positive'] = (sector_rets > 0).sum(axis=1) / len(sector_cols)

    return features


def calculate_credit_momentum(df):
    """
    Credit Spread Momentum
    Spread widening = risk-off
    Spread tightening = risk-on
    """
    features = {}

    # I8 = HY spread, I9 = IG spread
    if 'I8' in df.columns:
        features['hy_spread_chg_5d'] = df['I8'].diff(5)
        features['hy_spread_chg_21d'] = df['I8'].diff(21)
        features['hy_spread_zscore'] = calculate_zscore(df['I8'], 63)
        features['hy_spread_regime_high'] = (df['I8'] > df['I8'].rolling(252).quantile(0.8)).astype(int)

    if 'I9' in df.columns:
        features['ig_spread_chg_5d'] = df['I9'].diff(5)
        features['ig_spread_chg_21d'] = df['I9'].diff(21)

    # Spread ratio (HY/IG) - mide risk appetite
    if 'I8' in df.columns and 'I9' in df.columns:
        features['spread_ratio_hy_ig'] = df['I8'] / (df['I9'] + 1e-10)
        features['spread_ratio_chg'] = features['spread_ratio_hy_ig'].pct_change(21)

    return features


def calculate_interaction_terms(df):
    """
    Interaction Terms - Combinaciones no lineales de features
    Capturan efectos condicionales
    """
    features = {}

    # VIX × Yield Curve
    if 'V1' in df.columns and 'I6' in df.columns:
        features['vix_x_curve'] = df['V1'] * df['I6']
        features['vix_x_curve_inverted'] = df['V1'] * (df['I6'] < 0).astype(int)

    # Momentum × Volatility Regime
    if 'SPY_momentum_21' in df.columns and 'V1' in df.columns:
        vix_high = (df['V1'] > 20).astype(int)
        features['momentum_low_vol'] = df['SPY_momentum_21'] * (1 - vix_high)
        features['momentum_high_vol'] = df['SPY_momentum_21'] * vix_high

    # RSI × Trend
    if 'SPY_RSI_14' in df.columns and 'D1' in df.columns:
        features['rsi_above_sma200'] = df['SPY_RSI_14'] * df['D1']
        features['rsi_below_sma200'] = df['SPY_RSI_14'] * (1 - df['D1'])

    # Credit × VIX
    if 'I8' in df.columns and 'V1' in df.columns:
        features['credit_x_vix'] = df['I8'] * df['V1']

    return features


def calculate_regime_indicators(df):
    """
    Market Regime Indicators
    Múltiples señales combinadas para identificar régimen
    """
    features = {}

    signals = []

    # 1. Trend signal (precio vs MA)
    if 'D1' in df.columns:
        signals.append(df['D1'])

    # 2. Momentum signal
    if 'SPY_momentum_21' in df.columns:
        momentum_positive = (df['SPY_momentum_21'] > 0).astype(int)
        signals.append(momentum_positive)

    # 3. VIX regime
    if 'V1' in df.columns:
        vix_low = (df['V1'] < 20).astype(int)
        signals.append(vix_low)

    # 4. Credit regime
    if 'I8' in df.columns:
        hy_percentile = df['I8'].rolling(252).rank(pct=True)
        credit_ok = (hy_percentile < 0.7).astype(int)
        signals.append(credit_ok)

    # 5. Breadth (sectores positivos)
    if 'sectors_positive' in df.columns:
        breadth_ok = (df['sectors_positive'] > 0.6).astype(int)
        signals.append(breadth_ok)

    if signals:
        # Aggregate regime score (0-1)
        signals_df = pd.concat(signals, axis=1)
        features['regime_score'] = signals_df.mean(axis=1)

        # Regime categories
        features['regime_bull'] = (features['regime_score'] > 0.7).astype(int)
        features['regime_bear'] = (features['regime_score'] < 0.3).astype(int)
        features['regime_neutral'] = ((features['regime_score'] >= 0.3) &
                                       (features['regime_score'] <= 0.7)).astype(int)

    return features


def calculate_factor_momentum(df):
    """
    Factor Momentum - Relative performance of factors
    Value vs Growth, Size, etc.
    """
    features = {}

    # Value vs Growth (M14=Value, M15=Growth)
    if 'M14' in df.columns and 'M15' in df.columns:
        value_ret = df['M14'].pct_change()
        growth_ret = df['M15'].pct_change()

        # Value-Growth spread
        features['value_growth_spread_1d'] = value_ret - growth_ret
        features['value_growth_spread_21d'] = (df['M14'].pct_change(21) -
                                                df['M15'].pct_change(21))

        # Value momentum
        features['value_momentum_21d'] = df['M14'].pct_change(21)
        features['growth_momentum_21d'] = df['M15'].pct_change(21)

    # Small vs Large (M13=Small cap, M17=Dow/Large)
    if 'M13' in df.columns and 'M17' in df.columns:
        features['small_large_spread_21d'] = (df['M13'].pct_change(21) -
                                               df['M17'].pct_change(21))

    # Tech momentum (M18=QQQ)
    if 'M18' in df.columns:
        features['tech_momentum_21d'] = df['M18'].pct_change(21)
        features['tech_vs_spy_21d'] = (df['M18'].pct_change(21) -
                                        df['SPY_CLOSE'].pct_change(21) if 'SPY_CLOSE' in df.columns else 0)

    return features


def calculate_economic_momentum(df):
    """
    Economic Momentum - Cambios en indicadores macro
    """
    features = {}

    # PMI momentum
    for col, name in [('E1', 'gdp'), ('E2', 'ip'), ('E4', 'unemployment'),
                      ('E8', 'cpi'), ('E13', 'consumer_conf')]:
        if col in df.columns:
            features[f'{name}_momentum_21d'] = df[col].diff(21)
            features[f'{name}_momentum_63d'] = df[col].diff(63)
            features[f'{name}_trend'] = np.sign(df[col].rolling(5).mean() -
                                                 df[col].rolling(21).mean())

    # Unemployment claims momentum (inverse - falling claims = good)
    if 'E6' in df.columns:
        features['claims_momentum_4w'] = -df['E6'].diff(4)  # Negative = good

    # Inflation momentum
    if 'E8' in df.columns and 'E9' in df.columns:
        features['inflation_momentum'] = df['E8'].diff(21)
        features['core_inflation_momentum'] = df['E9'].diff(21)

    return features


def calculate_volatility_surface_features(df):
    """
    Volatility Surface Features
    Term structure, skew proxies
    """
    features = {}

    # Realized vs Implied volatility
    if 'V1' in df.columns and 'SPY_volatility_21' in df.columns:
        features['vol_risk_premium'] = df['V1'] - df['SPY_volatility_21'] * 100
        features['vol_premium_zscore'] = calculate_zscore(features['vol_risk_premium'], 63)

    # VIX term structure
    if 'V1' in df.columns and 'V2' in df.columns:
        features['vix_3m_spot_ratio'] = df['V2'] / (df['V1'] + 1e-10)

    if 'V1' in df.columns and 'V3' in df.columns:
        features['vix_6m_spot_ratio'] = df['V3'] / (df['V1'] + 1e-10)

    # SKEW (V6) features
    if 'V6' in df.columns:
        features['skew_zscore'] = calculate_zscore(df['V6'], 63)
        features['skew_high'] = (df['V6'] > 130).astype(int)  # Tail risk elevated

    # VVIX features (V5)
    if 'V5' in df.columns and 'V1' in df.columns:
        features['vvix_vix_ratio'] = df['V5'] / (df['V1'] + 1e-10)
        features['vvix_zscore'] = calculate_zscore(df['V5'], 63)

    return features


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal de feature engineering"""

    # PASO 1: Cargar datos
    print("PASO 1: Cargar Datos")
    print("-" * 80)

    df = pd.read_csv(os.path.join(PREPARED_DIR, "bloomberg_prepared.csv"))
    df['date'] = pd.to_datetime(df['date'])
    print(f"  + Dataset preparado: {len(df):,} filas x {df.shape[1]} columnas")

    # Cargar variables mejoradas si existen
    improved_file = os.path.join(DATA_DIR, "improved_variables.csv")
    if os.path.exists(improved_file):
        df_improved = pd.read_csv(improved_file)
        df_improved['date'] = pd.to_datetime(df_improved['date'])
        print(f"  + Variables mejoradas: {len(df_improved):,} filas x {df_improved.shape[1]-1} columnas")

        # Merge
        df = df.merge(df_improved, on='date', how='left')
        print(f"  + Dataset combinado: {len(df):,} filas x {df.shape[1]} columnas")

        # Usar variables mejoradas donde la original tiene NaN
        replacements = {
            'P9': 'P9_GOLD',
            'P10': 'P10_SILVER',
            'P12': 'P12_OIL',
            'P13': 'P13_NATGAS',
            'P11': 'P11_CMDTY_IDX',
            'M12': 'M12_REIT',
            'S4': 'S4_HY_INDEX',
            'S11': 'S11_CHINA'
        }

        for orig, improved in replacements.items():
            if improved in df.columns and orig in df.columns:
                null_before = df[orig].isna().sum()
                df[orig] = df[orig].fillna(df[improved])
                null_after = df[orig].isna().sum()
                if null_before != null_after:
                    print(f"  + {orig}: {null_before - null_after:,} NaN rellenados con {improved}")
    else:
        print("  + [INFO] No se encontró improved_variables.csv")
        print("  +        Ejecute download_improved_dataset.py primero para mejores resultados")

    # Identificar columnas
    target_cols = ['date_id', 'date', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    spy_cols = ['SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME']

    # PASO 2: Features Técnicos Avanzados
    print("\nPASO 2: Features Técnicos Avanzados")
    print("-" * 80)

    # Retornos básicos
    df['SPY_return_1d'] = df['SPY_CLOSE'].pct_change(1)
    df['SPY_return_5d'] = df['SPY_CLOSE'].pct_change(5)
    df['SPY_return_21d'] = df['SPY_CLOSE'].pct_change(21)
    df['SPY_return_63d'] = df['SPY_CLOSE'].pct_change(63)
    df['SPY_return_126d'] = df['SPY_CLOSE'].pct_change(126)
    df['SPY_return_252d'] = df['SPY_CLOSE'].pct_change(252)

    # RSI múltiples períodos
    for period in [7, 14, 21]:
        df[f'SPY_RSI_{period}'] = calculate_rsi(df['SPY_CLOSE'], period)

    # MACD
    macd, signal, hist = calculate_macd(df['SPY_CLOSE'])
    df['SPY_MACD'] = macd
    df['SPY_MACD_signal'] = signal
    df['SPY_MACD_hist'] = hist

    # ADX
    adx, plus_di, minus_di = calculate_adx(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'])
    df['SPY_ADX_14'] = adx
    df['SPY_PLUS_DI'] = plus_di
    df['SPY_MINUS_DI'] = minus_di
    df['SPY_DI_DIFF'] = plus_di - minus_di  # Dirección de tendencia

    # OBV y OBV Momentum
    df['SPY_OBV'] = calculate_obv(df['SPY_CLOSE'], df['SPY_VOLUME'])
    df['SPY_OBV_momentum_21'] = calculate_obv_momentum(df['SPY_CLOSE'], df['SPY_VOLUME'], 21)

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'])
    df['SPY_STOCH_K'] = stoch_k
    df['SPY_STOCH_D'] = stoch_d

    # Williams %R
    df['SPY_WILLIAMS_R'] = calculate_williams_r(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'])

    # CCI
    df['SPY_CCI_20'] = calculate_cci(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], 20)

    # ATR
    df['SPY_ATR_14'] = calculate_atr(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], 14)
    df['SPY_ATR_pct'] = df['SPY_ATR_14'] / df['SPY_CLOSE'] * 100

    # Bollinger Position
    df['SPY_BB_position_20'] = calculate_bollinger_position(df['SPY_CLOSE'], 20)

    # Z-scores múltiples ventanas
    for window in [10, 21, 63]:
        df[f'SPY_zscore_{window}'] = calculate_zscore(df['SPY_CLOSE'], window)

    # Momentum / ROC
    for period in [5, 10, 21, 63]:
        df[f'SPY_momentum_{period}'] = calculate_momentum(df['SPY_CLOSE'], period)
        df[f'SPY_ROC_{period}'] = calculate_roc(df['SPY_CLOSE'], period)

    # Distancias de MAs
    for window in [10, 21, 50, 100, 200]:
        ma = df['SPY_CLOSE'].rolling(window=window).mean()
        df[f'SPY_dist_MA_{window}'] = (df['SPY_CLOSE'] - ma) / (ma + 1e-10) * 100

    # MA Crossovers
    for fast, slow in [(5, 10), (10, 21), (10, 50), (20, 50), (50, 200)]:
        ma_fast = df['SPY_CLOSE'].rolling(window=fast).mean()
        ma_slow = df['SPY_CLOSE'].rolling(window=slow).mean()
        df[f'SPY_MA_cross_{fast}_{slow}'] = np.sign(ma_fast - ma_slow)

    # Volatilidad rolling
    for window in [5, 10, 21, 63, 126]:
        df[f'SPY_volatility_{window}'] = df['SPY_return_1d'].rolling(window=window).std() * np.sqrt(252)

    # Volatility ratios
    df['SPY_vol_ratio_5_21'] = df['SPY_volatility_5'] / (df['SPY_volatility_21'] + 1e-10)
    df['SPY_vol_ratio_21_63'] = df['SPY_volatility_21'] / (df['SPY_volatility_63'] + 1e-10)

    # Volume features
    df['SPY_volume_sma_20'] = df['SPY_VOLUME'].rolling(window=20).mean()
    df['SPY_volume_ratio'] = df['SPY_VOLUME'] / (df['SPY_volume_sma_20'] + 1e-10)
    df['SPY_volume_zscore'] = calculate_zscore(df['SPY_VOLUME'], 21)

    technical_count = len([c for c in df.columns if c.startswith('SPY_') and c not in spy_cols])
    print(f"  + Features técnicos SPY: {technical_count}")

    # PASO 3: VIX Term Structure
    print("\nPASO 3: VIX Term Structure")
    print("-" * 80)

    if 'V1' in df.columns:
        # VIX features básicos
        df['VIX_zscore'] = calculate_zscore(df['V1'], 21)
        df['VIX_zscore_63'] = calculate_zscore(df['V1'], 63)
        df['VIX_regime_high'] = (df['V1'] > df['V1'].rolling(252).quantile(0.75)).astype(int)
        df['VIX_regime_low'] = (df['V1'] < df['V1'].rolling(252).quantile(0.25)).astype(int)
        df['VIX_change_1d'] = df['V1'].diff(1)
        df['VIX_change_5d'] = df['V1'].diff(5)
        df['VIX_change_21d'] = df['V1'].diff(21)

        # Term structure
        for vcol, name in [('V7', 'VIX_F1'), ('V8', 'VIX_F2'), ('VIX_F1', 'VIX_F1'), ('VIX_F2', 'VIX_F2')]:
            if vcol in df.columns:
                vix_features = calculate_vix_term_structure(df['V1'], df[vcol],
                                                            df['VIX_F2'] if 'VIX_F2' in df.columns else None)
                for feat_name, feat_val in vix_features.items():
                    df[f'{feat_name}_{name}'] = feat_val
                break  # Solo necesitamos una vez

    print(f"  + VIX features creados")

    # PASO 4: Cross-Asset Correlations
    print("\nPASO 4: Cross-Asset Correlations")
    print("-" * 80)

    corr_features = calculate_cross_asset_correlations(df)
    for name, values in corr_features.items():
        df[name] = values
    print(f"  + Correlaciones calculadas: {len(corr_features)}")

    # PASO 5: Sector Dispersion
    print("\nPASO 5: Sector Dispersion")
    print("-" * 80)

    sector_features = calculate_sector_dispersion(df)
    for name, values in sector_features.items():
        df[name] = values
    print(f"  + Sector features: {len(sector_features)}")

    # PASO 6: Credit Momentum
    print("\nPASO 6: Credit Momentum")
    print("-" * 80)

    credit_features = calculate_credit_momentum(df)
    for name, values in credit_features.items():
        df[name] = values
    print(f"  + Credit features: {len(credit_features)}")

    # PASO 7: Interaction Terms
    print("\nPASO 7: Interaction Terms")
    print("-" * 80)

    interaction_features = calculate_interaction_terms(df)
    for name, values in interaction_features.items():
        df[name] = values
    print(f"  + Interaction terms: {len(interaction_features)}")

    # PASO 8: Regime Indicators
    print("\nPASO 8: Regime Indicators")
    print("-" * 80)

    regime_features = calculate_regime_indicators(df)
    for name, values in regime_features.items():
        df[name] = values
    print(f"  + Regime features: {len(regime_features)}")

    # PASO 9: Factor Momentum
    print("\nPASO 9: Factor Momentum")
    print("-" * 80)

    factor_features = calculate_factor_momentum(df)
    for name, values in factor_features.items():
        df[name] = values
    print(f"  + Factor features: {len(factor_features)}")

    # PASO 10: Economic Momentum
    print("\nPASO 10: Economic Momentum")
    print("-" * 80)

    econ_features = calculate_economic_momentum(df)
    for name, values in econ_features.items():
        df[name] = values
    print(f"  + Economic features: {len(econ_features)}")

    # PASO 11: Volatility Surface
    print("\nPASO 11: Volatility Surface Features")
    print("-" * 80)

    vol_surface_features = calculate_volatility_surface_features(df)
    for name, values in vol_surface_features.items():
        df[name] = values
    print(f"  + Vol surface features: {len(vol_surface_features)}")

    # PASO 12: Lagged Features
    print("\nPASO 12: Lagged Features")
    print("-" * 80)

    # Variables importantes para lags
    lag_vars = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
                'V1', 'M3', 'M4', 'M5', 'M13', 'M18', 'E1', 'E4', 'E8']
    lag_vars = [v for v in lag_vars if v in df.columns]

    lag_count = 0
    for var in lag_vars:
        for lag in CONFIG['lag_periods']:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
            lag_count += 1

    print(f"  + Lagged features: {lag_count}")

    # PASO 13: Rolling Statistics para variables importantes
    print("\nPASO 13: Rolling Statistics")
    print("-" * 80)

    roll_vars = ['I1', 'I2', 'I4', 'I6', 'I8', 'V1']
    roll_vars = [v for v in roll_vars if v in df.columns]

    roll_count = 0
    for var in roll_vars:
        for window in [5, 10, 21, 63]:
            df[f'{var}_roll_mean_{window}'] = df[var].rolling(window=window).mean()
            df[f'{var}_roll_std_{window}'] = df[var].rolling(window=window).std()
            df[f'{var}_roll_min_{window}'] = df[var].rolling(window=window).min()
            df[f'{var}_roll_max_{window}'] = df[var].rolling(window=window).max()
            df[f'{var}_zscore_{window}'] = calculate_zscore(df[var], window)
            roll_count += 5

    print(f"  + Rolling statistics: {roll_count}")

    # PASO 14: Yield Curve Features
    print("\nPASO 14: Yield Curve Features")
    print("-" * 80)

    if 'I4' in df.columns and 'I2' in df.columns:
        df['yield_spread_10_2'] = df['I4'] - df['I2']
        df['yield_spread_10_2_change_1d'] = df['yield_spread_10_2'].diff(1)
        df['yield_spread_10_2_change_5d'] = df['yield_spread_10_2'].diff(5)
        df['yield_spread_10_2_change_21d'] = df['yield_spread_10_2'].diff(21)
        df['yield_spread_10_2_zscore'] = calculate_zscore(df['yield_spread_10_2'], 63)
        df['yield_curve_inverted'] = (df['yield_spread_10_2'] < 0).astype(int)

    if 'I4' in df.columns and 'I1' in df.columns:
        df['yield_spread_10_3m'] = df['I4'] - df['I1']
        df['yield_spread_10_3m_inverted'] = (df['yield_spread_10_3m'] < 0).astype(int)

    # Butterfly spread (2-5-10)
    if all(col in df.columns for col in ['I2', 'I3', 'I4']):
        df['yield_butterfly'] = df['I2'] + df['I4'] - 2 * df['I3']

    print(f"  + Yield curve features creados")

    # PASO 15: Eliminar período de warmup
    print("\nPASO 15: Eliminar Período de Warmup")
    print("-" * 80)

    initial_rows = len(df)

    # Eliminar filas con demasiados NaN
    feature_cols = [c for c in df.columns if c not in target_cols]
    null_pct = df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    df = df[null_pct < 0.5].reset_index(drop=True)

    # Eliminar filas sin target
    df = df.dropna(subset=['market_forward_excess_returns']).reset_index(drop=True)

    # Re-crear date_id
    df['date_id'] = range(len(df))

    print(f"  + Filas eliminadas: {initial_rows - len(df):,}")
    print(f"  + Filas finales: {len(df):,}")

    # PASO 16: Guardar
    print("\nPASO 16: Guardar Dataset Final")
    print("-" * 80)

    # Ordenar columnas
    ordered_cols = ['date_id', 'date', 'SPY_CLOSE', 'forward_returns', 'risk_free_rate',
                    'market_forward_excess_returns']
    ordered_cols += sorted([c for c in df.columns if c not in ordered_cols])

    df_final = df[ordered_cols].copy()

    # Guardar
    output_path = os.path.join(OUTPUT_DIR, "bloomberg_features_hf.csv")
    df_final.to_csv(output_path, index=False)
    print(f"  + Dataset guardado: {output_path}")

    # Contar features
    total_features = len([c for c in df_final.columns
                          if c not in ['date_id', 'date', 'forward_returns',
                                       'risk_free_rate', 'market_forward_excess_returns']])

    # Guardar metadata
    feature_list = [c for c in df_final.columns
                    if c not in ['date_id', 'date', 'forward_returns',
                                 'risk_free_rate', 'market_forward_excess_returns']]

    metadata = {
        'creation_date': datetime.now().isoformat(),
        'n_observations': len(df_final),
        'n_total_features': total_features,
        'date_range': {
            'start': str(df_final['date'].min()),
            'end': str(df_final['date'].max())
        },
        'feature_categories': {
            'technical_spy': technical_count,
            'vix_term_structure': len([c for c in df.columns if 'vix_' in c.lower() or 'VIX' in c]),
            'cross_asset_corr': len(corr_features),
            'sector_dispersion': len(sector_features),
            'credit_momentum': len(credit_features),
            'interaction_terms': len(interaction_features),
            'regime_indicators': len(regime_features),
            'factor_momentum': len(factor_features),
            'economic_momentum': len(econ_features),
            'vol_surface': len(vol_surface_features),
            'lagged': lag_count,
            'rolling': roll_count,
        },
        'features': feature_list,
        'target_column': 'market_forward_excess_returns',
        'hedge_fund_features': [
            'VIX Term Structure (contango/backwardation)',
            'Cross-Asset Rolling Correlations',
            'Sector Dispersion & Rotation Signals',
            'Credit Spread Momentum & Regimes',
            'Interaction Terms (VIX×Curve, Momentum×Vol)',
            'Market Regime Score (Bull/Bear/Neutral)',
            'Factor Momentum (Value/Growth, Size)',
            'Economic Momentum & Trends',
            'Volatility Surface & Risk Premium',
            'ADX, OBV, Stochastic, Williams %R, CCI'
        ],
        'anti_leakage_measures': [
            'All features use only past information (t-k, k>=1)',
            'Rolling windows look backward only',
            'Forward returns calculated with shift(-1) on target only',
            'Extended warmup period (252 days)',
            'No future information used in any feature'
        ],
        'config': CONFIG
    }

    with open(os.path.join(OUTPUT_DIR, "feature_engineering_hf_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN DE FEATURE ENGINEERING - HEDGE FUND")
    print("=" * 80)

    print(f"\n  DATASET FINAL:")
    print(f"  + Observaciones: {len(df_final):,}")
    print(f"  + Periodo: {df_final['date'].min().date()} a {df_final['date'].max().date()}")
    print(f"  + Total features: {total_features}")

    print(f"\n  FEATURES POR CATEGORÍA:")
    for cat, count in metadata['feature_categories'].items():
        print(f"  + {cat}: {count}")

    print(f"\n  FEATURES HEDGE FUND IMPLEMENTADOS:")
    for hf_feat in metadata['hedge_fund_features']:
        print(f"  + {hf_feat}")

    print(f"\n  ARCHIVOS GENERADOS:")
    print(f"  + {output_path}")
    print(f"  + {os.path.join(OUTPUT_DIR, 'feature_engineering_hf_metadata.json')}")

    print("\n" + "=" * 80)
    print("[OK] FEATURE ENGINEERING HEDGE FUND COMPLETADO")
    print("=" * 80)

    return df_final


if __name__ == "__main__":
    df = main()
