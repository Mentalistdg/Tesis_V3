# -*- coding: utf-8 -*-
"""
================================================================================
INDICADORES ELDER TRIPLE PANTALLA - MULTI-FRECUENCIA (TA-Lib Optimizado)
================================================================================

Calcula indicadores de Alexander Elder en cada temporalidad:
- SEMANAL (Pantalla 1): Indicadores de tendencia
- DIARIO (Pantalla 2): Osciladores
- INTRADIARIO (Pantalla 3): Senales de entrada

Los features se agregan a frecuencia diaria para uso en ML.

OPTIMIZACION: Usa TA-Lib (implementacion en C) para calculos de indicadores
en lugar de implementaciones manuales con pandas, manteniendo resultados
equivalentes.

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import talib

# Detectar directorio base automaticamente (portabilidad entre PCs)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

print("=" * 80)
print("INDICADORES ELDER TRIPLE PANTALLA - MULTI-FRECUENCIA (TA-Lib)")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print(f"Data dir: {DATA_DIR}")
print()


# =============================================================================
# FUNCIONES DE INDICADORES TECNICOS
# =============================================================================

def ema(series, span):
    """
    Exponential Moving Average usando TA-Lib optimizado en C.

    Nota: TA-Lib EMA produce resultados equivalentes a pandas ewm(adjust=False)
    para la misma longitud de datos una vez que ambos han convergido.
    """
    values = series.values.astype(np.float64)
    result = talib.EMA(values, timeperiod=span)
    return pd.Series(result, index=series.index)


def macd(close, fast=12, slow=26, signal=9):
    """
    MACD, Signal Line y Histogram usando TA-Lib optimizado en C.

    TA-Lib MACD usa el mismo algoritmo estandar:
    - MACD Line = EMA(fast) - EMA(slow)
    - Signal Line = EMA(MACD Line, signal)
    - Histogram = MACD Line - Signal Line
    """
    values = close.values.astype(np.float64)
    macd_line, signal_line, histogram = talib.MACD(
        values,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )
    return (
        pd.Series(macd_line, index=close.index),
        pd.Series(signal_line, index=close.index),
        pd.Series(histogram, index=close.index)
    )


def force_index(close, volume, period=13):
    """
    Force Index de Elder usando TA-Lib EMA para suavizado.

    Formula: EMA(Close.diff() * Volume, period)
    """
    raw_force = (close.diff() * volume).values.astype(np.float64)
    result = talib.EMA(raw_force, timeperiod=period)
    return pd.Series(result, index=close.index)


def elder_ray(high, low, close, period=13):
    """
    Elder Ray (Bull Power y Bear Power) usando TA-Lib EMA.

    Formula:
    - Bull Power = High - EMA(Close, period)
    - Bear Power = Low - EMA(Close, period)
    """
    ema_values = talib.EMA(close.values.astype(np.float64), timeperiod=period)
    ema_close = pd.Series(ema_values, index=close.index)
    bull_power = high - ema_close
    bear_power = low - ema_close
    return bull_power, bear_power


def impulse_system(close, macd_hist, period=13):
    """
    Sistema Impulse de Elder usando TA-Lib EMA:
    - Verde (1): EMA subiendo Y MACD-H subiendo -> Comprar permitido
    - Rojo (-1): EMA bajando Y MACD-H bajando -> Vender permitido
    - Azul (0): Mixto -> Neutral
    """
    ema_values = talib.EMA(close.values.astype(np.float64), timeperiod=period)
    ema_close = pd.Series(ema_values, index=close.index)
    ema_direction = np.sign(ema_close.diff())
    macd_direction = np.sign(macd_hist.diff())

    impulse = np.where(
        (ema_direction > 0) & (macd_direction > 0), 1,  # Verde
        np.where(
            (ema_direction < 0) & (macd_direction < 0), -1,  # Rojo
            0  # Azul
        )
    )
    return pd.Series(impulse, index=close.index)


def atr(high, low, close, period=14):
    """
    Average True Range - Mantiene implementacion con SMA para resultados equivalentes.

    Nota: talib.ATR usa Wilder smoothing (EMA especial), que produce resultados
    diferentes al ATR clasico con SMA. Mantenemos SMA para consistencia con
    resultados anteriores.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# =============================================================================
# DETECTOR DE RACHAS CONSECUTIVAS (HSBC-ML Style)
# =============================================================================

def consecutive_moves_detector(series, threshold=0):
    """
    Detector de rachas consecutivas UP/DOWN.
    
    Inspirado en HSBC-ML up_and_down() function.
    
    Parameters:
    -----------
    series : pd.Series - Serie de precios o retornos
    threshold : float - Umbral minimo para considerar movimiento
    
    Returns:
    --------
    tuple: (up_streak, down_streak, streak_direction)
    - up_streak: Dias consecutivos al alza
    - down_streak: Dias consecutivos a la baja  
    - streak_direction: 1 (up), -1 (down), 0 (neutral)
    """
    changes = series.diff()
    direction = np.sign(changes)
    
    up_streak = pd.Series(0, index=series.index, dtype=float)
    down_streak = pd.Series(0, index=series.index, dtype=float)
    
    up_count = 0
    down_count = 0
    
    for i in range(1, len(series)):
        if direction.iloc[i] > threshold:
            up_count += 1
            down_count = 0
        elif direction.iloc[i] < -threshold:
            down_count += 1
            up_count = 0
        else:
            up_count = 0
            down_count = 0
        
        up_streak.iloc[i] = up_count
        down_streak.iloc[i] = down_count
    
    streak_direction = np.where(up_streak > 0, 1, np.where(down_streak > 0, -1, 0))
    
    return up_streak, down_streak, pd.Series(streak_direction, index=series.index)


def days_in_window(series, days_threshold, window, direction='up', comparison='>='):
    """
    Busca X dias de movimiento UP/DOWN dentro de una ventana de Y dias.
    
    Inspirado en HSBC-ML days_of_days() function.
    
    Parameters:
    -----------
    series : pd.Series - Serie de precios
    days_threshold : int - Numero de dias requeridos
    window : int - Tamano de la ventana
    direction : str - 'up' o 'down'
    comparison : str - '==' exactamente, '>=' al menos
    
    Returns:
    --------
    pd.Series: 1 donde se cumple la condicion, 0 en otro caso
    """
    changes = series.diff()
    
    if direction == 'up':
        condition = (changes > 0).astype(int)
    else:
        condition = (changes < 0).astype(int)
    
    # Contar dias en ventana
    count_in_window = condition.rolling(window=window).sum()
    
    if comparison == '==':
        result = (count_in_window == days_threshold).astype(int)
    elif comparison == '>=':
        result = (count_in_window >= days_threshold).astype(int)
    else:
        result = pd.Series(0, index=series.index)
    
    return result


def streak_signals(series, streak_thresholds=[3, 5, 7]):
    """
    Genera senales basadas en rachas de diferentes longitudes.
    
    Parameters:
    -----------
    series : pd.Series - Serie de precios
    streak_thresholds : list - Longitudes de racha a detectar
    
    Returns:
    --------
    dict: Diccionario de senales por longitud de racha
    """
    up_streak, down_streak, _ = consecutive_moves_detector(series)
    
    signals = {}
    for thresh in streak_thresholds:
        signals[f'up_streak_{thresh}'] = (up_streak == thresh).astype(int)
        signals[f'down_streak_{thresh}'] = (down_streak == thresh).astype(int)
        signals[f'up_streak_{thresh}plus'] = (up_streak >= thresh).astype(int)
        signals[f'down_streak_{thresh}plus'] = (down_streak >= thresh).astype(int)
    
    return signals


# =============================================================================
# PANTALLA 1: INDICADORES SEMANALES (TENDENCIA)
# =============================================================================

def calculate_weekly_indicators(df_weekly):
    """
    Calcula indicadores de tendencia en datos semanales.
    Estos determinan la direccion del mercado.
    """
    print("-" * 80)
    print("PANTALLA 1: Indicadores Semanales (Tendencia)")
    print("-" * 80)

    df = df_weekly.copy()

    # MACD semanal (el indicador principal de Elder para tendencia)
    df['W_MACD'], df['W_MACD_SIGNAL'], df['W_MACD_HIST'] = macd(df['W_CLOSE'])

    # Direccion del MACD Histogram (clave para Elder)
    df['W_MACD_HIST_DIRECTION'] = np.sign(df['W_MACD_HIST'].diff())

    # Sistema Impulse semanal
    df['W_IMPULSE'] = impulse_system(df['W_CLOSE'], df['W_MACD_HIST'])

    # EMAs para tendencia
    df['W_EMA13'] = ema(df['W_CLOSE'], 13)
    df['W_EMA26'] = ema(df['W_CLOSE'], 26)
    df['W_EMA_SLOPE'] = df['W_EMA13'].pct_change()

    # Posicion del precio respecto a EMAs
    df['W_PRICE_VS_EMA13'] = (df['W_CLOSE'] - df['W_EMA13']) / df['W_EMA13'] * 100
    df['W_PRICE_VS_EMA26'] = (df['W_CLOSE'] - df['W_EMA26']) / df['W_EMA26'] * 100

    # Tendencia basada en MACD-H
    # Si MACD-H sube -> tendencia alcista, si baja -> bajista
    df['W_TREND'] = np.where(df['W_MACD_HIST_DIRECTION'] > 0, 1,
                             np.where(df['W_MACD_HIST_DIRECTION'] < 0, -1, 0))

    # Fuerza de la tendencia (magnitud del MACD-H)
    df['W_TREND_STRENGTH'] = abs(df['W_MACD_HIST']) / df['W_CLOSE'] * 100

    # ATR semanal (volatilidad)
    df['W_ATR'] = atr(df['W_HIGH'], df['W_LOW'], df['W_CLOSE'], 14)
    df['W_ATR_PCT'] = df['W_ATR'] / df['W_CLOSE'] * 100

    print(f"  + Indicadores calculados: MACD, Impulse, EMAs, ATR (TA-Lib)")
    print(f"  + Registros: {len(df)}")

    return df


# =============================================================================
# PANTALLA 2: INDICADORES DIARIOS (OSCILADORES)
# =============================================================================

def calculate_daily_indicators(df_daily):
    """
    Calcula osciladores en datos diarios.
    Estos identifican puntos de entrada contra la tendencia semanal.
    """
    print("-" * 80)
    print("PANTALLA 2: Indicadores Diarios (Osciladores)")
    print("-" * 80)

    df = df_daily.copy()

    # Force Index (indicador favorito de Elder)
    df['D_FORCE_INDEX_2'] = force_index(df['SPY_CLOSE'], df['SPY_VOLUME'], period=2)
    df['D_FORCE_INDEX_13'] = force_index(df['SPY_CLOSE'], df['SPY_VOLUME'], period=13)

    # Normalizar Force Index
    df['D_FORCE_INDEX_2_NORM'] = df['D_FORCE_INDEX_2'] / df['D_FORCE_INDEX_2'].rolling(50).std()
    df['D_FORCE_INDEX_13_NORM'] = df['D_FORCE_INDEX_13'] / df['D_FORCE_INDEX_13'].rolling(50).std()

    # Elder Ray
    df['D_BULL_POWER'], df['D_BEAR_POWER'] = elder_ray(
        df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE']
    )

    # MACD diario
    df['D_MACD'], df['D_MACD_SIGNAL'], df['D_MACD_HIST'] = macd(df['SPY_CLOSE'])

    # Sistema Impulse diario
    df['D_IMPULSE'] = impulse_system(df['SPY_CLOSE'], df['D_MACD_HIST'])

    # EMAs diarias
    df['D_EMA13'] = ema(df['SPY_CLOSE'], 13)
    df['D_EMA26'] = ema(df['SPY_CLOSE'], 26)

    # ATR diario
    df['D_ATR'] = atr(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], 14)
    df['D_ATR_PCT'] = df['D_ATR'] / df['SPY_CLOSE'] * 100

    # Senales de entrada segun Elder:
    # En tendencia alcista (semanal): comprar cuando Force Index cae bajo cero
    # En tendencia bajista (semanal): vender cuando Force Index sube sobre cero
    df['D_FORCE_OVERSOLD'] = (df['D_FORCE_INDEX_2'] < 0).astype(int)
    df['D_FORCE_OVERBOUGHT'] = (df['D_FORCE_INDEX_2'] > 0).astype(int)

    # Elder Ray signals
    df['D_BULL_DIVERGENCE'] = ((df['D_BEAR_POWER'] < 0) &
                               (df['D_BEAR_POWER'] > df['D_BEAR_POWER'].shift(1))).astype(int)
    df['D_BEAR_DIVERGENCE'] = ((df['D_BULL_POWER'] > 0) &
                               (df['D_BULL_POWER'] < df['D_BULL_POWER'].shift(1))).astype(int)

    # Rachas consecutivas (HSBC-ML Style)
    up_streak, down_streak, streak_dir = consecutive_moves_detector(df['SPY_CLOSE'])
    df['D_UP_STREAK'] = up_streak
    df['D_DOWN_STREAK'] = down_streak
    df['D_STREAK_DIRECTION'] = streak_dir
    
    # Senales de rachas especificas
    streak_sigs = streak_signals(df['SPY_CLOSE'], streak_thresholds=[3, 5, 7])
    for sig_name, sig_values in streak_sigs.items():
        df[f'D_{sig_name.upper()}'] = sig_values
    
    # Dias UP/DOWN en ventanas
    df['D_UP_5_IN_10'] = days_in_window(df['SPY_CLOSE'], 5, 10, 'up', '>=')
    df['D_DOWN_5_IN_10'] = days_in_window(df['SPY_CLOSE'], 5, 10, 'down', '>=')

    print(f"  + Indicadores calculados: Force Index, Elder Ray, MACD, Impulse, Rachas (TA-Lib)")
    print(f"  + Registros: {len(df)}")

    return df


# =============================================================================
# PANTALLA 3: FEATURES INTRADIARIOS (ENTRADA)
# =============================================================================

def calculate_intraday_features(df_intraday_daily):
    """
    Calcula features de entrada basados en datos intradiarios.
    Estos refinan el timing de entrada.
    """
    print("-" * 80)
    print("PANTALLA 3: Features Intradiarios (Entrada)")
    print("-" * 80)

    if df_intraday_daily.empty:
        print("  [WARN] No hay datos intradiarios disponibles")
        return pd.DataFrame()

    df = df_intraday_daily.copy()

    # Renombrar columnas si es necesario
    if 'I_CLOSE' not in df.columns and 'close' in df.columns:
        df = df.rename(columns={
            'open': 'I_OPEN', 'high': 'I_HIGH',
            'low': 'I_LOW', 'close': 'I_CLOSE', 'volume': 'I_VOLUME'
        })

    # Breakout intradiario: cerro cerca del maximo o minimo del dia?
    if 'I_RANGE' not in df.columns:
        df['I_RANGE'] = df['I_HIGH'] - df['I_LOW']

    # Posicion del cierre en el rango del dia (0 = minimo, 1 = maximo)
    df['I_CLOSE_POSITION'] = np.where(
        df['I_RANGE'] > 0,
        (df['I_CLOSE'] - df['I_LOW']) / df['I_RANGE'],
        0.5
    )

    # Senales de breakout
    df['I_BREAKOUT_UP'] = (df['I_CLOSE_POSITION'] > 0.8).astype(int)  # Cerro cerca del maximo
    df['I_BREAKOUT_DOWN'] = (df['I_CLOSE_POSITION'] < 0.2).astype(int)  # Cerro cerca del minimo

    # Gap de apertura
    df['I_GAP'] = (df['I_OPEN'] - df['I_CLOSE'].shift(1)) / df['I_CLOSE'].shift(1) * 100
    df['I_GAP_UP'] = (df['I_GAP'] > 0.3).astype(int)
    df['I_GAP_DOWN'] = (df['I_GAP'] < -0.3).astype(int)

    # Rango como % del precio (volatilidad intradiaria)
    df['I_RANGE_PCT'] = df['I_RANGE'] / df['I_CLOSE'] * 100

    # Cambio de momentum intradiario
    if 'I_VOLATILITY' in df.columns:
        df['I_VOL_EXPANSION'] = df['I_VOLATILITY'] > df['I_VOLATILITY'].rolling(5).mean()

    print(f"  + Features calculados: Breakout, Gap, Range")
    print(f"  + Registros: {len(df)}")

    return df


# =============================================================================
# COMBINAR TODAS LAS PANTALLAS
# =============================================================================

def merge_multifreq_features(df_daily, df_weekly, df_intraday_daily):
    """
    Combina features de todas las temporalidades en frecuencia diaria.

    Estrategia:
    - Weekly: Se asigna el valor de la semana anterior (evita leakage)
    - Intraday: Se usa directamente (mismo dia)
    """
    print("-" * 80)
    print("COMBINANDO FEATURES MULTI-FRECUENCIA")
    print("-" * 80)

    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 1. Merge con datos semanales
    # Asignar fecha de fin de semana a cada dia
    df_weekly = df_weekly.copy()
    df_weekly['date'] = pd.to_datetime(df_weekly['date'])

    # Columnas semanales a usar (sin duplicar columnas base)
    weekly_cols = [c for c in df_weekly.columns if c.startswith('W_') and c not in
                   ['W_CLOSE', 'W_HIGH', 'W_LOW', 'W_OPEN', 'W_VOLUME']]
    weekly_cols = ['date'] + weekly_cols

    # Merge: cada dia obtiene el valor de la semana anterior
    # Usamos merge_asof para asignar la semana mas reciente
    df = df.sort_values('date')
    df_weekly_features = df_weekly[weekly_cols].sort_values('date')

    # Shift weekly data by one week to avoid leakage
    df_weekly_features_shifted = df_weekly_features.copy()
    df_weekly_features_shifted['date'] = df_weekly_features_shifted['date'] + pd.Timedelta(days=7)

    df = pd.merge_asof(
        df,
        df_weekly_features_shifted,
        on='date',
        direction='backward'
    )

    print(f"  + Features semanales agregados: {len(weekly_cols) - 1}")

    # 2. Merge con datos intradiarios
    if not df_intraday_daily.empty:
        df_intraday = df_intraday_daily.copy()
        df_intraday['date'] = pd.to_datetime(df_intraday['date'])

        # Columnas intradiarias a usar
        intraday_cols = [c for c in df_intraday.columns if c.startswith('I_')]
        intraday_cols = ['date'] + intraday_cols

        df = df.merge(
            df_intraday[intraday_cols],
            on='date',
            how='left'
        )

        print(f"  + Features intradiarios agregados: {len(intraday_cols) - 1}")

    print(f"  + Dataset final: {len(df)} filas x {df.shape[1]} columnas")

    return df


# =============================================================================
# GENERAR SENALES TRIPLE PANTALLA
# =============================================================================

def generate_triple_screen_signals(df):
    """
    Genera senales combinadas del sistema Triple Pantalla.

    Logica de Elder:
    1. Screen 1 (Weekly): Determina direccion (MACD-H subiendo = alcista)
    2. Screen 2 (Daily): Busca pullbacks (Force Index negativo en tendencia alcista)
    3. Screen 3 (Intraday): Timing de entrada (breakout)
    """
    print("-" * 80)
    print("GENERANDO SENALES TRIPLE PANTALLA")
    print("-" * 80)

    df = df.copy()

    # Verificar columnas necesarias
    has_weekly = 'W_TREND' in df.columns
    has_daily = 'D_FORCE_INDEX_2' in df.columns
    has_intraday = 'I_BREAKOUT_UP' in df.columns

    print(f"  + Datos semanales: {'SI' if has_weekly else 'NO'}")
    print(f"  + Datos diarios: {'SI' if has_daily else 'NO'}")
    print(f"  + Datos intradiarios: {'SI' if has_intraday else 'NO'}")

    # Senal de compra Triple Pantalla
    # 1. Tendencia semanal alcista (MACD-H subiendo)
    # 2. Pullback diario (Force Index negativo)
    # 3. Confirmacion intradiaria (breakout al alza)

    if has_weekly and has_daily:
        # Senal basica (sin intraday)
        df['TS_BUY_SETUP'] = (
            (df['W_TREND'] == 1) &  # Tendencia alcista
            (df['D_FORCE_OVERSOLD'] == 1)  # Pullback
        ).astype(int)

        df['TS_SELL_SETUP'] = (
            (df['W_TREND'] == -1) &  # Tendencia bajista
            (df['D_FORCE_OVERBOUGHT'] == 1)  # Rally
        ).astype(int)

        if has_intraday:
            # Senal completa (con intraday)
            df['TS_BUY_SIGNAL'] = (
                (df['TS_BUY_SETUP'] == 1) &
                (df['I_BREAKOUT_UP'] == 1)
            ).astype(int)

            df['TS_SELL_SIGNAL'] = (
                (df['TS_SELL_SETUP'] == 1) &
                (df['I_BREAKOUT_DOWN'] == 1)
            ).astype(int)
        else:
            df['TS_BUY_SIGNAL'] = df['TS_BUY_SETUP']
            df['TS_SELL_SIGNAL'] = df['TS_SELL_SETUP']

        # Senal neta
        df['TS_SIGNAL'] = df['TS_BUY_SIGNAL'] - df['TS_SELL_SIGNAL']

        # Contar senales
        n_buy = df['TS_BUY_SIGNAL'].sum()
        n_sell = df['TS_SELL_SIGNAL'].sum()
        print(f"  + Senales de compra: {n_buy}")
        print(f"  + Senales de venta: {n_sell}")

    # Features adicionales para ML

    # Alineacion de pantallas (todas en la misma direccion)
    if has_weekly and has_daily:
        df['TS_ALIGNMENT'] = np.where(
            (df['W_IMPULSE'] == 1) & (df['D_IMPULSE'] == 1), 1,  # Ambos verdes
            np.where(
                (df['W_IMPULSE'] == -1) & (df['D_IMPULSE'] == -1), -1,  # Ambos rojos
                0  # Mixto
            )
        )

    # Fuerza combinada
    if 'W_TREND_STRENGTH' in df.columns and 'D_FORCE_INDEX_13_NORM' in df.columns:
        df['TS_COMBINED_STRENGTH'] = (
            df['W_TREND_STRENGTH'].fillna(0) * 0.5 +
            abs(df['D_FORCE_INDEX_13_NORM'].fillna(0)) * 0.5
        )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Pipeline principal"""

    # Cargar datos
    print("PASO 1: Cargar Datos")
    print("-" * 80)

    # Datos semanales
    weekly_file = os.path.join(DATA_DIR, "spy_weekly.csv")
    if os.path.exists(weekly_file):
        df_weekly = pd.read_csv(weekly_file)
        df_weekly['date'] = pd.to_datetime(df_weekly['date'])
        print(f"  + Semanal: {len(df_weekly)} registros")
    else:
        print("  [ERROR] No se encontro spy_weekly.csv")
        return None

    # Datos intradiarios agregados
    intraday_file = os.path.join(DATA_DIR, "spy_intraday_daily_features.csv")
    if os.path.exists(intraday_file):
        df_intraday = pd.read_csv(intraday_file)
        df_intraday['date'] = pd.to_datetime(df_intraday['date'])
        print(f"  + Intradiario: {len(df_intraday)} dias")
    else:
        print("  [WARN] No se encontro spy_intraday_daily_features.csv")
        df_intraday = pd.DataFrame()

    # Datos diarios (dataset principal)
    daily_file = os.path.join(FINAL_DIR, "bloomberg_features_final.csv")
    if os.path.exists(daily_file):
        df_daily = pd.read_csv(daily_file)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        print(f"  + Diario: {len(df_daily)} registros, {df_daily.shape[1]} columnas")
    else:
        print("  [ERROR] No se encontro bloomberg_features_final.csv")
        return None

    # Calcular indicadores en cada temporalidad
    print()
    df_weekly = calculate_weekly_indicators(df_weekly)
    print()
    df_daily = calculate_daily_indicators(df_daily)
    print()
    df_intraday = calculate_intraday_features(df_intraday)

    # Combinar todo
    print()
    df_combined = merge_multifreq_features(df_daily, df_weekly, df_intraday)

    # Generar senales
    print()
    df_final = generate_triple_screen_signals(df_combined)

    # Guardar
    print()
    print("-" * 80)
    print("GUARDANDO DATASET FINAL")
    print("-" * 80)

    output_file = os.path.join(FINAL_DIR, "bloomberg_features_triple_screen.csv")
    df_final.to_csv(output_file, index=False)
    print(f"  + Guardado: {output_file}")
    print(f"  + Dimensiones: {len(df_final)} filas x {df_final.shape[1]} columnas")

    # Resumen de columnas nuevas
    triple_cols = [c for c in df_final.columns if c.startswith(('W_', 'I_', 'TS_', 'D_FORCE', 'D_BULL', 'D_BEAR', 'D_IMPULSE'))]
    print(f"  + Nuevas columnas Triple Pantalla: {len(triple_cols)}")

    # Estadisticas de senales
    print()
    print("=" * 80)
    print("RESUMEN TRIPLE PANTALLA")
    print("=" * 80)

    if 'TS_SIGNAL' in df_final.columns:
        signal_counts = df_final['TS_SIGNAL'].value_counts().sort_index()
        print(f"\n  Distribucion de senales:")
        print(f"    Venta (-1): {signal_counts.get(-1, 0)}")
        print(f"    Neutral (0): {signal_counts.get(0, 0)}")
        print(f"    Compra (1): {signal_counts.get(1, 0)}")

    # Missing values en nuevas columnas
    new_cols = [c for c in df_final.columns if c.startswith(('W_', 'I_', 'TS_'))]
    missing_summary = {c: df_final[c].isna().sum() for c in new_cols if df_final[c].isna().sum() > 0}
    if missing_summary:
        print(f"\n  Missing values en columnas nuevas:")
        for col, n in sorted(missing_summary.items(), key=lambda x: -x[1])[:5]:
            print(f"    {col}: {n} ({n/len(df_final)*100:.1f}%)")

    print()
    print("=" * 80)
    print("[OK] INDICADORES TRIPLE PANTALLA COMPLETADOS (TA-Lib)")
    print("=" * 80)

    return df_final


if __name__ == "__main__":
    df = main()
