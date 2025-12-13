# -*- coding: utf-8 -*-
"""
================================================================================
PROXY FEATURES DE VOLATILIDAD INTRADIARIA - METODOLOGÍA HEDGE FUND
================================================================================

Cuando no hay datos intradiarios históricos, los hedge funds utilizan
estimadores de volatilidad basados en datos OHLC diarios que capturan
información sobre el comportamiento intradiario.

ESTIMADORES IMPLEMENTADOS:

1. PARKINSON (1980)
   - Usa rango High-Low
   - 5x más eficiente que close-to-close

2. GARMAN-KLASS (1980)
   - Usa OHLC completo
   - 8x más eficiente que close-to-close

3. ROGERS-SATCHELL (1991)
   - Robusto a drift (tendencia)
   - Mejor para mercados con tendencia

4. YANG-ZHANG (2000)
   - Combina overnight + intraday
   - El más completo para acciones

5. FEATURES DE MICROESTRUCTURA
   - Posición del cierre en el rango
   - Gaps overnight
   - Patrones de velas japonesas

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

PROJECT_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FINAL_DIR = os.path.join(DATA_DIR, "final")

print("=" * 80)
print("PROXY FEATURES DE VOLATILIDAD INTRADIARIA")
print("Metodología: Hedge Fund Cuantitativo")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


def parkinson_volatility(high, low, window=21):
    """
    Parkinson (1980) - Estimador de volatilidad usando High-Low

    Fórmula: σ² = (1/4ln2) * ln(H/L)²

    5x más eficiente que volatilidad close-to-close
    """
    log_hl = np.log(high / low) ** 2
    factor = 1 / (4 * np.log(2))
    variance = factor * log_hl
    vol = np.sqrt(variance.rolling(window).mean() * 252)
    return vol


def garman_klass_volatility(open_price, high, low, close, window=21):
    """
    Garman-Klass (1980) - Estimador OHLC

    Combina información de apertura, máximo, mínimo y cierre.
    8x más eficiente que close-to-close.
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_price) ** 2

    variance = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    vol = np.sqrt(variance.rolling(window).mean() * 252)
    return vol


def rogers_satchell_volatility(open_price, high, low, close, window=21):
    """
    Rogers-Satchell (1991) - Robusto a drift

    No asume que el drift es cero, mejor para mercados con tendencia.
    """
    log_ho = np.log(high / open_price)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_price)
    log_lc = np.log(low / close)

    variance = log_ho * log_hc + log_lo * log_lc
    vol = np.sqrt(variance.rolling(window).mean() * 252)
    return vol


def yang_zhang_volatility(open_price, high, low, close, window=21):
    """
    Yang-Zhang (2000) - El estimador más completo

    Combina:
    - Volatilidad overnight (close-to-open)
    - Volatilidad open-to-close
    - Rogers-Satchell

    Maneja correctamente los gaps overnight.
    """
    # Overnight volatility (close anterior a open)
    log_oc = np.log(open_price / close.shift(1))
    overnight_var = log_oc.rolling(window).var()

    # Open-to-close volatility
    log_co = np.log(close / open_price)
    openclose_var = log_co.rolling(window).var()

    # Rogers-Satchell
    log_ho = np.log(high / open_price)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_price)
    log_lc = np.log(low / close)
    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()

    # Constante de Yang-Zhang
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Combinación
    variance = overnight_var + k * openclose_var + (1 - k) * rs_var
    vol = np.sqrt(variance * 252)

    return vol


def calculate_microstructure_features(df):
    """
    Features de microestructura de mercado basados en OHLC.

    Estos features capturan patrones que normalmente requerirían
    datos intradiarios.
    """
    features = pd.DataFrame(index=df.index)

    O = df['SPY_OPEN'] if 'SPY_OPEN' in df.columns else df['open']
    H = df['SPY_HIGH'] if 'SPY_HIGH' in df.columns else df['high']
    L = df['SPY_LOW'] if 'SPY_LOW' in df.columns else df['low']
    C = df['SPY_CLOSE'] if 'SPY_CLOSE' in df.columns else df['close']
    V = df['SPY_VOLUME'] if 'SPY_VOLUME' in df.columns else df.get('volume', pd.Series(1, index=df.index))

    # =========================================================================
    # 1. RANGO Y VOLATILIDAD REALIZADA
    # =========================================================================

    # Rango absoluto y normalizado
    features['INTRA_RANGE'] = H - L
    features['INTRA_RANGE_PCT'] = (H - L) / C * 100

    # True Range (incluye gaps)
    prev_close = C.shift(1)
    true_range = pd.concat([
        H - L,
        abs(H - prev_close),
        abs(L - prev_close)
    ], axis=1).max(axis=1)
    features['INTRA_TRUE_RANGE'] = true_range
    features['INTRA_ATR_5'] = true_range.rolling(5).mean()
    features['INTRA_ATR_21'] = true_range.rolling(21).mean()

    # =========================================================================
    # 2. ESTIMADORES DE VOLATILIDAD
    # =========================================================================

    features['VOL_PARKINSON_21'] = parkinson_volatility(H, L, 21)
    features['VOL_PARKINSON_63'] = parkinson_volatility(H, L, 63)
    features['VOL_GARMAN_KLASS_21'] = garman_klass_volatility(O, H, L, C, 21)
    features['VOL_GARMAN_KLASS_63'] = garman_klass_volatility(O, H, L, C, 63)
    features['VOL_ROGERS_SATCHELL_21'] = rogers_satchell_volatility(O, H, L, C, 21)
    features['VOL_YANG_ZHANG_21'] = yang_zhang_volatility(O, H, L, C, 21)
    features['VOL_YANG_ZHANG_63'] = yang_zhang_volatility(O, H, L, C, 63)

    # Ratio de volatilidades (corto vs largo plazo)
    features['VOL_RATIO_21_63'] = features['VOL_YANG_ZHANG_21'] / features['VOL_YANG_ZHANG_63']

    # =========================================================================
    # 3. POSICIÓN DEL PRECIO EN EL RANGO (MICROESTRUCTURA)
    # =========================================================================

    # Dónde cerró dentro del rango del día
    range_size = H - L
    features['INTRA_CLOSE_POSITION'] = np.where(
        range_size > 0,
        (C - L) / range_size,
        0.5
    )

    # Dónde abrió dentro del rango
    features['INTRA_OPEN_POSITION'] = np.where(
        range_size > 0,
        (O - L) / range_size,
        0.5
    )

    # =========================================================================
    # 4. GAPS OVERNIGHT
    # =========================================================================

    # Gap de apertura
    features['GAP_OVERNIGHT'] = (O - prev_close) / prev_close * 100
    features['GAP_OVERNIGHT_ABS'] = abs(features['GAP_OVERNIGHT'])

    # Gap como % del ATR (normalizado)
    features['GAP_VS_ATR'] = features['GAP_OVERNIGHT_ABS'] / (features['INTRA_ATR_21'] / C * 100)

    # Frecuencia de gaps grandes (>0.5%)
    features['GAP_LARGE_FLAG'] = (features['GAP_OVERNIGHT_ABS'] > 0.5).astype(int)
    features['GAP_LARGE_FREQ_21'] = features['GAP_LARGE_FLAG'].rolling(21).mean()

    # =========================================================================
    # 5. PATRONES DE VELAS JAPONESAS (CUANTIFICADOS)
    # =========================================================================

    body = C - O
    upper_shadow = H - pd.concat([O, C], axis=1).max(axis=1)
    lower_shadow = pd.concat([O, C], axis=1).min(axis=1) - L

    # Tamaño del cuerpo vs rango total
    features['CANDLE_BODY_PCT'] = np.where(
        range_size > 0,
        abs(body) / range_size,
        0
    )

    # Sombras
    features['CANDLE_UPPER_SHADOW_PCT'] = np.where(
        range_size > 0,
        upper_shadow / range_size,
        0
    )
    features['CANDLE_LOWER_SHADOW_PCT'] = np.where(
        range_size > 0,
        lower_shadow / range_size,
        0
    )

    # Dirección del cuerpo
    features['CANDLE_DIRECTION'] = np.sign(body)

    # Doji (cuerpo muy pequeño)
    features['CANDLE_DOJI'] = (features['CANDLE_BODY_PCT'] < 0.1).astype(int)

    # =========================================================================
    # 6. VOLUMEN INTRADIARIO PROXY
    # =========================================================================

    # Volumen relativo
    features['VOLUME_RELATIVE_21'] = V / V.rolling(21).mean()
    features['VOLUME_RELATIVE_63'] = V / V.rolling(63).mean()

    # Volumen * Rango (proxy de actividad)
    features['VOLUME_RANGE_PRODUCT'] = V * features['INTRA_RANGE_PCT']

    # Eficiencia del precio (cuánto se movió vs cuánto recorrió)
    net_change = abs(C - O)
    features['PRICE_EFFICIENCY'] = np.where(
        range_size > 0,
        net_change / range_size,
        0
    )

    # =========================================================================
    # 7. ESTADÍSTICAS ROLLING DE MICROESTRUCTURA
    # =========================================================================

    # Media y std del rango
    features['RANGE_PCT_MA_21'] = features['INTRA_RANGE_PCT'].rolling(21).mean()
    features['RANGE_PCT_STD_21'] = features['INTRA_RANGE_PCT'].rolling(21).std()
    features['RANGE_PCT_ZSCORE'] = (
        (features['INTRA_RANGE_PCT'] - features['RANGE_PCT_MA_21']) /
        features['RANGE_PCT_STD_21']
    )

    # Consistencia del cierre en el rango
    features['CLOSE_POSITION_MA_21'] = features['INTRA_CLOSE_POSITION'].rolling(21).mean()

    # =========================================================================
    # 8. INDICADORES DE PRESIÓN COMPRADORA/VENDEDORA
    # =========================================================================

    # Accumulation/Distribution proxy
    clv = np.where(
        range_size > 0,
        ((C - L) - (H - C)) / range_size,
        0
    )
    features['CLV'] = clv  # Close Location Value
    features['CLV_MA_21'] = pd.Series(clv, index=df.index).rolling(21).mean()

    # Presión de compra (cierres en parte alta del rango)
    features['BUYING_PRESSURE_21'] = (features['INTRA_CLOSE_POSITION'] > 0.6).rolling(21).mean()
    features['SELLING_PRESSURE_21'] = (features['INTRA_CLOSE_POSITION'] < 0.4).rolling(21).mean()

    return features


def main():
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    print("PASO 1: Cargando datos")
    print("-" * 80)

    # Dataset principal
    final_path = os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv")
    df = pd.read_csv(final_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Dataset cargado: {len(df):,} filas x {df.shape[1]} columnas")

    # Verificar columnas OHLC
    ohlc_cols = ['SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_CLOSE']
    missing_ohlc = [c for c in ohlc_cols if c not in df.columns]

    if missing_ohlc:
        print(f"  [WARN] Faltan columnas OHLC: {missing_ohlc}")
        print("  Intentando cargar de datos raw...")

        raw_path = os.path.join(DATA_DIR, "hull_dataset_raw.csv")
        df_raw = pd.read_csv(raw_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Merge OHLC
        for col in missing_ohlc:
            if col in df_raw.columns:
                df = df.merge(df_raw[['date', col]], on='date', how='left')

    print(f"  Columnas OHLC: {[c for c in df.columns if 'SPY_' in c and any(x in c for x in ['OPEN','HIGH','LOW','CLOSE'])]}")

    # =========================================================================
    # CALCULAR FEATURES
    # =========================================================================
    print("\nPASO 2: Calculando proxy features de volatilidad intradiaria")
    print("-" * 80)

    intra_features = calculate_microstructure_features(df)

    print(f"  Features calculados: {intra_features.shape[1]}")
    print(f"  Columnas: {list(intra_features.columns)}")

    # =========================================================================
    # COMBINAR CON DATOS INTRADIARIOS REALES (donde existen)
    # =========================================================================
    print("\nPASO 3: Combinando con datos intradiarios reales")
    print("-" * 80)

    # Cargar datos intradiarios reales de 30 min
    intra_30min_path = os.path.join(DATA_DIR, "spy_intraday_30min_max.csv")
    if os.path.exists(intra_30min_path):
        df_intra = pd.read_csv(intra_30min_path)
        df_intra['datetime'] = pd.to_datetime(df_intra['datetime'])
        df_intra['date'] = pd.to_datetime(df_intra['datetime'].dt.date)

        # Calcular volatilidad intradiaria REAL
        real_vol = df_intra.groupby('date', as_index=False).apply(
            lambda x: pd.Series({
                'INTRA_VOL_REAL_30MIN': x['close'].pct_change().std() * np.sqrt(len(x)) * np.sqrt(252)
            }),
            include_groups=False
        )

        # Agregar al dataframe de features
        intra_features['date'] = df['date'].values
        intra_features = intra_features.merge(real_vol, on='date', how='left')

        real_count = intra_features['INTRA_VOL_REAL_30MIN'].notna().sum()
        print(f"  Datos intradiarios reales disponibles: {real_count} días")
    else:
        intra_features['date'] = df['date'].values
        print("  [INFO] No hay datos intradiarios de 30 min")

    # =========================================================================
    # AGREGAR AL DATASET PRINCIPAL
    # =========================================================================
    print("\nPASO 4: Agregando features al dataset principal")
    print("-" * 80)

    # Agregar features
    for col in intra_features.columns:
        if col != 'date' and col not in df.columns:
            df[col] = intra_features[col].values

    print(f"  Dataset final: {len(df):,} filas x {df.shape[1]} columnas")

    # =========================================================================
    # GUARDAR
    # =========================================================================
    print("\nPASO 5: Guardando datasets actualizados")
    print("-" * 80)

    # Guardar dataset principal
    df.to_csv(final_path, index=False)
    print(f"  + {final_path}")

    # También en carpeta de transferencia
    transfer_path = os.path.join(PROJECT_DIR, "TRANSFER_TO_TRAINING_PC", "data", "bloomberg_triple_screen_core.csv")
    df.to_csv(transfer_path, index=False)
    print(f"  + {transfer_path}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN DE FEATURES DE VOLATILIDAD INTRADIARIA")
    print("=" * 80)

    # Listar features por categoría
    vol_features = [c for c in df.columns if 'VOL_' in c and 'INTRA' not in c]
    intra_features_list = [c for c in df.columns if 'INTRA_' in c]
    candle_features = [c for c in df.columns if 'CANDLE_' in c]
    gap_features = [c for c in df.columns if 'GAP_' in c]

    print(f"""
  ESTIMADORES DE VOLATILIDAD ({len(vol_features)} features):
    - Parkinson (21d, 63d)
    - Garman-Klass (21d, 63d)
    - Rogers-Satchell (21d)
    - Yang-Zhang (21d, 63d)

  FEATURES INTRADIARIOS PROXY ({len(intra_features_list)} features):
    - Rango, True Range, ATR
    - Posición del cierre/apertura
    - Eficiencia del precio

  PATRONES DE VELAS ({len(candle_features)} features):
    - Body %, Upper/Lower Shadow
    - Dirección, Doji

  GAPS OVERNIGHT ({len(gap_features)} features):
    - Gap %, Gap vs ATR
    - Frecuencia de gaps grandes

  DATOS REALES vs PROXY:
    - 2000-06-01: Proxy (estimadores OHLC)
    - 2025-06-02 en adelante: Datos reales 30 min + proxy

  Dataset final:
    - Filas: {len(df):,}
    - Columnas: {df.shape[1]}
""")

    print("=" * 80)
    print("[OK] PROXY FEATURES COMPLETADOS")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = main()
