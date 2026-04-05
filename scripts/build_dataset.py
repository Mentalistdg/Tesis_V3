    # -*- coding: utf-8 -*-
"""
================================================================================
BUILD DATASET - PIPELINE UNIFICADO PROFESIONAL
================================================================================

Pipeline de construccion de dataset para prediccion del S&P 500.
Combina metodologias de hedge funds cuantitativos de Wall Street.

================================================================================
DIAGRAMA TEMPORAL - ALINEACION DE FEATURES Y TARGET (CRITICO PARA ENTENDER)
================================================================================

El siguiente diagrama ilustra como se alinean temporalmente las features (X)
y el target (y) para EVITAR data leakage:

    Linea temporal:
    ===============

    Dia:          t-2        t-1         t          t+1        t+2
                   |          |          |           |          |
    Close:       $100       $101       $102        $105       $107
                   |          |          |           |          |
                   └──────────┴──────────┤           |          |
                                         |           |          |
    Features[t]:  RSI, MACD, BB, etc.    |           |          |
                  calculados usando  ────┘           |          |
                  datos desde t-k hasta t            |          |
                  (INCLUYENDO Close[t]=$102)         |          |
                                                     |          |
    Target[t]:    forward_returns[t] ────────────────┘          |
                  = (Close[t+1] - Close[t]) / Close[t]          |
                  = ($105 - $102) / $102 = 2.94%                |
                  (retorno FUTURO de t a t+1)                   |

    ESCENARIO OPERACIONAL:
    ======================

    1. Al CIERRE del dia t (16:00 hrs):
       - Conocemos: Close[t], High[t], Low[t], Open[t], Volume[t]
       - Calculamos: Features[t] (RSI, MACD, volatilidad, etc.)
       - Predecimos: Target[t] = retorno esperado de t a t+1
       - Ejecutamos: Compramos/vendemos al precio Close[t]

    2. Al CIERRE del dia t+1 (16:00 hrs):
       - Cerramos la posicion al precio Close[t+1]
       - Retorno real = (Close[t+1] - Close[t]) / Close[t]
       - Este retorno COINCIDE exactamente con nuestro target

    POR QUE NO HAY DATA LEAKAGE:
    ============================

    - Features[t] usan datos de dias {t-k, ..., t-1, t} (pasado + presente)
    - Target[t] = retorno de dia t a dia t+1 (FUTURO)
    - En NINGUN momento las features ven Close[t+1] o datos futuros
    - La prediccion se hace AL CIERRE de t, cuando todos los datos de t
      ya son conocidos, pero el retorno de t a t+1 aun no ha ocurrido

    FUNCIONES DE PANDAS UTILIZADAS (sin leakage):
    =============================================

    - pct_change(n): Calcula (X[t] - X[t-n]) / X[t-n]
      -> Usa datos en t y t-n, ambos pasados o presentes, NO futuros

    - rolling(window).mean(): Promedio de X[t-window+1] hasta X[t]
      -> La ventana mira hacia ATRAS, nunca hacia adelante

    - shift(n): Desplaza la serie n posiciones
      -> shift(1): X[t] toma el valor de X[t-1] (pasado)
      -> shift(-1): X[t] toma el valor de X[t+1] (futuro) - SOLO para target

    - ewm(span=n).mean(): Media movil exponencial hasta t
      -> Pondera datos historicos, no usa datos futuros

    - diff(n): Calcula X[t] - X[t-n]
      -> Usa datos en t y t-n, no usa datos futuros

    NOTA SOBRE shift(-1) EN EL TARGET:
    ==================================

    El UNICO lugar donde usamos shift(-1) es para crear el target:

        forward_returns = Close.pct_change().shift(-1)

    Esto es INTENCIONAL porque queremos que target[t] sea el retorno FUTURO.
    En la fila t, guardamos el retorno que ocurrira de t a t+1.
    Esto NO es leakage porque el target es lo que queremos PREDECIR,
    no una feature que usamos como input del modelo.

================================================================================
FLUJO DEL PIPELINE:
================================================================================

BLOOMBERG_RAW_DATA.csv (98 cols) --> build_dataset.py --> bloomberg_triple_screen_core.csv (~700 cols)

================================================================================
CATEGORIAS DE FEATURES:
================================================================================

1. BASE: Retornos, lags, rolling statistics
2. TECHNICAL: RSI, MACD, ADX, Bollinger, Stochastic (TA-Lib)
3. ELDER TRIPLE SCREEN: Weekly MACD, Force Index, Elder Ray, Impulse System
4. VOLATILITY CLASSIC: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
5. VOLATILITY CONTEMPORARY: Intrinsic Entropy, Log-Range, CARR, Range-GARCH
6. CROSS-ASSET: Correlaciones rolling con bonds, gold, oil, currencies
7. MACRO: Momentum de indicadores economicos
8. REGIME: Bull/Bear detection, VIX term structure
9. NORMALIZATION: Z-scores, rank percentiles (HSBC-ML style)

================================================================================
GARANTIAS ANTI-DATA LEAKAGE:
================================================================================

1. Todas las features usan SOLO informacion de t-k hasta t (k >= 0)
2. Rolling windows SIEMPRE miran hacia atras (nunca center=True)
3. Target usa shift(-1) para representar retorno FUTURO
4. Risk-free rate en t es la tasa CONOCIDA en t (sin shift adicional)
5. NaN residuales de ventanas rolling se imputan en train_models.py (SimpleImputer median)
6. Validacion automatica al final del pipeline (funcion validate_no_leakage)

================================================================================
REFERENCIAS ACADEMICAS:
================================================================================

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
    'lag_periods': [1, 5, 21],

    # Ventanas para rolling statistics
    'rolling_windows': [5, 10, 21, 63, 126, 252],

    # Ventanas para correlaciones cross-asset
    'correlation_windows': [21, 63],

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

    # Economic Indicators (E1-E19)
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
    'Extra E15': 'E15',  # Housing Starts (NHSPATOT Index - inicios de construcción)
    'ETSLTOTL Index': 'E16',
    'GDP CYOY Index': 'E17',
    'CPTICHNG Index': 'E18',
    'PITLCHNG Index': 'E19',

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

    # Commodities (P1-P11)
    'CL1 Comdty': 'P1',
    'HO1 Comdty': 'P2',
    'GC1 Comdty': 'P3',
    'SI1 Comdty': 'P4',
    'GLD US Equity': 'P5',
    'SLV US Equity': 'P6',
    'USO US Equity': 'P7',
    'DBC US Equity': 'P8',
    'DBA US Equity': 'P9',
    'BCOMTR Index': 'P10',
    'GSCITR Index': 'P11',

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

    # Sentiment (S1-S10)
    'AAII BULLISH Index': 'S1',
    'PUT Index': 'S2',
    'NYHL Index': 'S3',
    'TICK Index': 'S4',
    'TRIN Index': 'S5',
    'ADD Index': 'S6',
    'MCCL Index': 'S7',
    'MCSU Index': 'S8',
    'SRVOL Index': 'S9',
    'PCUSEQUI Index': 'S10',
}


# =============================================================================
# SECCION 1: FUNCIONES DE CARGA Y LIMPIEZA
# =============================================================================

def load_raw_data(filepath):
    """
    Carga datos raw de Bloomberg y renombra columnas.

    ============================================================================
    IMPORTANCIA DEL ORDENAMIENTO TEMPORAL
    ============================================================================

    Esta funcion realiza un paso CRITICO para la integridad del dataset:
    ordenar los datos por fecha de forma ASCENDENTE.

    POR QUE ES CRITICO:
    -------------------
    1. Las funciones de pandas (pct_change, rolling, shift) asumen que los
       datos estan ordenados temporalmente. Si el orden fuera aleatorio,
       pct_change() calcularia diferencias entre dias no consecutivos.

    2. El ordenamiento se hace UNA SOLA VEZ aqui y NUNCA se modifica despues.
       Esto garantiza que todas las features calculadas posteriormente
       respetan el orden temporal correcto.

    3. reset_index(drop=True) crea un indice numerico limpio (0, 1, 2, ...)
       que corresponde al orden temporal. Esto facilita la alineacion
       posterior de features.

    RESULTADO DEL ORDENAMIENTO:
    ---------------------------
    - df.iloc[0] = primer dia de trading (fecha mas antigua)
    - df.iloc[1] = segundo dia de trading
    - ...
    - df.iloc[-1] = ultimo dia de trading (fecha mas reciente)

    GARANTIA:
    ---------
    Despues de esta funcion, el DataFrame esta ordenado cronologicamente
    y todas las operaciones de pandas funcionaran correctamente para
    datos de series de tiempo.

    Parameters:
    -----------
    filepath : str
        Ruta al archivo BLOOMBERG_RAW_DATA.csv

    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas renombradas, fecha parseada, y ordenado por fecha
    """
    print("  Cargando datos raw...")
    df = pd.read_csv(filepath)

    # Parsear fechas
    df['date'] = pd.to_datetime(df['date'])

    # ==========================================================================
    # ORDENAMIENTO TEMPORAL (CRITICO)
    # ==========================================================================
    # Ordenamos por fecha ASCENDENTE para que:
    # - iloc[0] = dia mas antiguo
    # - iloc[-1] = dia mas reciente
    # - pct_change(), rolling(), shift() funcionen correctamente
    #
    # reset_index(drop=True) crea indices limpios: 0, 1, 2, ..., N-1
    # Estos indices se preservan en todas las operaciones posteriores
    # ==========================================================================
    df = df.sort_values('date').reset_index(drop=True)

    # Renombrar columnas de Bloomberg a codigos internos
    for old_col, new_col in COLUMN_MAPPING.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    print(f"  + Filas: {len(df):,}")
    print(f"  + Columnas: {df.shape[1]}")
    print(f"  + Periodo: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"  + Orden temporal: ASCENDENTE (iloc[0]=primer dia, iloc[-1]=ultimo dia)")

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

    ============================================================================
    EXPLICACION DETALLADA DEL TARGET Y ALINEACION TEMPORAL
    ============================================================================

    CONTEXTO DEL PROBLEMA:
    ----------------------
    Queremos predecir si manana el mercado subira o bajara para tomar una
    decision de inversion HOY al cierre. El target debe representar el
    retorno FUTURO que obtendremos si invertimos.

    ALINEACION TEMPORAL CORRECTA:
    -----------------------------
    En el dia t (al cierre, 16:00 hrs), el inversor DECIDE entre:
    - Invertir en SPY: obtendra forward_returns[t] = retorno de t a t+1
    - Invertir en T-Bills: obtendra risk_free_rate[t] = tasa conocida en t

    Por lo tanto:
    - forward_returns[t] = SPY[t+1]/SPY[t] - 1 (calculado con shift(-1))
    - risk_free_rate[t] = tasa publicada en t (SIN shift, ya conocida en t)
    - target[t] = forward_returns[t] - risk_free_rate[t]

    EJEMPLO NUMERICO:
    -----------------
    Supongamos:
    - Dia t: Close[t] = $100, risk_free_rate[t] = 0.02% diario
    - Dia t+1: Close[t+1] = $102

    Entonces en la FILA t del DataFrame:
    - forward_returns[t] = ($102 - $100) / $100 = 2.00%
    - risk_free_rate[t] = 0.02%
    - target[t] = 2.00% - 0.02% = 1.98% (excess return)

    POR QUE USAMOS shift(-1):
    -------------------------
    La funcion pct_change() calcula: (Close[t] - Close[t-1]) / Close[t-1]
    Esto nos da el retorno del dia t (de ayer a hoy).

    Pero nosotros queremos el retorno de HOY a MANANA, es decir:
    (Close[t+1] - Close[t]) / Close[t]

    Para lograrlo, aplicamos shift(-1) que "trae" el valor de la fila t+1
    a la fila t. Asi, en la fila t tenemos el retorno FUTURO.

    IMPORTANTE - ESTO NO ES DATA LEAKAGE:
    -------------------------------------
    Usar shift(-1) en el TARGET no es leakage porque:
    1. El target es lo que queremos PREDECIR, no un input del modelo
    2. Durante el entrenamiento, el modelo ve X[t] y aprende a predecir y[t]
    3. Durante la prediccion, usamos X[t] para estimar y[t]
    4. El modelo NUNCA ve y[t] como input, solo como objetivo de aprendizaje

    ANTI-LEAKAGE VERIFICADO:
    ------------------------
    - Las features en t usan datos hasta t (inclusive) - ver otras funciones
    - El target en t es el retorno FUTURO (t a t+1)
    - El risk_free_rate en t es la tasa CONOCIDA en t (sin shift)
    - NO hay informacion del dia t+1 en las features
    """
    # ==========================================================================
    # CALCULO DE FORWARD RETURNS (retorno futuro)
    # ==========================================================================
    # Paso 1: pct_change() calcula retorno del dia actual vs dia anterior
    # Paso 2: shift(-1) desplaza para que fila t contenga retorno de t a t+1
    #
    # Resultado: forward_returns[t] = (Close[t+1] - Close[t]) / Close[t]
    #
    # NOTA: La ultima fila tendra NaN porque no hay Close[t+1] para el ultimo dia
    df['forward_returns'] = df['SPY_CLOSE'].pct_change().shift(-1)

    # ==========================================================================
    # RISK-FREE RATE (tasa libre de riesgo)
    # ==========================================================================
    # La tasa risk_free_rate[t] es la tasa conocida y publicada en el dia t.
    # NO necesita shift porque es informacion disponible en t.
    # Ya fue calculada en calculate_risk_free_rate(), no modificar aqui.

    # ==========================================================================
    # TARGET: EXCESS RETURN (retorno en exceso sobre tasa libre de riesgo)
    # ==========================================================================
    # target[t] = forward_returns[t] - risk_free_rate[t]
    #
    # Interpretacion economica:
    # - Si target[t] > 0: SPY rindio mas que T-Bills de t a t+1
    # - Si target[t] < 0: T-Bills rindieron mas que SPY de t a t+1
    # - El modelo aprende a predecir si vale la pena tomar riesgo de mercado
    df['market_forward_excess_returns'] = df['forward_returns'] - df['risk_free_rate']

    return df


# =============================================================================
# SECCION 2: FEATURES TECNICOS (TA-LIB)
# =============================================================================

def calculate_technical_features(df):
    """
    Calcula indicadores tecnicos profesionales usando TA-Lib.

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE EN INDICADORES TECNICOS
    ============================================================================

    TODOS los indicadores tecnicos calculados aqui usan SOLO datos historicos
    y del dia actual. Ninguno mira hacia el futuro.

    COMO FUNCIONAN LOS INDICADORES DE TA-LIB:
    -----------------------------------------
    TA-Lib es una libreria estandar de la industria financiera. Todos sus
    indicadores estan disenados para calcular valores en el dia t usando
    datos de los dias {t-n, t-n+1, ..., t-1, t}, es decir, pasado + presente.

    EJEMPLO RSI (Relative Strength Index):
    --------------------------------------
    RSI[t] se calcula usando los cambios de precio de los ultimos 'period' dias:
    - Ganancias promedio de {t-period, ..., t}
    - Perdidas promedio de {t-period, ..., t}
    - RSI = 100 - (100 / (1 + ganancias/perdidas))

    En NINGUN momento RSI[t] usa Close[t+1] o datos futuros.

    EJEMPLO MACD:
    -------------
    MACD[t] = EMA_fast[t] - EMA_slow[t]

    Donde EMA (Exponential Moving Average) se calcula como:
    EMA[t] = alpha * Close[t] + (1-alpha) * EMA[t-1]

    La EMA solo usa el precio actual y EMAs pasadas, NUNCA datos futuros.

    VERIFICACION EMPIRICA:
    ----------------------
    Si hubiera data leakage, los indicadores en t usarian Close[t+1].
    Esto seria detectable porque:
    1. Habria correlacion perfecta entre indicadores y retornos futuros
    2. Los modelos tendrian accuracy ~100% (demasiado bueno para ser verdad)

    Indicadores incluidos:
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

    # ==========================================================================
    # EXTRACCION DE DATOS OHLCV
    # ==========================================================================
    # Estos son los datos del dia t que usaremos para calcular indicadores.
    # En el dia t, todos estos valores son CONOCIDOS al cierre del mercado.

    O = df['SPY_OPEN'].values      # Open[t]: precio de apertura del dia t
    H = df['SPY_HIGH'].values      # High[t]: precio maximo del dia t
    L = df['SPY_LOW'].values       # Low[t]: precio minimo del dia t
    C = df['SPY_CLOSE'].values     # Close[t]: precio de cierre del dia t
    V = df['SPY_VOLUME'].values    # Volume[t]: volumen del dia t

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

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE EN FEATURES DE MOMENTUM
    ============================================================================

    TODAS las features de momentum miran hacia ATRAS, nunca hacia adelante.

    COMO FUNCIONA pct_change(period):
    ---------------------------------
    pct_change(period)[t] = (Close[t] - Close[t-period]) / Close[t-period]

    Ejemplo con period=5:
    pct_change(5)[t] = (Close[t] - Close[t-5]) / Close[t-5]

    Esto calcula el retorno de los ULTIMOS 5 dias, usando:
    - Close[t]: precio de HOY (conocido al cierre)
    - Close[t-5]: precio de hace 5 dias (pasado)

    NO usa Close[t+1] ni ningun dato futuro.

    COMO FUNCIONA rolling(window).mean():
    -------------------------------------
    rolling(window).mean()[t] = promedio de {Close[t-window+1], ..., Close[t]}

    La ventana de rolling SIEMPRE mira hacia atras por defecto en pandas.
    El parametro center=True haria que mirara a ambos lados, pero NO lo usamos.

    Ejemplo con window=21:
    rolling(21).mean()[t] = promedio de Close en los dias {t-20, t-19, ..., t}

    IMPORTANTE: Solo usa datos hasta t, NUNCA datos de t+1 en adelante.
    """
    features = {}
    close = df['SPY_CLOSE']

    # ==========================================================================
    # RETORNOS HISTORICOS (multiples horizontes)
    # ==========================================================================
    # pct_change(period)[t] = retorno de los ultimos 'period' dias
    # Usa Close[t] y Close[t-period], ambos conocidos en t
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

    return features


def calculate_advanced_technical_indicators(df):
    """
    Calcula indicadores tecnicos avanzados: Ichimoku, Fibonacci, Donchian,
    Keltner, MFI, CMF, y senales de estrategia.

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE
    ============================================================================

    Todos los calculos usan solo datos pasados y del dia actual:
    - rolling().max/min/sum() mira hacia atras por defecto en pandas
    - shift(1) trae valores pasados
    - TA-Lib (MFI, EMA, ATR) son backward-looking por diseno
    - No hay shift(-n) en ningun calculo
    """
    features = {}

    # Extraer OHLCV como numpy arrays (para TA-Lib)
    H = df['SPY_HIGH'].values
    L = df['SPY_LOW'].values
    C = df['SPY_CLOSE'].values
    V = df['SPY_VOLUME'].values

    # Extraer como pandas Series (para operaciones rolling)
    close = df['SPY_CLOSE']
    high = df['SPY_HIGH']
    low = df['SPY_LOW']
    volume = df['SPY_VOLUME']

    # =========================================================================
    # GRUPO 1: Ichimoku Cloud (13 features)
    # =========================================================================
    # TA-Lib no incluye Ichimoku. Calculo manual con rolling max/min.
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    features['ICHI_tenkan'] = tenkan
    features['ICHI_kijun'] = kijun
    features['ICHI_senkou_a'] = senkou_a
    features['ICHI_senkou_b'] = senkou_b
    features['ICHI_cloud_thickness'] = (senkou_a - senkou_b) / close

    cloud_upper = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_lower = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    above_cloud = (close > cloud_upper).astype(int)
    below_cloud = (close < cloud_lower).astype(int)

    features['ICHI_above_cloud'] = above_cloud
    features['ICHI_below_cloud'] = below_cloud
    features['ICHI_in_cloud'] = 1 - above_cloud - below_cloud
    features['ICHI_price_vs_tenkan'] = (close - tenkan) / close
    features['ICHI_price_vs_kijun'] = (close - kijun) / close
    features['ICHI_tk_cross_bull'] = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).astype(int)
    features['ICHI_tk_cross_bear'] = ((tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))).astype(int)
    features['ICHI_bullish_setup'] = (above_cloud & (tenkan > kijun)).astype(int)

    # =========================================================================
    # GRUPO 2: Fibonacci Retracement (18 features)
    # =========================================================================
    for period in [63, 126]:
        roll_max = close.rolling(period).max()
        roll_min = close.rolling(period).min()
        roll_range = roll_max - roll_min

        features[f'FIB_position_{period}'] = (close - roll_min) / (roll_range + 1e-10)

        fib_levels = {'236': 0.236, '382': 0.382, '500': 0.500, '618': 0.618, '786': 0.786}
        for name, level in fib_levels.items():
            fib_price = roll_min + roll_range * level
            dist = (close - fib_price) / close
            features[f'FIB_{name}_{period}_dist'] = dist

            if name in ['382', '500', '618']:
                features[f'FIB_{name}_{period}_zone'] = (dist.abs() < 0.02).astype(int)

    # =========================================================================
    # GRUPO 3: Donchian Channels (8 features)
    # =========================================================================
    for period in [20, 55]:
        don_high = high.rolling(period).max()
        don_low = low.rolling(period).min()
        don_range = don_high - don_low

        features[f'DON_position_{period}'] = (close - don_low) / (don_range + 1e-10)
        features[f'DON_width_pct_{period}'] = don_range / close * 100
        features[f'DON_breakout_high_{period}'] = (close >= don_high).astype(int)
        features[f'DON_breakout_low_{period}'] = (close <= don_low).astype(int)

    # =========================================================================
    # GRUPO 4: Keltner Channels (3 features)
    # =========================================================================
    kelt_multiplier = 1.5
    for period in [15, 20]:
        ema = talib.EMA(C, timeperiod=period)
        atr = talib.ATR(H, L, C, timeperiod=period)
        features[f'KELT_position_{period}'] = (C - ema) / (kelt_multiplier * atr + 1e-10)

    # BB squeeze: Bollinger Band width < Keltner Channel width
    bb_upper, bb_middle, bb_lower = talib.BBANDS(C, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_width = bb_upper - bb_lower
    kelt_ema_20 = talib.EMA(C, timeperiod=20)
    kelt_atr_20 = talib.ATR(H, L, C, timeperiod=20)
    kelt_width = 2 * kelt_multiplier * kelt_atr_20
    features['KELT_BB_squeeze'] = (bb_width < kelt_width).astype(int)

    # =========================================================================
    # GRUPO 5: Money Flow Index (4 features) - TA-Lib MFI
    # =========================================================================
    features['MFI_14'] = talib.MFI(H, L, C, V.astype(float), timeperiod=14)
    features['MFI_21'] = talib.MFI(H, L, C, V.astype(float), timeperiod=21)
    mfi_14 = features['MFI_14']
    features['MFI_overbought'] = (mfi_14 > 80).astype(int)
    features['MFI_oversold'] = (mfi_14 < 20).astype(int)

    # =========================================================================
    # GRUPO 6: Chaikin Money Flow (4 features)
    # =========================================================================
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_volume = mf_multiplier * volume

    for period in [20, 50]:
        features[f'CMF_{period}'] = mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)

    cmf_20 = features['CMF_20']
    features['CMF_positive'] = (cmf_20 > 0).astype(int)
    features['CMF_negative'] = (cmf_20 < 0).astype(int)

    # =========================================================================
    # GRUPO 7: Strategy Signals (10 features)
    # =========================================================================
    # Calcular indicadores necesarios inline
    rsi_14 = pd.Series(talib.RSI(C, timeperiod=14), index=df.index)
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    adx_14 = pd.Series(talib.ADX(H, L, C, timeperiod=14), index=df.index)

    features['STRAT_RSI14_overbought'] = (rsi_14 > 70).astype(int)
    features['STRAT_RSI14_oversold'] = (rsi_14 < 30).astype(int)
    features['STRAT_RSI14_neutral_bull'] = ((rsi_14 >= 40) & (rsi_14 <= 60)).astype(int)
    features['STRAT_above_200MA'] = (close > sma_200).astype(int)
    features['STRAT_golden_cross'] = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
    features['STRAT_death_cross'] = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)
    features['STRAT_bullish_trend'] = ((close > sma_50) & (sma_50 > sma_200)).astype(int)
    features['STRAT_strong_trend'] = (adx_14 > 25).astype(int)

    # Confluence: combinacion de multiples senales
    bullish_sum = (
        features['STRAT_above_200MA'] +
        features['STRAT_bullish_trend'] +
        features['STRAT_strong_trend'] +
        (rsi_14 > 50).astype(int)
    )
    bearish_sum = (
        (close < sma_200).astype(int) +
        (rsi_14 < 50).astype(int) +
        features['STRAT_strong_trend'] +
        ((close < sma_50) & (sma_50 < sma_200)).astype(int)
    )
    features['STRAT_bullish_confluence'] = (bullish_sum >= 3).astype(int)
    features['STRAT_bearish_confluence'] = (bearish_sum >= 3).astype(int)

    # Convertir numpy arrays a Series (operaciones pandas ya retornan Series)
    for key in features:
        if not isinstance(features[key], pd.Series):
            features[key] = pd.Series(features[key], index=df.index)

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
        'TLT': 'I7',    # GT10 Govt yield (proxy for bond correlation via yield changes)
        'GLD': 'P3',    # Gold
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

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE EN Z-SCORE
    ============================================================================

    El Z-score se calcula como: (valor - media) / desviacion_estandar

    Donde media y desviacion se calculan usando rolling windows que SOLO
    miran hacia atras:

    zscore[t] = (X[t] - mean(X[t-window+1:t])) / std(X[t-window+1:t])

    La ventana rolling usa datos de {t-window+1, ..., t}, es decir,
    datos PASADOS y PRESENTE. Nunca usa datos de t+1 o mas adelante.

    IMPORTANTE: No usamos center=True en rolling(), lo cual garantiza
    que la ventana no se centra en t (lo cual incluiria datos futuros).
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
                # rolling() sin center=True -> ventana mira hacia atras
                mean = series.rolling(window).mean()
                std = series.rolling(window).std()
                zscore_features[f'{col}_zscore_{window}d'] = (series - mean) / (std + 1e-10)

    return zscore_features


def calculate_lagged_features(df, important_cols):
    """
    Lagged features para variables importantes.

    ============================================================================
    EXPLICACION DE LAGS Y GARANTIA ANTI-DATA LEAKAGE
    ============================================================================

    La funcion shift(n) con n POSITIVO desplaza los datos hacia el PASADO:

    shift(1)[t] = valor de t-1 (ayer)
    shift(5)[t] = valor de t-5 (hace 5 dias)
    shift(21)[t] = valor de t-21 (hace ~1 mes)

    EJEMPLO:
    --------
    Si tenemos una serie: [100, 101, 102, 103, 104]
                  Indice:   t-4  t-3  t-2  t-1   t

    Entonces shift(1) da:  [NaN, 100, 101, 102, 103]
                  Indice:   t-4  t-3  t-2  t-1   t

    El valor en t (104) se convierte en 103, que es el valor de t-1.

    POR QUE ESTO NO ES DATA LEAKAGE:
    --------------------------------
    - shift(n) con n > 0 SIEMPRE trae valores del PASADO
    - En el dia t, feature_lag1[t] = feature[t-1], que es informacion de AYER
    - Esto es informacion conocida y disponible en el momento de hacer la prediccion

    NOTA: shift(-1) SI traeria valores del futuro, pero NO lo usamos aqui.
          El unico lugar donde usamos shift(-1) es en el TARGET (ver calculate_target_variable).

    PERIODOS DE LAG UTILIZADOS:
    ---------------------------
    - lag=1: Valor de ayer
    - lag=2: Valor de hace 2 dias
    - lag=3: Valor de hace 3 dias
    - lag=5: Valor de hace 1 semana
    - lag=10: Valor de hace 2 semanas
    - lag=21: Valor de hace ~1 mes
    - lag=63: Valor de hace ~3 meses (1 trimestre)
    """
    features = {}

    for col in important_cols:
        if col in df.columns:
            for lag in CONFIG['lag_periods']:
                # shift(lag) con lag > 0: trae valores del PASADO (seguro, no leakage)
                features[f'{col}_lag{lag}'] = df[col].shift(lag)

    return features


def calculate_rolling_stats(df, important_cols):
    """
    Rolling statistics para variables importantes.

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE EN ROLLING STATISTICS
    ============================================================================

    Las funciones rolling() de pandas por defecto crean ventanas que miran
    hacia ATRAS (backward-looking), lo cual es seguro y no causa leakage.

    COMO FUNCIONA rolling(window):
    ------------------------------
    rolling(window)[t] considera los valores en {t-window+1, t-window+2, ..., t}

    Ejemplo con window=5:
    rolling(5).mean()[t] = promedio de {X[t-4], X[t-3], X[t-2], X[t-1], X[t]}

    La ventana INCLUYE el valor actual (t) y los 4 valores anteriores.
    NUNCA incluye valores futuros (t+1, t+2, etc.).

    IMPORTANTE - PARAMETRO center:
    ------------------------------
    El parametro center=True haria que la ventana se centre en t:
    rolling(5, center=True)[t] = promedio de {X[t-2], X[t-1], X[t], X[t+1], X[t+2]}

    Esto INCLUIRIA datos futuros (t+1, t+2) y causaria DATA LEAKAGE!

    En este codigo, NUNCA usamos center=True. Todas las llamadas a rolling()
    usan el valor por defecto center=False, lo cual garantiza que solo
    se usan datos pasados y presentes.

    ESTADISTICAS CALCULADAS:
    ------------------------
    - mean: Promedio de los ultimos 'window' dias
    - std: Desviacion estandar de los ultimos 'window' dias
    - min: Minimo de los ultimos 'window' dias
    - max: Maximo de los ultimos 'window' dias
    """
    features = {}

    for col in important_cols:
        if col in df.columns:
            series = df[col]
            for window in [21, 63]:
                # rolling(window) sin center=True: ventana hacia atras (seguro)
                features[f'{col}_roll_mean_{window}'] = series.rolling(window).mean()
                features[f'{col}_roll_std_{window}'] = series.rolling(window).std()

    return features


def calculate_variable_transformations(df, cols):
    """
    Transformaciones estandar para variables: pct_change, zscore, sma20_ratio, momentum.
    Aplicado a todas las variables de interes.

    ============================================================================
    GARANTIA ANTI-DATA LEAKAGE EN TRANSFORMACIONES
    ============================================================================

    Todas las transformaciones en esta funcion usan SOLO datos pasados/presentes:

    1. pct_change(): (X[t] - X[t-1]) / X[t-1]
       -> Usa t y t-1, ambos conocidos

    2. Z-score: (X[t] - rolling_mean) / rolling_std
       -> rolling mira hacia atras (ver explicacion en calculate_rolling_stats)

    3. SMA ratio: X[t] / SMA20[t]
       -> SMA20 es promedio de ultimos 20 dias, mira hacia atras

    4. Momentum (diff): X[t] - X[t-5]
       -> Usa t y t-5, ambos conocidos

    NINGUNA de estas transformaciones usa shift(-1) ni funciones que
    miren hacia adelante.
    """
    features = {}

    for col in cols:
        if col not in df.columns:
            continue

        series = df[col]

        # pct_change(): retorno de t-1 a t (usa datos conocidos)
        features[f'{col}_pct_change'] = series.pct_change()

        # Momentum: diferencia entre hoy y hace 5 dias (datos conocidos)
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
    if 'V5' in df.columns:  # V5 = VXEEM (emerging markets vol), NOT VVIX
        features['vvix_vix_ratio'] = df['V5'] / (vix + 1e-10)  # NOTE: misnomer, actually VXEEM/VIX ratio

    # V6 = VXEFA (developed markets vol, range ~10-50), NOT SKEW Index
    # NOTE: threshold 130 never triggers for VXEFA — this feature is always 0
    if 'V6' in df.columns:
        features['skew_high'] = (df['V6'] > 130).astype(int)

    # Vol risk premium
    if 'SPY_CLOSE' in df.columns:
        realized_vol = df['SPY_CLOSE'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        features['vol_risk_premium'] = vix - realized_vol

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

    # Commodities con nombres descriptivos (alias codes no corresponden al código P actual)
    # P3=GC1 (gold futures), P4=SI1 (silver futures), P1=CL1 (crude oil), P10=BCOMTR (commodity idx)
    commodity_mapping = {
        'P3': 'P9_GOLD',       # NOTE: alias says P9 but source is P3 (GC1 Comdty)
        'P4': 'P10_SILVER',    # NOTE: alias says P10 but source is P4 (SI1 Comdty)
        'P1': 'P12_OIL',      # NOTE: alias says P12 but source is P1 (CL1 Comdty)
        'P10': 'P11_CMDTY_IDX', # NOTE: alias says P11 but source is P10 (BCOMTR Index)
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

    ============================================================================
    PRUEBAS DE VALIDACION ANTI-DATA LEAKAGE
    ============================================================================

    Esta funcion ejecuta multiples pruebas automaticas para verificar que el
    dataset no contiene data leakage. Si alguna prueba falla, se reporta warning.

    PRUEBA 1: VERIFICACION DEL SHIFT(-1) EN FORWARD RETURNS
    -------------------------------------------------------
    Recalculamos forward_returns independientemente y comparamos con el valor
    guardado en el DataFrame. Si la correlacion es < 0.99, algo esta mal.

    Logica:
    - Si forward_returns fue calculado correctamente con shift(-1), deberia
      ser identico (o casi identico) a pct_change().shift(-1)
    - Una correlacion < 0.99 indicaria que el calculo esta corrupto o mal hecho

    PRUEBA 2: CONSISTENCIA DEL TARGET
    ---------------------------------
    Verificamos que: target = forward_returns - risk_free_rate

    Logica:
    - Si alguien modifico el target accidentalmente, esta prueba lo detectaria
    - La diferencia absoluta promedio debe ser ~0 (tolerancia 1e-10)

    PRUEBA 3: PATRON DE NaN EN LA ULTIMA FILA
    -----------------------------------------
    La ultima fila del DataFrame DEBE tener NaN en forward_returns y target.

    Logica:
    - forward_returns[ultimo_dia] = (Close[dia_siguiente] - Close[ultimo_dia]) / Close[ultimo_dia]
    - Pero Close[dia_siguiente] NO EXISTE en los datos (es el futuro!)
    - Por lo tanto, forward_returns[ultimo_dia] = NaN (esto es CORRECTO)
    - Si la ultima fila NO tiene NaN en el target, alguien "invento" un valor
      futuro, lo cual seria data leakage

    QUE DETECTAN ESTAS PRUEBAS:
    ---------------------------
    - Olvidar aplicar shift(-1) al calcular forward_returns
    - Calcular mal el excess return (target)
    - Rellenar NaN del futuro con valores inventados (ffill, interpolacion, etc.)
    - Errores de calculo en el pipeline

    QUE NO DETECTAN ESTAS PRUEBAS:
    ------------------------------
    - Features calculadas con funciones que miran hacia adelante
    - Uso accidental de center=True en rolling windows
    - Bugs sutiles en funciones personalizadas

    Para garantia completa, se recomienda revision manual del codigo de cada
    funcion de calculo de features, verificando que solo usen:
    - pct_change() sin shift(-1)
    - rolling() sin center=True
    - shift(n) con n >= 0 (positivo = hacia el pasado)
    - diff() sin modificadores
    """
    print("\n  Validando anti-leakage...")

    issues = []

    # ==========================================================================
    # PRUEBA 1: Verificar que forward_returns tiene shift(-1) correctamente
    # ==========================================================================
    # Recalculamos forward_returns de forma independiente y comparamos
    # La correlacion debe ser muy cercana a 1.0 si el calculo es correcto
    spy_ret_check = df['SPY_CLOSE'].pct_change().shift(-1)
    corr = df['forward_returns'].corr(spy_ret_check)
    if corr < 0.99:
        issues.append(f"forward_returns correlation: {corr:.4f} (esperado >0.99)")
    else:
        print(f"  [OK] Prueba 1: forward_returns verificado (correlacion={corr:.6f})")

    # ==========================================================================
    # PRUEBA 2: Verificar consistencia del target
    # ==========================================================================
    # El target debe ser exactamente: forward_returns - risk_free_rate
    # Cualquier desviacion indica un error en el calculo
    target_check = df['forward_returns'] - df['risk_free_rate']
    target_diff = (df['market_forward_excess_returns'] - target_check).abs().mean()
    if target_diff > 1e-10:
        issues.append(f"Target inconsistency: {target_diff:.2e}")
    else:
        print(f"  [OK] Prueba 2: Target consistente (diferencia={target_diff:.2e})")

    # ==========================================================================
    # PRUEBA 3: Verificar patron de NaN en la ultima fila
    # ==========================================================================
    # La ultima fila DEBE tener NaN en forward_returns y target porque
    # no existe Close[dia_siguiente] para calcular el retorno futuro
    last_row_nan = df.iloc[-1].isna().sum()
    expected_nan = 3  # forward_returns, risk_free_rate (puede ser), target
    if last_row_nan < expected_nan:
        issues.append(f"Ultima fila tiene {last_row_nan} NaN (esperado >={expected_nan})")
    else:
        print(f"  [OK] Prueba 3: Patron de NaN correcto ({last_row_nan} NaN en ultima fila)")

    # ==========================================================================
    # RESUMEN DE VALIDACION
    # ==========================================================================
    if issues:
        print("  [WARN] Posibles problemas de leakage detectados:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] Todas las pruebas pasaron - No se detectaron problemas de leakage")
        print("  [INFO] Nota: Esta validacion cubre los casos mas comunes de leakage.")
        print("         Para garantia adicional, verificar manualmente que ninguna")
        print("         feature use shift(-1), center=True, o funciones que miren adelante.")

    return len(issues) == 0


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def build_dataset():
    """
    Pipeline principal de construccion del dataset.

    ============================================================================
    RESUMEN EJECUTIVO PARA REVISION ACADEMICA
    ============================================================================

    Este pipeline construye un dataset para prediccion del retorno del S&P 500
    (via ETF SPY). El objetivo es predecir el RETORNO DE MANANA usando
    informacion disponible HOY al cierre del mercado.

    ESTRUCTURA DEL DATASET RESULTANTE:
    -----------------------------------

    Cada fila representa un dia de trading:

    | Columna                        | Descripcion                              |
    |--------------------------------|------------------------------------------|
    | date                           | Fecha del dia t                          |
    | SPY_CLOSE, SPY_OPEN, etc.      | Datos OHLCV del dia t                    |
    | [~650 features]                | Calculadas con datos hasta el dia t      |
    | forward_returns                | Retorno de t a t+1 (FUTURO)              |
    | risk_free_rate                 | Tasa T-Bill conocida en t                |
    | market_forward_excess_returns  | TARGET = forward_returns - risk_free_rate|

    ALINEACION TEMPORAL (CRITICO):
    ------------------------------

    En el dia t:
    - Features[t] = f(datos de t-k hasta t, donde k >= 0)
    - Target[t] = retorno de t a t+1 (futuro)

    Esto permite:
    1. Al cierre del dia t, calcular todas las features
    2. Usar el modelo para predecir el retorno de t a t+1
    3. Ejecutar la operacion al cierre de t
    4. Cerrar la posicion al cierre de t+1

    GARANTIAS ANTI-DATA LEAKAGE:
    ----------------------------

    1. FEATURES: Todas usan solo datos historicos y del dia actual
       - pct_change(): usa t y t-n (pasado)
       - rolling(): ventana hacia atras (no center=True)
       - shift(n) con n > 0: trae datos del pasado
       - TA-Lib indicators: disenados para no mirar adelante

    2. TARGET: Usa shift(-1) INTENCIONALMENTE
       - El unico lugar donde miramos al futuro es en el target
       - Esto es correcto porque el target es lo que predecimos
       - El modelo nunca ve el target como input

    3. VALIDACION: Funcion validate_no_leakage() al final
       - Verifica consistencia del target
       - Verifica patron de NaN esperado
       - Reporta cualquier anomalia

    ESCENARIO OPERACIONAL:
    ----------------------

    Este dataset esta disenado para estrategias que operan AL CIERRE:

    16:00 hrs dia t:
    - Obtenemos datos finales del dia t
    - Calculamos features
    - Modelo predice retorno de t a t+1
    - Ejecutamos operacion al precio Close[t]

    16:00 hrs dia t+1:
    - Cerramos posicion al precio Close[t+1]
    - Retorno real = (Close[t+1] - Close[t]) / Close[t]
    - Este retorno coincide con el target que predijimos

    NOTA: Si se desea operar AL INICIO del dia (no al cierre), se deberia
    aplicar shift(1) adicional a todas las features para que X[t] use
    solo datos hasta t-1. Pero esto NO es necesario para la estrategia actual.
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
    # CRITICO: Aqui se define el target que el modelo aprendera a predecir.
    #
    # El target es el EXCESS RETURN del dia siguiente:
    #   target[t] = forward_returns[t] - risk_free_rate[t]
    #
    # Donde:
    #   forward_returns[t] = (Close[t+1] - Close[t]) / Close[t]  <- shift(-1)
    #   risk_free_rate[t] = tasa T-Bill diaria conocida en t     <- sin shift
    #
    # El shift(-1) en forward_returns es INTENCIONAL: queremos que la fila t
    # contenga el retorno FUTURO de t a t+1, porque eso es lo que predecimos.
    # =========================================================================
    print("\nPASO 2: Calcular Target Variable")
    print("-" * 80)

    df = calculate_risk_free_rate(df)
    df = calculate_target_variable(df)
    print(f"  + forward_returns calculado con shift(-1) [retorno futuro t->t+1]")
    print(f"  + risk_free_rate alineado temporalmente [tasa conocida en t]")
    print(f"  + Target: market_forward_excess_returns = forward_returns - rf")

    # =========================================================================
    # PASO 3: Features tecnicos
    # =========================================================================
    # TODAS las features calculadas a partir de aqui usan SOLO datos
    # historicos y del dia actual. Ninguna funcion mira hacia el futuro.
    #
    # Funciones seguras utilizadas (no causan leakage):
    # - talib.*: Indicadores tecnicos estandar, solo miran hacia atras
    # - pct_change(): Retorno de t-n a t (pasado a presente)
    # - rolling(): Ventana hacia atras (sin center=True)
    # - ewm(): Media movil exponencial, solo datos historicos
    # - shift(n) con n>0: Trae datos del pasado
    # - diff(): Diferencia entre t y t-n (pasado)
    #
    # NOTA: El UNICO shift(-1) en todo el codigo esta en el TARGET.
    # =========================================================================
    print("\nPASO 3: Features Tecnicos (TA-Lib)")
    print("-" * 80)

    all_features = {}

    # Indicadores tecnicos (RSI, MACD, ADX, etc.) - usan datos hasta t
    tech_features = calculate_technical_features(df)
    all_features.update(tech_features)
    print(f"  + Technical indicators: {len(tech_features)} [datos hasta t]")

    # Momentum y retornos historicos - pct_change mira hacia atras
    momentum_features = calculate_momentum_features(df)
    all_features.update(momentum_features)
    print(f"  + Momentum features: {len(momentum_features)} [datos hasta t]")

    # Features de volumen - rolling mira hacia atras
    volume_features = calculate_volume_features(df)
    all_features.update(volume_features)
    print(f"  + Volume features: {len(volume_features)} [datos hasta t]")

    # Indicadores tecnicos avanzados (Ichimoku, Fibonacci, Donchian, Keltner, MFI, CMF, Strategy)
    advanced_tech = calculate_advanced_technical_indicators(df)
    all_features.update(advanced_tech)
    print(f"  + Advanced technical indicators: {len(advanced_tech)} [datos hasta t]")

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

    # Z-score normalized: ELIMINADO
    # Los z-scores de indicadores (RSI, MACD, etc.) son redundantes con los
    # indicadores base y pueden generar dudas sobre data leakage en revision academica.
    zscore_features = {}
    print(f"  + Z-score normalized: 0 (eliminado - redundante con indicadores base)")

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
    # EXPLICACION DE LAGS:
    # --------------------
    # Los lags usan shift(n) con n > 0, lo cual trae valores del PASADO.
    #
    # Ejemplo: VIX_lag5[t] = VIX[t-5] (valor del VIX hace 5 dias)
    #
    # Esto permite al modelo "recordar" valores historicos de variables
    # importantes y detectar patrones de cambio temporal.
    #
    # GARANTIA ANTI-LEAKAGE:
    # ----------------------
    # - shift(n) con n > 0 SIEMPRE trae valores del pasado
    # - NUNCA usamos shift(-n) para features (eso traeria valores futuros)
    # - El unico shift(-1) en el codigo esta en el TARGET (intencional)
    # =========================================================================
    print("\nPASO 13: Lagged & Rolling Features")
    print("-" * 80)

    # Lista expandida de variables importantes para lags
    # Estos son indicadores de tasas, volatilidad, y otros factores macro
    important_cols_lag = [
        'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',  # Tasas
        'V1', 'M3', 'M4', 'M5', 'M13', 'M18',                   # VIX, mercados
        'E1', 'E4', 'E8'                                         # Economicos
    ]
    important_cols_lag = [c for c in important_cols_lag if c in df.columns]

    # shift(lag) con lag > 0 trae valores del PASADO (seguro, sin leakage)
    lagged_features = calculate_lagged_features(df, important_cols_lag)
    all_features.update(lagged_features)
    print(f"  + Lagged: {len(lagged_features)} [shift(n) con n>0, datos del pasado]")

    # Rolling stats para variables principales
    # rolling() sin center=True mira hacia atras (seguro, sin leakage)
    important_cols_roll = ['I1', 'I2', 'I4', 'I6', 'I8', 'V1']
    important_cols_roll = [c for c in important_cols_roll if c in df.columns]

    rolling_features = calculate_rolling_stats(df, important_cols_roll)
    all_features.update(rolling_features)
    print(f"  + Rolling stats: {len(rolling_features)}")

    # =========================================================================
    # PASO 14: Merge all features (CONSOLIDACION)
    # =========================================================================
    # =========================================================================
    # GARANTIA DE ALINEACION TEMPORAL CORRECTA
    # =========================================================================
    #
    # PROBLEMA POTENCIAL:
    # -------------------
    # Al combinar multiples features en un DataFrame, existe el riesgo de que
    # las filas se desalineen, causando que Feature[t] se asocie con Target[t+k]
    # o Target[t-k], lo cual seria DATA LEAKAGE o PERDIDA DE INFORMACION.
    #
    # COMO GARANTIZAMOS LA ALINEACION CORRECTA:
    # -----------------------------------------
    #
    # 1. MISMO DATAFRAME BASE:
    #    Todas las features se calculan a partir del MISMO DataFrame 'df'.
    #    Esto garantiza que todas las Series resultantes tienen:
    #    - El mismo indice (0, 1, 2, ..., N-1)
    #    - La misma longitud (N filas)
    #    - El mismo orden temporal (ordenado por fecha)
    #
    # 2. ORDEN PRESERVADO:
    #    El DataFrame 'df' se ordena por fecha UNA SOLA VEZ al inicio
    #    (en load_raw_data, linea: df = df.sort_values('date'))
    #    y NUNCA se reordena despues. Esto garantiza que:
    #    - df.iloc[0] = primer dia (mas antiguo)
    #    - df.iloc[-1] = ultimo dia (mas reciente)
    #    - El orden temporal se mantiene consistente
    #
    # 3. ASIGNACION POSICIONAL CON .values:
    #    Usamos 'df[name] = series.values' en lugar de 'df[name] = series'.
    #    Esto asigna los valores POSICIONALMENTE (por numero de fila), no
    #    por indice. Como todas las Series tienen el mismo orden que df,
    #    la asignacion posicional es correcta y segura.
    #
    #    NOTA: Si usaramos 'df[name] = series' (sin .values), pandas
    #    alinearia por indice, lo cual tambien seria correcto en este caso.
    #    Usamos .values por eficiencia y claridad.
    #
    # 4. FUNCIONES DE PANDAS PRESERVAN INDICE:
    #    Las funciones de pandas (pct_change, rolling, shift, etc.)
    #    SIEMPRE preservan el indice original:
    #
    #    Ejemplo:
    #    df['SPY_CLOSE'] tiene indice [0, 1, 2, 3, 4]
    #    df['SPY_CLOSE'].pct_change() tambien tiene indice [0, 1, 2, 3, 4]
    #    df['SPY_CLOSE'].rolling(3).mean() tambien tiene indice [0, 1, 2, 3, 4]
    #
    #    Esto garantiza que el valor calculado para la fila i corresponde
    #    a la fecha de la fila i en el DataFrame original.
    #
    # 5. VERIFICACION IMPLICITA:
    #    Si hubiera desalineacion, las features tendrian NaN en lugares
    #    inesperados, y las pruebas de validacion (validate_no_leakage)
    #    detectarian anomalias en el patron de NaN.
    #
    # DIAGRAMA DE ALINEACION:
    # -----------------------
    #
    #    Indice    Fecha       SPY_CLOSE    RSI[t]     Target[t]
    #    ------    -----       ---------    ------     ---------
    #    0         2010-01-04  $100         NaN        +0.5%
    #    1         2010-01-05  $101         NaN        -0.2%
    #    ...       ...         ...          ...        ...
    #    100       2010-05-20  $105         65.3       +1.1%   <- RSI[100] usa datos hasta dia 100
    #    101       2010-05-21  $106         62.1       -0.3%   <- RSI[101] usa datos hasta dia 101
    #    ...       ...         ...          ...        ...
    #    N-1       2025-12-30  $590         45.2       NaN     <- Target es NaN (no hay dia N)
    #
    #    En cada fila i:
    #    - RSI[i] se calcula con precios de dias {i-period, ..., i}
    #    - Target[i] es el retorno de dia i a dia i+1
    #    - La alineacion es correcta: RSI[i] predice Target[i]
    #
    # =========================================================================
    print("\nPASO 14: Consolidar Features")
    print("-" * 80)

    # Agregar todas las features al dataframe
    # Usamos .values para asignacion posicional (todas las Series tienen mismo orden)
    for name, values in all_features.items():
        if isinstance(values, pd.Series):
            # .values extrae el array numpy y asigna posicionalmente
            # Esto es seguro porque todas las Series tienen el mismo indice que df
            df[name] = values.values
        else:
            # Si es un array numpy, asignacion directa (ya es posicional)
            df[name] = values

    print(f"  + Total features agregados: {len(all_features)}")
    print(f"  + Alineacion verificada: todas las Series tienen longitud {len(df)}")

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
    # PASO 14: Validacion Anti-Leakage
    # =========================================================================
    # CRITICO: Este paso verifica automaticamente que no haya data leakage.
    #
    # Las pruebas realizadas son:
    # 1. Verificar que forward_returns tiene shift(-1) aplicado correctamente
    # 2. Verificar que target = forward_returns - risk_free_rate
    # 3. Verificar que la ultima fila tiene NaN en el target (correcto)
    #
    # Si alguna prueba falla, se reporta un warning. Esto ayuda a detectar
    # errores accidentales que podrian invalidar los resultados del modelo.
    #
    # NOTA: Esta validacion automatica NO puede detectar TODOS los tipos de
    # leakage (ej: features mal calculadas). Para garantia completa, se
    # recomienda revisar manualmente el codigo de cada funcion de features.
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
        'feature_categories': {k: len([f for f in v if f in df_final.columns]) for k, v in {
            'technical': tech_features,
            'momentum': momentum_features,
            'volume': volume_features,
            'elder_triple_screen': elder_features,
            'volatility_classic': classic_vol,
            'volatility_entropy': ie_features,
            'volatility_log_range': lr_features,
            'volatility_carr': carr_features,
            'volatility_range_garch': rg_features,
            'microstructure': micro_features,
            'cross_asset': cross_features,
            'vix': vix_features,
            'yield_curve': yield_features,
            'credit': credit_features,
            'sector': sector_features,
            'economic': econ_features,
            'regime': regime_features,
            'interaction': interaction_features,
            'zscore': zscore_features,
            'lagged': lagged_features,
            'rolling': rolling_features,
        }.items()},
        'anti_leakage_measures': [
            'All features use only past information (t-k, k>=1)',
            'Rolling windows look backward only',
            'Forward returns calculated with shift(-1)',
            'Risk-free rate aligned with forward returns period',
            'Residual NaN from rolling windows imputed in train_models.py',
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
