# -*- coding: utf-8 -*-
"""
================================================================================
AUDITORIA COMPLETA DEL PIPELINE DE DATOS
================================================================================

Revision exhaustiva de:
1. Variables descargadas de Bloomberg
2. Mapeo de nombres de variables
3. Calidad de datos (missing, outliers)
4. Feature engineering
5. Indicadores Elder Triple Pantalla
6. Dataset final

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

DATA_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data"
FINAL_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data\final"

print("=" * 80)
print("AUDITORIA COMPLETA DEL PIPELINE DE DATOS")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()

# =============================================================================
# 1. MAPEO DE VARIABLES BLOOMBERG
# =============================================================================

# Mapeo de codigos internos a descripciones y tickers Bloomberg
VARIABLE_MAP = {
    # === MERCADO (M) ===
    "M1": {"ticker": "SPY US Equity", "desc": "S&P 500 ETF", "field": "PX_LAST"},
    "M2": {"ticker": "DIA US Equity", "desc": "Dow Jones ETF", "field": "PX_LAST"},
    "M3": {"ticker": "QQQ US Equity", "desc": "Nasdaq 100 ETF", "field": "PX_LAST"},
    "M4": {"ticker": "IWM US Equity", "desc": "Russell 2000 ETF", "field": "PX_LAST"},
    "M5": {"ticker": "EFA US Equity", "desc": "EAFE International ETF", "field": "PX_LAST"},
    "M6": {"ticker": "EEM US Equity", "desc": "Emerging Markets ETF", "field": "PX_LAST"},
    "M7": {"ticker": "XLF US Equity", "desc": "Financial Sector ETF", "field": "PX_LAST"},
    "M8": {"ticker": "XLK US Equity", "desc": "Technology Sector ETF", "field": "PX_LAST"},
    "M9": {"ticker": "XLE US Equity", "desc": "Energy Sector ETF", "field": "PX_LAST"},
    "M10": {"ticker": "XLV US Equity", "desc": "Healthcare Sector ETF", "field": "PX_LAST"},
    "M11": {"ticker": "XLI US Equity", "desc": "Industrial Sector ETF", "field": "PX_LAST"},
    "M12_REIT": {"ticker": "VNQ US Equity", "desc": "Real Estate ETF", "field": "PX_LAST"},
    "M13": {"ticker": "AGG US Equity", "desc": "US Aggregate Bond ETF", "field": "PX_LAST"},
    "M14": {"ticker": "IWD US Equity", "desc": "Russell 1000 Value ETF", "field": "PX_LAST"},
    "M15": {"ticker": "IWF US Equity", "desc": "Russell 1000 Growth ETF", "field": "PX_LAST"},
    "M16": {"ticker": "TLT US Equity", "desc": "20+ Year Treasury ETF", "field": "PX_LAST"},
    "M17": {"ticker": "SHY US Equity", "desc": "1-3 Year Treasury ETF", "field": "PX_LAST"},
    "M18": {"ticker": "DXY Curncy", "desc": "US Dollar Index", "field": "PX_LAST"},
    "NYSE_ADV": {"ticker": "ADVANCE Index", "desc": "NYSE Advancing Issues", "field": "PX_LAST"},
    "NYSE_DEC": {"ticker": "DECLINE Index", "desc": "NYSE Declining Issues", "field": "PX_LAST"},

    # === ECONOMICO (E) ===
    "E1": {"ticker": "GDP CQOQ Index", "desc": "US GDP QoQ", "field": "PX_LAST"},
    "E2": {"ticker": "USURTOT Index", "desc": "US Unemployment Rate", "field": "PX_LAST"},
    "E3": {"ticker": "NFP TCH Index", "desc": "US Nonfarm Payrolls", "field": "PX_LAST"},
    "E4": {"ticker": "CPI YOY Index", "desc": "US CPI YoY", "field": "PX_LAST"},
    "E5": {"ticker": "CPUPXCHG Index", "desc": "US Core CPI YoY", "field": "PX_LAST"},
    "E6": {"ticker": "PCE CYOY Index", "desc": "US PCE YoY", "field": "PX_LAST"},
    "E7": {"ticker": "NAPMPMI Index", "desc": "ISM Manufacturing PMI", "field": "PX_LAST"},
    "E8": {"ticker": "CONCCONF Index", "desc": "Consumer Confidence", "field": "PX_LAST"},
    "E9": {"ticker": "CONSSENT Index", "desc": "U. Michigan Sentiment", "field": "PX_LAST"},
    "E10": {"ticker": "RSTAMOM Index", "desc": "US Retail Sales MoM", "field": "PX_LAST"},
    "E11": {"ticker": "NHSPSTOT Index", "desc": "US Housing Starts", "field": "PX_LAST"},
    "E12": {"ticker": "IP CHNG Index", "desc": "Industrial Production MoM", "field": "PX_LAST"},
    "E13": {"ticker": "LEI TOTL Index", "desc": "Leading Economic Index", "field": "PX_LAST"},
    "E14": {"ticker": "NAPMNMI Index", "desc": "ISM Services PMI", "field": "PX_LAST"},
    "E15": {"ticker": "INJCJC Index", "desc": "Initial Jobless Claims", "field": "PX_LAST"},
    "E16": {"ticker": "ETSLTOTL Index", "desc": "Existing Home Sales", "field": "PX_LAST"},
    "E17": {"ticker": "GDP CYOY Index", "desc": "US GDP YoY", "field": "PX_LAST"},
    "E18": {"ticker": "CPTICHNG Index", "desc": "Capacity Utilization", "field": "PX_LAST"},
    "E19": {"ticker": "PRUSTOT Index", "desc": "Productivity", "field": "PX_LAST"},
    "E20": {"ticker": "PITLCHNG Index", "desc": "Personal Income MoM", "field": "PX_LAST"},

    # === TASAS DE INTERES (I) ===
    "I1": {"ticker": "FDTR Index", "desc": "Fed Funds Target Rate", "field": "PX_LAST"},
    "I2": {"ticker": "USGG2YR Index", "desc": "US 2-Year Treasury", "field": "PX_LAST"},
    "I3": {"ticker": "USGG5YR Index", "desc": "US 5-Year Treasury", "field": "PX_LAST"},
    "I4": {"ticker": "USGG10YR Index", "desc": "US 10-Year Treasury", "field": "PX_LAST"},
    "I5": {"ticker": "USGG30YR Index", "desc": "US 30-Year Treasury", "field": "PX_LAST"},
    "I6": {"ticker": "GB3 Govt", "desc": "US 3-Month T-Bill", "field": "PX_LAST"},
    "I7": {"ticker": "USYC2Y10 Index", "desc": "US 10Y-2Y Spread", "field": "PX_LAST"},
    "I8": {"ticker": "USYC3M10 Index", "desc": "US 10Y-3M Spread", "field": "PX_LAST"},
    "I9": {"ticker": "LF98OAS Index", "desc": "US HY Corporate OAS", "field": "PX_LAST"},
    "LIBOR_3M": {"ticker": "US0003M Index", "desc": "3-Month LIBOR", "field": "PX_LAST"},
    "TIPS_5Y": {"ticker": "GTII5 Govt", "desc": "US 5Y TIPS Yield", "field": "PX_LAST"},
    "TIPS_10Y": {"ticker": "GTII10 Govt", "desc": "US 10Y TIPS Yield", "field": "PX_LAST"},
    "BREAKEVEN_5Y": {"ticker": "USGGBE05 Index", "desc": "US 5Y Breakeven", "field": "PX_LAST"},
    "BREAKEVEN_10Y": {"ticker": "USGGBE10 Index", "desc": "US 10Y Breakeven", "field": "PX_LAST"},

    # === VALORACION (P) ===
    "P1": {"ticker": "SPX Index", "desc": "S&P 500 P/E Ratio", "field": "PE_RATIO"},
    "P2": {"ticker": "SPX Index", "desc": "S&P 500 Forward P/E", "field": "BEST_PE_RATIO"},
    "P3": {"ticker": "SPX Index", "desc": "S&P 500 P/B Ratio", "field": "PX_TO_BOOK_RATIO"},
    "P4": {"ticker": "SPX Index", "desc": "S&P 500 Dividend Yield", "field": "EQY_DVD_YLD_IND"},
    "P5": {"ticker": "SPXSPCAP Index", "desc": "Shiller CAPE", "field": "PX_LAST"},
    "P6": {"ticker": "SPX Index", "desc": "S&P 500 EV/EBITDA", "field": "CURR_ENTP_VAL_TO_EBITDA"},
    "P7": {"ticker": "SPX Index", "desc": "S&P 500 P/S Ratio", "field": "PX_TO_SALES_RATIO"},
    "P8": {"ticker": "SPX Index", "desc": "S&P 500 FCF Yield", "field": "FCF_YIELD"},
    "P9": {"ticker": "XAU Curncy", "desc": "Gold Spot (replaced GLD)", "field": "PX_LAST"},
    "P9_GOLD": {"ticker": "XAU Curncy", "desc": "Gold Spot Price", "field": "PX_LAST"},
    "P10": {"ticker": "SI1 Comdty", "desc": "Silver Futures (replaced SLV)", "field": "PX_LAST"},
    "P10_SILVER": {"ticker": "SI1 Comdty", "desc": "Silver Futures", "field": "PX_LAST"},
    "P11": {"ticker": "DBC US Equity", "desc": "Commodities ETF", "field": "PX_LAST"},
    "P11_CMDTY_IDX": {"ticker": "BCOMTR Index", "desc": "Bloomberg Commodity Index", "field": "PX_LAST"},
    "P12": {"ticker": "CL1 Comdty", "desc": "WTI Crude Oil (replaced USO)", "field": "PX_LAST"},
    "P12_OIL": {"ticker": "CL1 Comdty", "desc": "WTI Crude Oil Futures", "field": "PX_LAST"},
    "P13": {"ticker": "NG1 Comdty", "desc": "Natural Gas (replaced UNG)", "field": "PX_LAST"},
    "P13_NATGAS": {"ticker": "NG1 Comdty", "desc": "Natural Gas Futures", "field": "PX_LAST"},
    "COPPER": {"ticker": "HG1 Comdty", "desc": "Copper Futures", "field": "PX_LAST"},

    # === VOLATILIDAD (V) ===
    "V1": {"ticker": "VIX Index", "desc": "CBOE VIX", "field": "PX_LAST"},
    "V2": {"ticker": "VIX9D Index", "desc": "CBOE VIX 9-Day", "field": "PX_LAST"},
    "V4": {"ticker": "VIX3M Index", "desc": "CBOE VIX 3-Month", "field": "PX_LAST"},
    "V5": {"ticker": "VIX6M Index", "desc": "CBOE VIX 6-Month", "field": "PX_LAST"},
    "V6": {"ticker": "VIX1Y Index", "desc": "CBOE VIX 1-Year", "field": "PX_LAST"},
    "V7": {"ticker": "VVIX Index", "desc": "CBOE VVIX", "field": "PX_LAST"},
    "V8": {"ticker": "SKEW Index", "desc": "CBOE Skew Index", "field": "PX_LAST"},
    "V9": {"ticker": "MOVE Index", "desc": "MOVE Index (Bond Vol)", "field": "PX_LAST"},
    "V10": {"ticker": "CVIX Index", "desc": "Currency Vol Index", "field": "PX_LAST"},
    "V11": {"ticker": "OVX Index", "desc": "Oil Volatility Index", "field": "PX_LAST"},
    "VIX_F1": {"ticker": "UX1 Index", "desc": "VIX Futures Front", "field": "PX_LAST"},
    "VIX_F2": {"ticker": "UX2 Index", "desc": "VIX Futures 2nd", "field": "PX_LAST"},
    "VIX_F3": {"ticker": "UX3 Index", "desc": "VIX Futures 3rd", "field": "PX_LAST"},
    "VIX_F4": {"ticker": "UX4 Index", "desc": "VIX Futures 4th", "field": "PX_LAST"},
    "MOVE_INDEX": {"ticker": "MOVE Index", "desc": "Bond Vol Index", "field": "PX_LAST"},
    "CVIX": {"ticker": "CVIX Index", "desc": "Currency Vol Index", "field": "PX_LAST"},

    # === SENTIMIENTO (S) ===
    "S1": {"ticker": "PCUSEQTR Index", "desc": "Total Put/Call Ratio", "field": "PX_LAST"},
    "S2": {"ticker": "AABORINS Index", "desc": "AAII Bullish %", "field": "PX_LAST"},
    "S3": {"ticker": "AABOTINS Index", "desc": "AAII Bearish %", "field": "PX_LAST"},
    "S4": {"ticker": "HYG US Equity", "desc": "High Yield Bond ETF", "field": "PX_LAST"},
    "S4_HY_INDEX": {"ticker": "LQD US Equity", "desc": "Investment Grade Bond ETF", "field": "PX_LAST"},
    "S5": {"ticker": "INVILBLL Index", "desc": "II Bullish Advisors", "field": "PX_LAST"},
    "S6": {"ticker": "INVILBRS Index", "desc": "II Bearish Advisors", "field": "PX_LAST"},
    "S7": {"ticker": "MARGDEBT Index", "desc": "NYSE Margin Debt", "field": "PX_LAST"},
    "S8": {"ticker": "NAAIM Index", "desc": "NAAIM Exposure Index", "field": "PX_LAST"},
    "S9": {"ticker": "AABONINS Index", "desc": "AAII Neutral %", "field": "PX_LAST"},
    "S10": {"ticker": "PCUSINDX Index", "desc": "Index Put/Call Ratio", "field": "PX_LAST"},
    "S11": {"ticker": "FXI US Equity", "desc": "China ETF", "field": "PX_LAST"},
    "S11_CHINA": {"ticker": "SHCOMP Index", "desc": "Shanghai Composite", "field": "PX_LAST"},
    "S12": {"ticker": "PCUSEQUI Index", "desc": "Equity Put/Call Ratio", "field": "PX_LAST"},

    # === DUMMIES (D) - Calculadas ===
    "D1": {"desc": "Above SMA 200", "calc": "SPY > SMA(200)"},
    "D2": {"desc": "Above SMA 50", "calc": "SPY > SMA(50)"},
    "D3": {"desc": "Golden Cross", "calc": "SMA(50) > SMA(200)"},
    "D4": {"desc": "VIX > 20", "calc": "VIX > 20"},
    "D5": {"desc": "VIX > 30", "calc": "VIX > 30"},
    "D6": {"desc": "Yield Curve Inverted", "calc": "10Y-2Y < 0"},
    "D7": {"desc": "Bullish Sentiment", "calc": "AAII Bull > 50%"},
    "D8": {"desc": "Bearish Sentiment", "calc": "AAII Bear > 40%"},
    "D9": {"desc": "Positive Momentum", "calc": "ROC(20) > 0"},
}


def audit_variables():
    """Audita que las variables descargadas correspondan al mapa"""
    print("-" * 80)
    print("AUDITORIA 1: Variables Bloomberg")
    print("-" * 80)

    # Cargar dataset final
    df = pd.read_csv(os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv"))

    # Verificar variables base
    base_vars = ['M' + str(i) for i in range(1, 19)] + \
                ['E' + str(i) for i in range(1, 21)] + \
                ['I' + str(i) for i in range(1, 10)] + \
                ['P' + str(i) for i in range(1, 14)] + \
                ['V' + str(i) for i in range(1, 12)] + \
                ['S' + str(i) for i in range(1, 13)] + \
                ['D' + str(i) for i in range(1, 10)]

    found = []
    missing = []
    extras = []

    for var in base_vars:
        if var in df.columns:
            found.append(var)
        else:
            # Verificar si existe con otro nombre
            alternatives = [c for c in df.columns if c.startswith(var + '_') or c == var]
            if alternatives:
                found.append(var + f" (como {alternatives[0]})")
            else:
                missing.append(var)

    print(f"\n  Variables base encontradas: {len(found)}")
    print(f"  Variables base faltantes: {len(missing)}")

    if missing:
        print("\n  [WARN] Variables faltantes:")
        for var in missing:
            if var in VARIABLE_MAP:
                print(f"    - {var}: {VARIABLE_MAP[var].get('desc', 'N/A')}")
            else:
                print(f"    - {var}")

    # Variables adicionales importantes
    additional_vars = ['SPY_CLOSE', 'SPY_VOLUME', 'VIX_F1', 'TIPS_10Y',
                       'BREAKEVEN_10Y', 'COPPER', 'MOVE_INDEX', 'CVIX']

    print("\n  Variables adicionales:")
    for var in additional_vars:
        if var in df.columns:
            pct_nan = df[var].isna().sum() / len(df) * 100
            print(f"    [OK] {var}: {pct_nan:.1f}% NaN")
        else:
            print(f"    [MISSING] {var}")

    return found, missing


def audit_data_quality():
    """Audita calidad de los datos"""
    print("\n" + "-" * 80)
    print("AUDITORIA 2: Calidad de Datos")
    print("-" * 80)

    df = pd.read_csv(os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv"))
    df['date'] = pd.to_datetime(df['date'])

    # Missing values por categoria
    categories = {
        'Market (M)': [c for c in df.columns if c.startswith('M') and c[1:2].isdigit()],
        'Economic (E)': [c for c in df.columns if c.startswith('E') and c[1:2].isdigit()],
        'Interest (I)': [c for c in df.columns if c.startswith('I') and c[1:2].isdigit()],
        'Valuation (P)': [c for c in df.columns if c.startswith('P') and c[1:2].isdigit()],
        'Volatility (V)': [c for c in df.columns if c.startswith('V') and c[1:2].isdigit()],
        'Sentiment (S)': [c for c in df.columns if c.startswith('S') and c[1:2].isdigit()],
        'Dummy (D)': [c for c in df.columns if c.startswith('D') and c[1:2].isdigit()],
        'Elder Weekly (W_)': [c for c in df.columns if c.startswith('W_')],
        'Elder Daily (D_)': [c for c in df.columns if c.startswith('D_')],
        'Triple Screen (TS_)': [c for c in df.columns if c.startswith('TS_')],
    }

    print("\n  Missing Values por Categoria:")
    issues = []
    for cat, cols in categories.items():
        if cols:
            total_nan = sum(df[c].isna().sum() for c in cols if c in df.columns)
            total_cells = len(cols) * len(df)
            pct = total_nan / total_cells * 100 if total_cells > 0 else 0
            status = "[OK]" if pct < 5 else "[WARN]" if pct < 20 else "[CRITICAL]"
            print(f"    {status} {cat}: {pct:.2f}% NaN ({len(cols)} vars)")
            if pct > 20:
                issues.append(f"{cat}: {pct:.2f}% missing")

    # Outliers
    print("\n  Analisis de Outliers (>5 std):")
    outlier_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['date_id']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            n_outliers = (z_scores > 5).sum()
            if n_outliers > len(df) * 0.01:  # Mas del 1%
                outlier_cols.append((col, n_outliers, n_outliers/len(df)*100))

    if outlier_cols:
        print("    Variables con outliers significativos:")
        for col, n, pct in sorted(outlier_cols, key=lambda x: -x[1])[:10]:
            print(f"      - {col}: {n} outliers ({pct:.2f}%)")
    else:
        print("    [OK] No se detectaron outliers significativos")

    # Fechas
    print(f"\n  Rango de fechas: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"  Total dias: {len(df):,}")

    # Anos completos
    df['year'] = df['date'].dt.year
    years = df.groupby('year').size()
    incomplete_years = years[years < 200]  # Menos de 200 dias de trading
    if len(incomplete_years) > 0:
        print(f"\n  Anos con datos incompletos:")
        for year, count in incomplete_years.items():
            print(f"    - {year}: {count} dias")

    return issues


def audit_feature_engineering():
    """Audita el feature engineering"""
    print("\n" + "-" * 80)
    print("AUDITORIA 3: Feature Engineering")
    print("-" * 80)

    df = pd.read_csv(os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv"))

    # Verificar features derivados de SPY
    spy_features = [c for c in df.columns if c.startswith('SPY_')]
    print(f"\n  Features SPY: {len(spy_features)}")

    expected_spy = ['SPY_CLOSE', 'SPY_VOLUME', 'SPY_RSI_14', 'SPY_MACD',
                    'SPY_ATR_14', 'SPY_volatility_21', 'SPY_return_1d']
    for feat in expected_spy:
        status = "[OK]" if feat in df.columns else "[MISSING]"
        print(f"    {status} {feat}")

    # Verificar features de interaccion
    interaction_features = ['vix_x_curve', 'credit_x_vix', 'momentum_high_vol',
                           'momentum_low_vol', 'rsi_above_sma200']
    print(f"\n  Features de Interaccion:")
    for feat in interaction_features:
        status = "[OK]" if feat in df.columns else "[MISSING]"
        print(f"    {status} {feat}")

    # Verificar features de regimen
    regime_features = ['regime_bull', 'regime_bear', 'regime_neutral', 'regime_score']
    print(f"\n  Features de Regimen:")
    for feat in regime_features:
        status = "[OK]" if feat in df.columns else "[MISSING]"
        if feat in df.columns:
            unique = df[feat].nunique()
            print(f"    {status} {feat}: {unique} valores unicos")
        else:
            print(f"    {status} {feat}")

    # Verificar correlaciones
    print("\n  Features Cross-Asset:")
    corr_features = [c for c in df.columns if c.startswith('corr_')]
    print(f"    Total: {len(corr_features)}")
    for feat in corr_features[:5]:
        pct_nan = df[feat].isna().sum() / len(df) * 100
        print(f"      - {feat}: {pct_nan:.1f}% NaN")

    return True


def audit_elder_indicators():
    """Audita indicadores Elder Triple Pantalla"""
    print("\n" + "-" * 80)
    print("AUDITORIA 4: Indicadores Elder Triple Pantalla")
    print("-" * 80)

    df = pd.read_csv(os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv"))

    # Pantalla 1 - Semanal
    print("\n  PANTALLA 1 (Semanal - Tendencia):")
    weekly_indicators = ['W_MACD', 'W_MACD_SIGNAL', 'W_MACD_HIST', 'W_TREND',
                         'W_IMPULSE', 'W_EMA13', 'W_EMA26', 'W_ATR']
    for ind in weekly_indicators:
        if ind in df.columns:
            pct_nan = df[ind].isna().sum() / len(df) * 100
            unique = df[ind].nunique()
            print(f"    [OK] {ind}: {pct_nan:.1f}% NaN, {unique} valores unicos")
        else:
            print(f"    [MISSING] {ind}")

    # Pantalla 2 - Diario
    print("\n  PANTALLA 2 (Diario - Osciladores):")
    daily_indicators = ['D_FORCE_INDEX_2', 'D_FORCE_INDEX_13', 'D_BULL_POWER',
                        'D_BEAR_POWER', 'D_IMPULSE', 'D_MACD', 'D_MACD_HIST']
    for ind in daily_indicators:
        if ind in df.columns:
            pct_nan = df[ind].isna().sum() / len(df) * 100
            print(f"    [OK] {ind}: {pct_nan:.1f}% NaN")
        else:
            print(f"    [MISSING] {ind}")

    # Senales Triple Pantalla
    print("\n  SENALES COMBINADAS (TS_):")
    ts_signals = ['TS_BUY_SETUP', 'TS_SELL_SETUP', 'TS_BUY_SIGNAL',
                  'TS_SELL_SIGNAL', 'TS_SIGNAL', 'TS_ALIGNMENT']
    for sig in ts_signals:
        if sig in df.columns:
            value_counts = df[sig].value_counts()
            print(f"    [OK] {sig}:")
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                print(f"         {val}: {count:,} ({pct:.1f}%)")
        else:
            print(f"    [MISSING] {sig}")

    # Verificar logica de senales
    print("\n  Verificacion de Logica:")
    if 'TS_BUY_SETUP' in df.columns and 'W_TREND' in df.columns:
        # Buy setup debe ocurrir cuando W_TREND == 1
        buy_setups = df[df['TS_BUY_SETUP'] == 1]
        if len(buy_setups) > 0:
            pct_trend_up = (buy_setups['W_TREND'] == 1).mean() * 100
            print(f"    Buy setups con tendencia alcista: {pct_trend_up:.1f}%")
            if pct_trend_up < 90:
                print(f"    [WARN] Esperado >90%, hay inconsistencia")
            else:
                print(f"    [OK] Logica correcta")

    return True


def audit_target_variable():
    """Audita la variable target"""
    print("\n" + "-" * 80)
    print("AUDITORIA 5: Variable Target")
    print("-" * 80)

    df = pd.read_csv(os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv"))

    target = 'market_forward_excess_returns'
    if target not in df.columns:
        print(f"  [ERROR] Target '{target}' no encontrado")
        return False

    t = df[target]
    print(f"\n  Target: {target}")
    print(f"    - Registros: {len(t):,}")
    print(f"    - Missing: {t.isna().sum()}")
    print(f"    - Media: {t.mean()*100:.4f}%")
    print(f"    - Std: {t.std()*100:.4f}%")
    print(f"    - Min: {t.min()*100:.4f}%")
    print(f"    - Max: {t.max()*100:.4f}%")
    print(f"    - % Positivos: {(t > 0).sum() / len(t) * 100:.1f}%")

    # Verificar que sea retorno forward (no contemporaneo)
    if 'SPY_CLOSE' in df.columns:
        # El target debe estar correlacionado con el retorno del dia siguiente
        spy_return_next = df['SPY_CLOSE'].pct_change().shift(-1)
        corr = t.corr(spy_return_next)
        print(f"\n  Verificacion anti-leakage:")
        print(f"    Correlacion con retorno t+1: {corr:.4f}")
        if corr > 0.95:
            print(f"    [OK] Target correctamente calculado como forward return")
        else:
            print(f"    [WARN] Correlacion menor a esperada, revisar calculo")

        # No debe estar correlacionado con retorno actual
        spy_return_current = df['SPY_CLOSE'].pct_change()
        corr_current = t.corr(spy_return_current)
        print(f"    Correlacion con retorno t: {corr_current:.4f}")
        if abs(corr_current) < 0.1:
            print(f"    [OK] No hay leakage con retorno actual")
        else:
            print(f"    [WARN] Posible leakage con retorno actual")

    return True


def generate_audit_report():
    """Genera reporte completo de auditoria"""
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE DE AUDITORIA")
    print("=" * 80)

    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'variables': {},
        'data_quality': {},
        'feature_engineering': {},
        'elder_indicators': {},
        'target': {},
        'issues': [],
        'recommendations': []
    }

    # Ejecutar auditorias
    found, missing = audit_variables()
    audit_results['variables']['found'] = len(found)
    audit_results['variables']['missing'] = missing

    issues = audit_data_quality()
    audit_results['data_quality']['issues'] = issues

    audit_feature_engineering()
    audit_elder_indicators()
    audit_target_variable()

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE AUDITORIA")
    print("=" * 80)

    if missing:
        print(f"\n  [WARN] Variables faltantes: {len(missing)}")
        audit_results['issues'].append(f"{len(missing)} variables base faltantes")
    else:
        print(f"\n  [OK] Todas las variables base presentes")

    if issues:
        print(f"  [WARN] Issues de calidad: {len(issues)}")
        audit_results['issues'].extend(issues)
    else:
        print(f"  [OK] Calidad de datos aceptable")

    # Recomendaciones
    if missing:
        audit_results['recommendations'].append(
            "Algunas variables base no estan presentes. Verificar que los datos "
            "fueron correctamente descargados de Bloomberg."
        )

    # Guardar reporte
    report_file = os.path.join(FINAL_DIR, "audit_report.json")
    with open(report_file, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"\n  Reporte guardado: {report_file}")

    print("\n" + "=" * 80)
    print("[OK] AUDITORIA COMPLETADA")
    print("=" * 80)

    return audit_results


if __name__ == "__main__":
    results = generate_audit_report()
