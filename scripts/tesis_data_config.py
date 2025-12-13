# -*- coding: utf-8 -*-
"""
Configuracion de datos para Tesis
Taxonomia de variables para analisis del mercado estadounidense (SPY/S&P500)

Categorias:
- M*   : Dinamica de Mercado (Market Dynamics)
- E*   : Macroeconomicos (Economic)
- I*   : Tasas de Interes (Interest Rates)
- P*   : Precio/Valoracion (Price/Valuation)
- V*   : Volatilidad (Volatility)
- S*   : Sentimiento (Sentiment)
- MOM* : Momentum
- D*   : Dummy/Binarias (Dummy/Binary)
"""

# =============================================================================
# ACTIVO PRINCIPAL
# =============================================================================
MAIN_ASSET = {
    "ticker": "SPY US Equity",
    "name": "SPDR S&P 500 ETF Trust",
    "fields": ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"],
    "description": "ETF que replica el S&P 500"
}

# Alternativas al SPY
ALTERNATIVE_ASSETS = {
    "SPX": "SPX Index",          # S&P 500 Index directo
    "ES1": "ES1 Index",          # S&P 500 E-mini Futures
    "VOO": "VOO US Equity",      # Vanguard S&P 500 ETF
    "IVV": "IVV US Equity",      # iShares Core S&P 500 ETF
}

# =============================================================================
# M* - DINAMICA DE MERCADO (Market Dynamics)
# Indicadores tecnicos del mercado
# =============================================================================
M_MARKET_DYNAMICS = {
    # Breadth Indicators (Amplitud de mercado)
    "M_ADV_DEC": {
        "ticker": "ADD Index",
        "name": "NYSE Advance-Decline Issues",
        "fields": ["PX_LAST"],
        "description": "Diferencia entre acciones que suben vs bajan"
    },
    "M_ADV_DEC_LINE": {
        "ticker": "ADLN Index",
        "name": "NYSE Advance-Decline Line",
        "fields": ["PX_LAST"],
        "description": "Linea acumulativa A/D"
    },
    "M_NEW_HIGHS": {
        "ticker": "MAHP Index",
        "name": "NYSE New 52-Week Highs",
        "fields": ["PX_LAST"],
        "description": "Acciones en maximos de 52 semanas"
    },
    "M_NEW_LOWS": {
        "ticker": "MALP Index",
        "name": "NYSE New 52-Week Lows",
        "fields": ["PX_LAST"],
        "description": "Acciones en minimos de 52 semanas"
    },
    "M_MCCLELLAN": {
        "ticker": "MCOSI Index",
        "name": "McClellan Oscillator",
        "fields": ["PX_LAST"],
        "description": "Oscilador de amplitud de mercado"
    },
    "M_MCCLELLAN_SUM": {
        "ticker": "MCSM Index",
        "name": "McClellan Summation Index",
        "fields": ["PX_LAST"],
        "description": "Indice sumatorio McClellan"
    },
    # Volume indicators
    "M_UP_VOL": {
        "ticker": "UVOL Index",
        "name": "NYSE Up Volume",
        "fields": ["PX_LAST"],
        "description": "Volumen de acciones al alza"
    },
    "M_DOWN_VOL": {
        "ticker": "DVOL Index",
        "name": "NYSE Down Volume",
        "fields": ["PX_LAST"],
        "description": "Volumen de acciones a la baja"
    },
    "M_TICK": {
        "ticker": "TICK Index",
        "name": "NYSE TICK Index",
        "fields": ["PX_LAST"],
        "description": "Diferencia upticks vs downticks"
    },
    "M_TRIN": {
        "ticker": "TRIN Index",
        "name": "Arms Index (TRIN)",
        "fields": ["PX_LAST"],
        "description": "Trading Index - ratio A/D vs volumen"
    },
    # Sector performance
    "M_XLF": {
        "ticker": "XLF US Equity",
        "name": "Financial Select Sector SPDR",
        "fields": ["PX_LAST"],
        "description": "ETF sector financiero"
    },
    "M_XLK": {
        "ticker": "XLK US Equity",
        "name": "Technology Select Sector SPDR",
        "fields": ["PX_LAST"],
        "description": "ETF sector tecnologia"
    },
    "M_XLE": {
        "ticker": "XLE US Equity",
        "name": "Energy Select Sector SPDR",
        "fields": ["PX_LAST"],
        "description": "ETF sector energia"
    },
    "M_XLV": {
        "ticker": "XLV US Equity",
        "name": "Health Care Select Sector SPDR",
        "fields": ["PX_LAST"],
        "description": "ETF sector salud"
    },
}

# =============================================================================
# E* - MACROECONOMICOS (Economic Variables)
# Variables economicas agregadas
# =============================================================================
E_MACROECONOMIC = {
    # GDP
    "E_GDP": {
        "ticker": "GDP CQOQ Index",
        "name": "US GDP QoQ",
        "fields": ["PX_LAST"],
        "description": "PIB trimestral QoQ"
    },
    "E_GDP_YOY": {
        "ticker": "GDP CYOY Index",
        "name": "US GDP YoY",
        "fields": ["PX_LAST"],
        "description": "PIB anual YoY"
    },
    # Employment
    "E_UNEMPLOYMENT": {
        "ticker": "USURTOT Index",
        "name": "US Unemployment Rate",
        "fields": ["PX_LAST"],
        "description": "Tasa de desempleo"
    },
    "E_NFP": {
        "ticker": "NFP TCH Index",
        "name": "US Nonfarm Payrolls",
        "fields": ["PX_LAST"],
        "description": "Nominas no agricolas"
    },
    "E_INITIAL_CLAIMS": {
        "ticker": "INJCJC Index",
        "name": "Initial Jobless Claims",
        "fields": ["PX_LAST"],
        "description": "Solicitudes iniciales desempleo"
    },
    # Inflation
    "E_CPI": {
        "ticker": "CPI YOY Index",
        "name": "US CPI YoY",
        "fields": ["PX_LAST"],
        "description": "Inflacion IPC anual"
    },
    "E_CPI_CORE": {
        "ticker": "CPUPXCHG Index",
        "name": "US Core CPI YoY",
        "fields": ["PX_LAST"],
        "description": "Inflacion subyacente"
    },
    "E_PCE": {
        "ticker": "PCE CYOY Index",
        "name": "US PCE YoY",
        "fields": ["PX_LAST"],
        "description": "Deflactor PCE (preferido Fed)"
    },
    # Industrial/Manufacturing
    "E_INDUSTRIAL_PROD": {
        "ticker": "IP CHNG Index",
        "name": "US Industrial Production MoM",
        "fields": ["PX_LAST"],
        "description": "Produccion industrial"
    },
    "E_PMI_MFG": {
        "ticker": "NAPMPMI Index",
        "name": "ISM Manufacturing PMI",
        "fields": ["PX_LAST"],
        "description": "PMI manufacturero ISM"
    },
    "E_PMI_SERVICES": {
        "ticker": "NAPMNMI Index",
        "name": "ISM Services PMI",
        "fields": ["PX_LAST"],
        "description": "PMI servicios ISM"
    },
    # Consumer
    "E_CONSUMER_CONF": {
        "ticker": "CONCCONF Index",
        "name": "Consumer Confidence",
        "fields": ["PX_LAST"],
        "description": "Confianza del consumidor"
    },
    "E_UMICH_SENTIMENT": {
        "ticker": "CONSSENT Index",
        "name": "U. Michigan Consumer Sentiment",
        "fields": ["PX_LAST"],
        "description": "Sentimiento consumidor Michigan"
    },
    "E_RETAIL_SALES": {
        "ticker": "RSTAMOM Index",
        "name": "US Retail Sales MoM",
        "fields": ["PX_LAST"],
        "description": "Ventas minoristas"
    },
    # Housing
    "E_HOUSING_STARTS": {
        "ticker": "NHSPSTOT Index",
        "name": "US Housing Starts",
        "fields": ["PX_LAST"],
        "description": "Inicios de construccion"
    },
    "E_EXISTING_HOME_SALES": {
        "ticker": "ETSLTOTL Index",
        "name": "Existing Home Sales",
        "fields": ["PX_LAST"],
        "description": "Venta casas existentes"
    },
    # Leading Indicators
    "E_LEI": {
        "ticker": "LEI TOTL Index",
        "name": "Leading Economic Index",
        "fields": ["PX_LAST"],
        "description": "Indice economico lider"
    },
}

# =============================================================================
# I* - TASAS DE INTERES (Interest Rates)
# Indicadores de tipos de interes
# =============================================================================
I_INTEREST_RATES = {
    # Federal Reserve
    "I_FED_FUNDS": {
        "ticker": "FDTR Index",
        "name": "Federal Funds Target Rate",
        "fields": ["PX_LAST"],
        "description": "Tasa objetivo Fed Funds"
    },
    "I_FED_FUNDS_EFF": {
        "ticker": "FEDL01 Index",
        "name": "Fed Funds Effective Rate",
        "fields": ["PX_LAST"],
        "description": "Tasa efectiva Fed Funds"
    },
    # Treasury Yields
    "I_UST_3M": {
        "ticker": "GB3 Govt",
        "name": "US 3-Month T-Bill",
        "fields": ["PX_LAST"],
        "description": "Treasury 3 meses"
    },
    "I_UST_6M": {
        "ticker": "GB6 Govt",
        "name": "US 6-Month T-Bill",
        "fields": ["PX_LAST"],
        "description": "Treasury 6 meses"
    },
    "I_UST_1Y": {
        "ticker": "GB12 Govt",
        "name": "US 1-Year T-Bill",
        "fields": ["PX_LAST"],
        "description": "Treasury 1 ano"
    },
    "I_UST_2Y": {
        "ticker": "USGG2YR Index",
        "name": "US 2-Year Treasury Yield",
        "fields": ["PX_LAST"],
        "description": "Treasury 2 anos"
    },
    "I_UST_5Y": {
        "ticker": "USGG5YR Index",
        "name": "US 5-Year Treasury Yield",
        "fields": ["PX_LAST"],
        "description": "Treasury 5 anos"
    },
    "I_UST_10Y": {
        "ticker": "USGG10YR Index",
        "name": "US 10-Year Treasury Yield",
        "fields": ["PX_LAST"],
        "description": "Treasury 10 anos"
    },
    "I_UST_30Y": {
        "ticker": "USGG30YR Index",
        "name": "US 30-Year Treasury Yield",
        "fields": ["PX_LAST"],
        "description": "Treasury 30 anos"
    },
    # Yield Curve Spreads
    "I_SPREAD_10Y2Y": {
        "ticker": "USYC2Y10 Index",
        "name": "US 10Y-2Y Yield Spread",
        "fields": ["PX_LAST"],
        "description": "Spread curva 10Y-2Y (inversion)"
    },
    "I_SPREAD_10Y3M": {
        "ticker": "USYC3M10 Index",
        "name": "US 10Y-3M Yield Spread",
        "fields": ["PX_LAST"],
        "description": "Spread curva 10Y-3M"
    },
    # SOFR (reemplazo LIBOR)
    "I_SOFR": {
        "ticker": "SOFRRATE Index",
        "name": "SOFR Rate",
        "fields": ["PX_LAST"],
        "description": "Secured Overnight Financing Rate"
    },
    # Credit Spreads
    "I_IG_SPREAD": {
        "ticker": "LUACOAS Index",
        "name": "US IG Corporate OAS",
        "fields": ["PX_LAST"],
        "description": "Spread bonos investment grade"
    },
    "I_HY_SPREAD": {
        "ticker": "LF98OAS Index",
        "name": "US HY Corporate OAS",
        "fields": ["PX_LAST"],
        "description": "Spread bonos high yield"
    },
    # TED Spread
    "I_TED_SPREAD": {
        "ticker": "TEDSPRD Index",
        "name": "TED Spread",
        "fields": ["PX_LAST"],
        "description": "Spread LIBOR vs T-Bill (riesgo bancario)"
    },
    # Real Rates
    "I_TIPS_10Y": {
        "ticker": "GTII10 Govt",
        "name": "US 10Y TIPS Yield",
        "fields": ["PX_LAST"],
        "description": "Treasury real 10 anos"
    },
    "I_BREAKEVEN_10Y": {
        "ticker": "USGGBE10 Index",
        "name": "US 10Y Breakeven Inflation",
        "fields": ["PX_LAST"],
        "description": "Expectativa inflacion 10 anos"
    },
}

# =============================================================================
# P* - PRECIO/VALORACION (Price/Valuation)
# Metricas de valoracion de activos
# =============================================================================
P_VALUATION = {
    # P/E Ratios
    "P_PE_SPX": {
        "ticker": "SPX Index",
        "name": "S&P 500 P/E Ratio",
        "fields": ["PE_RATIO"],
        "description": "P/E ratio S&P 500"
    },
    "P_PE_FWD": {
        "ticker": "SPX Index",
        "name": "S&P 500 Forward P/E",
        "fields": ["BEST_PE_RATIO"],
        "description": "P/E forward 12 meses"
    },
    # Price/Book
    "P_PB_SPX": {
        "ticker": "SPX Index",
        "name": "S&P 500 P/B Ratio",
        "fields": ["PX_TO_BOOK_RATIO"],
        "description": "Price to Book S&P 500"
    },
    # Dividend Yield
    "P_DIV_YIELD": {
        "ticker": "SPX Index",
        "name": "S&P 500 Dividend Yield",
        "fields": ["EQY_DVD_YLD_IND"],
        "description": "Rendimiento por dividendos"
    },
    # Earnings Yield
    "P_EARNINGS_YIELD": {
        "ticker": "SPX Index",
        "name": "S&P 500 Earnings Yield",
        "fields": ["EARN_YLD"],
        "description": "Rendimiento por ganancias (1/PE)"
    },
    # CAPE (Shiller P/E) - puede requerir calculo manual
    "P_CAPE": {
        "ticker": "SPXSPCAP Index",
        "name": "S&P 500 Shiller CAPE",
        "fields": ["PX_LAST"],
        "description": "Cyclically Adjusted P/E (Shiller)"
    },
    # EV/EBITDA
    "P_EV_EBITDA": {
        "ticker": "SPX Index",
        "name": "S&P 500 EV/EBITDA",
        "fields": ["CURR_ENTP_VAL_TO_EBITDA"],
        "description": "Enterprise Value / EBITDA"
    },
    # Price/Sales
    "P_PS_SPX": {
        "ticker": "SPX Index",
        "name": "S&P 500 P/S Ratio",
        "fields": ["PX_TO_SALES_RATIO"],
        "description": "Price to Sales"
    },
    # Free Cash Flow Yield
    "P_FCF_YIELD": {
        "ticker": "SPX Index",
        "name": "S&P 500 FCF Yield",
        "fields": ["FCF_YIELD"],
        "description": "Free Cash Flow Yield"
    },
}

# =============================================================================
# V* - VOLATILIDAD (Volatility)
# Medidas de volatilidad del mercado
# =============================================================================
V_VOLATILITY = {
    # VIX Index
    "V_VIX": {
        "ticker": "VIX Index",
        "name": "CBOE Volatility Index",
        "fields": ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW"],
        "description": "VIX - Indice de miedo"
    },
    # VIX Term Structure
    "V_VIX9D": {
        "ticker": "VIX9D Index",
        "name": "CBOE VIX 9-Day",
        "fields": ["PX_LAST"],
        "description": "VIX 9 dias"
    },
    "V_VIX3M": {
        "ticker": "VIX3M Index",
        "name": "CBOE VIX 3-Month",
        "fields": ["PX_LAST"],
        "description": "VIX 3 meses"
    },
    "V_VIX6M": {
        "ticker": "VIX6M Index",
        "name": "CBOE VIX 6-Month",
        "fields": ["PX_LAST"],
        "description": "VIX 6 meses"
    },
    "V_VIX1Y": {
        "ticker": "VIX1Y Index",
        "name": "CBOE VIX 1-Year",
        "fields": ["PX_LAST"],
        "description": "VIX 1 ano"
    },
    # VVIX - Volatility of Volatility
    "V_VVIX": {
        "ticker": "VVIX Index",
        "name": "CBOE VVIX Index",
        "fields": ["PX_LAST"],
        "description": "Volatilidad del VIX"
    },
    # Skew
    "V_SKEW": {
        "ticker": "SKEW Index",
        "name": "CBOE Skew Index",
        "fields": ["PX_LAST"],
        "description": "Asimetria opciones (tail risk)"
    },
    # VIX Futures Term Structure
    "V_VX1": {
        "ticker": "UX1 Index",
        "name": "VIX Future Front Month",
        "fields": ["PX_LAST"],
        "description": "Futuro VIX mes 1"
    },
    "V_VX2": {
        "ticker": "UX2 Index",
        "name": "VIX Future 2nd Month",
        "fields": ["PX_LAST"],
        "description": "Futuro VIX mes 2"
    },
    # Realized Volatility (se calculara)
    "V_REALIZED_20D": {
        "ticker": "SPY US Equity",
        "name": "SPY 20-Day Realized Vol",
        "fields": ["VOLATILITY_20D"],
        "description": "Volatilidad realizada 20 dias"
    },
    "V_REALIZED_60D": {
        "ticker": "SPY US Equity",
        "name": "SPY 60-Day Realized Vol",
        "fields": ["VOLATILITY_60D"],
        "description": "Volatilidad realizada 60 dias"
    },
    # MOVE Index (bond volatility)
    "V_MOVE": {
        "ticker": "MOVE Index",
        "name": "ICE BofA MOVE Index",
        "fields": ["PX_LAST"],
        "description": "Volatilidad bonos (VIX de renta fija)"
    },
    # Currency Volatility
    "V_CVIX": {
        "ticker": "CVIX Index",
        "name": "Currency Volatility Index",
        "fields": ["PX_LAST"],
        "description": "Volatilidad divisas"
    },
}

# =============================================================================
# S* - SENTIMIENTO (Sentiment)
# Indicadores de sentimiento de mercado
# =============================================================================
S_SENTIMENT = {
    # Put/Call Ratios
    "S_PC_TOTAL": {
        "ticker": "PCUSEQTR Index",
        "name": "CBOE Total Put/Call Ratio",
        "fields": ["PX_LAST"],
        "description": "Ratio Put/Call total"
    },
    "S_PC_EQUITY": {
        "ticker": "PCUSEQUI Index",
        "name": "CBOE Equity Put/Call Ratio",
        "fields": ["PX_LAST"],
        "description": "Ratio Put/Call acciones"
    },
    "S_PC_INDEX": {
        "ticker": "PCUSINDX Index",
        "name": "CBOE Index Put/Call Ratio",
        "fields": ["PX_LAST"],
        "description": "Ratio Put/Call indices"
    },
    # AAII Sentiment Survey
    "S_AAII_BULL": {
        "ticker": "AABORINS Index",
        "name": "AAII Bullish Sentiment",
        "fields": ["PX_LAST"],
        "description": "Sentimiento alcista AAII"
    },
    "S_AAII_BEAR": {
        "ticker": "AABOTINS Index",
        "name": "AAII Bearish Sentiment",
        "fields": ["PX_LAST"],
        "description": "Sentimiento bajista AAII"
    },
    "S_AAII_NEUTRAL": {
        "ticker": "AABONINS Index",
        "name": "AAII Neutral Sentiment",
        "fields": ["PX_LAST"],
        "description": "Sentimiento neutral AAII"
    },
    # Investor Intelligence
    "S_II_BULL": {
        "ticker": "INVILBLL Index",
        "name": "II Bullish Advisors",
        "fields": ["PX_LAST"],
        "description": "Asesores alcistas II"
    },
    "S_II_BEAR": {
        "ticker": "INVILBRS Index",
        "name": "II Bearish Advisors",
        "fields": ["PX_LAST"],
        "description": "Asesores bajistas II"
    },
    # Short Interest
    "S_SHORT_SPY": {
        "ticker": "SPY US Equity",
        "name": "SPY Short Interest Ratio",
        "fields": ["SHORT_INT_RATIO"],
        "description": "Ratio interes corto SPY"
    },
    # Fund Flows (ETF flows)
    "S_FLOW_SPY": {
        "ticker": "SPY US Equity",
        "name": "SPY Fund Flow",
        "fields": ["FUND_FLOW"],
        "description": "Flujo de fondos SPY"
    },
    # Margin Debt
    "S_MARGIN_DEBT": {
        "ticker": "MARGDEBT Index",
        "name": "NYSE Margin Debt",
        "fields": ["PX_LAST"],
        "description": "Deuda margen NYSE"
    },
    # CNN Fear & Greed components (proxy via indices)
    "S_SAFE_HAVEN": {
        "ticker": "TLT US Equity",
        "name": "iShares 20+ Year Treasury",
        "fields": ["PX_LAST"],
        "description": "Proxy demanda activos seguros"
    },
    "S_JUNK_DEMAND": {
        "ticker": "HYG US Equity",
        "name": "iShares iBoxx High Yield",
        "fields": ["PX_LAST"],
        "description": "Proxy demanda bonos basura"
    },
}

# =============================================================================
# MOM* - MOMENTUM
# Indicadores de momentum (muchos se calculan de SPY)
# =============================================================================
MOM_MOMENTUM = {
    # Estos se calcularan a partir del precio de SPY
    # Aqui definimos indices de momentum pre-calculados

    # Relative Strength vs Other Markets
    "MOM_SPY_VS_EFA": {
        "ticker": "EFA US Equity",
        "name": "iShares MSCI EAFE",
        "fields": ["PX_LAST"],
        "description": "Para calcular RS vs mercados desarrollados"
    },
    "MOM_SPY_VS_EEM": {
        "ticker": "EEM US Equity",
        "name": "iShares MSCI Emerging Markets",
        "fields": ["PX_LAST"],
        "description": "Para calcular RS vs emergentes"
    },
    "MOM_SPY_VS_AGG": {
        "ticker": "AGG US Equity",
        "name": "iShares Core US Aggregate Bond",
        "fields": ["PX_LAST"],
        "description": "Para calcular RS vs bonos"
    },
    # Size/Style Factors
    "MOM_QQQ": {
        "ticker": "QQQ US Equity",
        "name": "Invesco QQQ (Nasdaq 100)",
        "fields": ["PX_LAST"],
        "description": "Growth/Tech momentum"
    },
    "MOM_IWM": {
        "ticker": "IWM US Equity",
        "name": "iShares Russell 2000",
        "fields": ["PX_LAST"],
        "description": "Small cap momentum"
    },
    "MOM_IWD": {
        "ticker": "IWD US Equity",
        "name": "iShares Russell 1000 Value",
        "fields": ["PX_LAST"],
        "description": "Value momentum"
    },
    "MOM_IWF": {
        "ticker": "IWF US Equity",
        "name": "iShares Russell 1000 Growth",
        "fields": ["PX_LAST"],
        "description": "Growth momentum"
    },
    # Commodities for cross-asset momentum
    "MOM_GLD": {
        "ticker": "GLD US Equity",
        "name": "SPDR Gold Shares",
        "fields": ["PX_LAST"],
        "description": "Oro momentum"
    },
    "MOM_USO": {
        "ticker": "USO US Equity",
        "name": "United States Oil Fund",
        "fields": ["PX_LAST"],
        "description": "Petroleo momentum"
    },
    "MOM_DBC": {
        "ticker": "DBC US Equity",
        "name": "Invesco DB Commodity Index",
        "fields": ["PX_LAST"],
        "description": "Commodities momentum"
    },
    # Dollar
    "MOM_DXY": {
        "ticker": "DXY Curncy",
        "name": "US Dollar Index",
        "fields": ["PX_LAST"],
        "description": "Dolar momentum"
    },
    "MOM_UUP": {
        "ticker": "UUP US Equity",
        "name": "Invesco DB US Dollar Index",
        "fields": ["PX_LAST"],
        "description": "ETF dolar momentum"
    },
}

# =============================================================================
# D* - DUMMY/BINARIAS
# Variables categoricas binarias (se calcularan)
# =============================================================================
D_DUMMY = {
    # Estas variables se calcularan basandose en los datos descargados
    # Definicion de las reglas:

    "D_RECESSION": {
        "description": "1 si economia en recesion (NBER), 0 si no",
        "calculation": "Basado en indicadores macro"
    },
    "D_ABOVE_SMA200": {
        "description": "1 si SPY > SMA(200), 0 si no",
        "calculation": "SPY.close > SPY.close.rolling(200).mean()"
    },
    "D_ABOVE_SMA50": {
        "description": "1 si SPY > SMA(50), 0 si no",
        "calculation": "SPY.close > SPY.close.rolling(50).mean()"
    },
    "D_GOLDEN_CROSS": {
        "description": "1 si SMA(50) > SMA(200), 0 si no",
        "calculation": "SMA50 > SMA200"
    },
    "D_VIX_HIGH": {
        "description": "1 si VIX > 20, 0 si no",
        "calculation": "VIX > 20"
    },
    "D_VIX_EXTREME": {
        "description": "1 si VIX > 30, 0 si no",
        "calculation": "VIX > 30"
    },
    "D_YIELD_INVERTED": {
        "description": "1 si curva invertida (10Y-2Y < 0), 0 si no",
        "calculation": "I_SPREAD_10Y2Y < 0"
    },
    "D_BULL_SENTIMENT": {
        "description": "1 si AAII Bull > 50%, 0 si no",
        "calculation": "S_AAII_BULL > 50"
    },
    "D_BEAR_SENTIMENT": {
        "description": "1 si AAII Bear > 40%, 0 si no",
        "calculation": "S_AAII_BEAR > 40"
    },
    "D_HIGH_SPREAD": {
        "description": "1 si HY spread > percentil 75, 0 si no",
        "calculation": "I_HY_SPREAD > percentile(75)"
    },
    "D_MOMENTUM_POS": {
        "description": "1 si ROC(20) > 0, 0 si no",
        "calculation": "(SPY.close / SPY.close.shift(20) - 1) > 0"
    },
    "D_TREND_UP": {
        "description": "1 si higher highs y higher lows, 0 si no",
        "calculation": "Analisis de estructura de precio"
    },
}

# =============================================================================
# AGREGACION DE TODAS LAS CATEGORIAS
# =============================================================================
ALL_CATEGORIES = {
    "M": M_MARKET_DYNAMICS,
    "E": E_MACROECONOMIC,
    "I": I_INTEREST_RATES,
    "P": P_VALUATION,
    "V": V_VOLATILITY,
    "S": S_SENTIMENT,
    "MOM": MOM_MOMENTUM,
    "D": D_DUMMY,
}

def get_all_tickers():
    """Retorna lista de todos los tickers unicos a descargar"""
    tickers = set()
    tickers.add(MAIN_ASSET["ticker"])

    for category_name, category_data in ALL_CATEGORIES.items():
        if category_name == "D":  # Dummies se calculan, no se descargan
            continue
        for var_name, var_data in category_data.items():
            if "ticker" in var_data:
                tickers.add(var_data["ticker"])

    return sorted(list(tickers))

def get_tickers_by_category(category: str):
    """Retorna tickers de una categoria especifica"""
    if category not in ALL_CATEGORIES:
        raise ValueError(f"Categoria invalida: {category}. Usar: {list(ALL_CATEGORIES.keys())}")

    if category == "D":
        return []  # Dummies se calculan

    tickers = []
    for var_name, var_data in ALL_CATEGORIES[category].items():
        if "ticker" in var_data:
            tickers.append({
                "variable": var_name,
                "ticker": var_data["ticker"],
                "name": var_data["name"],
                "fields": var_data["fields"]
            })
    return tickers

def print_summary():
    """Imprime resumen de todas las variables"""
    print("=" * 70)
    print("RESUMEN DE VARIABLES PARA TESIS")
    print("=" * 70)

    total = 0
    for cat_name, cat_data in ALL_CATEGORIES.items():
        count = len(cat_data)
        total += count
        print(f"\n{cat_name}* - {count} variables")
        for var_name, var_data in cat_data.items():
            if "ticker" in var_data:
                print(f"  {var_name}: {var_data['ticker']} - {var_data['description']}")
            else:
                print(f"  {var_name}: [CALCULADA] - {var_data['description']}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total} variables")
    print(f"Tickers unicos a descargar: {len(get_all_tickers())}")
    print("=" * 70)


if __name__ == "__main__":
    print_summary()
