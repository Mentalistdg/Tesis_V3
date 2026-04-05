# Diccionario de Variables — Dataset Final

**Archivo:** `bloomberg_triple_screen_core.csv`
**Total:** 546 columnas = 536 features + 10 columnas auxiliares
**Observaciones:** 6,507 (Feb 2000 — Dic 2025)
**Generado por:** `scripts/build_dataset.py`

> **Nota:** `build_dataset.py` genera algunas features adicionales que se eliminan automáticamente por tener >50% valores faltantes (e.g., CARR_range_up, CARR_range_down, CARR_range_asymmetry, M12_REIT). Los conteos en este diccionario reflejan las columnas que sobreviven en el CSV final.

---

## Columnas Auxiliares (10)

| Columna | Descripción |
|---------|-------------|
| `date` | Fecha del día de trading |
| `date_id` | Índice numérico secuencial (0, 1, 2, ...) |
| `SPY_CLOSE` | Precio de cierre SPY |
| `SPY_OPEN` | Precio de apertura SPY |
| `SPY_HIGH` | Precio máximo del día SPY |
| `SPY_LOW` | Precio mínimo del día SPY |
| `SPY_VOLUME` | Volumen de SPY |
| `forward_returns` | Retorno del día t al t+1 (TARGET bruto) |
| `market_forward_excess_returns` | Retorno forward menos tasa libre de riesgo (TARGET del modelo) |
| `risk_free_rate` | Tasa libre de riesgo diaria (BIL, alineada con forward) |

> **Nota:** `risk_free_rate_raw` (tasa libre de riesgo sin alinear) se cuenta como feature, no como auxiliar. `train_models.py` la incluye en el feature matrix.

---

## Parte 1: Variables Raw de Bloomberg (92)

### Mercados y ETFs (M1-M18)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| M1 | SPY US Equity | S&P 500 ETF |
| M2 | QQQ US Equity | Nasdaq 100 ETF |
| M3 | IWM US Equity | Russell 2000 (small caps) |
| M4 | EFA US Equity | MSCI EAFE (desarrollados ex-US) |
| M5 | EEM US Equity | Mercados emergentes ETF |
| M6 | VGK US Equity | FTSE Europe ETF |
| M7 | EWJ US Equity | MSCI Japan ETF |
| M8 | FXI US Equity | FTSE China 50 ETF |
| M9 | DXY Curncy | US Dollar Index |
| M10 | EURUSD Curncy | Euro / Dólar |
| M11 | USDJPY Curncy | Dólar / Yen |
| M12 | FNERTR Index | FTSE NAREIT (REITs) — **eliminada por >50% NaN** |
| M13 | XLF US Equity | Sector Financiero ETF |
| M14 | XLK US Equity | Sector Tecnología ETF |
| M15 | XLE US Equity | Sector Energía ETF |
| M16 | XLV US Equity | Sector Salud ETF |
| M17 | XLI US Equity | Sector Industrial ETF |
| M18 | XLU US Equity | Sector Utilities ETF |

### Indicadores Económicos (E1-E19)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| E1 | GDP CQOQ Index | GDP trimestral (cambio QoQ) |
| E2 | NAPMPMI Index | ISM Manufacturing PMI |
| E3 | CONSSENT Index | U. Michigan Consumer Sentiment |
| E4 | NFP TCH Index | Nonfarm Payrolls (empleo) |
| E5 | USURTOT Index | Tasa de desempleo total US |
| E6 | CPI YOY Index | Inflación CPI interanual |
| E7 | PCE CRCH Index | Core PCE (inflación preferida por la Fed) |
| E8 | RSTAMOM Index | Retail Sales (ventas minoristas) |
| E9 | LEI TOTL Index | Leading Economic Indicators |
| E10 | INJCJC Index | Initial Jobless Claims (solicitudes desempleo semanales) |
| E11 | NHSPSTOT Index | New Home Sales |
| E12 | IP CHNG Index | Industrial Production (cambio mensual) |
| E13 | CONCCONF Index | Conference Board Consumer Confidence |
| E14 | NAPMNMI Index | ISM Non-Manufacturing (servicios) |
| E15 | NHSPATOT Index | Housing Starts (inicios de construcción, anualizado) |
| E16 | ETSLTOTL Index | Existing Home Sales |
| E17 | GDP CYOY Index | GDP año contra año |
| E18 | CPTICHNG Index | Capacity Utilization (cambio) |
| E19 | PITLCHNG Index | Personal Income (cambio) |

### Tasas de Interés (I1-I20)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| I1 | FDTR Index | Fed Funds Target Rate |
| I2 | GB3 Govt | Treasury 3 meses |
| I3 | GB6 Govt | Treasury 6 meses |
| I4 | GB12 Govt | Treasury 1 año |
| I5 | GT2 Govt | Treasury 2 años |
| I6 | GT5 Govt | Treasury 5 años |
| I7 | GT10 Govt | Treasury 10 años |
| I8 | GT30 Govt | Treasury 30 años |
| I9 | USGG10YR Index | Rendimiento Treasury 10Y (genérico) |
| I10 | USGG2YR Index | Rendimiento Treasury 2Y (genérico) |
| I11 | USYC2Y10 Index | Spread curva 2Y-10Y |
| I12 | USYC3M10 Index | Spread curva 3M-10Y |
| I13 | US0003M Index | LIBOR 3 meses |
| I14 | USSWAP10 Index | Swap rate 10 años |
| I15 | CDX IG CDSI GEN 5Y | CDS Investment Grade 5Y |
| I16 | CDX HY CDSI GEN 5Y | CDS High Yield 5Y |
| I17 | LF98TRUU Index | US Aggregate Bond Total Return |
| I18 | LF98OAS Index | US Aggregate Bond OAS (spread) |
| I19 | LUACTRUU Index | US Corporate Bond Total Return |
| I20 | LQD US Equity | ETF bonos corporativos IG |

### Commodities (P1-P11)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| P1 | CL1 Comdty | Petróleo WTI crudo (futures) |
| P2 | HO1 Comdty | Heating Oil (petróleo calefacción) |
| P3 | GC1 Comdty | Oro (Gold futures) |
| P4 | SI1 Comdty | Plata (Silver futures) |
| P5 | GLD US Equity | SPDR Gold Shares ETF |
| P6 | SLV US Equity | iShares Silver Trust ETF |
| P7 | USO US Equity | United States Oil Fund ETF |
| P8 | DBC US Equity | PowerShares DB Commodity Index |
| P9 | DBA US Equity | PowerShares DB Agriculture |
| P10 | BCOMTR Index | Bloomberg Commodity Total Return |
| P11 | GSCITR Index | S&P GSCI Total Return |

### Volatilidad (V1-V13)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| V1 | VIX Index | CBOE Volatility Index (S&P 500) |
| V2 | VIX3M Index | VIX 3 meses |
| V3 | VIX1M Index | VIX 1 mes |
| V4 | VXV Index | VIX 3M (variante) |
| V5 | VXEEM Index | VIX mercados emergentes |
| V6 | VXEFA Index | VIX mercados desarrollados (ex-US) |
| V7 | GVZ Index | Gold Volatility Index |
| V8 | OVX Index | Oil Volatility Index |
| V9 | TYVIX Index | Treasury Volatility Index |
| V10 | MOVE Index | Merrill Lynch Option Volatility (bonos) |
| V11 | CVIX Index | Currency Volatility Index (divisas) |
| V12 | VIY1 Index | VIX Futures front-month |
| V13 | V2X Index | Euro Stoxx 50 Volatility |

### Sentimiento (S1-S10)

| Código | Bloomberg Ticker | Descripción |
|--------|-----------------|-------------|
| S1 | AAII BULLISH Index | % inversores alcistas (AAII survey) |
| S2 | PUT Index | Put/Call ratio |
| S3 | NYHL Index | NYSE New Highs - New Lows |
| S4 | TICK Index | NYSE TICK (transacciones up - down) |
| S5 | TRIN Index | Arms Index (TRIN) |
| S6 | ADD Index | NYSE Advance/Decline |
| S7 | MCCL Index | McClellan Oscillator |
| S8 | MCSU Index | McClellan Summation Index |
| S9 | SRVOL Index | Short-Range Volatility |
| S10 | PCUSEQUI Index | Equity Put/Call Ratio |

---

## Parte 2: Features Construidos (536)

Todas las features usan exclusivamente datos pasados y del día actual. No hay data leakage.
- `shift(n)` con n > 0: trae valores del pasado
- `rolling(window)`: ventana backward-looking (sin `center=True`)
- `pct_change(n)`: retorno entre t-n y t
- `diff(n)`: diferencia entre t y t-n
- TA-Lib: todos los indicadores son backward-looking por diseño

### 1. Indicadores Técnicos SPY — TA-Lib (19)

| Variable | Descripción |
|----------|-------------|
| `SPY_RSI_7` | Relative Strength Index 7 días |
| `SPY_RSI_14` | Relative Strength Index 14 días |
| `SPY_RSI_21` | RSI 21 días |
| `SPY_MACD` | MACD line (EMA12 - EMA26) |
| `SPY_MACD_signal` | MACD signal line (EMA9 del MACD) |
| `SPY_MACD_hist` | MACD histograma (MACD - Signal) |
| `SPY_ADX_14` | Average Directional Index 14d (fuerza de tendencia) |
| `SPY_PLUS_DI` | DI+ (presión compradora direccional) |
| `SPY_MINUS_DI` | DI- (presión vendedora direccional) |
| `SPY_DI_DIFF` | DI+ menos DI- |
| `SPY_STOCH_K` | Stochastic %K |
| `SPY_STOCH_D` | Stochastic %D |
| `SPY_WILLIAMS_R` | Williams %R (oscilador de momentum) |
| `SPY_CCI_20` | Commodity Channel Index 20 días |
| `SPY_ATR_14` | Average True Range 14 días |
| `SPY_ATR_pct` | ATR como % del precio |
| `SPY_OBV` | On-Balance Volume |
| `SPY_BB_position` | Posición del precio dentro de Bollinger Bands (0-1) |
| `SPY_BB_width` | Ancho de Bollinger Bands / middle × 100 |

### 2. Momentum y Retornos SPY (32)

| Variable | Descripción |
|----------|-------------|
| `SPY_return_1d` | Retorno 1 día |
| `SPY_return_5d` | Retorno 5 días (~1 semana) |
| `SPY_return_21d` | Retorno 21 días (~1 mes) |
| `SPY_return_63d` | Retorno 63 días (~1 trimestre) |
| `SPY_return_126d` | Retorno 126 días (~6 meses) |
| `SPY_return_252d` | Retorno 252 días (~1 año) |
| `SPY_momentum_5` | Retorno acumulado 5 días |
| `SPY_momentum_10` | Retorno acumulado 10 días |
| `SPY_momentum_21` | Retorno acumulado 21 días |
| `SPY_momentum_63` | Retorno acumulado 63 días |
| `SPY_ROC_5` | Rate of Change 5 días (%) |
| `SPY_ROC_10` | Rate of Change 10 días |
| `SPY_ROC_21` | Rate of Change 21 días |
| `SPY_ROC_63` | Rate of Change 63 días |
| `SPY_dist_MA_10` | Distancia % del precio a MA 10 días |
| `SPY_dist_MA_21` | Distancia % a MA 21 días |
| `SPY_dist_MA_50` | Distancia % a MA 50 días |
| `SPY_dist_MA_100` | Distancia % a MA 100 días |
| `SPY_dist_MA_200` | Distancia % a MA 200 días |
| `SPY_MA_cross_5_10` | Cruce MA5/MA10: +1 alcista, -1 bajista |
| `SPY_MA_cross_10_21` | Cruce MA10/MA21 |
| `SPY_MA_cross_20_50` | Cruce MA20/MA50 |
| `SPY_MA_cross_50_200` | Cruce MA50/MA200 (Golden/Death Cross) |
| `SPY_volatility_5` | Volatilidad realizada 5 días (anualizada) |
| `SPY_volatility_10` | Volatilidad realizada 10 días |
| `SPY_volatility_21` | Volatilidad realizada 21 días |
| `SPY_volatility_63` | Volatilidad realizada 63 días |
| `SPY_volatility_126` | Volatilidad realizada 126 días |
| `SPY_vol_ratio_5_21` | Ratio volatilidad corta/media (5d/21d) |
| `SPY_vol_ratio_21_63` | Ratio volatilidad media/larga (21d/63d) |
| `SPY_volume_ratio_20` | Volumen / media 20d |
| `SPY_volume_ratio_63` | Volumen / media 63d |

### 3. Ichimoku Cloud (13)

| Variable | Descripción |
|----------|-------------|
| `ICHI_tenkan` | Línea Tenkan-sen (conversión, media H-L 9 períodos) |
| `ICHI_kijun` | Línea Kijun-sen (base, media H-L 26 períodos) |
| `ICHI_senkou_a` | Senkou Span A (media de Tenkan y Kijun) |
| `ICHI_senkou_b` | Senkou Span B (media H-L 52 períodos) |
| `ICHI_cloud_thickness` | Grosor de la nube normalizado por precio |
| `ICHI_above_cloud` | 1 si precio encima de la nube (alcista) |
| `ICHI_below_cloud` | 1 si precio debajo de la nube (bajista) |
| `ICHI_in_cloud` | 1 si precio dentro de la nube (indecisión) |
| `ICHI_price_vs_tenkan` | Distancia % del precio a Tenkan-sen |
| `ICHI_price_vs_kijun` | Distancia % del precio a Kijun-sen |
| `ICHI_tk_cross_bull` | 1 si Tenkan cruza por encima de Kijun |
| `ICHI_tk_cross_bear` | 1 si Tenkan cruza por debajo de Kijun |
| `ICHI_bullish_setup` | 1 si above cloud + Tenkan > Kijun |

### 4. Fibonacci Retracement (18)

Para ventanas de 63 y 126 días:

| Variable | Descripción |
|----------|-------------|
| `FIB_position_{63,126}` | Posición del precio en rango (0=mín, 1=máx) |
| `FIB_236_{63,126}_dist` | Distancia % al nivel 23.6% de retroceso |
| `FIB_382_{63,126}_dist` | Distancia al nivel 38.2% |
| `FIB_382_{63,126}_zone` | 1 si precio cerca del nivel 38.2% |
| `FIB_500_{63,126}_dist` | Distancia al nivel 50% |
| `FIB_500_{63,126}_zone` | 1 si precio cerca del 50% |
| `FIB_618_{63,126}_dist` | Distancia al nivel 61.8% (golden ratio) |
| `FIB_618_{63,126}_zone` | 1 si precio cerca del 61.8% |
| `FIB_786_{63,126}_dist` | Distancia al nivel 78.6% |

### 5. Donchian Channels (8)

Para ventanas de 20 y 55 días:

| Variable | Descripción |
|----------|-------------|
| `DON_position_{20,55}` | Posición del precio en canal Donchian (0-1) |
| `DON_width_pct_{20,55}` | Ancho del canal como % del precio |
| `DON_breakout_high_{20,55}` | 1 si precio rompe máximo del período |
| `DON_breakout_low_{20,55}` | 1 si precio rompe mínimo del período |

### 6. Keltner Channels (3)

| Variable | Descripción |
|----------|-------------|
| `KELT_position_15` | Posición del precio en canal Keltner 15d (EMA ± 1.5×ATR) |
| `KELT_position_20` | Posición en canal Keltner 20d |
| `KELT_BB_squeeze` | 1 si BB dentro de Keltner (compresión de volatilidad) |

### 7. Money Flow (8)

| Variable | Descripción |
|----------|-------------|
| `MFI_14` | Money Flow Index 14 días — TA-Lib (RSI ponderado por volumen) |
| `MFI_21` | Money Flow Index 21 días — TA-Lib |
| `MFI_overbought` | 1 si MFI > 80 |
| `MFI_oversold` | 1 si MFI < 20 |
| `CMF_20` | Chaikin Money Flow 20 días |
| `CMF_50` | Chaikin Money Flow 50 días |
| `CMF_positive` | 1 si CMF_20 > 0 (acumulación) |
| `CMF_negative` | 1 si CMF_20 < 0 (distribución) |

### 8. Señales de Estrategia (10)

| Variable | Descripción |
|----------|-------------|
| `STRAT_RSI14_overbought` | 1 si RSI 14 > 70 |
| `STRAT_RSI14_oversold` | 1 si RSI 14 < 30 |
| `STRAT_RSI14_neutral_bull` | 1 si RSI entre 40-60 |
| `STRAT_above_200MA` | 1 si precio > MA 200 |
| `STRAT_golden_cross` | 1 si MA 50 cruzó por encima de MA 200 |
| `STRAT_death_cross` | 1 si MA 50 cruzó por debajo de MA 200 |
| `STRAT_bullish_trend` | 1 si precio > MA50 y MA50 > MA200 |
| `STRAT_strong_trend` | 1 si ADX > 25 |
| `STRAT_bullish_confluence` | 1 si >= 3 señales alcistas activas |
| `STRAT_bearish_confluence` | 1 si >= 3 señales bajistas activas |

### 9. Triple Screen Elder — Pantalla Semanal (11)

| Variable | Descripción |
|----------|-------------|
| `W_EMA_13` | EMA 65d (simula EMA13 semanal) |
| `W_MACD` | MACD semanal (EMA 60d - EMA 130d) |
| `W_MACD_SIGNAL` | Signal line del MACD semanal |
| `W_MACD_HIST` | Histograma MACD semanal |
| `W_MACD_HIST_DIRECTION` | Dirección del histograma (subiendo/bajando) |
| `W_TREND` | Tendencia semanal: +1 alcista, -1 bajista |
| `W_TREND_STRENGTH` | Fuerza absoluta de la tendencia |
| `W_ABOVE_EMA` | 1 si precio > EMA 65d |
| `W_ATR_PCT` | ATR semanal como % del precio |
| `W_PRICE_VS_EMA13` | Distancia precio a EMA semanal |

### 10. Triple Screen Elder — Pantalla Diaria (16)

| Variable | Descripción |
|----------|-------------|
| `D_EMA13` | EMA 13 diaria |
| `D_EMA26` | EMA 26 diaria |
| `D1` | 1 si precio > EMA 13 |
| `D2` | 1 si precio > EMA 26 |
| `D3` | Rango diario (H-L) como % del precio |
| `D4` | Posición del cierre en rango H-L |
| `D5` | Retorno diario (pct_change) |
| `D_FORCE_INDEX_2` | Force Index 2 días (precio × volumen, EMA 2) |
| `D_FORCE_INDEX_13` | Force Index 13 días (EMA 13) |
| `D_FORCE_INDEX_2_NORM` | Force Index 2d normalizado (/ std 63d) |
| `D_FORCE_INDEX_13_NORM` | Force Index 13d normalizado |
| `D_FORCE_OVERBOUGHT` | 1 si Force Index > media + 2σ |
| `D_FORCE_OVERSOLD` | 1 si Force Index < media - 2σ |
| `D_BULL_POWER` | Elder Bull Power: High - EMA13 |
| `D_BEAR_POWER` | Elder Bear Power: Low - EMA13 |
| `D_ELDER_RAY` | Bull Power + Bear Power |
| `D_IMPULSE` | Impulse: +1 (ambos subiendo), -1 (ambos bajando), 0 (mixto) |

### 11. Triple Screen — Señales Combinadas (12)

| Variable | Descripción |
|----------|-------------|
| `TS_BUY_SETUP` | 1 si tendencia alcista + FI oversold |
| `TS_SELL_SETUP` | 1 si tendencia bajista + FI overbought |
| `TS_BUY_SIGNAL` | Compra: marea alcista + FI oversold + Bull Power > 0 |
| `TS_SELL_SIGNAL` | Venta: marea bajista + FI overbought + Bear Power < 0 |
| `TS_BUY_SIGNAL_ELDER` | Compra Elder clásica |
| `TS_SELL_SIGNAL_ELDER` | Venta Elder clásica |
| `TS_NEUTRAL` | 1 si sin señal |
| `TS_STRENGTH` | Fuerza de señal combinada |
| `TS_STRENGTH_ELDER` | Fuerza Elder |
| `TS_ALIGNMENT` | Alineación entre pantallas |
| `TS_COMBINED_STRENGTH` | Promedio de fuerzas |
| `TS_SIGNAL` | Señal final: +1 compra, -1 venta, 0 neutro |

### 12. Microestructura y Velas (30)

| Variable | Descripción |
|----------|-------------|
| `INTRA_RANGE` | Rango intradiario (High - Low) |
| `INTRA_RANGE_PCT` | Rango como % del precio |
| `INTRA_TRUE_RANGE` | True Range |
| `INTRA_ATR_5` | ATR 5 días |
| `INTRA_ATR_21` | ATR 21 días |
| `INTRA_CLOSE_POSITION` | Posición del cierre en rango H-L |
| `INTRA_OPEN_POSITION` | Posición de apertura en rango H-L |
| `RANGE_PCT_MA_21` | Media del rango % en 21 días |
| `RANGE_PCT_STD_21` | Desv. estándar del rango % en 21 días |
| `CANDLE_BODY_PCT` | Tamaño del cuerpo como % del rango |
| `CANDLE_DIRECTION` | +1 alcista, -1 bajista |
| `CANDLE_DOJI` | 1 si doji (cuerpo < 10% del rango) |
| `CANDLE_UPPER_SHADOW_PCT` | % de sombra superior |
| `CANDLE_LOWER_SHADOW_PCT` | % de sombra inferior |
| `CLV` | Close Location Value (-1 a +1) |
| `CLV_MA_21` | Media 21d del CLV |
| `CLOSE_POSITION_MA_21` | Media 21d de la posición del cierre |
| `BUYING_PRESSURE_21` | Frecuencia de cierre en tercio superior (21d) |
| `SELLING_PRESSURE_21` | Frecuencia de cierre en tercio inferior (21d) |
| `PRICE_EFFICIENCY` | Retorno / rango intradiario |
| `VOLUME_RELATIVE_21` | Volumen / media 21d |
| `VOLUME_RELATIVE_63` | Volumen / media 63d |
| `VOLUME_RANGE_PRODUCT` | Rango % × Volumen relativo |
| `GAP_OVERNIGHT` | Gap overnight (%) |
| `GAP_OVERNIGHT_ABS` | Gap overnight absoluto |
| `GAP_VS_ATR` | Gap / ATR 14 |
| `GAP_DIRECTION` | +1 up, -1 down, 0 sin gap |
| `GAP_FILLED` | 1 si gap cerrado durante el día |
| `GAP_LARGE_FLAG` | 1 si gap > ATR |
| `GAP_LARGE_FREQ_21` | Conteo de gaps grandes en 21 días |

### 13. Volatilidad — Estimadores Clásicos (9)

| Variable | Descripción |
|----------|-------------|
| `VOL_PARKINSON_21` | Volatilidad Parkinson 21d (basada en H-L) |
| `VOL_PARKINSON_63` | Volatilidad Parkinson 63d |
| `VOL_GARMAN_KLASS_21` | Volatilidad Garman-Klass 21d (usa OHLC) |
| `VOL_GARMAN_KLASS_63` | Volatilidad Garman-Klass 63d |
| `VOL_ROGERS_SATCHELL_21` | Volatilidad Rogers-Satchell 21d |
| `VOL_ROGERS_SATCHELL_63` | Volatilidad Rogers-Satchell 63d |
| `VOL_YANG_ZHANG_21` | Volatilidad Yang-Zhang 21d (incluye overnight) |
| `VOL_YANG_ZHANG_63` | Volatilidad Yang-Zhang 63d |
| `VOL_RATIO_21_63` | Ratio vol 21d/63d (compresión/expansión) |

### 14. Volatilidad — Log-Range, Alizadeh, Brandt & Diebold 2002 (20)

| Variable | Descripción |
|----------|-------------|
| `LR_log_range` | ln(High/Low) — estimador de volatilidad |
| `LR_adjusted` | Log-range ajustado |
| `LR_ma{5,10,21,63}` | Medias móviles del log-range |
| `LR_std{5,10,21,63}` | Desv. estándar rolling del log-range |
| `LR_momentum_{5,21}` | Cambio del log-range |
| `LR_ratio_{5_21,21_63}` | Ratios entre ventanas |
| `LR_percentile_{63,252}` | Percentil del log-range |
| `LR_volatility_{21,63}` | Volatilidad derivada del log-range |
| `LR_skew_63` | Asimetría del log-range (63d) |
| `LR_kurt_63` | Curtosis del log-range (63d) |

### 15. Volatilidad — CARR, Chou 2005 (13)

> **Nota:** `build_dataset.py` genera 16 features CARR, pero 3 se eliminan por >50% NaN: `CARR_range_up`, `CARR_range_down`, `CARR_range_asymmetry`.

| Variable | Descripción |
|----------|-------------|
| `CARR_range_pct` | Rango como % del precio |
| `CARR_ema_range_{5,21}` | EMA del rango |
| `CARR_range_lag{1,2,5}` | Rango rezagado |
| `CARR_range_vs_atr` | Rango / ATR |
| `CARR_standardized` | Rango estandarizado |
| `CARR_innovation` | Innovación CARR (rango - EMA) |
| `CARR_innovation_sq` | Innovación al cuadrado |
| `CARR_shock` | Shock: innovación > 2σ |
| `CARR_expansion` | Ratio rango actual / rango anterior |
| `CARR_large_range` | 1 si rango > percentil 90% |

### 16. Volatilidad — Range-GARCH, Fiszeder & Perczak 2016 (16)

| Variable | Descripción |
|----------|-------------|
| `RG_intraday_var` | Varianza intradiaria |
| `RG_overnight_var` | Varianza overnight |
| `RG_combined_var` | Varianza combinada intraday + overnight |
| `RG_combined_var_ma21` | MA 21d de varianza combinada |
| `RG_overnight_ratio` | Ratio overnight / total |
| `RG_high_contribution` | Contribución del High a la varianza |
| `RG_low_contribution` | Contribución del Low |
| `RG_hl_balance` | Balance High-Low |
| `RG_hlc_volatility` | Volatilidad HLC |
| `RG_gap_impact` | Impacto del gap en volatilidad |
| `RG_efficiency` | Eficiencia del rango |
| `RG_efficiency_ma21` | MA 21d de eficiencia |
| `RG_intraday_bias` | Sesgo intradiario |
| `RG_intraday_bias_ma21` | MA 21d del sesgo |
| `RG_vol_regime_high` | 1 si régimen de alta volatilidad (>percentil 80%) |
| `RG_vol_regime_low` | 1 si régimen de baja volatilidad (<percentil 20%) |

### 17. Volatilidad — Entropía Intrínseca, Vinte & Ausloos 2021 (5)

| Variable | Descripción |
|----------|-------------|
| `IE_price_entropy` | Entropía intrínseca del precio |
| `IE_price_entropy_ma21` | MA 21d de la entropía |
| `IE_price_concentration` | Concentración del precio |
| `IE_range_entropy` | Entropía del rango |
| `IE_vol_weighted` | Entropía ponderada por volumen |

### 18. VIX y Estructura Temporal (41)

**Niveles raw:**

| Variable | Descripción |
|----------|-------------|
| `V1` — `V13` | 13 índices de volatilidad (ver mapeo Bloomberg arriba) |
| `VIX_F1` — `VIX_F4` | VIX Futures (front-month a 4to contrato) |

**Features derivados:**

| Variable | Descripción |
|----------|-------------|
| `V1_lag{1,5,21}` | VIX rezagado 1, 5, 21 días |
| `V1_roll_mean_{21,63}` | Media rolling del VIX |
| `V1_roll_std_{21,63}` | Desv. estándar rolling del VIX |
| `VIX_change_{1d,5d,21d}` | Cambio absoluto del VIX |
| `VIX_regime_high` | 1 si VIX > percentil 75% (252d) |
| `VIX_regime_low` | 1 si VIX < percentil 25% (252d) |
| `vix_f1_spot_ratio` | Ratio futuro VIX / spot |
| `vix_3m_spot_ratio` | Ratio VIX 3M / spot |
| `vix_contango` | 1 si estructura temporal en contango |
| `vix_contango_VIX_F1` | 1 si VIX 3M > VIX spot |
| `vix_roll_yield` | Roll yield de la curva VIX |
| `vix_roll_yield_VIX_F1` | Roll yield (VIX 3M / spot - 1) × 12 |
| `vix_term_ratio_VIX_F1` | Ratio VIX 3M / VIX spot |
| `vix_term_slope_VIX_F1` | Pendiente de la curva VIX |
| `vvix_vix_ratio` | Ratio VXEEM / VIX |
| `skew_high` | 1 si Skew > 130 |
| `vol_risk_premium` | VIX - volatilidad realizada 21d |
| `vix_x_curve` | Interacción VIX × yield spread 10-2 |

### 19. Otros Índices (3)

| Variable | Descripción |
|----------|-------------|
| `DXY_ALT` | US Dollar Index (alternativo) |
| `MOVE_INDEX` | MOVE — volatilidad implícita Treasuries |
| `CVIX` | CVIX — volatilidad implícita divisas |

### 20. Correlaciones Cross-Asset (10)

| Variable | Descripción |
|----------|-------------|
| `corr_SPY_TLT_{21d,63d}` | Correlación rolling SPY vs Bonos |
| `corr_SPY_GLD_{21d,63d}` | Correlación SPY vs Oro |
| `corr_SPY_DXY_{21d,63d}` | Correlación SPY vs Dólar |
| `corr_SPY_VIX_{21d,63d}` | Correlación SPY vs VIX |
| `corr_SPY_OIL_{21d,63d}` | Correlación SPY vs Petróleo |

### 21. Yield Curve y Crédito (16)

| Variable | Descripción |
|----------|-------------|
| `yield_spread_10_2` | Spread Treasury 10Y - 2Y |
| `yield_spread_10_3m` | Spread Treasury 10Y - 3M (indicador de recesión) |
| `yield_spread_10_3m_inverted` | 1 si spread 10Y-3M invertido |
| `yield_curve_inverted` | 1 si curva invertida (10Y < 2Y) |
| `yield_butterfly` | Butterfly spread: 2Y + 30Y - 2×5Y |
| `yield_spread_10_2_change_{1d,5d,21d}` | Cambios del spread 10Y-2Y |
| `spread_ratio_hy_ig` | Ratio spread HY / IG |
| `spread_ratio_chg` | Cambio del ratio HY/IG |
| `hy_spread_chg_{5d,21d}` | Cambio del spread High Yield |
| `hy_spread_regime_high` | 1 si HY spread > percentil 80% (252d) |
| `ig_spread_chg_{5d,21d}` | Cambio del spread Investment Grade |
| `credit_x_vix` | CDX HY × VIX (estrés combinado) |
| `yield_spread_10_2_change_1d` | Cambio diario spread 10Y-2Y (adicional) |

### 22. Tasas de Interés — Niveles Raw (20)

Todas las tasas I1-I20 están presentes como niveles (ver mapeo Bloomberg arriba).

### 23. Tasas de Interés — Transformaciones (22)

Para I1, I2, I10-I20 se generan:
- `{col}_pct_change`: cambio % diario
- `{col}_momentum_5`: cambio absoluto en 5 días

### 24. Tasas de Interés — Lags (27)

Para I1-I9 se generan lags con `shift(n)`:
- `{col}_lag{1,5,21}`: valor hace 1, 5, 21 días

### 25. Tasas de Interés — Rolling Stats (24)

Para I1, I2, I4, I6, I8, V1 se generan:
- `{col}_roll_mean_{21,63}`: media rolling
- `{col}_roll_std_{21,63}`: desv. estándar rolling

### 26. Mercados y ETFs (32)

**Niveles raw:** M1-M18 (18 variables — ver mapeo Bloomberg)

**Lags:** M3, M4, M5, M13, M18 × lag{1,5,21} = 15 variables

Ejemplos: `M3_lag1` (Russell 2000 hace 1 día), `M13_lag21` (XLF hace 21 días)

### 27. Commodities (15)

**Niveles raw:** P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11 (11 variables)

**Alias descriptivos:**
| Variable | Origen | Descripción |
|----------|--------|-------------|
| `P9_GOLD` | GLD | Oro ETF (alias) |
| `P10_SILVER` | SI1 | Plata futures (alias) |
| `P11_CMDTY_IDX` | BCOMTR | Bloomberg Commodity TR (alias) |
| `P12_OIL` | CL1 | Petróleo WTI (alias) |

### 28. Indicadores Económicos (52)

**Niveles raw:** E1-E19 (19 variables)

**Lags:** E1 × lag{1,5,21}, E4 × lag{1,5,21}, E8 × lag{1,5,21} = 9 variables

**Momentum derivado (24):**

| Variable | Descripción |
|----------|-------------|
| `gdp_momentum_{21d,63d}` | Cambio GDP |
| `gdp_trend` | Tendencia GDP: sign(MA5 - MA21) |
| `pmi_momentum_{21d,63d}` | Cambio PMI |
| `pmi_trend` | Tendencia PMI |
| `nfp_momentum_{21d,63d}` | Cambio Nonfarm Payrolls |
| `nfp_trend` | Tendencia NFP |
| `unemployment_momentum_{21d,63d}` | Cambio desempleo |
| `unemployment_trend` | Tendencia desempleo |
| `cpi_momentum_{21d,63d}` | Cambio CPI |
| `cpi_trend` | Tendencia CPI |
| `ip_momentum_{21d,63d}` | Cambio producción industrial |
| `ip_trend` | Tendencia producción industrial |
| `consumer_conf_momentum_{21d,63d}` | Cambio confianza consumidor |
| `consumer_conf_trend` | Tendencia confianza consumidor |
| `claims_momentum_4w` | Mejora en solicitudes desempleo |
| `inflation_momentum` | Cambio CPI en 21 días |
| `core_inflation_momentum` | Cambio Core PCE en 21 días |

### 29. Sentimiento y Sectores (17)

| Variable | Descripción |
|----------|-------------|
| `S1` | AAII Bullish (% inversores alcistas) |
| `S4` | Put/Call ratio |
| `S4_HY_INDEX` | CDX HY 5Y (proxy aversión al riesgo) |
| `S5` | NYSE New Highs - New Lows |
| `S6` | NYSE TICK |
| `S7` | Arms Index (TRIN) |
| `S8` | NYSE Advance/Decline |
| `S9` | McClellan Oscillator |
| `S10` | McClellan Summation Index |
| `S11` | Short-Range Volatility |
| `S11_CHINA` | Volatilidad China (derivado) |
| `S12` | Equity Put/Call Ratio |
| `sector_dispersion_{1d,5d,21d}` | Dispersión de retornos entre sectores |
| `sector_rotation_range` | Rango max-min de retornos sectoriales |
| `sectors_positive` | Fracción de sectores con retorno positivo |

### 30. Régimen de Mercado (4)

| Variable | Descripción |
|----------|-------------|
| `regime_score` | Score 0-1: promedio de señales (trend, momentum, sectores, VIX) |
| `regime_bull` | 1 si regime_score > 0.7 |
| `regime_bear` | 1 si regime_score < 0.3 |
| `regime_neutral` | 1 si 0.3 ≤ regime_score ≤ 0.7 |

### 31. Factor e Interacciones (9)

| Variable | Descripción |
|----------|-------------|
| `value_growth_spread_{1d,21d}` | Retorno Value (XLF) - Growth (XLK) |
| `value_momentum_21d` | Momentum 21d del sector Value |
| `growth_momentum_21d` | Momentum 21d del sector Growth |
| `small_large_spread_21d` | Retorno Small Caps (IWM) - Large (SPY) |
| `tech_momentum_21d` | Momentum 21d sector Tech |
| `tech_vs_spy_21d` | Alpha Tech vs SPY |
| `momentum_high_vol` | Momentum 21d en entorno VIX alto |
| `momentum_low_vol` | Momentum 21d en entorno VIX bajo |
