# Identificación de series Bloomberg (7-oct-2026)

Investigación que identificó, en la Terminal Bloomberg, la serie real detrás de cada una de las 97
columnas de `BLOOMBERG_RAW_DATA.csv` (repo Tesis_V3). **Los encabezados de ese CSV no corresponden
a su contenido**; la fuente de verdad es `scripts/mapping.py`.

## Resultado

| Grupo | Columnas | Verificación 2000–2025 |
|---|---|---|
| Mercado (ETFs, tasas, spreads, volatilidad, SPY OHLCV) | 73 | Calce exacto 100% (UX1 y CVIX: 1–17 días con diferencia mínima) |
| Macro (EE.UU. + PMI China) | 20 | Misma serie y fechas; ρ 0.996–1.000; diferencias = revisiones posteriores |
| Valoración SPX (PE, P/B, earnings yield, PE forward) | 4 | ρ 0.95–0.99; Bloomberg revisa utilidades del índice |

Con el CSV crudo original, `build_dataset.py` de la tesis ejecutado con **pandas 2.3.3** reproduce las
estadísticas de entrenamiento del scaler en 541/541 features, y `produccion_lstm.py` reproduce las
predicciones (diff máx 1.5e-8) y el backtest de la tesis (+395.4%, Sharpe 1.319). El CSV que
acompaña al paquete difiere en 12 columnas (regenerado con pandas 3).

## Reglas de extracción (necesarias para reproducir el CSV crudo)

- **Precios de ETFs:** ajustados solo por splits (`adjustmentSplit=True`, sin dividendos), según la
  configuración vigente al extraer (~dic-2025). Excepción: IWF calza sin ajuste (split posterior).
- **Macro y fundamentales:** `PX_LAST` vigente al extraer, fechado al cierre del período, unido al
  calendario de trading de SPY **por fecha exacta** y luego forward-fill. Las observaciones fechadas
  en fin de semana se pierden (p.ej. PMI China feb-2020).
- Las fechas de período implican **sesgo de anticipación** (el dato aparece antes de su publicación).

## Contenido

- `scripts/mapping.py` — mapeo verificado columna cruda → (ticker, campo, ajuste, tipo).
  Correcciones finales incluidas: E3 = `CPMINDX Index`, E19 = `DGNOXTCH Index`; ajuste `split`
  para ETFs (ver `reportes/` para la evidencia; `full_report.csv` es anterior a esas correcciones).
- `scripts/bbg.py` — cliente mínimo `HistoricalDataRequest` (blpapi).
- `scripts/match*.py`, `macro*.py`, `screen*.py`, `search_eco.py` — rondas de identificación.
- `scripts/verify_full.py` — verificación de historia completa.
- `scripts/compare_build.py`, `run_prod_with.py` — reproducción del dataset y del backtest.
- `reportes/` — salidas de las rondas y búsquedas de tickers.

Entornos: `C:\Users\itau_lab\cronos-python\cpython-3.12-windows-x86_64-none` (producción) y
`C:\Users\itau_lab\cronos-python\thesis-env` (versiones exactas de la tesis).
