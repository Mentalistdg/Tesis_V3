# Spec — Pipeline de producción CRONOS (Bloomberg → señal diaria → app)

- **Fecha:** 2026-10-07
- **Autor:** David González Cañón (con Claude Code)
- **Estado:** borrador para revisión

## 1. Objetivo

Que el modelo LSTM_Attention de la tesis opere con datos actuales en este PC (el que tiene la
Terminal Bloomberg):

1. Extraer desde la Terminal, sin intervención manual, las 97 series que alimentan el modelo.
2. Extender el dataset de forma **idéntica** a como se construyó para entrenar.
3. Emitir cada día hábil la señal **CASH / SPY (1x) / UPRO (3x)** para que el usuario decida si
   toma posición (no hay ejecución automática de órdenes).
4. Mostrar las señales en la app CRONOS (primero local, después en la nube).
5. Primer uso: responder qué señales habría dado el modelo desde el 12-dic-2025 hasta hoy.

A futuro se agregarán otros activos al seguimiento; hoy solo existe el S&P 500.

## 2. Contexto verificado (7-oct-2026)

Detalle y evidencia en `research/identificacion_series/README.md`.

- Los encabezados de `BLOOMBERG_RAW_DATA.csv` (repo Tesis_V3) **no corresponden** a su contenido.
  Las 97 columnas fueron identificadas contra Bloomberg (historia 2000–2025):
  73 de mercado con calce exacto 100%; 20 macro con la misma serie y fechas (ρ 0.996–1.000,
  diferencias por revisiones); 4 de valoración SPX con ρ 0.95–0.99.
- Reglas del CSV crudo: precios de ETFs ajustados solo por splits; series diarias unidas al
  calendario de SPY por fecha exacta **sin relleno**; macro y fundamentales con `PX_LAST` fechado al
  cierre del período, unidos por fecha exacta y **con forward-fill** (observaciones fechadas en fin
  de semana se pierden).
- `build_dataset.py` de la tesis con **pandas 2.3.3** reproduce las estadísticas de entrenamiento
  (541/541 features) y el modelo reproduce la tesis exactamente (+395.4%, Sharpe 1.319). El CSV
  `data/bloomberg_triple_screen_core.csv` distribuido con el paquete difiere en 12 columnas y es la
  causa del resultado +364.2% que entrega hoy `run.bat`.
- La macro en el entrenamiento está fechada al cierre del período, antes de su publicación
  (sesgo de anticipación: ISM ~1 día, CPI ~15 días, PIB ~1 mes).
- Bloomberg entrega `ECO_RELEASE_DT` histórico para 19 de las 20 series macro (falta LEI); para el
  PIB corresponde a la última revisión, no a la primera publicación.

## 3. Criterios de éxito

1. **Regresión exacta:** con la historia congelada, el pipeline produce features con estadísticas de
   entrenamiento idénticas al scaler (541/541) y predicciones/backtest idénticos a la tesis
   (diff máx de predicción < 1e-6; retorno +395.4%, Sharpe 1.319).
2. **Extracción completa:** una corrida descarga las 97 series; en la ventana de solapamiento las
   series de nivel calzan exacto con lo almacenado y las de precio calzan tras el encadenado.
3. **Reporte del período nuevo:** señales diarias A y B desde el 12-dic-2025 hasta el último día
   disponible, con entradas, distribución de días por instrumento y retorno con costos vs SPY.
4. **Operación desatendida:** la tarea programada corre lunes a viernes, se pone al día sola tras
   días perdidos y deja la señal en `logs/senales.csv` y en la app.
5. **App local:** la página "Señales" muestra la señal del día, su cambio respecto del día anterior,
   el estado de los datos y el historial en vivo.

## 4. Arquitectura

```
 Terminal Bloomberg (blpapi, localhost:8194)
        │
 [1] extraer      activos/<id>/series.csv → descarga incremental + ventana de solapamiento
 [2] empalmar     historia congelada + días nuevos → data/<id>/raw_extendido.csv (98 columnas)
 [3] validar      solapamiento, splits, series caídas, días faltantes → estado verde/amarillo/rojo
 [4] features     build_dataset.py de la tesis (+ parche mínimo) → dataset_A.csv y dataset_B.csv
 [5] señal        modelo + Meta-KNN → señal B (operativa) y A (diagnóstico)
 [6] exportar     logs/senales.csv + JSON para la app
```

### 4.1 Estructura de carpetas

```
CRONOS_PRODUCCION/
  activos/
    spx/
      series.csv              mapeo verificado de las 97 series (ver 5.1)
      config.yaml             instrumentos, costos, umbrales, ancla de empalme, rezagos de publicación
      historia_congelada.csv  copia de BLOOMBERG_RAW_DATA.csv (hasta 2025-12-12), solo lectura
      modelo/                 LSTM_Attention.pt, preprocessors.joblib, resultados_esperados.json
  pipeline/
    bloomberg.py              cliente blpapi (lotes, reintentos, ajustes, errores explícitos)
    extraer.py                etapa 1
    empalmar.py               etapa 2
    validar.py                etapa 3
    features.py               etapa 4 (envuelve tesis/build_dataset.py)
    senal.py                  etapa 5 (lógica de produccion_lstm.py)
    exportar.py               etapa 6
    cronos.py                 punto de entrada: python -m pipeline.cronos actualizar [--ensayo]
    tesis/build_dataset.py    copia sin cambios de lógica + parche documentado (ver 7.1)
  app/                        app CRONOS copiada desde Tesis_V3 + página "Señales"
  data/<id>/                  raw_extendido.csv, dataset_A.csv, dataset_B.csv, descargas/AAAA-MM-DD/
  logs/                       senales.csv, pipeline.log
  tests/
  research/                   identificación de series (ya existe)
  actualizar.bat, abrir_app.bat, programar_tarea.bat, run.bat
```

Sumar un activo = agregar `activos/<id>/` con sus series, modelo y config. El motor recorre los
activos habilitados. Entrenar modelos de activos nuevos está fuera de alcance.

### 4.2 Decisiones de base

- **La historia no se reescribe:** las filas hasta el 12-dic-2025 provienen siempre de
  `historia_congelada.csv`.
- **Un entorno con las versiones de la tesis** (sección 10).
- **Ubicación de ejecución:** clon en `C:\Users\itau_lab\CRONOS_PRODUCCION` (tarea programada y app).
  El USB es copia portátil; ambas se sincronizan vía el repo privado de GitHub.

## 5. Extracción

### 5.1 `series.csv`

Una fila por columna cruda. Columnas:

| Campo | Descripción |
|---|---|
| `columna_cruda` | Encabezado en el CSV crudo (clave; `build_dataset.py` lo espera así) |
| `codigo` | Código interno que asigna `COLUMN_MAPPING` (M1, E3, I15…) |
| `ticker`, `campo` | Serie Bloomberg real verificada (p.ej. `XLK US Equity`, `PX_LAST`) |
| `ajuste` | `none` / `split` / `default` (configuración que reprodujo el CSV) |
| `tipo` | `precio` / `volumen` / `nivel` / `fundamental` / `macro` |
| `relleno` | `no` (series diarias) / `ffill` (macro y fundamentales) |
| `rezago_pub_dias` | Rezago fijo de publicación para la versión B cuando `ECO_RELEASE_DT` falta o refleja una revisión (PIB, LEI) |
| `notas` | Evidencia de la verificación |

Se genera desde `research/identificacion_series/scripts/mapping.py` (con las correcciones finales:
E3 = `CPMINDX Index`, E19 = `DGNOXTCH Index`, ETFs con ajuste `split` salvo IWF `none`).

### 5.2 Descarga

- Rango por corrida: desde (última fecha almacenada − 30 días hábiles) hasta hoy. Primera corrida:
  desde 2025-11-01 (solapamiento con la historia) hasta hoy (~205 días hábiles nuevos).
- Pedidos `HistoricalDataRequest` agrupados por (campo, ajuste), lotes de hasta 25 tickers,
  3 reintentos con espera. Macro: además `ECO_RELEASE_DT`.
- Calendario: días con dato de `SPY US Equity PX_LAST`.
- Cada respuesta se guarda sin procesar en `data/<id>/descargas/AAAA-MM-DD/` (auditoría: Bloomberg
  revisa datos).
- Una serie sin respuesta o con `securityError` se reporta por nombre; nunca se sustituye en
  silencio.

## 6. Empalme y fechado

### 6.1 Reglas por tipo (días posteriores al ancla 2025-12-12)

| Tipo | Regla |
|---|---|
| `precio` (ETFs, SPY OHLC) | Encadenar retornos: `v[t] = v_congelado[ancla] × bbg[t] / bbg[ancla]`, con `bbg` descargado hoy en su ajuste. Inmune a splits posteriores a la extracción. |
| `volumen` (SPY) | Valor directo; un split de SPY se reporta como advertencia. |
| `nivel` (tasas, spreads, VIX, OAS, CDX, put/call, UX1/UX2, índices TR) | Valor directo. |
| `fundamental` (SPX PE, P/B, EY, PE forward) | Valor directo; salto > 1% en el solapamiento → advertencia. |
| `macro` | Valor directo; períodos ya presentes en la historia congelada no se reescriben. |
| US0003M (LIBOR) | Permanece NaN (discontinuada en 2023, igual que en el entrenamiento). |

Unión al calendario: por fecha exacta; `relleno = ffill` solo para macro y fundamentales.

### 6.2 Versiones A y B de la macro

- **A (diagnóstico, réplica del entrenamiento):** macro fechada al cierre del período, como en la
  historia congelada.
- **B (operativa, point-in-time en fechas):** cada observación macro se ubica en su fecha de
  publicación (`ECO_RELEASE_DT`, o cierre de período + `rezago_pub_dias` si falta o es una revisión)
  y desde ahí se rellena. B se aplica a **toda** la historia para que lags y ventanas móviles sean
  coherentes. Solo afecta a `tipo = macro`; los fundamentales del SPX son diarios y no se desplazan.
- Construcción de B: una descarga única de la historia completa de cada serie macro (`PX_LAST` +
  `ECO_RELEASE_DT`) entrega la lista de observaciones (fecha de período, fecha de publicación). El
  valor de cada observación es el de la historia congelada en esa fecha de período cuando existe;
  si no existe (p.ej. observaciones fechadas en fin de semana que A descartó, como el PMI China de
  feb-2020) o es posterior al ancla, se usa el valor de Bloomberg.
- Limitación: los *valores* siguen siendo los de la vintage de extracción (Bloomberg no entrega la
  primera publicación de forma confiable).
- La señal operativa es la B. La diferencia A–B en el período nuevo (y sobre la historia de la
  tesis) mide el efecto del sesgo de anticipación.

## 7. Features y señal

### 7.1 Features (`features.py`)

`pipeline/tesis/build_dataset.py` es copia del original con un parche mínimo y documentado
(`pipeline/tesis/PARCHE.md` con el diff):

1. No descartar filas con target NaN (la última fila es la de la señal del día).
2. Conjunto de columnas fijo: las 541 de `preprocessors.joblib` (vía `LEGACY_MAP`), en ese orden;
   falta de cualquiera → error bloqueante. Se desactiva la selección dinámica por % de NaN.

Entrada en memoria (raw extendido A o B), salida `dataset_A.csv` / `dataset_B.csv`.

### 7.2 Señal (`senal.py`)

- Reutiliza de `produccion_lstm.py`: arquitectura, preprocesamiento, Meta-KNN entrenado solo con
  datos de entrenamiento, percentiles rolling 63d y modelo de costos.
- Predice todas las filas, incluida la última sin target; backtest solo con filas con target.
- Corre tras el cierre de EE.UU. (tarea 18:30 Chile).
- `produccion_lstm.py` y `run.bat` siguen funcionando como verificación de la tesis; su CSV se
  reemplaza por el dataset reconstruido con pandas 2.3.3 (reproduce +395.4%).

### 7.3 Salidas

- `logs/senales.csv` (una fila por activo y día): fecha, activo, señal B, señal A, predicción B/A,
  percentil, umbrales, cambio vs día anterior, estado de datos, advertencias.
- Reporte del período nuevo (texto + JSON).
- JSON para la app (sección 9).

## 8. Validación y errores

| Nivel | Casos | Efecto |
|---|---|---|
| Bloqueante (rojo) | Sin conexión a Bloomberg o Terminal sin sesión; falta OHLCV de SPY del día; serie de nivel no calza en el solapamiento; columna del modelo ausente; falla la prueba de regresión | No emite señal, no modifica archivos, registra causa |
| Advertencia (amarillo) | Serie de mercado no crítica sin dato del día (queda NaN, como en el original); salto de fundamentales > 1%; split detectado y encadenado | Emite señal con la advertencia |
| Normal (verde) | Macro sin publicación nueva | — |

- Idempotente: correr dos veces el mismo día da el mismo resultado; `raw_extendido.csv` solo se
  actualiza si la validación pasa (escritura a archivo temporal + renombrado).
- Sin día nuevo (fin de semana, feriado, ya procesado) → termina sin cambios.

## 9. App (fase 1: local)

- Se copia `app/` desde Tesis_V3 al repo de producción.
- Backend: nuevos endpoints `GET /api/senales` (estado y señal del día por activo) y
  `GET /api/senales/{activo}/historial` (serie en vivo A/B, posiciones, retornos, trades). FastAPI
  sirve también el frontend compilado (sin nginx en Windows).
- Frontend: página **"Señales"** — tarjeta por activo con señal del día, cambio vs ayer, predicción y
  percentil vs umbrales, señal A junto a B, estado de datos; historial del período en vivo. El
  período en vivo se expone además como un "modelo" en `daily_data.json` para reutilizar las
  páginas Trades, Riesgo y Costos.
- Node portable instalado en `C:` (sin admin) solo para compilar el frontend.
- `abrir_app.bat` levanta uvicorn y abre el navegador en `localhost`.

**Fase 2 (fuera de alcance de esta implementación):** publicar la app en la nube con el Dockerfile
existente; el pipeline sube a GitHub solo los JSON de señales/resultados (nunca datos crudos de
Bloomberg); acceso protegido con contraseña.

## 10. Operación en este PC

- Tarea del Programador de tareas para el usuario `itau_lab` (sin admin), lunes a viernes 18:30 con
  reintentos 19:30 y 21:00, ejecutando `actualizar.bat` desde el clon en `C:`. Corre solo con la
  sesión de Windows iniciada. `programar_tarea.bat` crea/elimina la tarea.
- Requisito operativo: Terminal Bloomberg abierta y con sesión iniciada.
- Entorno: Python 3.12 portable (uv) en `C:\Users\itau_lab\cronos-python`, con pandas 2.3.3,
  numpy 2.3.5, scikit-learn 1.7.2, torch 2.6.0 (CPU), TA-Lib 0.6.8, blpapi, fastapi, uvicorn.
  `requirements.txt` y `run.bat` se actualizan a estas versiones.

## 11. Pruebas

- **Unitarias** (sin Bloomberg, datos sintéticos): encadenado con split simulado; unión por fecha
  exacta con/sin relleno y pérdida de fechas de fin de semana; desplazamiento B por fecha de
  publicación y por rezago fijo; clasificación de errores; idempotencia.
- **Regresión** (sin Bloomberg; también antes de cada señal): criterio de éxito 1.
- **Integración** (con Bloomberg, manual): descarga de las 97 series y calce en el solapamiento.
- **Ensayo:** `actualizar --ensayo` ejecuta todo sin escribir.

## 12. Repositorio

- Git en el proyecto; repo privado nuevo `Mentalistdg/CRONOS_PRODUCCION` (push solo con aprobación
  del usuario). Clon de trabajo en `C:`; el USB es otro clon.
- `.gitignore`: entornos de Python, `__pycache__/`, `.idea/`, `data/*/descargas/`, `node_modules/`,
  build del frontend, `tools/` (uv.exe, 38 MB; `run.bat`/`actualizar.bat` lo descargan si falta).
- Se versionan código, `series.csv`, modelos, historia congelada, datasets reconstruidos y
  `logs/senales.csv`. El repo contiene datos de Bloomberg: debe permanecer privado.

## 13. Fuera de alcance

- Ejecución automática de órdenes.
- Entrenar o reentrenar modelos (incluidos los de activos nuevos).
- Fase 2 (nube) y notificaciones (correo, celular).
- Recuperar la primera publicación de los datos macro (vintages).

## 14. Riesgos

| Riesgo | Mitigación |
|---|---|
| Límites de uso de la API de la Terminal | Descarga incremental (~97 series × ~30 días por corrida) |
| Terminal cerrada o sin sesión a la hora de la tarea | Error bloqueante + reintentos + puesta al día automática |
| Ticker discontinuado o renombrado | Error por nombre; corrección en `series.csv` sin tocar código |
| Splits o eventos corporativos nuevos | Encadenado de retornos + advertencia |
| Revisiones de datos macro y fundamentales | Historia congelada; descargas diarias guardadas; diferencias reportadas |
| Distribución de la macro B distinta a la del entrenamiento | Señal A en paralelo para medir el efecto |
