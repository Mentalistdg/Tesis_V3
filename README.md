# CRONOS — Producción (LSTM + Attention sobre el S&P 500)

Pipeline de producción del modelo ganador de la tesis *"Predicción de retornos del S&P 500 con
Triple Screen de Elder + Machine Learning"* (David González Cañón, FEN — Universidad de Chile).

Cada día hábil, en este PC (el que tiene la Terminal Bloomberg):

1. **Extrae** desde Bloomberg las 97 series que usa el modelo (`activos/spx/series.csv`, verificadas una a una
   contra la historia 2000–2025; ver `research/identificacion_series/`).
2. **Empalma** los días nuevos sobre la historia congelada con la que se entrenó (hasta el 11-dic-2025).
3. **Valida** los datos (verde / amarillo / rojo).
4. **Construye las 541 features** con el `build_dataset.py` de la tesis (parche mínimo documentado en
   `pipeline/tesis/PARCHE.md`).
5. **Emite la señal** CASH / SPY (1x) / UPRO (3x) para el **día hábil NYSE siguiente** al último dato y la muestra en la app.

> La predicción del día *t* usa datos hasta *t−1*: la señal se calcula en la noche y se ejecuta con una
> **orden MOC al cierre del día indicado** (antes de 15:50 NY), igual que en el backtest de la tesis.
> No hay ejecución automática de órdenes: la decisión de tomar posición es del usuario.

---

## Estado al 7-oct-2026 (para retomar)

**En producción y funcionando.** Última corrida: verde, último dato 2026-10-06, señal para 2026-10-07 = **SPY 1x**
(cambió desde UPRO). Suite de tests: 88/88 (incluye pruebas con la Terminal en vivo).

### Dónde está cada cosa

| Qué | Dónde |
|---|---|
| **Copia de producción** (la que usan las tareas y la app) | `C:\Users\itau_lab\CRONOS_PRODUCCION` — rama `main` |
| Copia portátil | USB `E:\CRONOS_PRODUCCION` (FAT32: git exige `-c safe.directory=E:/CRONOS_PRODUCCION`) |
| **GitHub** | Repo privado `Mentalistdg/Tesis_V3`, rama **`cronos-produccion`** (remote `github` en la copia de C:; `main` la sigue) |
| Python 3.12 + dependencias | `C:\Users\itau_lab\cronos-python\cpython-3.12-windows-x86_64-none\python.exe` |
| Entorno con versiones exactas de la tesis | `C:\Users\itau_lab\cronos-python\thesis-env` |
| Node portable (solo para compilar el frontend) | `C:\Users\itau_lab\cronos-python\node` · carpeta de compilación `C:\Users\itau_lab\cronos-build\frontend` |
| GitHub CLI portable (sin sesión iniciada) | `C:\Users\itau_lab\cronos-python\gh\bin\gh.exe` |
| Repo original de la tesis (clon) | `C:\Users\itau_lab\Tesis_V3` |
| Tareas programadas | `CRONOS_actualizar_1930`, `_2130`, `_2330`, `_0930`, `_1130` (lun–vie, hora de Chile) |
| App | `http://127.0.0.1:8000/senales` (usar 127.0.0.1; `localhost` puede fallar por IPv6) |

### Cómo retomar mañana

1. Abrir la Terminal Bloomberg con sesión iniciada (sin ella las corridas quedan en rojo).
2. Revisar que las corridas nocturnas funcionaron:
   `programar_tarea.bat estado` y las últimas líneas de `logs\pipeline.log` (debe decir `estado=verde` o `amarillo`).
3. Abrir la app (`abrir_app.bat`) **después de las corridas de 09:30 / 11:30** y leer la señal del día.
   Si aparece un aviso rojo ("última corrida falló" o "señal vencida"), no operar y revisar `logs\pipeline.log`.
4. Para subir cambios a GitHub: `! git -C C:/Users/itau_lab/CRONOS_PRODUCCION push` (el `!` es necesario para
   iniciar sesión en el navegador; la terminal de Claude no puede pedir la contraseña).

### Primer resultado en vivo (11-dic-2025 → 5-oct-2026, 204 días)

| | Entradas | Retorno con costos | Sharpe | Máx. drawdown | UPRO/SPY/CASH |
|---|---|---|---|---|---|
| **Versión B (operativa)** | 15 | **+15,6%** | 0,72 | −21,6% | 18/54/28% |
| Versión A (diagnóstico) | 14 | −5,5% | −0,58 | −23,0% | 19/52/29% |
| SPY buy & hold | — | +13,0% | — | — | — |

Con 1 día de retraso en la ejecución: B +15,1%, A +13,1%. Detalle en `logs/reporte_spx.txt`.

### Hallazgos clave (no volver a investigar)

- **Los encabezados de `BLOOMBERG_RAW_DATA.csv` (Tesis_V3) no corresponden a su contenido** (p.ej. "IWM" contiene
  XLF, "CL1 Comdty" contiene el P/E del SPX). El mapeo real verificado está en `activos/spx/series.csv`.
  E3 = PMI manufacturero de China (`CPMINDX`), E19 = órdenes de bienes duraderos ex transporte (`DGNOXTCH`).
- **Reglas del CSV original:** precios de ETFs ajustados solo por splits; series diarias unidas por fecha exacta
  sin relleno; macro con valor vigente al extraer, fechada al cierre del período y con relleno (las fechas de fin
  de semana se pierden).
- **El CSV de features que venía con el paquete difería del de entrenamiento** (regenerado con pandas 3). Con
  pandas 2.3.3 se reproduce exacto (541/541 features, +395,4%). Por eso pandas 2.3.3 es obligatorio.
- **Momento de ejecución:** la predicción de *t* usa features hasta *t−1* (verificado). El backtest es operable:
  señal de noche, orden MOC al cierre del día siguiente. (Una alarma anterior de "+77% con 1 día de retraso"
  fue un error de interpretación; ver spec §16.)
- **Meta-KNN en vivo** usa la ventana de mercado hasta *t−2* (el retorno *t−1→t* aún no se conoce al decidir):
  en la prueba de la tesis da +412,5% vs +395,4% (14 de 1.301 días distintos).
- **Series que Bloomberg publica tarde** (hora de Chile, 7-oct): MOVE 17:49, SKEW 18:19, índices de bonos
  (LF98OAS, LF98TRUU, LUACTRUU, LUACOAS) 18:49; CVIX y put/call total (PCUSEQTR) aún no publicadas a las 18:49.
  La cola mutable (5 días hábiles) las completa en corridas posteriores.
- El 12-dic-2025 de la historia original tenía el CVIX sin publicar; por eso el ancla congelada es el **11-dic-2025**.

### Pendientes (menores, detectados en la revisión final)

- `version_b`: ordenar por `["pub", "periodo"]` (orden no determinista si dos datos se publican el mismo día).
- A y B se alinean por posición en `senales_recalculadas.csv` e historial: unir por `date`.
- `guardar_descarga` crea la carpeta de auditoría aun cuando la corrida queda en rojo.
- Registrar el `responseError` de blpapi (hoy "Terminal sin sesión" aparece como "sin respuesta").
- `--hasta` sin protecciones: puede usar barras intradía o recortar `raw_extendido` con una fecha pasada.
- Horarios: 19:30 Chile siempre cae antes de 20:00 NY; entre nov y mar solo la de 23:30 sirve de noche
  (respaldan las de la mañana). Evaluar agregar 22:30.
- `logs\salida.log` no se rota.
- Falta advertencia por split en el volumen de SPY (spec §6.1).
- "Cambió vs ayer" de la tarjeta (recalculado) y del log (emitido) pueden discrepar; no se avisa si la señal
  cambia entre la corrida de la noche y la de la mañana.
- Conectar la copia del USB a GitHub (`git remote add github ...`) si se quiere sincronizar por ahí.
- Fase 2 (fuera de alcance por ahora): publicar la app en la nube (Dockerfile existente) subiendo solo los JSON de señales.
- A futuro: agregar otros activos (`activos/<id>/` con sus series, modelo entrenado y configuración).

---

## Uso diario

| Comando | Qué hace |
|---|---|
| `abrir_app.bat` | Abre la app en `http://127.0.0.1:8000/senales` (señal del día, estado de datos, historial en vivo). |
| `actualizar.bat` | Corre el pipeline a mano (`--ensayo` = sin escribir; `--hasta AAAA-MM-DD`). Salida en `logs\salida.log`. |
| `programar_tarea.bat crear` | Programa las corridas automáticas (lun–vie 19:30, 21:30, 23:30, 09:30, 11:30, hora de Chile). |
| `programar_tarea.bat estado` / `eliminar` / `probar` | Ver, quitar o lanzar ahora las corridas programadas. |
| `run.bat` | Verificación de la tesis: reproduce el backtest publicado (+395,4%, Sharpe 1,319). |

**Requisitos:** Terminal Bloomberg abierta y con sesión iniciada; sesión de Windows del usuario iniciada a la
hora de las tareas. El entorno de Python (3.12 portable en `%USERPROFILE%\cronos-python`) se instala solo la
primera vez (`preparar_entorno.bat`).

**Frontend:** `app/frontend/dist/` no se versiona. Para recompilar: copiar `app/frontend` a
`C:\Users\itau_lab\cronos-build\frontend`, correr `npm install` y `npm run build` con Node de
`C:\Users\itau_lab\cronos-python\node`, y copiar `dist/` de vuelta (en el USB FAT32 npm es inviable).

## Cómo leer las señales

- **Versión B (operativa):** los datos macro entran en su fecha real de publicación (point-in-time).
- **Versión A (diagnóstico):** los datos macro entran en la fecha del período, como en el entrenamiento
  (eso implica conocerlos antes de su publicación). La diferencia A–B mide ese sesgo.
- **Estado de datos:** verde = completo; amarillo = advertencias (p.ej. CVIX sin dato en feriados del Reino
  Unido, revisiones de utilidades del SPX); **rojo = no se emitió señal** (ver `logs\pipeline.log`).
- La app avisa si la última corrida falló (`app/backend/data/estado_corrida.json`) o si la señal está vencida.
- `logs/senales_emitidas.csv`: registro solo-agregar de cada señal mostrada (con hora y estado).
- `logs/reporte_spx.txt`: desempeño del período en vivo (entradas, retorno vs SPY, escenario con 1 día de retraso).

## Mantención

- **Una serie cambia de ticker o se discontinúa:** corregir la fila en `activos/spx/series.csv` (no el código).
  `US0003M` (LIBOR) y `USSWAP10` están discontinuadas desde 2023 y ya eran NaN en el entrenamiento.
- **Pruebas:** `%CRONOS_PY% -m pytest -m "not bloomberg"` (sin Terminal) y `-m bloomberg` (con Terminal).
  La prueba de regresión contra la tesis corre sola en el pipeline cuando cambia el código o el entorno
  (resultado cacheado en `logs/regresion.json`).
- **pandas 2.3.3 es obligatorio:** con otra versión las features cambian en silencio; el pipeline se niega a correr.
- **Calendario NYSE** en `pipeline/calendario.py` (feriados regulares); los cierres extraordinarios (p.ej. duelo
  nacional) darán rojo ese día por falta de barra de SPY.
- Agregar un activo = nueva carpeta `activos/<id>/` con sus series, modelo y configuración.

## Estructura

| Ruta | Contenido |
|---|---|
| `pipeline/` | Motor: `bloomberg`, `extraer`, `empalmar`, `validar`, `features`, `senal`, `exportar`, `calendario`, `cronos` |
| `pipeline/tesis/` | `build_dataset.py` de la tesis + parche documentado |
| `activos/spx/` | Series verificadas, configuración, modelo entrenado e historia congelada |
| `app/` | App CRONOS (FastAPI + React) con la página "Señales" |
| `data/spx/` | Datos generados (no versionados): raw extendido, datasets A/B, publicaciones macro, descargas diarias |
| `logs/` | `pipeline.log`, `salida.log`, `senales_emitidas.csv` y `reporte_spx.txt` (estos dos sí se versionan) |
| `research/` | Identificación y verificación de las series en Bloomberg |
| `docs/superpowers/` | Spec (con adendas §15–16) y plan de implementación |
| `produccion_lstm.py` | Script de verificación de la tesis (usa `data/bloomberg_triple_screen_core.csv`, el dataset de entrenamiento reconstruido) |

**Limitación conocida:** en la versión B las *fechas* son point-in-time, pero los *valores* son los vigentes en
Bloomberg (con revisiones posteriores); Bloomberg no entrega de forma confiable la primera publicación.
