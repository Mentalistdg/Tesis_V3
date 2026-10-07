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
5. **Emite la señal** CASH / SPY (1x) / UPRO (3x) para el **día hábil siguiente** al último dato y la muestra en la app.

> La predicción del día *t* usa datos hasta *t−1*: la señal se calcula en la noche y se ejecuta con una
> **orden MOC al cierre del día indicado** (antes de 15:50 NY), igual que en el backtest de la tesis.
> No hay ejecución automática de órdenes: la decisión de tomar posición es del usuario.

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

## Cómo leer las señales

- **Versión B (operativa):** los datos macro entran en su fecha real de publicación (point-in-time).
- **Versión A (diagnóstico):** los datos macro entran en la fecha del período, como en el entrenamiento
  (eso implica conocerlos antes de su publicación). La diferencia A–B mide ese sesgo.
- **Estado de datos:** verde = completo; amarillo = advertencias (p.ej. CVIX sin dato en feriados del Reino
  Unido, revisiones de utilidades del SPX); **rojo = no se emitió señal** (ver `logs\pipeline.log`).
- `logs/senales_emitidas.csv`: registro solo-agregar de cada señal mostrada (con hora y estado).
- `logs/reporte_spx.txt`: desempeño del período en vivo (entradas, retorno vs SPY, escenario con 1 día de retraso).

## Mantención

- **Una serie cambia de ticker o se discontinúa:** corregir la fila en `activos/spx/series.csv` (no el código).
  `US0003M` (LIBOR) y `USSWAP10` están discontinuadas desde 2023 y ya eran NaN en el entrenamiento.
- **Pruebas:** `%CRONOS_PY% -m pytest -m "not bloomberg"` (sin Terminal) y `-m bloomberg` (con Terminal).
  La prueba de regresión contra la tesis corre sola en el pipeline cuando cambia el código o el entorno.
- **pandas 2.3.3 es obligatorio:** con otra versión las features cambian en silencio; el pipeline se niega a correr.
- Agregar un activo = nueva carpeta `activos/<id>/` con sus series, modelo y configuración.

## Estructura

| Ruta | Contenido |
|---|---|
| `pipeline/` | Motor: `bloomberg`, `extraer`, `empalmar`, `validar`, `features`, `senal`, `exportar`, `cronos` |
| `pipeline/tesis/` | `build_dataset.py` de la tesis + parche documentado |
| `activos/spx/` | Series verificadas, configuración, modelo entrenado e historia congelada |
| `app/` | App CRONOS (FastAPI + React) con la página "Señales" |
| `data/spx/` | Datos generados (no versionados): raw extendido, datasets A/B, descargas diarias de Bloomberg |
| `research/` | Identificación y verificación de las series en Bloomberg |
| `docs/superpowers/` | Spec y plan de implementación |
| `produccion_lstm.py` | Script de verificación de la tesis (usa `data/bloomberg_triple_screen_core.csv`, el dataset de entrenamiento reconstruido) |

**Limitación conocida:** en la versión B las *fechas* son point-in-time, pero los *valores* son los vigentes en
Bloomberg (con revisiones posteriores); Bloomberg no entrega de forma confiable la primera publicación.
