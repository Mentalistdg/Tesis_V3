# Pipeline de producción CRONOS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extraer diariamente desde la Terminal Bloomberg las 97 series del modelo, extender el dataset de forma idéntica al entrenamiento, emitir la señal CASH/SPY/UPRO (versión B operativa y A diagnóstico) y mostrarla en la app CRONOS local, con ejecución programada en este PC.

**Architecture:** Paquete `pipeline/` con etapas puras y testeables (extraer → empalmar → validar → features → señal → exportar) orquestadas por `pipeline/cronos.py`; configuración por activo en `activos/spx/`; `build_dataset.py` de la tesis copiado con un parche mínimo; la app CRONOS (FastAPI + React) copiada desde Tesis_V3 con una página "Señales".

**Tech Stack:** Python 3.12, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.7.2, torch 2.6.0 (CPU), TA-Lib 0.6.8, blpapi, PyYAML, pytest, FastAPI/uvicorn/httpx; React 18 + TypeScript + Vite (Node portable).

**Spec:** `docs/superpowers/specs/2026-10-07-pipeline-produccion-design.md` (leer junto con este plan). Evidencia de las series: `research/identificacion_series/`.

## Global Constraints

- Raíz del repo: `E:\CRONOS_PRODUCCION` hasta la Task 12; desde ahí, clon en `C:\Users\itau_lab\CRONOS_PRODUCCION`.
- Mientras el repo esté en el USB (FAT32), todo comando git va como `git -c safe.directory=E:/CRONOS_PRODUCCION ...`; no modificar la configuración global de git (cuenta compartida).
- Intérprete: `C:\Users\itau_lab\cronos-python\cpython-3.12-windows-x86_64-none\python.exe` (en adelante `$PY`). Tests: `$PY -m pytest`.
- Versiones exactas: pandas==2.3.3, numpy==2.3.5, scikit-learn==1.7.2, torch==2.6.0 (CPU), TA-Lib==0.6.8, joblib==1.5.3, scipy==1.17.1, blpapi (repo pip de Bloomberg), PyYAML, pytest, fastapi, uvicorn, httpx.
- Ancla de empalme inicial: **2025-12-12** (última fila de la historia congelada). Fin de entrenamiento: **2020-10-06** (5.205 filas del dataset de entrenamiento).
- La historia congelada (`activos/spx/historia_congelada.csv`) es de solo lectura; filas ya escritas en `raw_extendido.csv` no se reescriben.
- Señal operativa = versión **B**; versión A = diagnóstico.
- Meta-KNN siempre entrenado con las predicciones de entrenamiento del dataset **A** (artefacto fijo), también al evaluar B.
- Nombres de módulos, funciones y mensajes en español (como en la spec).
- No hacer `git push` ni crear el repo de GitHub sin aprobación explícita del usuario.
- Tolerancias: calce exacto = `np.isclose(rtol=1e-4, atol=1e-6)`; predicciones idénticas = diff máx < 1e-6.

## Review Focus

1. **Corrida antes del cierre de EE.UU.** (alguien ejecuta `actualizar` a mediodía): el día en curso no debe usarse con precios intradía; el último día procesable es hoy solo si la hora de Nueva York ≥ 16:15. → test en Task 4.
2. **Terminal abierta sin sesión / Bloomberg responde vacío:** SPY sin datos debe dar estado rojo, sin señal y sin tocar archivos. → tests en Task 6 y Task 10.
3. **Se ejecuta con pandas ≠ 2.3.3:** las features derivan en silencio (caso real: el CSV distribuido). `features.py` debe negarse a correr. → test en Task 7.
4. **Split de un ETF en un día nuevo (incluido IWF, cuyo ajuste histórico es `none`):** el precio encadenado no debe saltar. → test en Task 5.
5. **Proceso interrumpido a mitad de escritura** (PC se apaga): los CSV previos quedan intactos y la siguiente corrida se pone al día. → test en Task 10.

---

### Task 1: Reestructurar el paquete, fijar el entorno de la tesis y reemplazar el dataset

**Files:**
- Create: `activos/spx/modelo/` (mover `models/LSTM_Attention.pt`, `models/preprocessors.joblib`, `reference/resultados_esperados.json`)
- Create: `activos/spx/historia_congelada.csv` (copia de `C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv`)
- Create: `activos/spx/columnas_dataset.json` (lista ordenada de las 546 columnas del dataset de entrenamiento)
- Modify: `data/bloomberg_triple_screen_core.csv` → reemplazar por el reconstruido con pandas 2.3.3 (hoy en el scratchpad de la sesión: `thesis_copy/data/bloomberg_triple_screen_core.csv`; si no existe, regenerarlo corriendo `C:\Users\itau_lab\Tesis_V3\scripts\build_dataset.py` sobre una copia con el entorno de la tesis)
- Modify: `produccion_lstm.py:55-58` (rutas a `activos/spx/modelo/`)
- Modify: `requirements.txt`, `run.bat`
- Create: `pytest.ini`, `tests/test_regresion_tesis.py`

**Interfaces:**
- Produces: estructura `activos/spx/{modelo/,historia_congelada.csv,columnas_dataset.json}`; `$PY` con las versiones de Global Constraints; marcadores pytest `lento` y `bloomberg`.

- [ ] **Step 1: Escribir el test de regresión**

```python
# tests/test_regresion_tesis.py
import subprocess, sys, pytest

@pytest.mark.lento
def test_produccion_reproduce_tesis():
    out = subprocess.run([sys.executable, "-W", "ignore", "produccion_lstm.py"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    assert "Predicciones identicas a las de la tesis" in out
    assert "Reproduccion EXACTA de los resultados publicados" in out
    assert "+395.4%" in out
```

`pytest.ini`: `markers = lento: tarda minutos | bloomberg: requiere la Terminal`; `testpaths = tests`.

- [ ] **Step 2: Correr y verificar que falla**

Run: `$PY -m pytest tests/test_regresion_tesis.py -v`
Expected: FAIL (hoy entrega +364.2% y "[ALERTA] Predicciones divergen").

- [ ] **Step 3: Reestructurar, fijar versiones y reemplazar el CSV**

`requirements.txt` con las versiones de Global Constraints (torch desde `https://download.pytorch.org/whl/cpu`, blpapi desde `https://blpapi.bloomberg.com/repository/releases/python/simple/`). Instalar con `tools\uv.exe pip install --python $PY --break-system-packages -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --extra-index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ --index-strategy unsafe-best-match`. `run.bat`: descargar `tools\uv.exe` desde `https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip` si falta; misma instalación. `columnas_dataset.json` = encabezado del CSV reconstruido (546 nombres, en orden). Borrar `models/` y `reference/` vacíos.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `$PY -m pytest tests/test_regresion_tesis.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A activos data produccion_lstm.py requirements.txt run.bat pytest.ini tests
git commit -m "feat: estructura por activo, entorno de la tesis y dataset reconstruido (+395.4%)"
```

---

### Task 2: `series.csv` y su cargador

**Files:**
- Create: `activos/spx/series.csv`, `activos/spx/config.yaml`
- Create: `pipeline/__init__.py`, `pipeline/configuracion.py`
- Test: `tests/test_configuracion.py`

**Interfaces:**
- Produces:
  - `cargar_series(dir_activo: Path) -> pd.DataFrame` con columnas `columna_cruda, codigo, ticker, campo, ajuste, tipo, relleno, rezago_pub_dias, notas`, en el orden del encabezado de `historia_congelada.csv`.
  - `cargar_config(dir_activo: Path) -> dict`.
  - Constantes `TIPOS = {"precio","volumen","nivel","fundamental","macro"}`, `AJUSTES = {"none","split","default"}`.

- [ ] **Step 1: Tests**

```python
def test_series_cubre_todas_las_columnas_crudas():
    s = cargar_series(SPX)
    cols = list(pd.read_csv(SPX / "historia_congelada.csv", nrows=0).columns[1:])
    assert list(s.columna_cruda) == cols and len(s) == 97

def test_series_valores_validos():
    s = cargar_series(SPX)
    assert set(s.tipo) <= TIPOS and set(s.ajuste) <= AJUSTES
    assert (s.loc[s.tipo.isin(["macro", "fundamental"]), "relleno"] == "ffill").all()
    assert (s.loc[~s.tipo.isin(["macro", "fundamental"]), "relleno"] == "no").all()

def test_series_correcciones_finales():
    s = cargar_series(SPX).set_index("columna_cruda")
    assert s.loc["CONSSENT Index", "ticker"] == "CPMINDX Index"
    assert s.loc["PITLCHNG Index", "ticker"] == "DGNOXTCH Index"
    assert s.loc["XLE US Equity", ("ticker", "ajuste")].tolist() == ["IWF US Equity", "none"]
    assert s.loc["EFA US Equity", ("ticker", "ajuste")].tolist() == ["XLK US Equity", "split"]

def test_cargar_series_rechaza_tipo_invalido(tmp_path): ...  # ValueError con el nombre de la columna
```

- [ ] **Step 2: Run** `$PY -m pytest tests/test_configuracion.py -v` → FAIL (módulo inexistente).

- [ ] **Step 3: Generar `series.csv` e implementar el cargador**

Generar desde `research/identificacion_series/scripts/mapping.py` (ya con E3/E19 corregidos). Reglas: `codigo` desde `COLUMN_MAPPING` de `build_dataset.py`; ETFs `precio` con `ajuste=split` salvo `IWF US Equity`=`none`; SPY OHLC `precio`/`none`; SPY volumen `volumen`; SPX fundamentales `fundamental`; macro `macro`; resto `nivel`. `rezago_pub_dias`: `GDP CQOQ Index`=30, `LEI TOTL Index`=20, vacío en el resto. `config.yaml`:

```yaml
activo: spx
instrumentos: {UPRO: {expense_ratio: 0.0091, bid_ask: 0.0005}, SPY: {expense_ratio: 0.0009, bid_ask: 0.0002}, CASH: {expense_ratio: 0.0, bid_ask: 0.0}}
ancla_inicial: "2025-12-12"
fin_entrenamiento: "2020-10-06"
solapamiento_dias_habiles: 30
umbral_salto_fundamental: 0.01
calendario: {ticker: "SPY US Equity", campo: PX_LAST}
```

`cargar_series` valida enums, unicidad de `columna_cruda` y orden contra la historia; error = `ValueError` con detalle.

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: series.csv verificado y cargador de configuracion`

---

### Task 3: Cliente Bloomberg

**Files:**
- Create: `pipeline/bloomberg.py`
- Test: `tests/test_bloomberg.py`

**Interfaces:**
- Produces:
  - `class ErrorBloomberg(Exception)`.
  - `class ClienteBloomberg(host="localhost", puerto=8194, lote=25, reintentos=3, espera_s=5)`.
  - `ClienteBloomberg.historico(tickers: list[str], campo: str, inicio: date, fin: date, ajuste: str) -> tuple[dict[str, pd.Series], dict[str, str]]` (series por ticker con índice `Timestamp`; errores por ticker). `ajuste`: `none` → normal/abnormal/split = False; `split` → solo split = True; `default` → sin overrides.
  - `ClienteBloomberg._pedir_lote(tickers, campo, inicio, fin, ajuste) -> tuple[dict, dict]` (única función que toca blpapi; los tests la sustituyen).
  - `ECO_RELEASE_DT` se devuelve como `Timestamp` (Bloomberg lo entrega como número `AAAAMMDD.0`).

- [ ] **Step 1: Tests (sin Terminal)**

```python
def test_divide_en_lotes_de_25(): ...           # 60 tickers → 3 llamadas a _pedir_lote (25, 25, 10)
def test_reintenta_y_luego_falla():             # _pedir_lote lanza 3 veces → ErrorBloomberg
    ...
def test_errores_por_ticker_se_acumulan(): ...  # securityError en 1 ticker no aborta los demás
def test_eco_release_dt_se_parsea():            # 20250128.0 → Timestamp("2025-01-28")
    ...

@pytest.mark.bloomberg
def test_integracion_spy_cierre():
    s, err = ClienteBloomberg().historico(["SPY US Equity"], "PX_LAST", date(2025,12,8), date(2025,12,12), "none")
    assert not err and s["SPY US Equity"].loc["2025-12-10"] == pytest.approx(687.57)
```

- [ ] **Step 2: Run** `$PY -m pytest tests/test_bloomberg.py -v -m "not bloomberg"` → FAIL.
- [ ] **Step 3: Implementar** a partir de `research/identificacion_series/scripts/bbg.py` (misma petición `HistoricalDataRequest`, timeout 120 s por evento; sesión que no inicia → `ErrorBloomberg("Terminal Bloomberg no disponible: ...")`).
- [ ] **Step 4: Run** `-m "not bloomberg"` → PASS; luego `-m bloomberg` con la Terminal abierta → PASS.
- [ ] **Step 5: Commit** `feat: cliente Bloomberg con lotes, reintentos y errores por ticker`

---

### Task 4: Extracción

**Files:**
- Create: `pipeline/extraer.py`
- Test: `tests/test_extraer.py`, `tests/fakes.py` (cliente falso reutilizado por Tasks 4–10)

**Interfaces:**
- Consumes: `cargar_series`, `ClienteBloomberg.historico`.
- Produces:
  - `@dataclass Descarga: calendario: pd.DatetimeIndex; valores: dict[str, pd.Series]` (clave = `columna_cruda`, fechas Bloomberg sin unir); `publicaciones: pd.DataFrame` (`columna_cruda, periodo, publicacion, valor`); `errores: dict[str, str]`.
  - `ultimo_dia_procesable(ahora: datetime) -> date`: hoy si la hora de `America/New_York` ≥ 16:15 y es día hábil, si no el día hábil anterior.
  - `extraer(series: pd.DataFrame, inicio: date, fin: date, cliente) -> Descarga`. Precios (`tipo=precio`) se descargan **siempre con ajuste `split`** (base irrelevante al encadenar; evita saltos por splits nuevos, incluido IWF); el resto con su `ajuste`. Macro: también `ECO_RELEASE_DT`.
  - `extraer_publicaciones_completas(series, cliente) -> pd.DataFrame` (historia macro completa desde 1999-12-01 con `PX_LAST` + `ECO_RELEASE_DT`; insumo de la versión B).
  - `guardar_descarga(d: Descarga, carpeta: Path) -> None` (un CSV por columna cruda + `errores.json` + `publicaciones.csv`).
- `tests/fakes.py`: `ClienteFalso(historia: pd.DataFrame, series: pd.DataFrame, rezago_pub=5, vacio=set(), splits={})` que responde con los valores de la historia cruda para el ticker de cada columna (macro: valores diarios de la historia; `ECO_RELEASE_DT` = fecha + `rezago_pub` días; `splits={ticker: (fecha, factor)}` divide los precios anteriores a la fecha; `vacio` = tickers que responden sin datos).

- [ ] **Step 1: Tests**

```python
def test_ultimo_dia_antes_del_cierre_es_ayer():
    assert ultimo_dia_procesable(datetime(2026,10,7,15,0, tzinfo=NY)) == date(2026,10,6)
def test_ultimo_dia_despues_del_cierre_es_hoy():
    assert ultimo_dia_procesable(datetime(2026,10,7,16,20, tzinfo=NY)) == date(2026,10,7)
def test_ultimo_dia_fin_de_semana_es_viernes():
    assert ultimo_dia_procesable(datetime(2026,10,10,18,0, tzinfo=NY)) == date(2026,10,9)
def test_extraer_calendario_es_spy(): ...       # calendario == fechas con SPY PX_LAST
def test_precios_se_piden_con_split(): ...      # el ClienteFalso registra ajuste="split" para tipo precio
def test_serie_vacia_queda_en_errores(): ...    # ticker en `vacio` → errores[columna] no vacío
```

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run (PASS).
- [ ] **Step 5: Commit** `feat: extraccion incremental con corte de cierre NY y publicaciones macro`

---

### Task 5: Empalme (versión A) y versión B

**Files:**
- Create: `pipeline/empalmar.py`
- Test: `tests/test_empalmar.py`

**Interfaces:**
- Consumes: `Descarga`, `cargar_series`.
- Produces:
  - `unir_calendario(serie: pd.Series, calendario: pd.DatetimeIndex, relleno: bool) -> pd.Series` — unión por **fecha exacta**; `relleno=True` → forward-fill después de unir (observaciones en fines de semana se pierden).
  - `encadenar(valor_ancla: float, serie_bbg: pd.Series, ancla: pd.Timestamp) -> pd.Series` — `v[t] = valor_ancla * bbg[t] / bbg[ancla]` para `t > ancla`.
  - `empalmar(raw_previo: pd.DataFrame, d: Descarga, series: pd.DataFrame) -> pd.DataFrame` — devuelve raw (formato de 98 columnas, `date` + columnas crudas) con filas nuevas posteriores a la última fecha de `raw_previo`; ancla = última fecha de `raw_previo`. Reglas por tipo según spec 6.1; filas existentes intactas.
  - `version_b(raw_a: pd.DataFrame, publicaciones: pd.DataFrame, series: pd.DataFrame, ancla_congelada: pd.Timestamp) -> pd.DataFrame` — solo columnas `tipo=macro`: cada observación se ubica en `publicacion` (o `periodo + rezago_pub_dias` si `rezago_pub_dias` no está vacío o `publicacion` falta); valor = el de `raw_a` en la fecha de período si existe y `periodo <= ancla_congelada`, si no el de Bloomberg; luego `unir_calendario(..., relleno=True)` sobre toda la historia.

- [ ] **Step 1: Tests**

```python
def test_union_exacta_pierde_fin_de_semana():
    s = pd.Series([35.7, 52.0], index=pd.to_datetime(["2020-02-29", "2020-03-31"]))
    cal = pd.to_datetime(["2020-02-28", "2020-03-02", "2020-03-31"])
    r = unir_calendario(s, cal, relleno=True)
    assert 35.7 not in r.values and np.isnan(r.iloc[0]) and r.iloc[2] == 52.0
def test_union_sin_relleno_deja_nan(): ...
def test_encadenar_sobrevive_split():
    bbg = pd.Series([100., 101., 50.5], index=fechas)  # split 2:1 ya ajustado → bbg en base nueva
    assert encadenar(200., bbg, fechas[0]).tolist() == [202., 101.]
def test_empalme_no_reescribe_filas_previas(): ...
def test_empalme_split_nuevo_en_iwf_sin_salto():   # ClienteFalso(splits={"IWF US Equity": (dia_nuevo, 4)})
    ...                                            # retorno encadenado del día del split == retorno real
def test_empalme_nivel_directo_y_macro_ffill(): ...
def test_version_b_desplaza_a_publicacion(): ...   # valor aparece en la fecha de publicación, no antes
def test_version_b_rezago_fijo_pib(): ...          # GDP CQOQ: periodo + 30 días
def test_version_b_recupera_observacion_de_fin_de_semana(): ...  # PMI China 2020-02-29 publicada en día hábil → aparece en B
```

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run (PASS).
- [ ] **Step 5: Commit** `feat: empalme encadenado, union por fecha exacta y version B point-in-time`

---

### Task 6: Validación

**Files:**
- Create: `pipeline/validar.py`
- Test: `tests/test_validar.py`

**Interfaces:**
- Consumes: `Descarga`, `cargar_series`, `cargar_config`.
- Produces:
  - `@dataclass Resultado: estado: str` (`"verde"|"amarillo"|"rojo"`)`; mensajes: list[str]`.
  - `validar(raw_previo: pd.DataFrame, raw_nuevo: pd.DataFrame, d: Descarga, series: pd.DataFrame, config: dict) -> Resultado`.
- Reglas (spec 8): **rojo** — SPY OHLCV ausente en algún día nuevo; ticker con error o sin datos en toda la ventana (se nombra la columna); serie `nivel` que no calza exacto con `raw_previo` en el solapamiento; precio cuyos retornos diarios en el solapamiento difieren de los almacenados (rtol 1e-6). **amarillo** — serie de mercado sin dato en un día nuevo puntual (queda NaN); salto de `fundamental` > `umbral_salto_fundamental` en el solapamiento; split detectado (razón de precios no constante en Bloomberg); split de SPY en volumen. **verde** — resto. `US0003M Index` se excluye de "sin datos".

- [ ] **Step 1: Tests** — uno por regla, con el `ClienteFalso`:

```python
def test_spy_vacio_es_rojo(): ...                 # ClienteFalso(vacio={"SPY US Equity"}) → estado "rojo", mensaje menciona SPY
def test_nivel_que_no_calza_es_rojo(): ...
def test_ticker_con_error_es_rojo_y_nombra_columna(): ...
def test_dato_puntual_faltante_es_amarillo(): ...
def test_salto_fundamental_es_amarillo(): ...
def test_libor_sin_datos_no_alerta(): ...
def test_todo_bien_es_verde(): ...
```

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run (PASS).
- [ ] **Step 5: Commit** `feat: validacion con niveles verde/amarillo/rojo`

---

### Task 7: Features con `build_dataset.py` parchado

**Files:**
- Create: `pipeline/tesis/build_dataset.py` (copia de `C:\Users\itau_lab\Tesis_V3\scripts\build_dataset.py`), `pipeline/tesis/PARCHE.md`, `pipeline/tesis/__init__.py`
- Create: `pipeline/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - Parche en `build_dataset()` → `build_dataset(df_raw: pd.DataFrame | None = None, columnas_fijas: list[str] | None = None, conservar_sin_target: bool = False, guardar: bool = True) -> pd.DataFrame`. Con `df_raw=None` y defaults el comportamiento es el original. `df_raw` reemplaza la lectura de archivo (pasa por `load_raw_data` sin leer disco: extraer el cuerpo posterior a `read_csv` a `preparar_raw(df)`). `columnas_fijas` reemplaza los pasos 13.1 y 13.3 por `df = df[columnas_fijas_disponibles]` y lanza `KeyError` si falta alguna; el filtro de filas 13.4 se calcula sobre esas mismas columnas. `conservar_sin_target=True` omite el paso 13.5. `guardar=False` no escribe archivos. Nada más cambia; `PARCHE.md` contiene el diff.
  - `construir_features(raw: pd.DataFrame, columnas: list[str]) -> pd.DataFrame` — exige `pandas.__version__ == "2.3.3"` (si no, `RuntimeError("build_dataset requiere pandas 2.3.3 ...")`), llama a `build_dataset(raw, columnas, conservar_sin_target=True, guardar=False)` y devuelve las 546 columnas en el orden de `columnas`.

- [ ] **Step 1: Tests**

```python
@pytest.mark.lento
def test_reproduce_dataset_de_entrenamiento():
    raw = pd.read_csv(SPX / "historia_congelada.csv")
    df = construir_features(raw, COLUMNAS)
    ref = pd.read_csv("data/bloomberg_triple_screen_core.csv")
    assert len(df) == 6508 and df.iloc[-1]["date"] == "2025-12-12"
    assert np.isnan(df.iloc[-1]["market_forward_excess_returns"])
    pd.testing.assert_frame_equal(df.iloc[:6507].reset_index(drop=True), ref, check_exact=False, rtol=1e-12)

def test_falta_columna_lanza_error(): ...          # raw sin "VIX Index" → KeyError
def test_rechaza_pandas_distinto(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "3.0.1")
    with pytest.raises(RuntimeError, match="pandas 2.3.3"):
        construir_features(pd.DataFrame(), [])
```

- [ ] **Step 2–4:** Run (FAIL) → copiar + parchar + implementar → Run (PASS).
- [ ] **Step 5: Commit** `feat: features con build_dataset de la tesis parchado (columnas fijas, conserva ultima fila)`

---

### Task 8: Señal

**Files:**
- Create: `pipeline/modelo.py` (copia literal desde `produccion_lstm.py` de: `LuongAttention`, `LSTMAttention`, `prepare_sequences`, `predict_in_batches`, `compute_rolling_percentiles`, `compute_rolling_std`, `evaluate_combo`, `compute_meta_features`, `build_meta_dataset`, `train_meta_knn`, `meta_knn_filtered_backtest`, `calculate_returns_with_costs`, `compute_metrics` y sus constantes)
- Modify: `produccion_lstm.py` (importar esas piezas desde `pipeline.modelo`; sin cambio de comportamiento)
- Create: `pipeline/senal.py`
- Test: `tests/test_senal.py`

**Interfaces:**
- Consumes: datasets de `construir_features`; `activos/spx/modelo/`; `config["fin_entrenamiento"]`.
- Produces:
  - `@dataclass ModeloCargado: red, feature_cols, imputer, scaler, y_scaler` y `cargar_modelo(dir_modelo: Path, dataset_a: pd.DataFrame, fin_entrenamiento: str) -> ModeloCargado` (`y_scaler` ajustado con el target de entrenamiento; columnas vía `LEGACY_MAP`).
  - `predecir(m: ModeloCargado, dataset: pd.DataFrame) -> pd.Series` (predicción por fecha para filas ≥ 31.ª, lookback 30).
  - `entrenar_meta(m: ModeloCargado, dataset_a: pd.DataFrame, fin_entrenamiento: str) -> dict` (Meta-KNN con predicciones de entrenamiento del dataset A).
  - `generar_senales(m, meta, dataset: pd.DataFrame, fin_entrenamiento: str) -> pd.DataFrame` con columnas `date, prediccion, percentil, q_ext, q_mod, posicion, senal_valida, forward_returns, risk_free_rate` para toda fecha > fin de entrenamiento (incluida la última sin target). El corte entrenamiento/prueba es **por fecha**, no 80/20.
  - `metricas(senales: pd.DataFrame, config: dict, desde: str | None = None) -> dict` (retorno, Sharpe, drawdown, % días por instrumento, n.º trades y n.º de **entradas** = transiciones de CASH a SPY/UPRO; usa solo filas con target).

- [ ] **Step 1: Tests**

```python
@pytest.mark.lento
def test_senales_reproducen_backtest_tesis():
    s = generar_senales(M, META, DATASET_ENTRENAMIENTO_CON_ULTIMA_FILA, "2020-10-06")
    mt = metricas(s[s.date <= "2025-12-10"], CONFIG)
    assert mt["total_return"] == pytest.approx(3.9538, abs=1e-4)
    assert mt["sharpe"] == pytest.approx(1.3192, abs=1e-4)
def test_ultima_fila_sin_target_tiene_senal(): ...   # fila 2025-12-12 presente con posicion en {0,1,3}
def test_corte_por_fecha_no_por_porcentaje(): ...    # agregar 300 filas al final no cambia predicciones previas
```

Además `tests/test_regresion_tesis.py` (Task 1) debe seguir pasando tras modificar `produccion_lstm.py`.

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run `tests/test_senal.py tests/test_regresion_tesis.py` (PASS).
- [ ] **Step 5: Commit** `feat: generacion de senales con corte por fecha y Meta-KNN fijo`

---

### Task 9: Exportación (log de señales, reporte y JSON de la app)

**Files:**
- Create: `pipeline/exportar.py`, `pipeline/io.py`
- Test: `tests/test_exportar.py`

**Interfaces:**
- Consumes: salidas de `generar_senales`, `metricas`, `Resultado`.
- Produces:
  - `escribir_atomico(path: Path, contenido: bytes | str) -> None` y `escribir_csv_atomico(df: pd.DataFrame, path: Path) -> None` (archivo temporal en la misma carpeta + `os.replace`).
  - `registrar_senales(log: Path, activo: str, senales_b: pd.DataFrame, senales_a: pd.DataFrame, resultado: Resultado) -> None` — upsert por (`fecha`, `activo`) en `logs/senales.csv` con columnas `fecha, activo, senal_b, senal_a, prediccion_b, prediccion_a, percentil_b, umbral_upro, umbral_spy, cambio_vs_ayer, estado, advertencias`; las señales se escriben como `CASH|SPY|UPRO`.
  - `reporte_periodo(senales_a, senales_b, config, desde="2025-12-12") -> dict` y `reporte_texto(rep: dict) -> str` (entradas, días por instrumento, retorno con costos vs SPY buy&hold, para A y B).
  - `exportar_app(dir_datos_app: Path, activo: str, senales_b, senales_a, resultado, reporte) -> None` — escribe `senales.json` (`{activos: [{activo, fecha, senal_b, senal_a, cambio_vs_ayer, prediccion, percentil, umbrales, estado, advertencias}]}`) y `vivo.json` (una entrada por versión, `"LSTM_Attention_vivo_B"` y `"LSTM_Attention_vivo_A"`, con el **mismo esquema por modelo** que `daily_data.json`/`models_summary.json`; reutilizar `extract_trades` y el armado de resumen de `C:\Users\itau_lab\Tesis_V3\paper\update_backend_data.py`, copiándolos a `pipeline/exportar.py`).

- [ ] **Step 1: Tests**

```python
def test_registrar_es_idempotente(tmp_path): ...       # registrar dos veces el mismo día → 1 fila
def test_cambio_vs_ayer(): ...                          # CASH→SPY marca cambio_vs_ayer=True
def test_entradas_cuenta_transiciones_desde_cash(): ... # [0,1,1,0,3,3,0] → 2 entradas
def test_escritura_atomica_no_deja_parcial(tmp_path, monkeypatch):
    # os.replace lanza OSError → el archivo original queda intacto y no quedan .tmp
    ...
def test_vivo_json_tiene_esquema_de_daily_data(): ...   # mismas claves que una entrada de daily_data.json
```

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run (PASS).
- [ ] **Step 5: Commit** `feat: log de senales, reporte del periodo y exportacion para la app`

---

### Task 10: Orquestador `cronos actualizar`

**Files:**
- Create: `pipeline/cronos.py`, `actualizar.bat`
- Test: `tests/test_cronos.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces:
  - CLI: `$PY -m pipeline.cronos actualizar [--activo spx] [--ensayo] [--hasta AAAA-MM-DD]`. Códigos de salida: 0 = verde/amarillo o sin día nuevo; 2 = rojo; 1 = error inesperado.
  - `actualizar(dir_raiz: Path, activo: str, cliente, ahora: datetime, ensayo: bool) -> Resultado`. Flujo: cargar config → `raw_previo` = `data/<activo>/raw_extendido.csv` o, si no existe, `historia_congelada.csv` → fin = `ultimo_dia_procesable(ahora)` (o `--hasta`) → si fin ≤ última fecha: salir verde "sin día nuevo" → `extraer` desde (última fecha − `solapamiento_dias_habiles`) → `guardar_descarga` → `empalmar` → `validar` (rojo: registrar en `logs/pipeline.log` y salir sin escribir nada más) → publicaciones macro (`data/<activo>/publicaciones_macro.csv`; completa la primera vez con `extraer_publicaciones_completas`, después incremental) → `version_b` → `construir_features` A y B → `generar_senales` A y B → escribir atómicamente `raw_extendido.csv`, `dataset_A.csv`, `dataset_B.csv` → `registrar_senales` → `reporte_periodo` (`logs/reporte_<activo>.txt` y `.json`) → `exportar_app(app/backend/data, ...)`. `--ensayo`: todo en memoria, nada en disco.
  - `actualizar.bat`: `cd /d %~dp0`, usa `$PY` (ruta de Global Constraints; si no existe, misma instalación que `run.bat`), corre `actualizar`, agrega salida a `logs/pipeline.log`.

- [ ] **Step 1: Tests (extremo a extremo con `ClienteFalso`)**

```python
@pytest.mark.lento
def test_e2e_reproduce_historia(tmp_path):
    # raíz temporal con historia_congelada truncada en 2025-11-28; ClienteFalso sirve la historia completa
    r = actualizar(raiz, "spx", ClienteFalso(HIST, SERIES), ahora=datetime(2025,12,12,17,0,tzinfo=NY), ensayo=False)
    raw = pd.read_csv(raiz / "data/spx/raw_extendido.csv")
    pd.testing.assert_frame_equal(raw, HIST_HASTA_2025_12_12, check_exact=False, rtol=1e-9)
    ds = pd.read_csv(raiz / "data/spx/dataset_A.csv")
    pd.testing.assert_frame_equal(ds.iloc[:-1], DATASET_ENTRENAMIENTO, check_exact=False, rtol=1e-9)

def test_sin_dia_nuevo_no_escribe(tmp_path): ...
def test_rojo_no_modifica_archivos(tmp_path): ...       # ClienteFalso(vacio={"SPY US Equity"}) → exit 2, mtime/contenido iguales
def test_corte_por_hora_ny(tmp_path): ...                 # ahora 15:00 NY → no procesa el día en curso
def test_interrupcion_en_escritura_deja_previos(tmp_path, monkeypatch): ...  # falla escribir dataset_B → raw_extendido previo intacto; la corrida siguiente completa
def test_dos_corridas_mismo_dia_idempotentes(tmp_path): ...
def test_ensayo_no_escribe(tmp_path): ...
```

Para la escritura multi-archivo: escribir todos los `.tmp` primero y reemplazar al final en orden `dataset_A, dataset_B, raw_extendido` (el `raw_extendido` último marca la corrida como completa).

- [ ] **Step 2–4:** Run (FAIL) → implementar → Run (PASS); además toda la suite `-m "not bloomberg"`.
- [ ] **Step 5: Commit** `feat: orquestador cronos actualizar con escritura atomica e idempotencia`

---

### Task 11: Primera corrida real — período 12-dic-2025 → hoy

**Files:**
- Create (generados): `data/spx/raw_extendido.csv`, `data/spx/dataset_A.csv`, `data/spx/dataset_B.csv`, `data/spx/publicaciones_macro.csv`, `logs/senales.csv`, `logs/reporte_spx.txt`, `logs/reporte_spx.json`
- Test: `tests/test_integracion_bloomberg.py`

- [ ] **Step 1: Test de integración**

```python
@pytest.mark.bloomberg
def test_97_series_y_solapamiento_exacto():
    # extrae 2025-11-01 → 2025-12-12 y compara con historia_congelada
    # nivel: calce exacto; precio: retornos diarios iguales (rtol 1e-6); errores vacíos (salvo US0003M sin datos)
    ...
```

- [ ] **Step 2: Run** `$PY -m pytest -m bloomberg -v` con la Terminal abierta → PASS (si falla, corregir `series.csv`; no el código).
- [ ] **Step 3: Ensayo** `$PY -m pipeline.cronos actualizar --ensayo` → estado verde o amarillo con advertencias explicadas.
- [ ] **Step 4: Corrida real** `$PY -m pipeline.cronos actualizar` → `logs/reporte_spx.txt` con entradas, días por instrumento y retorno vs SPY para A y B desde 2025-12-12; mostrar el reporte al usuario.
- [ ] **Step 5: Commit** `data: primera extension del dataset y senales 2025-12-12 a hoy`

---

### Task 12: App local con página "Señales"

**Files:**
- Create: `app/` (copiar `C:\Users\itau_lab\Tesis_V3\app\` completo)
- Modify: `app/backend/main.py` (endpoints nuevos; fusionar `vivo.json` en `/api/models` y `/api/models/{name}`; servir `app/frontend/dist` con `StaticFiles` y fallback a `index.html`)
- Create: `app/frontend/src/pages/SenalesPage.tsx`; Modify: `App.tsx`, `components/Layout.tsx`, `types/index.ts`, `services/api.ts`
- Create: `abrir_app.bat`
- Test: `tests/test_app_backend.py`

**Interfaces:**
- Consumes: `senales.json`, `vivo.json` (Task 9).
- Produces: `GET /api/senales` → contenido de `senales.json`; `GET /api/senales/{activo}/historial` → entrada `LSTM_Attention_vivo_B` + `_A` de `vivo.json` filtradas por activo; 404 si no existe el activo.

- [ ] **Step 1: Tests backend**

```python
def test_api_senales(cliente_con_datos): assert r.json()["activos"][0]["activo"] == "spx"
def test_api_historial_404_activo_inexistente(cliente_con_datos): assert r.status_code == 404
def test_models_incluye_vivo(cliente_con_datos): ...  # "LSTM_Attention_vivo_B" en /api/models
def test_sin_senales_json_responde_503(cliente_vacio): ...  # mensaje "aun no hay corridas"
```

- [ ] **Step 2–4:** Run (FAIL) → implementar backend → Run (PASS).
- [ ] **Step 5: Página "Señales"** — tarjeta por activo: señal B grande (`CASH` / `SPY 1x` / `UPRO 3x`), fecha del dato, insignia "CAMBIÓ" si `cambio_vs_ayer`, predicción y percentil vs umbrales, señal A al lado, estado verde/amarillo/rojo con advertencias; debajo, gráfico de equity del período en vivo (B, A y SPY) y tabla de señales diarias. Estilo y componentes existentes (tema "Axe Capital", lightweight-charts). Ruta `/senales` como inicio (redirigir `/` a `/senales`).
- [ ] **Step 6: Compilar** — Node portable (zip oficial `node-v20.x-win-x64.zip` en `C:\Users\itau_lab\cronos-python\node\`), `npm --prefix app/frontend install` y `npm --prefix app/frontend run build` sin errores; `npm --prefix app/frontend run lint` sin errores. `abrir_app.bat`: `$PY -m uvicorn app.backend.main:app --port 8000` y abre `http://localhost:8000/senales`. Agregar `fastapi`, `uvicorn`, `httpx` a `requirements.txt`.
- [ ] **Step 7: Verificación manual** — `abrir_app.bat`, la página muestra la señal de la Task 11; capturar pantalla para el usuario.
- [ ] **Step 8: Commit** `feat: app CRONOS local con pagina Senales`

---

### Task 13: Operación en este PC y repositorio

**Files:**
- Create: `programar_tarea.bat`; Modify: `README.md`

- [ ] **Step 1: `programar_tarea.bat`** — `crear`: `schtasks /Create /TN "CRONOS\actualizar_<HHMM>" /TR "\"<raiz>\actualizar.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST <HH:MM> /F` para 18:30, 19:30 y 21:00 (nombres `actualizar_1830`, `actualizar_1930`, `actualizar_2100`; usuario actual, sin admin); `eliminar`: `schtasks /Delete /TN ... /F` para las tres; `estado`: `schtasks /Query /TN ...`.
- [ ] **Step 2: README** — qué hace el pipeline, requisitos (Terminal abierta y con sesión, sesión de Windows iniciada), comandos (`actualizar.bat`, `abrir_app.bat`, `programar_tarea.bat crear|eliminar|estado`, `run.bat`), cómo leer `logs/senales.csv` y los estados, cómo corregir una serie en `series.csv`, y la limitación de la versión B (fechas point-in-time, valores de la vintage de extracción).
- [ ] **Step 3: Commit** `feat: programacion de la tarea diaria y documentacion de operacion`
- [ ] **Step 4: GitHub (requiere aprobación explícita del usuario)** — el usuario crea el repo **privado** `Mentalistdg/CRONOS_PRODUCCION` (web) o autoriza crearlo; `git remote add origin https://Mentalistdg@github.com/Mentalistdg/CRONOS_PRODUCCION.git` y `git push -u origin main` (el usuario se autentica en el navegador).
- [ ] **Step 5: Clon en C:** `git clone https://Mentalistdg@github.com/Mentalistdg/CRONOS_PRODUCCION.git C:/Users/itau_lab/CRONOS_PRODUCCION`; `tools\uv.exe` se descarga solo; correr `$PY -m pytest -m "not bloomberg and not lento"` en el clon → PASS.
- [ ] **Step 6: Programar desde C:** `C:\Users\itau_lab\CRONOS_PRODUCCION\programar_tarea.bat crear`; `schtasks /Run /TN "CRONOS\actualizar_1830"`; verificar en `logs/pipeline.log` una corrida verde/amarillo o "sin día nuevo".
- [ ] **Step 7: Commit/push** de cualquier ajuste con aprobación del usuario.

---

## Enmiendas v2 (tras revisión previa; prevalecen sobre las tareas de arriba)

Ver spec §15. Cambios por tarea:

- **Global:** `.gitattributes` (`*.csv -text`, `*.bat eol=crlf`, `*.pt binary`, `*.joblib binary`) en Task 1. `.gitignore` agrega `data/*/` salvo nada (los datasets no se versionan; `activos/` sí). blpapi siempre con `adjustmentFollowDPDF=False`. Toda lectura/escritura de texto con `encoding="utf-8"`.
- **Task 1:** el test corre `produccion_lstm.py` con `env={**os.environ, "PYTHONUTF8": "1"}` y `cwd` = raíz. `run.bat` sin reinstalar si ya está instalado (marcador `.deps_ok` con hash de `requirements.txt`); `uv` fijo `0.12.23` (`https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-pc-windows-msvc.zip`).
- **Task 2:** `series.csv` agrega `fecha_pub` (`eco` | `rezago`) y `rezago_pub_dias` completo para toda serie macro (mediana observada de `ECO_RELEASE_DT` − período; `GDP CQOQ Index` = `rezago` 30; `LEI TOTL Index` = `rezago` 20). `ajuste` de precios queda documental (la descarga de `precio` y `fundamental` para encadenar usa `split`/`default`). Columna `publica_tarde` (`si` para las 9 series: LF98OAS, LF98TRUU, LUACTRUU, LUACOAS, MOVE, CVIX, SKEW, PCUSEQTR —incluidas sus columnas duplicadas— ) usada por el modo provisional. `config.yaml` agrega `cola_mutable_dias_habiles: 5`, `ventana_provisional_ny: ["15:30", "15:45"]`, `hora_final_ny: "20:00"`.
- **Task 4:** `ultimo_dia_procesable` se reemplaza por `modo_corrida(ahora) -> tuple[str, date]` con `modo ∈ {"provisional", "final", "nada"}`: provisional si hoy es hábil y la hora NY ∈ ventana; final para el último día hábil cuya hora NY ≥ `hora_final_ny` (hoy si ya pasó, si no ayer); `nada` en otro caso. Test de feriado: día sin barra de SPY → final del día hábil anterior, no rojo. `extraer` pide `ECO_RELEASE_DT` y descarta rezagos fuera de [−15, 120] días.
- **Task 5:** `encadenar` usa el **último valor válido** de la columna en la parte inmutable como ancla. Test corregido: precios Bloomberg en base nueva `[50.0, 50.5, 25.5]` con split 2:1 aplicado retroactivamente (bbg ya ajustado) → la serie encadenada desde un valor histórico 200 en el ancla da `[202.0, 204.0]` (retornos +1% y +0.99…%, sin salto). `tipo=fundamental` también se encadena (luego ffill). `empalmar` reescribe las filas de la cola mutable (últimos `cola_mutable_dias_habiles` días hábiles posteriores al ancla congelada) y conserva las anteriores. La macro **no** se guarda en `raw_extendido.csv` para fechas posteriores al ancla: `macro_a(historia, publicaciones, series, calendario)` y `version_b(...)` la derivan en cada corrida desde `publicaciones_macro.csv` (A: fecha de período + unión exacta + ffill; B: fecha de publicación). Test adicional: observación mensual fechada en sábado no aparece en A y sí en B.
- **Task 6:** el calce exacto `nivel`/retornos de `precio` se exige solo en fechas ≤ ancla congelada; diferencias dentro de la cola mutable = informativo. Feriado ≠ rojo.
- **Task 7:** `columnas_fijas` = las 546 columnas de `columnas_dataset.json`; si falta alguna → `KeyError`; el denominador del filtro de filas usa solo las 536 no-meta (como el original). Tests comparan `str(df.iloc[-1]["date"])[:10] == "2025-12-12"` y convierten `date` a string antes de `assert_frame_equal`. Test de causalidad: construir con `raw[raw.date <= k]` para k ∈ {2015-06-30, 2023-12-29} y comparar exacto las filas ≤ k−1.
- **Task 8:** `entrenar_meta` copia literal el desalineamiento de `produccion_lstm.py:602-605` (`[:n_tr]`) y `meta_knn_filtered_backtest` el de `:346-349` (`[-n_tr:]`). `metricas` acepta `retraso: int = 0` (desplaza posiciones) para los escenarios.
- **Task 9:** dos logs: `logs/senales_emitidas.csv` (solo agregado: `emitida_en, modo, fecha, activo, senal_b, senal_a, prediccion_b, percentil_b, umbral_upro, umbral_spy, cambio_vs_ultima_final, estado, advertencias`) y `data/<activo>/senales_recalculadas.csv` (upsert, interno). `escribir_atomico` reintenta `os.replace` 5 veces cada 2 s ante `PermissionError` y luego lanza. `vivo.json` sigue la convención de fechas de `update_backend_data.py:146-152` (fecha de realización del retorno). Reporte con escenarios ideal / MOC (series `publica_tarde` rezagadas 1 día) / t+1, nota de que usa datos revisados.
- **Task 10:** `actualizar(..., modo)`. Provisional: descarga desde la última fila inmutable hasta hoy (intradía), series `publica_tarde` con su último valor, construye features en memoria, emite señal provisional (log emitidas + `senales.json`), **no** escribe raw/datasets. Final: flujo completo con cola mutable. Lock `logs/cronos.lock` (PID + timestamp; vencido si > 2 h o PID inexistente). Regresión cacheada en `logs/regresion.json` por hash (`pipeline/**/*.py`, `activos/spx/**`, versiones de pandas/numpy/torch/sklearn/talib); si el hash cambió, corre la regresión antes de emitir y, si falla → rojo. `ClienteFalso` entrega la macro como observaciones mensuales en fecha de período (incluidas fechas de fin de semana) con `ECO_RELEASE_DT`.
- **Task 12:** `main.py` abre JSON con `encoding="utf-8"`. La página muestra la señal **provisional del día** (si existe, con hora) y la **final** más reciente, con aviso "para orden MOC antes de 15:50 NY".
- **Task 13:** una sola tarea `CRONOS\actualizar` cada 15 min lun–vie 09:00–23:45 (`/SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:00 /RI 15 /DU 14:45`), ejecutando `wscript.exe //B "<raiz>\actualizar_oculto.vbs"` (lanza `actualizar.bat` sin ventana). Verificar que la creación funciona sin admin; si la carpeta `\CRONOS` no se permite, usar nombre `CRONOS_actualizar`.
