# Parche CRONOS a `build_dataset.py`

`build_dataset.py` es copia de `Tesis_V3/scripts/build_dataset.py` (commit 3809a19). La lógica de
features **no cambia**. El parche (8 cambios, marcados `[PARCHE CRONOS]`) solo permite usarlo en
producción:

1. `preparar_raw(df)`: cuerpo de `load_raw_data` sin leer disco (entrada en memoria).
2. `build_dataset(df_raw=None, columnas_fijas=None, conservar_sin_target=False, guardar=True)`.
   Con los valores por defecto el comportamiento es el original.
3. `columnas_fijas`: usa exactamente las columnas del entrenamiento (`activos/spx/columnas_dataset.json`)
   en vez de decidir qué columnas sobreviven por % de NaN (esa decisión cambia al agregar datos:
   p.ej. `hy_spread_chg_21d` desaparece con historia hasta 2023). Falta una → `KeyError`.
   El filtro de filas (13.4) usa las mismas 536 columnas no-meta que el original.
4. `conservar_sin_target=True`: no elimina la última fila (sin retorno futuro), que es la de la señal del día.
5. `guardar=False`: no escribe archivos; devuelve el DataFrame.

Verificado por `tests/test_features.py`: con la historia congelada reproduce el dataset de
entrenamiento (6.507 filas) y agregar filas no cambia valores históricos (causalidad).

## Diff

```diff
--- build_dataset_original.py	2026-10-07 17:05:22.000000000 -0300
+++ build_dataset.py	2026-10-07 17:05:22.000000000 -0300
@@ -338,8 +338,11 @@
         DataFrame con columnas renombradas, fecha parseada, y ordenado por fecha
     """
     print("  Cargando datos raw...")
-    df = pd.read_csv(filepath)
+    return preparar_raw(pd.read_csv(filepath))
 
+
+def preparar_raw(df):
+    """[PARCHE CRONOS] Cuerpo de load_raw_data a partir de un DataFrame ya cargado."""
     # Parsear fechas
     df['date'] = pd.to_datetime(df['date'])
 
@@ -2222,7 +2225,7 @@
 # PIPELINE PRINCIPAL
 # =============================================================================
 
-def build_dataset():
+def build_dataset(df_raw=None, columnas_fijas=None, conservar_sin_target=False, guardar=True):
     """
     Pipeline principal de construccion del dataset.
 
@@ -2312,8 +2315,11 @@
     print("PASO 1: Cargar Datos Raw de Bloomberg")
     print("-" * 80)
 
-    raw_file = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
-    df = load_raw_data(raw_file)
+    if df_raw is not None:  # [PARCHE CRONOS] entrada en memoria
+        df = preparar_raw(df_raw.copy())
+    else:
+        raw_file = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
+        df = load_raw_data(raw_file)
 
     # =========================================================================
     # PASO 2: Calcular target variable
@@ -2697,7 +2703,7 @@
     initial_rows = len(df)
 
     # 13.1: Eliminar columnas con 100% NaN (no aportan nada)
-    full_nan_cols = df.columns[df.isna().all()].tolist()
+    full_nan_cols = [] if columnas_fijas is not None else df.columns[df.isna().all()].tolist()  # [PARCHE CRONOS]
     if full_nan_cols:
         df = df.drop(columns=full_nan_cols)
         print(f"  + Columnas 100% NaN eliminadas: {len(full_nan_cols)}")
@@ -2713,7 +2719,14 @@
     meta_cols = ['date', 'date_id', 'SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME',
                  'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
     feature_cols = [c for c in df.columns if c not in meta_cols]
-    high_nan_cols = [c for c in feature_cols if df[c].isna().sum() / len(df) > 0.5]
+    if columnas_fijas is not None:  # [PARCHE CRONOS] conjunto de columnas fijo (el del entrenamiento)
+        faltan = [c for c in columnas_fijas if c not in df.columns and c != 'date_id']
+        if faltan:
+            raise KeyError(f"Columnas del modelo ausentes en el dataset: {faltan}")
+        df = df[[c for c in columnas_fijas if c != 'date_id']]
+        high_nan_cols = []
+    else:
+        high_nan_cols = [c for c in feature_cols if df[c].isna().sum() / len(df) > 0.5]
     if high_nan_cols:
         df = df.drop(columns=high_nan_cols)
         print(f"  + Columnas >50% NaN eliminadas: {len(high_nan_cols)}")
@@ -2724,7 +2737,8 @@
     df = df[null_pct < CONFIG['max_nan_pct']].reset_index(drop=True)
 
     # 13.5: Eliminar filas sin target
-    df = df.dropna(subset=['market_forward_excess_returns']).reset_index(drop=True)
+    if not conservar_sin_target:  # [PARCHE CRONOS] la ultima fila es la de la senal del dia
+        df = df.dropna(subset=['market_forward_excess_returns']).reset_index(drop=True)
 
     # 13.6: Crear date_id
     df['date_id'] = range(len(df))
@@ -2770,10 +2784,14 @@
     ordered_cols = meta_cols + other_cols
     ordered_cols = [c for c in ordered_cols if c in df.columns]
 
+    if columnas_fijas is not None:  # [PARCHE CRONOS]
+        ordered_cols = list(columnas_fijas)
     df_final = df[ordered_cols].copy()
 
     # Guardar
     output_path = os.path.join(DATA_DIR, "bloomberg_triple_screen_core.csv")
+    if not guardar:  # [PARCHE CRONOS]
+        return df_final
     df_final.to_csv(output_path, index=False)
     print(f"  + Dataset guardado: {output_path}")
 
```
