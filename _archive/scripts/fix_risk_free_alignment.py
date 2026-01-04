# -*- coding: utf-8 -*-
"""
================================================================================
CORRECCIÓN DE ALINEACIÓN TEMPORAL - RISK FREE RATE
================================================================================

Este script corrige el problema de alineación temporal en el cálculo del
target variable (market_forward_excess_returns).

PROBLEMA:
---------
El target original calculaba:
    market_forward_excess_returns[t] = forward_returns[t] - risk_free_rate[t]

Donde:
    - forward_returns[t] = SPY return from t to t+1 (tiene shift(-1) aplicado)
    - risk_free_rate[t] = BIL return from t-1 to t (SIN shift)

Esto mezcla dos períodos diferentes, lo cual es metodológicamente incorrecto.

SOLUCIÓN:
---------
Aplicar shift(-1) a risk_free_rate para que ambas variables cubran el mismo período:
    market_forward_excess_returns[t] = forward_returns[t] - risk_free_rate.shift(-1)[t]
                                     = (SPY return t→t+1) - (BIL return t→t+1)

================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import shutil

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

print("=" * 80)
print("CORRECCIÓN DE ALINEACIÓN TEMPORAL - RISK FREE RATE")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()

# =============================================================================
# PASO 1: Crear backup del dataset original
# =============================================================================
print("PASO 1: Crear backup del dataset original")
print("-" * 80)

core_file = os.path.join(DATA_DIR, "bloomberg_triple_screen_core.csv")
backup_file = os.path.join(DATA_DIR, "bloomberg_triple_screen_core_BACKUP_before_rf_fix.csv")

if not os.path.exists(backup_file):
    shutil.copy2(core_file, backup_file)
    print(f"  + Backup creado: {backup_file}")
else:
    print(f"  + Backup ya existe: {backup_file}")

# =============================================================================
# PASO 2: Cargar dataset
# =============================================================================
print("\nPASO 2: Cargar dataset")
print("-" * 80)

df = pd.read_csv(core_file, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
print(f"  + Dataset cargado: {len(df):,} filas x {df.shape[1]} columnas")

# Guardar valores originales para comparación
original_rf = df['risk_free_rate'].copy()
original_target = df['market_forward_excess_returns'].copy()

# =============================================================================
# PASO 3: Verificar el problema actual
# =============================================================================
print("\nPASO 3: Verificar alineación actual")
print("-" * 80)

# forward_returns ya tiene shift(-1) aplicado
# Verificamos que sea consistente con el precio de SPY
spy_return_check = df['SPY_CLOSE'].pct_change().shift(-1)
forward_corr = df['forward_returns'].corr(spy_return_check)
print(f"  + Correlación forward_returns vs SPY.pct_change().shift(-1): {forward_corr:.6f}")

if forward_corr > 0.99:
    print("  + [OK] forward_returns tiene shift(-1) aplicado correctamente")
else:
    print("  + [WARN] forward_returns puede tener un problema")

# =============================================================================
# PASO 4: Aplicar corrección
# =============================================================================
print("\nPASO 4: Aplicar corrección de shift(-1) a risk_free_rate")
print("-" * 80)

# Guardar la tasa original (t-1 → t) en una columna separada para referencia
df['risk_free_rate_original'] = df['risk_free_rate'].copy()

# Aplicar shift(-1) para obtener la tasa del período t → t+1
# Después del shift, risk_free_rate[t] = BIL return from t to t+1
df['risk_free_rate'] = df['risk_free_rate'].shift(-1)

# El último valor será NaN (no conocemos la tasa del día después del último)
last_valid_idx = df['risk_free_rate'].last_valid_index()
print(f"  + shift(-1) aplicado a risk_free_rate")
print(f"  + Último índice válido después del shift: {last_valid_idx}")

# =============================================================================
# PASO 5: Recalcular target
# =============================================================================
print("\nPASO 5: Recalcular market_forward_excess_returns")
print("-" * 80)

# Guardar el target original para referencia
df['market_forward_excess_returns_original'] = original_target

# Recalcular el target: ambos ahora cubren el período t → t+1
df['market_forward_excess_returns'] = df['forward_returns'] - df['risk_free_rate']

# Estadísticas del cambio
target_diff = df['market_forward_excess_returns'] - original_target
print(f"  + Target recalculado")
print(f"  + Diferencia promedio: {target_diff.mean() * 10000:.4f} bps")
print(f"  + Diferencia máxima: {target_diff.abs().max() * 10000:.4f} bps")
print(f"  + Diferencia std: {target_diff.std() * 10000:.4f} bps")

# =============================================================================
# PASO 6: Limpiar datos (eliminar filas sin target válido)
# =============================================================================
print("\nPASO 6: Limpiar datos")
print("-" * 80)

initial_rows = len(df)
df = df.dropna(subset=['market_forward_excess_returns'])
df = df.reset_index(drop=True)
df['date_id'] = range(len(df))

removed_rows = initial_rows - len(df)
print(f"  + Filas eliminadas (sin target válido): {removed_rows}")
print(f"  + Filas finales: {len(df):,}")

# =============================================================================
# PASO 7: Verificar consistencia
# =============================================================================
print("\nPASO 7: Verificar consistencia")
print("-" * 80)

# Verificar que el target es forward_returns - risk_free_rate
check = df['forward_returns'] - df['risk_free_rate']
consistency = np.allclose(check.dropna(), df['market_forward_excess_returns'].dropna())
print(f"  + Consistencia target = forward - rf: {'OK' if consistency else 'ERROR'}")

# Estadísticas del nuevo target
print(f"\n  ESTADÍSTICAS DEL TARGET CORREGIDO:")
print(f"  + Media diaria: {df['market_forward_excess_returns'].mean() * 100:.4f}%")
print(f"  + Std diaria: {df['market_forward_excess_returns'].std() * 100:.4f}%")
print(f"  + % Positivos: {(df['market_forward_excess_returns'] > 0).mean() * 100:.1f}%")

# =============================================================================
# PASO 8: Eliminar columnas auxiliares
# =============================================================================
print("\nPASO 8: Eliminar columnas auxiliares")
print("-" * 80)

# Eliminar las columnas originales que guardamos para verificación
columns_to_drop = ['risk_free_rate_original', 'market_forward_excess_returns_original']
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
print(f"  + Columnas auxiliares eliminadas")

# =============================================================================
# PASO 9: Guardar dataset corregido
# =============================================================================
print("\nPASO 9: Guardar dataset corregido")
print("-" * 80)

df.to_csv(core_file, index=False)
print(f"  + Dataset guardado: {core_file}")

# También actualizar bloomberg_prepared.csv si existe
prepared_file = os.path.join(DATA_DIR, "prepared", "bloomberg_prepared.csv")
if os.path.exists(prepared_file):
    df_prepared = pd.read_csv(prepared_file)
    df_prepared['date'] = pd.to_datetime(df_prepared['date'])

    # Backup
    prepared_backup = prepared_file.replace('.csv', '_BACKUP_before_rf_fix.csv')
    if not os.path.exists(prepared_backup):
        shutil.copy2(prepared_file, prepared_backup)

    # Aplicar la misma corrección
    df_prepared['risk_free_rate'] = df_prepared['risk_free_rate'].shift(-1)
    df_prepared['market_forward_excess_returns'] = df_prepared['forward_returns'] - df_prepared['risk_free_rate']
    df_prepared = df_prepared.dropna(subset=['market_forward_excess_returns'])
    df_prepared = df_prepared.reset_index(drop=True)
    df_prepared['date_id'] = range(len(df_prepared))

    df_prepared.to_csv(prepared_file, index=False)
    print(f"  + Dataset preparado actualizado: {prepared_file}")

# También actualizar bloomberg_features_hf.csv si existe
features_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")
if os.path.exists(features_file):
    df_features = pd.read_csv(features_file, low_memory=False)
    df_features['date'] = pd.to_datetime(df_features['date'])

    # Backup
    features_backup = features_file.replace('.csv', '_BACKUP_before_rf_fix.csv')
    if not os.path.exists(features_backup):
        shutil.copy2(features_file, features_backup)

    # Aplicar la misma corrección
    df_features['risk_free_rate'] = df_features['risk_free_rate'].shift(-1)
    df_features['market_forward_excess_returns'] = df_features['forward_returns'] - df_features['risk_free_rate']
    df_features = df_features.dropna(subset=['market_forward_excess_returns'])
    df_features = df_features.reset_index(drop=True)
    df_features['date_id'] = range(len(df_features))

    df_features.to_csv(features_file, index=False)
    print(f"  + Dataset features actualizado: {features_file}")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE LA CORRECCIÓN")
print("=" * 80)

print(f"""
  CAMBIO REALIZADO:
  -----------------
  ANTES: target[t] = forward_returns[t] - risk_free_rate[t]
         = (SPY return t→t+1) - (BIL return t-1→t)
         ❌ Períodos diferentes

  DESPUÉS: target[t] = forward_returns[t] - risk_free_rate.shift(-1)[t]
           = (SPY return t→t+1) - (BIL return t→t+1)
           ✓ Mismo período

  ARCHIVOS ACTUALIZADOS:
  ----------------------
  + {core_file}
  + {prepared_file if os.path.exists(prepared_file) else 'N/A'}
  + {features_file if os.path.exists(features_file) else 'N/A'}

  BACKUPS CREADOS:
  ----------------
  + {backup_file}

  IMPACTO NUMÉRICO:
  -----------------
  + Diferencia promedio en target: {target_diff.mean() * 10000:.4f} bps
  + El impacto es pequeño pero la metodología es ahora correcta

  NOTA IMPORTANTE:
  ----------------
  Si has entrenado modelos con el dataset anterior, deberás re-entrenarlos
  con el dataset corregido para mantener consistencia metodológica.
""")

print("=" * 80)
print("[OK] CORRECCIÓN COMPLETADA")
print("=" * 80)
