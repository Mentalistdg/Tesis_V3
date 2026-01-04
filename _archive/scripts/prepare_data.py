# -*- coding: utf-8 -*-
"""
================================================================================
PREPARACION DE DATOS - Bloomberg Raw to Prepared
================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Detectar directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
PREPARED_DIR = os.path.join(BASE_DIR, "data", "prepared")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

# Crear directorios si no existen
os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

print("=" * 80)
print("PREPARACION DE DATOS BLOOMBERG")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()

# =============================================================================
# CARGAR DATOS RAW
# =============================================================================
print("PASO 1: Cargar datos raw")
print("-" * 80)

raw_file = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
df = pd.read_csv(raw_file)
df['date'] = pd.to_datetime(df['date'])
print(f"  + Archivo: {raw_file}")
print(f"  + Filas: {len(df):,}")
print(f"  + Columnas: {df.shape[1]}")

# =============================================================================
# RENOMBRAR COLUMNAS
# =============================================================================
print("\nPASO 2: Renombrar columnas")
print("-" * 80)

# Mapeo de columnas SPY
col_mapping = {
    'SPY US Equity (CLOSE)': 'SPY_CLOSE',
    'SPY US Equity (OPEN)': 'SPY_OPEN',
    'SPY US Equity (HIGH)': 'SPY_HIGH',
    'SPY US Equity (LOW)': 'SPY_LOW',
    'SPY US Equity (VOLUME)': 'SPY_VOLUME',
    'BIL US Equity (Risk Free)': 'risk_free_rate',
    'VIX Index': 'V1',
    'VIX3M Index': 'V2',
    'VIX1M Index': 'V3',
    'VXV Index': 'V4',
    'MOVE Index': 'V5',
    'GT2 Govt': 'I2',
    'GT5 Govt': 'I3',
    'GT10 Govt': 'I4',
    'GT30 Govt': 'I5',
    'GB3 Govt': 'I1',
    'FDTR Index': 'I6',
    'US0003M Index': 'I7',
    'USGG10YR Index': 'I8',
    'CL1 Comdty': 'P12',  # Oil
    'GC1 Comdty': 'P9',   # Gold
    'SI1 Comdty': 'P10',  # Silver
    'BCOMTR Index': 'P11',  # Commodities
    'CDX IG CDSI GEN 5Y': 'S1',  # Credit IG
    'CDX HY CDSI GEN 5Y': 'S2',  # Credit HY
    'LF98OAS Index': 'S3',
    'AAII BULLISH Index': 'S5',
    'PUT Index': 'S6',
    'ADD Index': 'M1',
    'TICK Index': 'M2',
    'TRIN Index': 'M3',
    'MCCL Index': 'M4',
    'MCSU Index': 'M5',
}

# Renombrar columnas que existen
for old_col, new_col in col_mapping.items():
    if old_col in df.columns:
        df = df.rename(columns={old_col: new_col})

print(f"  + Columnas renombradas: {len(col_mapping)}")

# =============================================================================
# CALCULAR TARGET (Forward Returns)
# =============================================================================
print("\nPASO 3: Calcular target (forward returns)")
print("-" * 80)

# Forward returns (retorno del dia siguiente)
df['forward_returns'] = df['SPY_CLOSE'].pct_change().shift(-1)

# Risk free rate ajustado (diario)
if 'risk_free_rate' in df.columns:
    df['risk_free_rate'] = df['risk_free_rate'] / 100 / 252  # Convertir a diario
else:
    df['risk_free_rate'] = 0.0

# Excess returns sobre risk free
df['market_forward_excess_returns'] = df['forward_returns'] - df['risk_free_rate']

print(f"  + forward_returns calculado")
print(f"  + market_forward_excess_returns calculado")

# =============================================================================
# CREAR DATE_ID
# =============================================================================
print("\nPASO 4: Crear identificadores")
print("-" * 80)

df = df.sort_values('date').reset_index(drop=True)
df['date_id'] = range(len(df))

print(f"  + date_id creado")

# =============================================================================
# SELECCIONAR COLUMNAS FINALES
# =============================================================================
print("\nPASO 5: Seleccionar columnas")
print("-" * 80)

# Columnas esenciales
essential_cols = ['date_id', 'date', 'SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME',
                  'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

# Columnas de features (las que fueron renombradas)
feature_cols = [col for col in df.columns if col not in essential_cols and
                (col.startswith(('V', 'I', 'P', 'S', 'M', 'E')) and len(col) <= 4 or
                 col in ['V1', 'V2', 'V3', 'V4', 'V5'])]

final_cols = essential_cols + feature_cols
final_cols = [c for c in final_cols if c in df.columns]

df_prepared = df[final_cols].copy()

print(f"  + Columnas seleccionadas: {len(final_cols)}")

# =============================================================================
# LIMPIAR DATOS
# =============================================================================
print("\nPASO 6: Limpiar datos")
print("-" * 80)

# Eliminar filas sin target
initial_rows = len(df_prepared)
df_prepared = df_prepared.dropna(subset=['market_forward_excess_returns'])
print(f"  + Filas eliminadas (sin target): {initial_rows - len(df_prepared)}")

# Re-crear date_id
df_prepared = df_prepared.reset_index(drop=True)
df_prepared['date_id'] = range(len(df_prepared))

print(f"  + Filas finales: {len(df_prepared):,}")

# =============================================================================
# GUARDAR
# =============================================================================
print("\nPASO 7: Guardar")
print("-" * 80)

output_file = os.path.join(PREPARED_DIR, "bloomberg_prepared.csv")
df_prepared.to_csv(output_file, index=False)
print(f"  + Guardado: {output_file}")

# Resumen
print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)
print(f"  + Periodo: {df_prepared['date'].min().date()} a {df_prepared['date'].max().date()}")
print(f"  + Filas: {len(df_prepared):,}")
print(f"  + Columnas: {df_prepared.shape[1]}")
print(f"\n  + Columnas disponibles:")
for col in df_prepared.columns:
    null_pct = df_prepared[col].isna().sum() / len(df_prepared) * 100
    print(f"      {col}: {null_pct:.1f}% NaN")

print("\n" + "=" * 80)
print("[OK] DATOS PREPARADOS")
print("=" * 80)
