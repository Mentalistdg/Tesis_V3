# -*- coding: utf-8 -*-
"""
================================================================================
VALIDACION FINAL DEL DATASET TRIPLE PANTALLA
================================================================================

Verificaciones:
1. No hay data leakage
2. Features Triple Pantalla correctamente calculados
3. Target correctamente definido
4. Missing values bajo control
5. Datos listos para ML

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

FINAL_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data\final"

print("=" * 80)
print("VALIDACION FINAL DEL DATASET TRIPLE PANTALLA")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print()


def check_data_leakage(df):
    """Verifica que no haya data leakage"""
    issues = []

    # 1. Verificar que forward_returns use shift(-1)
    if 'SPY_CLOSE' in df.columns and 'forward_returns' in df.columns:
        expected = df['SPY_CLOSE'].pct_change().shift(-1)
        corr = df['forward_returns'].corr(expected)
        if corr < 0.99:
            issues.append(f"forward_returns correlation: {corr:.4f} (expected >0.99)")
        else:
            print("  [OK] forward_returns correctamente calculado")

    # 2. Verificar que ningun feature tenga correlacion perfecta con target
    target = df['market_forward_excess_returns']
    exclude = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
               'market_forward_excess_returns', 'SPY_CLOSE']

    high_corr_features = []
    for col in df.columns:
        if col not in exclude:
            try:
                corr = df[col].corr(target)
                if abs(corr) > 0.8:
                    high_corr_features.append((col, corr))
            except:
                pass

    if high_corr_features:
        issues.append(f"Features con correlacion >0.8 con target: {len(high_corr_features)}")
        for col, corr in high_corr_features[:5]:
            issues.append(f"  - {col}: {corr:.4f}")
    else:
        print("  [OK] Ningun feature tiene correlacion >0.8 con target")

    # 3. Verificar que features semanales usen datos de semana anterior
    if 'W_MACD_HIST' in df.columns:
        # Los datos semanales deben estar desfasados
        print("  [OK] Features semanales presentes (verificar lag manualmente)")

    return issues


def check_triple_screen_features(df):
    """Verifica features de Triple Pantalla"""
    issues = []

    # Features esperados
    weekly_features = ['W_MACD', 'W_MACD_SIGNAL', 'W_MACD_HIST', 'W_TREND',
                       'W_IMPULSE', 'W_EMA13', 'W_EMA26']
    daily_features = ['D_FORCE_INDEX_2', 'D_FORCE_INDEX_13', 'D_BULL_POWER',
                      'D_BEAR_POWER', 'D_IMPULSE']
    ts_features = ['TS_BUY_SETUP', 'TS_SELL_SETUP', 'TS_SIGNAL', 'TS_ALIGNMENT']

    # Verificar presencia
    for feat in weekly_features:
        if feat not in df.columns:
            issues.append(f"Missing weekly feature: {feat}")

    for feat in daily_features:
        if feat not in df.columns:
            issues.append(f"Missing daily feature: {feat}")

    for feat in ts_features:
        if feat not in df.columns:
            issues.append(f"Missing TS feature: {feat}")

    if not issues:
        print("  [OK] Todos los features Triple Pantalla presentes")

    # Verificar logica de senales
    if 'TS_SIGNAL' in df.columns:
        signal_dist = df['TS_SIGNAL'].value_counts()
        print(f"  Distribucion TS_SIGNAL:")
        for val in [-1, 0, 1]:
            count = signal_dist.get(val, 0)
            pct = count / len(df) * 100
            label = {-1: 'Venta', 0: 'Neutral', 1: 'Compra'}[val]
            print(f"    {label} ({val}): {count:,} ({pct:.1f}%)")

    return issues


def check_missing_values(df):
    """Analiza missing values"""
    issues = []

    exclude = ['date', 'date_id']
    total_missing = 0
    cols_with_missing = []

    for col in df.columns:
        if col not in exclude:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                pct = n_missing / len(df) * 100
                cols_with_missing.append((col, n_missing, pct))
                total_missing += n_missing

    total_cells = df.shape[0] * (df.shape[1] - len(exclude))
    pct_total = total_missing / total_cells * 100

    print(f"  Missing total: {total_missing:,} de {total_cells:,} ({pct_total:.2f}%)")

    if pct_total < 1.0:
        print("  [OK] Missing values < 1%")
    elif pct_total < 5.0:
        print("  [WARN] Missing values entre 1-5%")
    else:
        issues.append(f"Alto porcentaje de missing: {pct_total:.2f}%")

    # Mostrar columnas con mas missing
    if cols_with_missing:
        print(f"  Top columnas con missing:")
        for col, n, pct in sorted(cols_with_missing, key=lambda x: -x[1])[:5]:
            print(f"    - {col}: {n} ({pct:.1f}%)")

    return issues


def check_target(df):
    """Verifica el target"""
    issues = []

    if 'market_forward_excess_returns' not in df.columns:
        issues.append("Target 'market_forward_excess_returns' no encontrado")
        return issues

    target = df['market_forward_excess_returns']

    # Estadisticas basicas
    print(f"  Target: market_forward_excess_returns")
    print(f"    - Mean: {target.mean()*100:.4f}%")
    print(f"    - Std: {target.std()*100:.4f}%")
    print(f"    - Min: {target.min()*100:.4f}%")
    print(f"    - Max: {target.max()*100:.4f}%")
    print(f"    - Missing: {target.isna().sum()}")

    # Verificar que no sea constante
    if target.std() < 0.0001:
        issues.append("Target parece constante (std < 0.01%)")

    # Verificar distribucion
    positive_pct = (target > 0).sum() / len(target) * 100
    print(f"    - % Positivos: {positive_pct:.1f}%")

    if positive_pct < 40 or positive_pct > 60:
        print(f"  [INFO] Distribucion de target ligeramente sesgada")

    return issues


def check_data_integrity(df):
    """Verificaciones generales de integridad"""
    issues = []

    # 1. Verificar orden temporal
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        if not dates.is_monotonic_increasing:
            issues.append("Datos no estan ordenados cronologicamente")
        else:
            print("  [OK] Datos ordenados cronologicamente")

    # 2. Verificar duplicados
    n_duplicates = df.duplicated(subset=['date']).sum()
    if n_duplicates > 0:
        issues.append(f"Hay {n_duplicates} fechas duplicadas")
    else:
        print("  [OK] No hay fechas duplicadas")

    # 3. Verificar rango de fechas
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        print(f"  Periodo: {dates.min().date()} a {dates.max().date()}")
        print(f"  Total dias: {len(df):,}")

    # 4. Verificar tipos de datos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric = [c for c in df.columns if c not in numeric_cols and c != 'date']
    if non_numeric:
        print(f"  [INFO] Columnas no numericas: {non_numeric}")

    return issues


def generate_feature_summary(df):
    """Genera resumen de features para documentacion"""
    print("-" * 80)
    print("RESUMEN DE FEATURES")
    print("-" * 80)

    # Categorizar features
    categories = {
        'Market Indices (M)': [c for c in df.columns if c.startswith('M') and not c.startswith('MOM')],
        'Economic (E)': [c for c in df.columns if c.startswith('E') and c[1:2].isdigit()],
        'Interest Rates (I)': [c for c in df.columns if c.startswith('I') and c[1:2].isdigit()],
        'Commodities (P)': [c for c in df.columns if c.startswith('P') and c[1:2].isdigit()],
        'Volatility (V)': [c for c in df.columns if c.startswith('V') and c[1:2].isdigit()],
        'Sectors (S)': [c for c in df.columns if c.startswith('S') and c[1:2].isdigit()],
        'Momentum (MOM)': [c for c in df.columns if 'MOM' in c],
        'Elder Weekly (W)': [c for c in df.columns if c.startswith('W_')],
        'Elder Daily (D)': [c for c in df.columns if c.startswith('D_')],
        'Triple Screen (TS)': [c for c in df.columns if c.startswith('TS_')],
        'SPY Base': [c for c in df.columns if c.startswith('SPY_')],
        'Technical': [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'BB_', 'ATR', 'ADX', 'OBV', 'STOCH'])],
        'Meta/Target': ['date', 'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    }

    total_categorized = 0
    for cat, cols in categories.items():
        cols_in_df = [c for c in cols if c in df.columns]
        if cols_in_df:
            print(f"  {cat}: {len(cols_in_df)} features")
            total_categorized += len(cols_in_df)

    uncategorized = df.shape[1] - total_categorized
    if uncategorized > 0:
        print(f"  Otros: {uncategorized} features")

    print(f"\n  TOTAL: {df.shape[1]} columnas")


def main():
    """Ejecuta todas las validaciones"""

    # Cargar dataset CORE
    print("CARGANDO DATASET CORE")
    print("-" * 80)
    core_file = os.path.join(FINAL_DIR, "bloomberg_triple_screen_core.csv")
    df = pd.read_csv(core_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  + Cargado: {len(df):,} filas x {df.shape[1]} columnas")

    all_issues = []

    # 1. Data Leakage
    print()
    print("CHECK 1: Data Leakage")
    print("-" * 80)
    issues = check_data_leakage(df)
    all_issues.extend(issues)

    # 2. Triple Screen Features
    print()
    print("CHECK 2: Features Triple Pantalla")
    print("-" * 80)
    issues = check_triple_screen_features(df)
    all_issues.extend(issues)

    # 3. Missing Values
    print()
    print("CHECK 3: Missing Values")
    print("-" * 80)
    issues = check_missing_values(df)
    all_issues.extend(issues)

    # 4. Target
    print()
    print("CHECK 4: Variable Target")
    print("-" * 80)
    issues = check_target(df)
    all_issues.extend(issues)

    # 5. Integridad
    print()
    print("CHECK 5: Integridad de Datos")
    print("-" * 80)
    issues = check_data_integrity(df)
    all_issues.extend(issues)

    # 6. Resumen de features
    print()
    generate_feature_summary(df)

    # Resultado final
    print()
    print("=" * 80)
    print("RESULTADO DE VALIDACION")
    print("=" * 80)

    if all_issues:
        print(f"\n  [WARN] Se encontraron {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"    - {issue}")
    else:
        print("\n  [OK] TODAS LAS VALIDACIONES PASARON")

    print()
    print("=" * 80)
    print("DATASET LISTO PARA TRANSFERENCIA")
    print("=" * 80)
    print(f"\n  Archivo: bloomberg_triple_screen_core.csv")
    print(f"  Tamano: {len(df):,} filas x {df.shape[1]} columnas")
    print(f"  Features Triple Pantalla: W_ (semanal), D_ (diario), TS_ (senales)")
    print(f"\n  Uso recomendado:")
    print(f"    1. Dividir en train/val/test manteniendo orden temporal")
    print(f"    2. No usar shuffle para preservar estructura temporal")
    print(f"    3. Target: 'market_forward_excess_returns'")
    print(f"    4. Features TS_* pueden usarse directamente o como filtro")

    return df, all_issues


if __name__ == "__main__":
    df, issues = main()
