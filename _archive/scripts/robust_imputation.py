# -*- coding: utf-8 -*-
"""
================================================================================
IMPUTACIÓN ROBUSTA DE DATOS - NIVEL HEDGE FUND
================================================================================

Pipeline de imputación que preserva la integridad temporal de los datos
financieros y evita introducir leakage.

PRINCIPIOS:
1. Forward-fill primero (más común en datos financieros)
2. Backward-fill limitado (solo para primeros valores)
3. Rolling mean para gaps persistentes
4. NUNCA usar información futura
5. Documentar qué se imputó

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

DATA_DIR = r"C:\Users\salas\PycharmProjects\Tesis_2\data\final"


def analyze_missing(df, exclude_cols=None):
    """Analiza patrones de missing values"""
    if exclude_cols is None:
        exclude_cols = ['date_id', 'date', 'forward_returns', 'risk_free_rate',
                        'market_forward_excess_returns']

    cols = [c for c in df.columns if c not in exclude_cols]

    analysis = {}
    for col in cols:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100

        if n_missing > 0:
            # Primera y última fecha con datos
            first_valid = df[col].first_valid_index()
            last_valid = df[col].last_valid_index()

            if first_valid is not None:
                first_date = df.loc[first_valid, 'date']
                last_date = df.loc[last_valid, 'date']
            else:
                first_date = None
                last_date = None

            analysis[col] = {
                'n_missing': n_missing,
                'pct_missing': pct_missing,
                'first_valid_date': first_date,
                'last_valid_date': last_date
            }

    return analysis


def impute_financial_timeseries(df, max_ffill=10, max_bfill=5):
    """
    Imputa datos financieros preservando integridad temporal.

    Estrategia por tipo de variable:
    - Precios/Niveles: Forward-fill (último valor conocido)
    - Retornos: 0 para gaps pequeños, NaN para gaps grandes
    - Ratios: Forward-fill + Rolling mean
    - Indicadores económicos: Forward-fill (se publican con rezago)
    """
    df = df.copy()

    # Identificar tipos de columnas
    price_cols = [c for c in df.columns if any(x in c for x in
                  ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'M1', 'M2', 'M3', 'M4', 'M5',
                   'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14',
                   'M15', 'M16', 'M17', 'M18', 'P7', 'P8', 'P9', 'P10', 'P11',
                   'P12', 'P13', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
                   'S11', 'GLD', 'GOLD', 'SILVER', 'OIL', 'COPPER'])]

    return_cols = [c for c in df.columns if 'return' in c.lower() or 'momentum' in c.lower()]

    rate_cols = [c for c in df.columns if c.startswith('I') and c[1:].split('_')[0].isdigit()]

    volatility_cols = [c for c in df.columns if c.startswith('V') or 'vol' in c.lower() or 'VIX' in c]

    economic_cols = [c for c in df.columns if c.startswith('E') and c[1:].split('_')[0].isdigit()]

    exclude = ['date_id', 'date', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

    imputation_log = {}

    # 1. Precios y niveles: Forward-fill
    for col in price_cols:
        if col in df.columns and col not in exclude:
            before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=max_ffill)
            after = df[col].isna().sum()
            if before != after:
                imputation_log[col] = f'ffill: {before - after} valores'

    # 2. Tasas de interés: Forward-fill (cambian poco día a día)
    for col in rate_cols:
        if col in df.columns and col not in exclude:
            before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=max_ffill * 2)  # Más tolerante
            after = df[col].isna().sum()
            if before != after:
                imputation_log[col] = f'ffill: {before - after} valores'

    # 3. Económicos: Forward-fill extenso (se publican mensual/trimestral)
    for col in economic_cols:
        if col in df.columns and col not in exclude:
            before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=30)  # Hasta 30 días (más de un mes)
            after = df[col].isna().sum()
            if before != after:
                imputation_log[col] = f'ffill: {before - after} valores'

    # 4. Volatilidad: Forward-fill moderado
    for col in volatility_cols:
        if col in df.columns and col not in exclude:
            before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=max_ffill)
            after = df[col].isna().sum()
            if before != after:
                imputation_log[col] = f'ffill: {before - after} valores'

    # 5. Retornos: Poner 0 para gaps pequeños (asumimos sin cambio)
    for col in return_cols:
        if col in df.columns and col not in exclude:
            before = df[col].isna().sum()
            # Solo para gaps pequeños (máximo 3 días)
            mask = df[col].isna() & (df[col].ffill(limit=3) == df[col].ffill(limit=3))
            df.loc[mask, col] = 0
            after = df[col].isna().sum()
            if before != after:
                imputation_log[col] = f'zero-fill: {before - after} valores'

    # 6. Backward-fill limitado para primeros valores
    for col in df.columns:
        if col not in exclude:
            before = df[col].isna().sum()
            df[col] = df[col].bfill(limit=max_bfill)
            after = df[col].isna().sum()
            if before != after:
                if col in imputation_log:
                    imputation_log[col] += f', bfill: {before - after}'
                else:
                    imputation_log[col] = f'bfill: {before - after} valores'

    # 7. Para gaps persistentes: Rolling mean (solo datos pasados)
    for col in df.columns:
        if col not in exclude and df[col].isna().sum() > 0:
            before = df[col].isna().sum()
            # Rolling mean de los últimos 21 días (solo pasado)
            rolling_mean = df[col].rolling(window=21, min_periods=5).mean()
            df[col] = df[col].fillna(rolling_mean)
            after = df[col].isna().sum()
            if before != after:
                if col in imputation_log:
                    imputation_log[col] += f', rolling_mean: {before - after}'
                else:
                    imputation_log[col] = f'rolling_mean: {before - after} valores'

    return df, imputation_log


def remove_high_nan_columns(df, threshold=0.30, exclude_cols=None):
    """Elimina columnas con más de threshold% de NaN"""
    if exclude_cols is None:
        exclude_cols = ['date_id', 'date', 'forward_returns', 'risk_free_rate',
                        'market_forward_excess_returns', 'SPY_CLOSE']

    cols_to_drop = []
    for col in df.columns:
        if col not in exclude_cols:
            pct_nan = df[col].isna().sum() / len(df)
            if pct_nan > threshold:
                cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop


def validate_no_leakage(df):
    """Verifica que no haya data leakage"""
    issues = []

    # 1. Verificar que forward_returns esté correctamente calculado
    if 'SPY_CLOSE' in df.columns and 'forward_returns' in df.columns:
        expected = df['SPY_CLOSE'].pct_change().shift(-1)
        correlation = df['forward_returns'].corr(expected)
        if correlation < 0.99:
            issues.append(f"forward_returns correlation: {correlation:.4f} (expected >0.99)")

    # 2. Verificar que features no estén correlacionados perfectamente con target
    target = df['market_forward_excess_returns']
    for col in df.columns:
        if col not in ['date_id', 'date', 'forward_returns', 'risk_free_rate',
                       'market_forward_excess_returns']:
            corr = df[col].corr(target)
            if abs(corr) > 0.5:
                issues.append(f"{col} has high correlation with target: {corr:.4f}")

    return issues


def create_imputed_dataset(input_file=None, output_file=None):
    """Pipeline completo de imputación"""

    print("=" * 80)
    print("IMPUTACIÓN ROBUSTA DE DATOS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Cargar datos
    if input_file is None:
        input_file = os.path.join(DATA_DIR, "bloomberg_features_hf.csv")

    if not os.path.exists(input_file):
        # Fallback al archivo original
        input_file = os.path.join(DATA_DIR, "bloomberg_features.csv")

    print(f"PASO 1: Cargar datos de {input_file}")
    print("-" * 80)

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  + Cargado: {len(df):,} filas x {df.shape[1]} columnas")

    # Analizar missing antes
    print("\nPASO 2: Análisis de Missing Values (antes)")
    print("-" * 80)

    analysis_before = analyze_missing(df)
    high_missing = {k: v for k, v in analysis_before.items() if v['pct_missing'] > 10}
    print(f"  + Variables con >10% missing: {len(high_missing)}")
    for col, info in sorted(high_missing.items(), key=lambda x: -x[1]['pct_missing'])[:10]:
        print(f"    - {col}: {info['pct_missing']:.1f}%")

    # Eliminar columnas con demasiados NaN
    print("\nPASO 3: Eliminar columnas con >30% NaN")
    print("-" * 80)

    df, dropped_cols = remove_high_nan_columns(df, threshold=0.30)
    print(f"  + Columnas eliminadas: {len(dropped_cols)}")
    if dropped_cols:
        for col in dropped_cols[:10]:
            print(f"    - {col}")
        if len(dropped_cols) > 10:
            print(f"    ... y {len(dropped_cols) - 10} más")

    # Imputar
    print("\nPASO 4: Imputación")
    print("-" * 80)

    df_imputed, imputation_log = impute_financial_timeseries(df)
    print(f"  + Variables imputadas: {len(imputation_log)}")

    # Analizar missing después
    print("\nPASO 5: Análisis de Missing Values (después)")
    print("-" * 80)

    analysis_after = analyze_missing(df_imputed)
    remaining_missing = {k: v for k, v in analysis_after.items() if v['n_missing'] > 0}
    print(f"  + Variables con missing restante: {len(remaining_missing)}")

    # Eliminar filas restantes con NaN en target
    df_imputed = df_imputed.dropna(subset=['market_forward_excess_returns'])

    # Validar no leakage
    print("\nPASO 6: Validación Anti-Leakage")
    print("-" * 80)

    issues = validate_no_leakage(df_imputed)
    if issues:
        print("  [WARN] Posibles issues detectados:")
        for issue in issues[:5]:
            print(f"    - {issue}")
    else:
        print("  [OK] No se detectaron problemas de leakage")

    # Guardar
    print("\nPASO 7: Guardar Dataset Imputado")
    print("-" * 80)

    if output_file is None:
        output_file = os.path.join(DATA_DIR, "bloomberg_features_final.csv")

    df_imputed.to_csv(output_file, index=False)
    print(f"  + Guardado: {output_file}")
    print(f"  + Dimensiones finales: {len(df_imputed):,} filas x {df_imputed.shape[1]} columnas")

    # Guardar metadata de imputación
    imputation_metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_file': input_file,
        'output_file': output_file,
        'rows_input': len(df),
        'rows_output': len(df_imputed),
        'columns_input': df.shape[1] + len(dropped_cols),
        'columns_output': df_imputed.shape[1],
        'columns_dropped': dropped_cols,
        'imputation_log': imputation_log,
        'remaining_missing': {k: v['n_missing'] for k, v in remaining_missing.items()},
        'validation_issues': issues
    }

    meta_file = output_file.replace('.csv', '_imputation_log.json')
    with open(meta_file, 'w') as f:
        json.dump(imputation_metadata, f, indent=2, default=str)

    print(f"  + Log guardado: {meta_file}")

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE IMPUTACIÓN")
    print("=" * 80)
    print(f"  + Filas: {len(df):,} -> {len(df_imputed):,}")
    print(f"  + Columnas: {df.shape[1] + len(dropped_cols)} -> {df_imputed.shape[1]}")
    print(f"  + Columnas eliminadas: {len(dropped_cols)}")
    print(f"  + Variables imputadas: {len(imputation_log)}")

    # Missing final
    total_missing = df_imputed.isnull().sum().sum()
    total_cells = df_imputed.shape[0] * df_imputed.shape[1]
    print(f"  + Missing final: {total_missing:,} de {total_cells:,} ({total_missing/total_cells*100:.4f}%)")

    print("\n" + "=" * 80)
    print("[OK] IMPUTACIÓN COMPLETADA")
    print("=" * 80)

    return df_imputed


if __name__ == "__main__":
    df = create_imputed_dataset()

    if df is not None:
        print("\nMuestra de datos finales:")
        print(df.head().to_string())
