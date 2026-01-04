# -*- coding: utf-8 -*-
"""
================================================================================
DIAGNÓSTICO DE DATA LEAKAGE
================================================================================
Detecta posibles fugas de datos en el pipeline de features.

Tests realizados:
1. Correlación de features con target (correlación > 0.3 es sospechosa)
2. Verificar que features no usen información futura
3. Verificar cálculo correcto del target
4. Test de información futura en features
================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

print("=" * 80)
print("DIAGNÓSTICO DE DATA LEAKAGE")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n1. CARGANDO DATOS")
print("-" * 80)

# Datos raw
raw_file = os.path.join(DATA_DIR, "BLOOMBERG_RAW_DATA.csv")
df_raw = pd.read_csv(raw_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
print(f"  Raw data: {len(df_raw):,} filas")

# Datos con features
features_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")
df = pd.read_csv(features_file)
df['date'] = pd.to_datetime(df['date'])
print(f"  Features data: {len(df):,} filas x {df.shape[1]} columnas")

# =============================================================================
# TEST 1: VERIFICAR CÁLCULO DEL TARGET
# =============================================================================
print("\n2. VERIFICAR CÁLCULO DEL TARGET")
print("-" * 80)

# Recalcular target desde cero
spy_close = df['SPY_CLOSE'].values
actual_returns = np.diff(spy_close) / spy_close[:-1]  # Retorno del día actual
forward_returns_recalc = np.append(actual_returns, np.nan)  # Alinear: retorno[t] = (precio[t+1] - precio[t]) / precio[t]

# El target debería ser el retorno FUTURO, es decir:
# target[t] = (precio[t+1] - precio[t]) / precio[t]
# Lo que equivale a: precio.pct_change().shift(-1)

# Verificar
target_in_file = df['market_forward_excess_returns'].values
forward_in_file = df['forward_returns'].values if 'forward_returns' in df.columns else None

# Calcular forward returns correctamente
correct_forward = np.zeros(len(spy_close))
correct_forward[:-1] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
correct_forward[-1] = np.nan

print(f"  Target en archivo (primeros 5): {target_in_file[:5]}")
print(f"  Forward recalculado (primeros 5): {correct_forward[:5]}")

# Verificar que el target sea forward-looking
# Alinear arrays
mask = ~np.isnan(target_in_file) & ~np.isnan(correct_forward)
if mask.sum() > 0:
    correlation_target_forward = np.corrcoef(target_in_file[mask], correct_forward[mask])[0,1]
else:
    correlation_target_forward = 0
print(f"  Correlación target vs forward recalculado: {correlation_target_forward:.6f}")

if abs(correlation_target_forward) > 0.99:
    print("  [OK] Target es forward-looking (retorno futuro)")
else:
    print("  [WARNING] Target podría no ser forward-looking correctamente")

# =============================================================================
# TEST 2: BUSCAR FEATURES CON CORRELACIÓN SOSPECHOSA AL TARGET
# =============================================================================
print("\n3. BUSCAR FEATURES CON CORRELACIÓN ALTA AL TARGET")
print("-" * 80)

target_col = 'market_forward_excess_returns'
exclude_cols = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'SPY_CLOSE', 'SPY_OPEN',
                'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME']

feature_cols = [c for c in df.columns if c not in exclude_cols]

# Calcular correlaciones
correlations = {}
for col in feature_cols:
    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        mask = ~(df[col].isna() | df[target_col].isna())
        if mask.sum() > 100:
            corr = df.loc[mask, col].corr(df.loc[mask, target_col])
            if not np.isnan(corr):
                correlations[col] = corr

# Ordenar por valor absoluto
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"  Total features analizados: {len(correlations)}")
print(f"\n  TOP 20 correlaciones más altas (sospechosas si > 0.1):")
print("-" * 60)

suspicious_features = []
for i, (feat, corr) in enumerate(sorted_corr[:20]):
    status = "[!] SOSPECHOSO" if abs(corr) > 0.1 else ""
    print(f"  {i+1:2d}. {feat:<45} {corr:>8.4f} {status}")
    if abs(corr) > 0.1:
        suspicious_features.append((feat, corr))

if len(suspicious_features) == 0:
    print("\n  [OK] No hay features con correlación sospechosamente alta (>0.1)")
else:
    print(f"\n  [WARNING] {len(suspicious_features)} features con correlación > 0.1")

# =============================================================================
# TEST 3: VERIFICAR QUE FEATURES NO USEN INFORMACIÓN FUTURA
# =============================================================================
print("\n4. TEST DE INFORMACIÓN FUTURA")
print("-" * 80)

# Test: Si un feature usa información futura, su valor en t debería estar
# más correlacionado con el retorno en t que con el retorno en t+1

print("  Verificando si features están correlacionados con retorno actual vs futuro...")

# Retorno actual (contemporáneo) - NO debería predecirse con features de t
current_return = df['SPY_CLOSE'].pct_change()

leakage_suspects = []

# Probar algunos features clave
test_features = [
    'SPY_return_1d', 'SPY_RSI_14', 'SPY_MACD', 'SPY_ADX_14',
    'SPY_momentum_21', 'SPY_volatility_21', 'VIX_zscore',
    'SPY_BB_position_20', 'SPY_zscore_21'
]
test_features = [f for f in test_features if f in df.columns]

print(f"\n  {'Feature':<30} {'Corr Target(t+1)':<18} {'Corr Return(t)':<18} {'Ratio':<10} {'Status'}")
print("-" * 100)

for feat in test_features:
    mask = ~(df[feat].isna() | df[target_col].isna() | current_return.isna())

    if mask.sum() > 100:
        corr_target = df.loc[mask, feat].corr(df.loc[mask, target_col])  # Correlación con futuro
        corr_current = df.loc[mask, feat].corr(current_return[mask])  # Correlación con presente

        # Si el feature está más correlacionado con el retorno presente que con el futuro,
        # es normal. Pero si está MUY correlacionado con el presente, podría haber leakage

        ratio = abs(corr_current) / (abs(corr_target) + 1e-10) if abs(corr_target) > 0.001 else 999

        status = ""
        if abs(corr_current) > 0.5:
            status = "[!] POSIBLE LEAKAGE"
            leakage_suspects.append((feat, corr_current, corr_target))
        elif abs(corr_current) > 0.3:
            status = "[!] REVISAR"
        else:
            status = "OK"

        print(f"  {feat:<30} {corr_target:>15.4f}   {corr_current:>15.4f}   {ratio:>8.2f}   {status}")

# =============================================================================
# TEST 4: VERIFICAR SECUENCIA TEMPORAL
# =============================================================================
print("\n5. VERIFICAR SECUENCIA TEMPORAL")
print("-" * 80)

# Verificar que las fechas estén ordenadas
dates_sorted = df['date'].is_monotonic_increasing
print(f"  Fechas ordenadas cronológicamente: {'SI' if dates_sorted else 'NO'}")

# Verificar que no haya gaps extraños
date_diffs = df['date'].diff().dt.days
max_gap = date_diffs.max()
print(f"  Gap máximo entre fechas: {max_gap} días")

if max_gap > 10:
    print(f"  [WARNING] Hay gaps de más de 10 días (podría ser OK si hay feriados)")

# =============================================================================
# TEST 5: VERIFICAR CÁLCULO DE FEATURES ROLLING
# =============================================================================
print("\n6. VERIFICAR ROLLING WINDOWS (LOOK-AHEAD BIAS)")
print("-" * 80)

# Recalcular manualmente un rolling y comparar
if 'SPY_volatility_21' in df.columns:
    # Calcular volatilidad rolling manualmente
    returns = df['SPY_CLOSE'].pct_change()
    manual_vol = returns.rolling(window=21, min_periods=21).std() * np.sqrt(252)

    # Comparar
    mask = ~(manual_vol.isna() | df['SPY_volatility_21'].isna())
    corr = manual_vol[mask].corr(df['SPY_volatility_21'][mask])
    diff = (manual_vol[mask] - df['SPY_volatility_21'][mask]).abs().mean()

    print(f"  SPY_volatility_21:")
    print(f"    Correlación con recálculo manual: {corr:.6f}")
    print(f"    Diferencia media absoluta: {diff:.6f}")

    if corr > 0.999:
        print("    [OK] Rolling window calculado correctamente")
    else:
        print("    [WARNING] Diferencia en cálculo de rolling")

# Verificar RSI
if 'SPY_RSI_14' in df.columns:
    print(f"\n  SPY_RSI_14:")
    rsi_range = df['SPY_RSI_14'].describe()
    print(f"    Min: {rsi_range['min']:.2f}, Max: {rsi_range['max']:.2f}")

    if rsi_range['min'] >= 0 and rsi_range['max'] <= 100:
        print("    [OK] RSI en rango válido 0-100")
    else:
        print("    [WARNING] RSI fuera de rango")

# =============================================================================
# TEST 6: TEST DE PREDICTABILIDAD IMPOSIBLE
# =============================================================================
print("\n7. TEST DE PREDICTABILIDAD IMPOSIBLE")
print("-" * 80)

# Si algún feature puede predecir el target con R² > 0.5 de forma individual,
# es muy sospechoso

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("  Probando predictabilidad individual de features...")

high_r2_features = []

for feat in sorted_corr[:50]:  # Top 50 correlacionados
    feat_name = feat[0]

    mask = ~(df[feat_name].isna() | df[target_col].isna())
    X = df.loc[mask, feat_name].values.reshape(-1, 1)
    y = df.loc[mask, target_col].values

    if len(X) > 100:
        # Split temporal simple
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        if r2 > 0.05:  # R² > 5% es ya sospechoso para un solo feature en finanzas
            high_r2_features.append((feat_name, r2))

if high_r2_features:
    print(f"\n  [WARNING] Features con R² > 5% (sospechoso):")
    for feat, r2 in sorted(high_r2_features, key=lambda x: x[1], reverse=True):
        print(f"    {feat}: R² = {r2:.4f}")
else:
    print("  [OK] Ningún feature individual tiene R² > 5% (normal en finanzas)")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DEL DIAGNÓSTICO")
print("=" * 80)

issues = []

if abs(correlation_target_forward) < 0.99:
    issues.append("Target no es forward-looking correcto")

if len(suspicious_features) > 0:
    issues.append(f"{len(suspicious_features)} features con correlación > 0.1 al target")

if len(leakage_suspects) > 0:
    issues.append(f"{len(leakage_suspects)} features con posible look-ahead bias")

if len(high_r2_features) > 0:
    issues.append(f"{len(high_r2_features)} features con R² individual > 5%")

if not dates_sorted:
    issues.append("Fechas no ordenadas cronológicamente")

if len(issues) == 0:
    print("\n  [OK] NO SE DETECTARON PROBLEMAS DE DATA LEAKAGE")
    print("\n  El pipeline parece estar correctamente configurado.")
else:
    print(f"\n  [WARNING] SE DETECTARON {len(issues)} POSIBLES PROBLEMAS:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

    print("\n  RECOMENDACIONES:")
    print("    - Revisar features con alta correlación al target")
    print("    - Verificar cálculos de rolling windows")
    print("    - Considerar eliminar features sospechosos")

print("\n" + "=" * 80)
print("[OK] DIAGNÓSTICO COMPLETADO")
print("=" * 80)
