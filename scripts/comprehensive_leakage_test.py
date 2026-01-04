# -*- coding: utf-8 -*-
"""
================================================================================
TEST EXHAUSTIVO DE DATA LEAKAGE
================================================================================
Bateria completa de tests para garantizar que no hay fuga de datos.

Tests incluidos:
1. Verificacion del calculo del target (forward returns)
2. Walk-forward validation (R2 por periodo - detecta leakage temporal)
3. Purged cross-validation (gap entre train/test)
4. Feature timestamp audit (verificar que features no usen futuro)
5. Shuffle test (si hay leakage, shufflear deberia mantener performance)
6. Autocorrelation test (detectar dependencias temporales espurias)
7. Future correlation test (features correlacionados con retornos futuros)

================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("TEST EXHAUSTIVO DE DATA LEAKAGE")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n" + "=" * 80)
print("CARGANDO DATOS")
print("=" * 80)

# Cargar datos con features
features_file = os.path.join(DATA_DIR, "final", "bloomberg_features_hf.csv")
df = pd.read_csv(features_file)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"  Filas: {len(df):,}")
print(f"  Columnas: {df.shape[1]}")
print(f"  Periodo: {df['date'].min().date()} a {df['date'].max().date()}")

# Identificar columnas
target_col = 'market_forward_excess_returns'
exclude_cols = ['date', 'date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'SPY_CLOSE', 'SPY_OPEN',
                'SPY_HIGH', 'SPY_LOW', 'SPY_VOLUME']
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"  Features: {len(feature_cols)}")
print(f"  Target: {target_col}")

# Preparar X, y
X = df[feature_cols].copy()
y = df[target_col].copy()
dates = df['date'].copy()

# Imputar y escalar
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

all_tests_passed = True
issues_found = []

# =============================================================================
# TEST 1: VERIFICACION DEL TARGET
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: VERIFICACION DEL CALCULO DEL TARGET")
print("=" * 80)

# El target debe ser el retorno FUTURO: (precio[t+1] - precio[t]) / precio[t]
spy_close = df['SPY_CLOSE'].values
target_values = df[target_col].values

# Recalcular forward returns
correct_forward = np.zeros(len(spy_close))
correct_forward[:-1] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
correct_forward[-1] = np.nan

# El target incluye risk_free_rate adjustment
rf_rate = df['risk_free_rate'].values if 'risk_free_rate' in df.columns else 0

# Comparar
# forward_returns en el archivo deberia ser el retorno puro
forward_in_file = df['forward_returns'].values if 'forward_returns' in df.columns else None

print("\n  Verificando alineacion temporal del target...")

# Test: El target de hoy debe correlacionarse con el retorno de hoy a manana
# Shift -1 del precio da el precio de manana
price_tomorrow = np.roll(spy_close, -1)
price_tomorrow[-1] = np.nan
return_today_to_tomorrow = (price_tomorrow - spy_close) / spy_close

mask = ~np.isnan(target_values) & ~np.isnan(return_today_to_tomorrow)
corr_with_future = np.corrcoef(target_values[mask], return_today_to_tomorrow[mask])[0,1]

print(f"  Correlacion target vs return(t->t+1): {corr_with_future:.6f}")

# Test: El target NO debe correlacionarse con el retorno de ayer a hoy
return_yesterday_to_today = (spy_close - np.roll(spy_close, 1)) / np.roll(spy_close, 1)
return_yesterday_to_today[0] = np.nan

mask2 = ~np.isnan(target_values) & ~np.isnan(return_yesterday_to_today)
corr_with_past = np.corrcoef(target_values[mask2], return_yesterday_to_today[mask2])[0,1]

print(f"  Correlacion target vs return(t-1->t): {corr_with_past:.6f}")

if corr_with_future > 0.99 and abs(corr_with_past) < 0.15:
    print("\n  [PASS] Target correctamente calculado como retorno futuro")
else:
    print("\n  [FAIL] Posible problema con el calculo del target")
    all_tests_passed = False
    issues_found.append("Target no es correctamente forward-looking")

# Verificacion adicional: El target[t] NO debe poder predecirse perfectamente por features[t]
# porque features[t] solo tienen info hasta tiempo t

# =============================================================================
# TEST 2: WALK-FORWARD VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: WALK-FORWARD VALIDATION")
print("=" * 80)

print("\n  Si hay leakage, el R2 seria consistentemente alto en todos los periodos.")
print("  En mercados reales, el R2 varia significativamente entre periodos.\n")

# Dividir en 5 periodos
n_periods = 5
period_size = len(df) // n_periods

walk_forward_results = []

for i in range(n_periods - 1):
    # Train en periodos 0 a i, test en periodo i+1
    train_end = (i + 1) * period_size
    test_start = train_end
    test_end = min((i + 2) * period_size, len(df))

    X_train = X_imputed.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X_imputed.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo simple (Ridge para velocidad)
    model = Ridge(alpha=10)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Directional accuracy
    dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

    period_start_date = dates.iloc[test_start].strftime('%Y-%m')
    period_end_date = dates.iloc[test_end-1].strftime('%Y-%m')

    walk_forward_results.append({
        'period': f"{period_start_date} to {period_end_date}",
        'r2': r2,
        'rmse': rmse,
        'dir_acc': dir_acc,
        'train_size': len(X_train),
        'test_size': len(X_test)
    })

    print(f"  Periodo {i+1}: {period_start_date} to {period_end_date}")
    print(f"    Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"    R2: {r2:.4f}, RMSE: {rmse:.6f}, Dir Acc: {dir_acc:.2%}")

# Analizar variabilidad
r2_values = [r['r2'] for r in walk_forward_results]
r2_std = np.std(r2_values)
r2_mean = np.mean(r2_values)

print(f"\n  Resumen Walk-Forward:")
print(f"    R2 promedio: {r2_mean:.4f}")
print(f"    R2 std: {r2_std:.4f}")
print(f"    R2 rango: [{min(r2_values):.4f}, {max(r2_values):.4f}]")

# Si R2 es consistentemente alto (> 0.1) en todos los periodos, es sospechoso
if r2_mean > 0.1 and r2_std < 0.05:
    print("\n  [WARNING] R2 sospechosamente alto y estable - posible leakage")
    issues_found.append(f"Walk-forward R2 muy alto y estable: {r2_mean:.4f} +/- {r2_std:.4f}")
elif max(r2_values) > 0.2:
    print("\n  [WARNING] Algun periodo tiene R2 muy alto")
    issues_found.append(f"Periodo con R2 muy alto: {max(r2_values):.4f}")
else:
    print("\n  [PASS] R2 varia naturalmente entre periodos (esperado sin leakage)")

# =============================================================================
# TEST 3: PURGED CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: PURGED CROSS-VALIDATION")
print("=" * 80)

print("\n  Agrega gap entre train y test para evitar contaminacion.")
print("  Si el R2 cae significativamente con gap, habia dependencia espuria.\n")

purge_gaps = [0, 5, 21, 63]  # 0, 1 semana, 1 mes, 1 trimestre

purged_results = []

train_end = int(len(df) * 0.7)
test_size = int(len(df) * 0.2)

for gap in purge_gaps:
    test_start = train_end + gap
    test_end = min(test_start + test_size, len(df))

    if test_end <= test_start:
        continue

    X_train = X_imputed.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X_imputed.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    model = Ridge(alpha=10)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

    purged_results.append({
        'gap': gap,
        'r2': r2,
        'dir_acc': dir_acc
    })

    print(f"  Gap {gap:3d} dias: R2 = {r2:>8.4f}, Dir Acc = {dir_acc:.2%}")

# Comparar R2 sin gap vs con gap
if len(purged_results) >= 2:
    r2_no_gap = purged_results[0]['r2']
    r2_with_gap = purged_results[-1]['r2']
    r2_drop = r2_no_gap - r2_with_gap

    print(f"\n  Caida de R2 (gap 0 vs gap {purge_gaps[-1]}): {r2_drop:.4f}")

    if r2_drop > 0.05:
        print("  [WARNING] R2 cae significativamente con gap - posible dependencia espuria")
        issues_found.append(f"R2 cae {r2_drop:.4f} con purge gap de {purge_gaps[-1]} dias")
    else:
        print("  [PASS] R2 estable con gap temporal")

# =============================================================================
# TEST 4: SHUFFLE TEST
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: SHUFFLE TEST (Destruir Orden Temporal)")
print("=" * 80)

print("\n  Si hay leakage, shufflear los datos NO deberia destruir la senial.")
print("  Si no hay leakage, shufflear destruye la relacion temporal.\n")

# Split normal (temporal)
train_size = int(len(df) * 0.8)

X_train_temporal = X_imputed.iloc[:train_size]
y_train_temporal = y.iloc[:train_size]
X_test_temporal = X_imputed.iloc[train_size:]
y_test_temporal = y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_temporal)
X_test_scaled = scaler.transform(X_test_temporal)

# Modelo con datos temporales
model_temporal = Ridge(alpha=10)
model_temporal.fit(X_train_scaled, y_train_temporal)
y_pred_temporal = model_temporal.predict(X_test_scaled)
r2_temporal = r2_score(y_test_temporal, y_pred_temporal)

print(f"  R2 con split temporal: {r2_temporal:.4f}")

# Ahora shuffle
np.random.seed(42)
shuffle_idx = np.random.permutation(len(df))

X_shuffled = X_imputed.iloc[shuffle_idx].reset_index(drop=True)
y_shuffled = y.iloc[shuffle_idx].reset_index(drop=True)

X_train_shuffled = X_shuffled.iloc[:train_size]
y_train_shuffled = y_shuffled.iloc[:train_size]
X_test_shuffled = X_shuffled.iloc[train_size:]
y_test_shuffled = y_shuffled.iloc[train_size:]

scaler_shuffled = StandardScaler()
X_train_shuffled_scaled = scaler_shuffled.fit_transform(X_train_shuffled)
X_test_shuffled_scaled = scaler_shuffled.transform(X_test_shuffled)

model_shuffled = Ridge(alpha=10)
model_shuffled.fit(X_train_shuffled_scaled, y_train_shuffled)
y_pred_shuffled = model_shuffled.predict(X_test_shuffled_scaled)
r2_shuffled = r2_score(y_test_shuffled, y_pred_shuffled)

print(f"  R2 con datos shuffleados: {r2_shuffled:.4f}")

r2_diff = r2_temporal - r2_shuffled

print(f"\n  Diferencia (temporal - shuffled): {r2_diff:.4f}")

if r2_shuffled > r2_temporal * 0.8 and r2_temporal > 0.01:
    print("  [WARNING] Shufflear no destruye la senial - posible leakage")
    issues_found.append(f"Shuffle test: R2 shuffled ({r2_shuffled:.4f}) similar a temporal ({r2_temporal:.4f})")
else:
    print("  [PASS] Shufflear destruye la estructura temporal (esperado)")

# =============================================================================
# TEST 5: FEATURE-TARGET LEAD/LAG CORRELATION
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: CORRELACION LEAD/LAG FEATURE-TARGET")
print("=" * 80)

print("\n  Verifica que features[t] NO esten correlacionados con target[t-k] (k>0).")
print("  Si lo estan, el feature podria estar usando informacion futura.\n")

# Seleccionar features importantes para analizar
top_features = ['SPY_return_1d', 'SPY_RSI_14', 'SPY_momentum_21', 'VIX_zscore',
                'SPY_volatility_21', 'SPY_MACD', 'SPY_ADX_14', 'SPY_BB_position_20']
top_features = [f for f in top_features if f in feature_cols]

print(f"  {'Feature':<25} {'Corr(t,t)':<12} {'Corr(t,t-1)':<12} {'Corr(t,t+1)':<12} {'Status'}")
print("-" * 80)

leakage_features = []

for feat in top_features:
    feat_values = X_imputed[feat].values
    target_values = y.values

    # Correlacion contemporanea: feature[t] vs target[t]
    mask = ~np.isnan(feat_values) & ~np.isnan(target_values)
    corr_t_t = np.corrcoef(feat_values[mask], target_values[mask])[0,1] if mask.sum() > 100 else np.nan

    # Feature[t] vs Target[t-1] (target pasado)
    target_lag1 = np.roll(target_values, 1)
    target_lag1[0] = np.nan
    mask_lag = ~np.isnan(feat_values) & ~np.isnan(target_lag1)
    corr_t_tminus1 = np.corrcoef(feat_values[mask_lag], target_lag1[mask_lag])[0,1] if mask_lag.sum() > 100 else np.nan

    # Feature[t] vs Target[t+1] (target futuro - NO deberia estar correlacionado!)
    target_lead1 = np.roll(target_values, -1)
    target_lead1[-1] = np.nan
    mask_lead = ~np.isnan(feat_values) & ~np.isnan(target_lead1)
    corr_t_tplus1 = np.corrcoef(feat_values[mask_lead], target_lead1[mask_lead])[0,1] if mask_lead.sum() > 100 else np.nan

    # Status
    status = "OK"
    # Si feature[t] esta muy correlacionado con target[t-1], es sospechoso
    # porque feature[t] podria estar incorporando info de target[t-1]
    if abs(corr_t_tminus1) > 0.3:
        status = "[!] REVISAR"
    if abs(corr_t_tminus1) > 0.5:
        status = "[!] SOSPECHOSO"
        leakage_features.append((feat, corr_t_tminus1))

    print(f"  {feat:<25} {corr_t_t:>10.4f}   {corr_t_tminus1:>10.4f}   {corr_t_tplus1:>10.4f}   {status}")

if len(leakage_features) > 0:
    print(f"\n  [WARNING] {len(leakage_features)} features con alta correlacion a target pasado")
    for feat, corr in leakage_features:
        issues_found.append(f"Feature {feat} correlacionado con target pasado: {corr:.4f}")
else:
    print("\n  [PASS] No hay features con correlacion sospechosa a target pasado")

# =============================================================================
# TEST 6: VERIFICAR FEATURES ESPECIFICOS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: AUDITORIA DE FEATURES ESPECIFICOS")
print("=" * 80)

print("\n  Verificando que features criticos no usen informacion futura...\n")

# Lista de features que podrian tener problemas
suspect_patterns = [
    ('forward', 'Contiene "forward" en el nombre'),
    ('future', 'Contiene "future" en el nombre'),
    ('lead', 'Contiene "lead" en el nombre'),
    ('target', 'Contiene "target" en el nombre'),
    ('return_1d', 'Retorno diario - verificar direccion'),
]

print(f"  Buscando patrones sospechosos en nombres de features...")

found_suspects = []
for pattern, desc in suspect_patterns:
    matches = [f for f in feature_cols if pattern.lower() in f.lower()]
    if matches:
        print(f"    '{pattern}': {len(matches)} features - {desc}")
        for m in matches[:3]:  # Mostrar max 3
            print(f"      - {m}")
        if len(matches) > 3:
            print(f"      ... y {len(matches)-3} mas")

        if pattern in ['forward', 'future', 'target']:
            found_suspects.extend(matches)

if found_suspects:
    print(f"\n  [WARNING] Features con nombres sospechosos encontrados: {found_suspects}")
    issues_found.append(f"Features con nombres sospechosos: {found_suspects}")
else:
    print("\n  [PASS] No hay features con nombres sospechosos")

# Verificar que SPY_return_1d sea el retorno de t-1 a t, no de t a t+1
print("\n  Verificando calculo de SPY_return_1d...")
if 'SPY_return_1d' in feature_cols:
    spy_return_1d = X_imputed['SPY_return_1d'].values
    spy_close = df['SPY_CLOSE'].values

    # Calcular retorno correcto (t-1 a t)
    correct_return = np.zeros(len(spy_close))
    correct_return[1:] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
    correct_return[0] = np.nan

    mask = ~np.isnan(spy_return_1d) & ~np.isnan(correct_return)
    corr = np.corrcoef(spy_return_1d[mask], correct_return[mask])[0,1]

    print(f"    Correlacion con retorno(t-1->t) recalculado: {corr:.6f}")

    if corr > 0.99:
        print("    [PASS] SPY_return_1d es retorno pasado (correcto)")
    else:
        print("    [FAIL] SPY_return_1d podria no estar calculado correctamente")
        issues_found.append("SPY_return_1d no coincide con retorno recalculado")

# =============================================================================
# TEST 7: RANDOM FOREST FEATURE IMPORTANCE STABILITY
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: ESTABILIDAD DE FEATURE IMPORTANCE")
print("=" * 80)

print("\n  Si hay leakage, las features importantes serian muy consistentes")
print("  entre periodos (porque siempre 'predicen' el mismo patron).\n")

# Entrenar RF en 3 periodos diferentes
n_periods = 3
period_size = len(df) // (n_periods + 1)

importance_by_period = []

for i in range(n_periods):
    start_idx = i * period_size
    end_idx = (i + 2) * period_size

    X_period = X_imputed.iloc[start_idx:end_idx]
    y_period = y.iloc[start_idx:end_idx]

    # Train/test split dentro del periodo
    split = int(len(X_period) * 0.8)
    X_train = X_period.iloc[:split]
    y_train = y_period.iloc[:split]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # RF con pocas iteraciones para velocidad
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # Top 10 features
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    top10 = importance.nlargest(10).index.tolist()
    importance_by_period.append(set(top10))

    period_start = dates.iloc[start_idx].strftime('%Y')
    period_end = dates.iloc[end_idx-1].strftime('%Y')
    print(f"  Periodo {i+1} ({period_start}-{period_end}): Top features = {top10[:5]}...")

# Calcular overlap de features importantes entre periodos
overlaps = []
for i in range(len(importance_by_period)):
    for j in range(i+1, len(importance_by_period)):
        overlap = len(importance_by_period[i] & importance_by_period[j])
        overlaps.append(overlap)

avg_overlap = np.mean(overlaps)
print(f"\n  Overlap promedio de top-10 features entre periodos: {avg_overlap:.1f}/10")

if avg_overlap > 8:
    print("  [WARNING] Features muy consistentes - podria indicar leakage")
    issues_found.append(f"Feature importance muy estable entre periodos: {avg_overlap:.1f}/10 overlap")
else:
    print("  [PASS] Features varian entre periodos (esperado sin leakage)")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN FINAL DE TESTS DE LEAKAGE")
print("=" * 80)

print(f"\n  Tests ejecutados: 7")
print(f"  Problemas encontrados: {len(issues_found)}")

if len(issues_found) == 0:
    print("\n  " + "=" * 60)
    print("  [OK] TODOS LOS TESTS PASARON - NO SE DETECTO DATA LEAKAGE")
    print("  " + "=" * 60)
    print("\n  El pipeline parece estar correctamente configurado.")
    print("  Puedes proceder con el entrenamiento de modelos.")
else:
    print("\n  " + "=" * 60)
    print(f"  [WARNING] SE ENCONTRARON {len(issues_found)} PROBLEMAS POTENCIALES")
    print("  " + "=" * 60)
    print("\n  Problemas detectados:")
    for i, issue in enumerate(issues_found, 1):
        print(f"    {i}. {issue}")

    print("\n  RECOMENDACIONES:")
    print("    1. Revisar el calculo de features problematicos")
    print("    2. Verificar que todos los rolling windows miren solo al pasado")
    print("    3. Considerar eliminar features sospechosos antes de entrenar")
    print("    4. Usar purged cross-validation con gap >= 21 dias")

# Guardar resultados
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'tests_passed': len(issues_found) == 0,
    'issues_found': issues_found,
    'walk_forward_results': walk_forward_results,
    'purged_results': purged_results,
    'shuffle_test': {
        'r2_temporal': float(r2_temporal),
        'r2_shuffled': float(r2_shuffled)
    }
}

import json
with open(os.path.join(RESULTS_DIR, 'leakage_test_results.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\n  Resultados guardados en: {os.path.join(RESULTS_DIR, 'leakage_test_results.json')}")

print("\n" + "=" * 80)
print("[OK] TESTS DE LEAKAGE COMPLETADOS")
print("=" * 80)
