"""Compara dataset reconstruido vs CSV produccion, CSV repo y estadisticas del scaler."""
import sys, importlib.util
import numpy as np
import pandas as pd
import joblib

rebuilt_path, label = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
reb = pd.read_csv(rebuilt_path, low_memory=False)
prod = pd.read_csv(r"E:\CRONOS_PRODUCCION\data\bloomberg_triple_screen_core.csv", low_memory=False)
repo = pd.read_csv(r"C:\Users\itau_lab\Tesis_V3\data\bloomberg_triple_screen_core.csv", low_memory=False)
spec = importlib.util.spec_from_file_location("p", r"E:\CRONOS_PRODUCCION\produccion_lstm.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
pre = joblib.load(r"E:\CRONOS_PRODUCCION\models\preprocessors.joblib")
fc, sc, imp = pre["feature_cols"], pre["scaler"], pre["imputer"]

print(f"[{label}] reconstruido {reb.shape} | produccion {prod.shape} | repo {repo.shape}")
print("  fechas iguales a produccion:", reb["date"].equals(prod["date"]))
for name, ref in [("produccion", prod), ("repo", repo)]:
    common = [c for c in ref.columns if c in reb.columns and c not in ("date",)]
    only_ref = [c for c in ref.columns if c not in reb.columns]
    only_reb = [c for c in reb.columns if c not in ref.columns]
    diff = []
    for c in common:
        a, b = reb[c].astype(float), ref[c].astype(float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            diff.append((c, int((~same).sum())))
    print(f"  vs {name}: {len(common)} columnas comunes, {len(common)-len(diff)} identicas, {len(diff)} distintas;"
          f" solo en {name}: {only_ref}; solo en reconstruido: {only_reb[:10]}{'...' if len(only_reb)>10 else ''}")
    for c, n in diff[:25]:
        print(f"      {c:32s} filas distintas: {n}")

# scaler: estadisticas de train con el reconstruido (nombres actuales via LEGACY_MAP)
df = reb.dropna(subset=[p.TARGET_COL]).reset_index(drop=True)
tr = df.iloc[:int(len(df) * 0.8)]
bad = []
for i, c in enumerate(fc):
    src = p.LEGACY_MAP.get(c, c)
    if src not in tr.columns:
        bad.append((c, "FALTA")); continue
    x = tr[src].astype(float).fillna(imp.statistics_[i])
    if not (np.isclose(x.mean(), sc.mean_[i], rtol=1e-7, atol=1e-10) and np.isclose(x.std(ddof=0), sc.scale_[i], rtol=1e-7, atol=1e-10)):
        bad.append((c, f"media {x.mean():.6g} vs {sc.mean_[i]:.6g}"))
print(f"  scaler: {len(fc)-len(bad)}/{len(fc)} features con estadisticas de entrenamiento identicas")
for b in bad[:20]: print("     ", b)
