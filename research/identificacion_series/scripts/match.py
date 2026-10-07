"""Paso 2: compara cada columna cruda contra todas las series descargadas.
Uso: python match.py <pickle1> [<pickle2> ...]   -> imprime top-3 por columna y guarda match.csv"""
import pickle, sys
import numpy as np
import pandas as pd

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date")
W0, W1 = "2023-12-01", "2025-12-12"
rawW = raw.loc[W0:W1]

cands = {}
for path in sys.argv[1:]:
    res = pickle.load(open(path, "rb"))
    for adj, data in res.items():
        for (tk, f), s in data.items():
            if len(s):
                s = s[~s.index.duplicated()].sort_index()
                cands[f"{tk}|{f}|adj={adj}"] = s

def score(r, s):
    """r: serie cruda (indice fechas crudas). s: candidato. Prueba desfases de -3..+3 dias habiles."""
    best = (0, 0, np.nan)
    sr = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
    for k in range(-3, 4):
        x = sr.shift(k)
        m = r.notna() & x.notna()
        if m.sum() < 20:
            continue
        a, b = r[m].values, x[m].values
        exact = np.mean(np.isclose(a, b, rtol=1e-4, atol=1e-6))
        if exact > best[0] or (exact == best[0] and best[1] != 0 and abs(k) < abs(best[1])):
            best = (exact, k, np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
    return best

rows = []
for c in raw.columns:
    r = rawW[c].dropna()
    if len(r) < 20:
        rows.append(dict(raw_col=c, n=len(r), best="(sin datos en ventana)", exact=np.nan))
        continue
    sc = sorted(((score(r, s), name) for name, s in cands.items()), key=lambda z: -z[0][0])[:3]
    (e, k, rho), name = sc[0]
    rows.append(dict(raw_col=c, n=len(r), best=name, exact=e, shift=k, corr=rho,
                     second=sc[1][1], exact2=sc[1][0][0]))
    print(f"{c:28s} -> {name:45s} exact={e:6.1%} shift={k:+d} | 2do {sc[1][1]:40s} {sc[1][0][0]:6.1%}")
pd.DataFrame(rows).to_csv(sys.argv[1].replace(".pkl", "_match.csv"), index=False)
