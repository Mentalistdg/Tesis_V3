"""Paso 3: para columnas sin calce exacto, ranking por correlacion de cambios diarios
y razon de niveles. Uso: python match2.py <match.csv> <pkl> [<pkl>...]"""
import pickle, sys
import numpy as np
import pandas as pd

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").loc["2023-12-01":"2025-12-12"]
done = pd.read_csv(sys.argv[1])
todo = done.loc[~(done["exact"] >= 0.99), "raw_col"].tolist()

cands = {}
for path in sys.argv[2:]:
    for adj, data in pickle.load(open(path, "rb")).items():
        for (tk, f), s in data.items():
            if len(s):
                cands[f"{tk}|{f}|adj={adj}"] = s[~s.index.duplicated()].sort_index()

for c in todo:
    r = raw[c].dropna()
    if len(r) < 20:
        print(f"{c:28s} (sin datos en ventana)"); continue
    dr = r.diff()
    res = []
    for name, s in cands.items():
        x = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
        m = x.notna()
        if m.sum() < 20: continue
        dx = x.diff()
        mm = dr.notna() & dx.notna() & (dr != 0)
        if mm.sum() < 15 or dx[mm].std() == 0: continue
        rho = np.corrcoef(dr[mm], dx[mm])[0, 1]
        ratio = (r[m] / x[m]).replace([np.inf, -np.inf], np.nan).dropna()
        res.append((rho, name, ratio.median(), ratio.std() / abs(ratio.median()) if len(ratio) else np.nan))
    res.sort(key=lambda z: -abs(z[0]))
    top = " | ".join(f"{n.split('|')[0]}[{n.split('=')[1]}] rho={p:.3f} ratio={q:.4g}±{v:.1%}" for p, n, q, v in res[:2])
    print(f"{c:28s} last={r.iloc[-1]:<10.6g} {top}")
