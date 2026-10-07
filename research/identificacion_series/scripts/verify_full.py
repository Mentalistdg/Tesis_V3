"""Verificacion historia completa: descarga cada serie del mapeo (2000-2025) y compara con el CSV crudo."""
import pickle, sys, os
import numpy as np
import pandas as pd
import bbg
from mapping import MAPPING

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date")
ADJ = {"none": False, "default": None, "split": "split", "all": True}
cache = sys.argv[1]

if not os.path.exists(cache):
    groups = {}
    for col, (tk, f, adj, kind) in MAPPING.items():
        if tk: groups.setdefault((adj, f), set()).add(tk)
    data = {}
    for (adj, f), tks in groups.items():
        tks = sorted(tks)
        for i in range(0, len(tks), 25):
            d, err = bbg.bdh(tks[i:i+25], [f], "19991201", "20251212", adjust=ADJ[adj])
            for (tk, ff), s in d.items():
                data[(tk, ff, adj)] = s[~s.index.duplicated()].sort_index()
            if err: print("errores", adj, f, err)
    pickle.dump(data, open(cache, "wb"))
data = pickle.load(open(cache, "rb"))

rows = []
for col, (tk, f, adj, kind) in MAPPING.items():
    r = raw[col].dropna()
    if not tk:
        rows.append(dict(columna=col, ticker="(sin identificar)", tipo=kind, n=len(r))); continue
    s = data.get((tk, f, adj))
    if s is None or not len(s):
        rows.append(dict(columna=col, ticker=tk, campo=f, tipo=kind, n=len(r), nota="sin datos")); continue
    if kind == "derivada":
        s = s.diff()
    x = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
    m = x.notna()
    a, b = r[m].values, x[m].values
    exact = np.mean(np.isclose(a, b, rtol=1e-4, atol=1e-6))
    near = np.mean(np.isclose(a, b, rtol=5e-3, atol=1e-3))
    rho = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
    first_bad = r[m].index[~np.isclose(a, b, rtol=1e-4, atol=1e-6)]
    rows.append(dict(columna=col, ticker=tk, campo=f, ajuste=adj, tipo=kind, n=len(r), n_comp=int(m.sum()),
                     exacto=exact, dentro_05pct=near, rho=rho,
                     primer_desvio=first_bad.min().date() if len(first_bad) else None,
                     ultimo_desvio=first_bad.max().date() if len(first_bad) else None))
df = pd.DataFrame(rows)
df.to_csv(cache.replace(".pkl", "_report.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.to_string(float_format=lambda v: f"{v:.4f}"))
