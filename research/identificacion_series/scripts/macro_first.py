"""Primera publicacion (ACTUAL_RELEASE) + fecha de publicacion (ECO_RELEASE_DT) vs columnas crudas."""
import pickle, sys
import numpy as np
import pandas as pd
import bbg

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").loc["2015-01-01":"2025-12-12"]
COLS = ["SPY US Equity", "QQQ US Equity", "GDP CQOQ Index", "NAPMPMI Index", "CONSSENT Index",
        "NFP TCH Index", "USURTOT Index", "CPI YOY Index", "PCE CRCH Index", "RSTAMOM Index",
        "LEI TOTL Index", "INJCJC Index", "NHSPSTOT Index", "IP CHNG Index", "CONCCONF Index",
        "NAPMNMI Index", "Extra E15", "ETSLTOTL Index", "GDP CYOY Index", "CPTICHNG Index", "PITLCHNG Index"]
d2 = pickle.load(open("macro2.pkl", "rb")); d1 = pickle.load(open("macro.pkl", "rb"))
TK = sorted({tk for (tk, f), s in list(d1.items()) + list(d2.items()) if len(s)})

if len(sys.argv) > 2 and sys.argv[2] == "fetch":
    d, err = bbg.bdh(TK, ["ACTUAL_RELEASE", "PX_LAST"], "20140101", "20251212", adjust=None)
    print("errores:", {k: v for k, v in err.items()})
    pickle.dump(d, open(sys.argv[1], "wb"))
d = pickle.load(open(sys.argv[1], "rb"))
n_ar = sum(1 for (tk, f), s in d.items() if f == "ACTUAL_RELEASE" and len(s))
print("series con ACTUAL_RELEASE:", n_ar)
cands = {f"{tk}|{f}": s[~s.index.duplicated()].sort_index() for (tk, f), s in d.items() if len(s)}

for c in COLS:
    r = raw[c].dropna()
    best = []
    for name, s in cands.items():
        sr = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
        top = (0, 0)
        for k in range(-3, 45):
            x = sr.shift(k); m = x.notna()
            if m.sum() < 500: continue
            e = np.mean(np.isclose(r[m], x[m], rtol=1e-4, atol=1e-6))
            if e > top[0]: top = (e, k)
        best.append((top[0], top[1], name))
    best.sort(reverse=True)
    print(f"{c:18s} " + " | ".join(f"{t} {e:.1%} k={k}" for e, k, t in best[:3]))
