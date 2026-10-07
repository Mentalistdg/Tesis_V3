"""Paso 1: descarga ~2 anos de los tickers del encabezado (sin ajuste y default)."""
import pickle, sys
import pandas as pd
from bbg import bdh

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
cols = pd.read_csv(RAW, nrows=0).columns[1:]
FIX = {"Extra E15": "NHSPATOT Index",
       "CDX IG CDSI GEN 5Y": "CDX IG CDSI GEN 5Y Corp",
       "CDX HY CDSI GEN 5Y": "CDX HY CDSI GEN 5Y Corp"}
tickers = []
for c in cols:
    t = c.split(" (")[0]
    t = FIX.get(t, t)
    if t not in tickers:
        tickers.append(t)
print(len(tickers), "tickers")

res = {}
for adj in (False, None):
    data, err = bdh(tickers, ["PX_LAST"], "20231201", "20251212", adjust=adj)
    res[str(adj)] = data
    print("adjust", adj, "series:", sum(len(v) > 0 for v in data.values()), "errores:", err)
pickle.dump(res, open(sys.argv[1], "wb"))
