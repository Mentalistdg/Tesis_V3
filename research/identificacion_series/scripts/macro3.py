import pickle, numpy as np, pandas as pd
raw = pd.read_csv(r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv", parse_dates=["date"]).set_index("date").loc["2005-01-01":"2025-12-12"]
d = {}
for p in ["macro.pkl", "macro2.pkl"]:
    d.update(pickle.load(open(p, "rb")))
cands = {tk: s[~s.index.duplicated()].sort_index() for (tk, f), s in d.items() if len(s)}
for c in ["USURTOT Index", "IP CHNG Index", "PITLCHNG Index", "CONSSENT Index"]:
    r = raw[c].dropna().resample("ME").last()
    res = []
    for tk, s in cands.items():
        x = s.resample("ME").last()
        for lag in (0, 1, -1):
            j = pd.concat([r, x.shift(lag)], axis=1).dropna()
            if len(j) < 60 or j.iloc[:, 1].std() == 0: continue
            res.append((j.corr().iloc[0, 1], lag, tk, len(j), np.mean(np.isclose(j.iloc[:,0], j.iloc[:,1], rtol=1e-3))))
    res.sort(key=lambda z: -abs(z[0]))
    print(f"{c:16s} rango {r.min():.4g}..{r.max():.4g} desde {raw[c].first_valid_index().date()} | " +
          " | ".join(f"{t} rho={p:.3f} lag={l} n={n} exact={e:.0%}" for p, l, t, n, e in res[:3]))
