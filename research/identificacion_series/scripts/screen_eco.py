"""Tamiza candidatos 2018-2021 contra E3 (col CONSSENT) y E19 (col PITLCHNG)."""
import json, pickle, re, sys, os
import numpy as np
import pandas as pd
import bbg

cands = json.load(open("eco_search.json"))
SKIP = re.compile(r"Kalshi|Polymarket|Crude Oil|OSP|Forecast|Fcst|Survey Median|Consensus|BofA|Option|Future|"
                  r"ETF|Fund|Swap|Bond|Yield|Spread|Rate Prob|Probability", re.I)
tks = sorted({re.sub(r"Index$", " Index", k).replace("  ", " ") for k, v in cands.items() if not SKIP.search(v)})
print("candidatos tras filtro:", len(tks))

cache = "screen.pkl"
if os.path.exists(cache):
    data = pickle.load(open(cache, "rb"))
else:
    data = {}
    for i in range(0, len(tks), 100):
        d, err = bbg.bdh(tks[i:i+100], ["PX_LAST"], "20171201", "20211231", adjust=None, timeout_ms=120000)
        for (tk, f), s in d.items():
            if len(s): data[tk] = s[~s.index.duplicated()].sort_index()
        print(f"  lote {i//100+1}: acumulado {len(data)} series")
    pickle.dump(data, open(cache, "wb"))

raw = pd.read_csv(r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv", parse_dates=["date"]).set_index("date")
win = raw.loc["2018-01-01":"2021-12-31"]
for col in ["CONSSENT Index", "PITLCHNG Index"]:
    r = win[col]
    res = []
    for tk, s in data.items():
        x = s.reindex(win.index).ffill()            # regla del CSV: fecha exacta + ffill
        m = r.notna() & x.notna()
        if m.sum() < 400 or x[m].std() == 0: continue
        exact = np.isclose(r[m], x[m], rtol=1e-4, atol=1e-6).mean()
        near = np.isclose(r[m], x[m], rtol=0, atol=0.15).mean()
        rho = np.corrcoef(r[m], x[m])[0, 1]
        res.append((exact, near, rho, tk))
    res.sort(key=lambda z: (-z[0], -z[1], -abs(z[2])))
    print(f"\n== {col}: top por calce exacto")
    for e, n, p, tk in res[:8]:
        print(f"   {tk:22s} exacto={e:6.1%} |dif|<=0.15: {n:6.1%} rho={p:+.3f}  {cands.get(tk.replace(' Index','Index'), '')[:70]}")
    print(f"== {col}: top por correlacion")
    for e, n, p, tk in sorted(res, key=lambda z: -abs(z[2]))[:6]:
        print(f"   {tk:22s} exacto={e:6.1%} |dif|<=0.15: {n:6.1%} rho={p:+.3f}  {cands.get(tk.replace(' Index','Index'), '')[:70]}")
