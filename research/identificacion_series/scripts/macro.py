"""Macro: descarga candidatos (5 anos) y busca el desfase (dias habiles) que maximiza calce exacto.
Tambien prueba alinear por fecha de publicacion (ECO_RELEASE_DT)."""
import pickle, sys
import numpy as np
import pandas as pd
import bbg

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").loc["2021-01-01":"2025-12-12"]
COLS = ["SPY US Equity", "QQQ US Equity", "GDP CQOQ Index", "NAPMPMI Index", "CONSSENT Index",
        "NFP TCH Index", "USURTOT Index", "CPI YOY Index", "PCE CRCH Index", "RSTAMOM Index",
        "LEI TOTL Index", "INJCJC Index", "NHSPSTOT Index", "IP CHNG Index", "CONCCONF Index",
        "NAPMNMI Index", "Extra E15", "ETSLTOTL Index", "GDP CYOY Index", "CPTICHNG Index",
        "PITLCHNG Index", "AAII BULLISH Index"]
TK = """NAPMPMI NAPMNMI CONSSENT CONCCONF USURTOT NFP\\ TCH NFP\\ PCH NFP\\ P CPI\\ YOY CPI\\ XYOY CPI\\ CHNG CPUPXCHG
PCE\\ CRCH PCE\\ CYOY PCE\\ DEFY PCE\\ CMOM RSTAMOM RSTAXMOM LEI\\ TOTL LEI\\ CHNG INJCJC INJCSP NHSPSTOT NHSPATOT
NHSLTOT ETSLTOTL IP\\ CHNG IP\\ YOY CPTICHNG PITLCHNG PIDSDPS GDP\\ CQOQ GDP\\ CYOY PPI\\ YOY PPI\\ CHNG FDIUFDYO
AHE\\ YOY USHEYOY MPMIUSMA MPMIUSSA MPMIUSCA CHPMINDX EMPRGBCI OUTFGAF NAPMNEWO NAPMEMPL NAPMPRIC SBOITOTL
USHBMIDX NHCHATCH DGNOCHNG TMNOCHNG CONSEXP CONSCURR COSTLIER JOLTTOTL ADP\\ CHNG USEMNCHG""".replace("\\ ", "~").split()
TK = [t.replace("~", " ") + " Index" for t in TK]

if len(sys.argv) > 2 and sys.argv[2] == "fetch":
    d, err = bbg.bdh(TK, ["PX_LAST"], "20200101", "20251212", adjust=None)
    print("errores:", err)
    pickle.dump(d, open(sys.argv[1], "wb"))
d = pickle.load(open(sys.argv[1], "rb"))
cands = {tk: s[~s.index.duplicated()].sort_index() for (tk, f), s in d.items() if len(s)}

for c in COLS:
    r = raw[c].dropna()
    best = []
    for tk, s in cands.items():
        sr = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
        top = (0, 0)
        for k in range(-5, 70):
            x = sr.shift(k); m = x.notna()
            if m.sum() < 100: continue
            e = np.mean(np.isclose(r[m], x[m], rtol=1e-4, atol=1e-6))
            if e > top[0]: top = (e, k)
        best.append((top[0], top[1], tk))
    best.sort(reverse=True)
    print(f"{c:22s} last={r.iloc[-1]:<9.6g} " + " | ".join(f"{t} {e:.1%} k={k}" for e, k, t in best[:3]))
