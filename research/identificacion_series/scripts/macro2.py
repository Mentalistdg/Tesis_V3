"""Macro sin resolver: puntaje con tolerancia (|diff| <= 3% del rango) en fechas de periodo."""
import pickle, sys
import numpy as np
import pandas as pd
import bbg

RAW = r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv"
raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").loc["2015-01-01":"2025-12-12"]
COLS = ["CONSSENT Index", "USURTOT Index", "INJCJC Index", "IP CHNG Index", "Extra E15",
        "GDP CYOY Index", "CPTICHNG Index", "PITLCHNG Index", "GDP CQOQ Index", "ETSLTOTL Index"]
EXTRA = """NHCHATCH NHSPSTOT NHSPATOT NHSLTOT ETSLTOTL LEI TOTL PCE CYOY PCE DEFY PCE CRCH PCE CMOM PITLCHNG
PIDSDPS RSTAMOM RSTAXMOM RSTAXAG% NFP TCH NFP PCH ADP CHNG USEMNCHG CPTICHNG IP CHNG IP YOY GDP CYOY GDP CQOQ
GDPCTOT% NAPMPMI CHPMINDX MPMIUSMA MPMIUSSA MPMIUSCA SBOITOTL USHBMIDX CONSSENT CONCCONF CONSEXP CONSCURR
CPI XYOY CPI YOY CPUPXCHG CPI CHNG PPI YOY PPI CHNG FDIUFDYO FDIUSGYO DGNOCHNG CGNOXAI% TMNOCHNG USTBTOT
JOLTTOTL ECI SA% PRODNFR% COSTNFR% USCABAL USPHTMOM SPCS20Y% HPIMMOM% DFEDGBA KCLSIFNC STLFSI4 NFCIINDX
CFNAI USCRWTIC""".replace("LEI TOTL", "LEI~TOTL").replace("PCE CYOY", "PCE~CYOY").replace("PCE DEFY", "PCE~DEFY") \
    .replace("PCE CRCH", "PCE~CRCH").replace("PCE CMOM", "PCE~CMOM").replace("NFP TCH", "NFP~TCH") \
    .replace("NFP PCH", "NFP~PCH").replace("ADP CHNG", "ADP~CHNG").replace("IP CHNG", "IP~CHNG") \
    .replace("IP YOY", "IP~YOY").replace("GDP CYOY", "GDP~CYOY").replace("GDP CQOQ", "GDP~CQOQ") \
    .replace("CPI XYOY", "CPI~XYOY").replace("CPI YOY", "CPI~YOY").replace("CPI CHNG", "CPI~CHNG") \
    .replace("PPI YOY", "PPI~YOY").replace("PPI CHNG", "PPI~CHNG").replace("ECI SA%", "ECI~SA%").split()
TK = [t.replace("~", " ") + " Index" for t in EXTRA]

if len(sys.argv) > 2 and sys.argv[2] == "fetch":
    d, err = bbg.bdh(TK, ["PX_LAST"], "20140101", "20251212", adjust=None)
    print("errores:", err)
    pickle.dump(d, open(sys.argv[1], "wb"))
d = pickle.load(open(sys.argv[1], "rb"))
cands = {tk: s[~s.index.duplicated()].sort_index() for (tk, f), s in d.items() if len(s)}

for c in COLS:
    r = raw[c].dropna()
    tol = 0.03 * (r.max() - r.min())
    best = []
    for tk, s in cands.items():
        sr = s.reindex(r.index.union(s.index)).ffill().reindex(r.index)
        top = (0, 0, 0)
        for k in range(-5, 70, 1):
            x = sr.shift(k); m = x.notna()
            if m.sum() < 500: continue
            e = np.mean(np.abs(r[m] - x[m]) <= tol)
            if e > top[0]:
                top = (e, k, np.corrcoef(r[m], x[m])[0, 1])
        best.append((top[0], top[1], top[2], tk))
    best.sort(reverse=True)
    print(f"{c:18s} last={r.iloc[-1]:<9.6g} " + " | ".join(f"{t} {e:.0%} k={k} rho={p:.3f}" for e, k, p, t in best[:3]))
