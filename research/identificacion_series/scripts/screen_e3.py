"""Segunda busqueda para E3 (col CONSSENT): sin filtro 'US', ventana 2008-2010; luego verifica E19 completo."""
import json, pickle, re, os
import blpapi
import numpy as np
import pandas as pd
import bbg

QUERIES = """ISM|ISM manufacturing|ISM services|ISM non-manufacturing|ISM prices paid|ISM employment|ISM new orders|
ISM inventories|ISM customers inventories|ISM supplier deliveries|ISM backlog|ISM new export|ISM imports|
ISM business activity|ISM inventory sentiment|NAPM|Institute for Supply|Conference Board|consumer confidence|
University of Michigan|Michigan sentiment|Michigan expectations|Michigan current|NFIB|small business optimism|
IBD|TIPP|economic optimism|diffusion index|diffusion|Chicago purchasing|Milwaukee|Kansas City|Richmond|
Philadelphia|Empire|Dallas|Atlanta Fed|Cleveland Fed|St Louis Fed|Chicago Fed|employment trends|
help wanted|Monster|Manpower|hiring|ADP|Challenger|Gallup|Rasmussen|ABC|Langer|Bloomberg comfort|
homebuilder|NAHB|mortgage|Fannie Mae|purchasing managers|composite|business conditions|outlook|
CEO|CFO|Duke|Business Roundtable|NABE|Beige|Senior Loan Officer|lending standards|credit conditions""".replace("\n", "").split("|")

if not os.path.exists("eco_search2.json"):
    opts = blpapi.SessionOptions(); opts.setServerHost("localhost"); opts.setServerPort(8194)
    s = blpapi.Session(opts); s.start(); s.openService("//blp/instruments")
    svc = s.getService("//blp/instruments"); found = {}
    for q in QUERIES:
        req = svc.createRequest("instrumentListRequest")
        req.set("query", q.strip()); req.set("yellowKeyFilter", "YK_FILTER_INDX"); req.set("maxResults", 300)
        s.sendRequest(req)
        while True:
            ev = s.nextEvent(30000)
            for msg in ev:
                if msg.hasElement("results"):
                    res = msg.getElement("results")
                    for i in range(res.numValues()):
                        it = res.getValueAsElement(i)
                        found[re.sub(r"<[Ii]ndex>$", " Index", it.getElementAsString("security"))] = it.getElementAsString("description")
            if ev.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.TIMEOUT): break
    s.stop()
    json.dump(found, open("eco_search2.json", "w"), indent=1)
found = json.load(open("eco_search2.json"))
SKIP = re.compile(r"Kalshi|Polymarket|Forecast|Fcst|Option|Future|ETF|Fund|Swap|Bond", re.I)
tested = set(pickle.load(open("screen.pkl", "rb")).keys())
tks = sorted(k for k, v in found.items() if not SKIP.search(v) and k not in tested)
print("encontrados:", len(found), "| nuevos a probar:", len(tks))

cache = "screen_e3.pkl"
if os.path.exists(cache):
    data = pickle.load(open(cache, "rb"))
else:
    data = {}
    for i in range(0, len(tks), 150):
        d, err = bbg.bdh(tks[i:i+150], ["PX_LAST"], "20071201", "20101231", adjust=None, timeout_ms=120000)
        for (tk, f), s in d.items():
            if len(s): data[tk] = s[~s.index.duplicated()].sort_index()
    pickle.dump(data, open(cache, "wb"))
print("series descargadas:", len(data))

raw = pd.read_csv(r"C:\Users\itau_lab\Tesis_V3\data\BLOOMBERG_RAW_DATA.csv", parse_dates=["date"]).set_index("date")
win = raw.loc["2008-01-01":"2010-12-31"]; r = win["CONSSENT Index"]
res = []
for tk, s in data.items():
    x = s.reindex(win.index).ffill(); m = r.notna() & x.notna()
    if m.sum() < 300 or x[m].std() == 0: continue
    res.append((np.isclose(r[m], x[m], rtol=1e-4, atol=1e-6).mean(), np.corrcoef(r[m], x[m])[0, 1], tk))
res.sort(key=lambda z: (-z[0], -abs(z[1])))
print("== E3 top por calce exacto (2008-2010)")
for e, p, tk in res[:10]: print(f"   {tk:22s} exacto={e:6.1%} rho={p:+.3f}  {found.get(tk, '')[:80]}")
print("== E3 top por correlacion")
for e, p, tk in sorted(res, key=lambda z: -abs(z[1]))[:6]: print(f"   {tk:22s} exacto={e:6.1%} rho={p:+.3f}  {found.get(tk, '')[:80]}")

# E19 historia completa con la regla del CSV
d, err = bbg.bdh(["DGNOXTCH Index"], ["PX_LAST"], "19991201", "20251212", adjust=None)
s = d[("DGNOXTCH Index", "PX_LAST")]; s = s[~s.index.duplicated()]
r19 = raw["PITLCHNG Index"]; x = s.reindex(raw.index).ffill(); m = r19.notna() & x.notna()
print(f"\nE19 vs DGNOXTCH 2000-2025: n={m.sum()} exacto={np.isclose(r19[m], x[m], rtol=1e-4, atol=1e-6).mean():.1%} "
      f"|dif|<=0.15: {np.isclose(r19[m], x[m], rtol=0, atol=0.15).mean():.1%} rho={np.corrcoef(r19[m], x[m])[0,1]:.4f}")
