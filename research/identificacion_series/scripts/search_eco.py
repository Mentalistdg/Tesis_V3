"""Busca tickers de indicadores economicos US via //blp/instruments y los guarda."""
import json, sys
import blpapi

QUERIES = """US PMI|ISM|ISM manufacturing|ISM services|ISM prices|ISM employment|ISM new orders|ISM inventories|
ISM supplier deliveries|ISM backlog|ISM export|ISM import|Chicago PMI|Empire State|Philadelphia Fed|Richmond Fed|
Kansas City Fed|Dallas Fed|Texas|Fed manufacturing|Fed services|NFIB|small business|optimism|Conference Board|
consumer confidence|present situation|expectations|labor differential|jobs plentiful|jobs hard to get|
University of Michigan|sentiment|current conditions|consumer expectations|inflation expectations|IBD TIPP|
economic optimism|Gallup|Langer|Bloomberg consumer comfort|CEO confidence|business roundtable|
durable goods|capital goods|factory orders|new orders|shipments|inventories|business inventories|wholesale|
retail inventories|industrial production|manufacturing production|capacity utilization|personal spending|
personal consumption|real personal|disposable income|personal income|PCE|vehicle sales|auto sales|
housing starts|building permits|new home sales|existing home sales|pending home sales|NAHB|construction spending|
mortgage applications|MBA|import price|export price|PPI|CPI|average hourly earnings|average weekly hours|
productivity|unit labor|labor costs|employment cost|JOLTS|quits|challenger|ADP|continuing claims|
leading index|coincident|lagging|CFNAI|ECRI|weekly leading|ADS|trade balance|current account|
GDP|GDP price|final sales|consumer credit|M2|money supply|budget|treasury|retail sales|control group|
Redbook|ICSC|chain store|freight|Cass|truck tonnage|rail|ATA|electricity|steel""".replace("\n", "").split("|")

opts = blpapi.SessionOptions(); opts.setServerHost("localhost"); opts.setServerPort(8194)
s = blpapi.Session(opts); s.start(); s.openService("//blp/instruments")
svc = s.getService("//blp/instruments")
found = {}
for q in QUERIES:
    req = svc.createRequest("instrumentListRequest")
    req.set("query", q.strip()); req.set("yellowKeyFilter", "YK_FILTER_INDX"); req.set("maxResults", 100)
    s.sendRequest(req)
    while True:
        ev = s.nextEvent(30000)
        for msg in ev:
            if msg.hasElement("results"):
                res = msg.getElement("results")
                for i in range(res.numValues()):
                    it = res.getValueAsElement(i)
                    sec = it.getElementAsString("security"); desc = it.getElementAsString("description")
                    found[sec.replace("<index>", "Index").replace("<Index>", "Index")] = desc
        if ev.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.TIMEOUT):
            break
s.stop()
# solo US
us = {k: v for k, v in found.items() if (" US " in f" {v} " or v.upper().startswith("US") or "U.S." in v or "United States" in v or "UNITED STATES" in v.upper())}
print("encontrados:", len(found), "| US:", len(us))
json.dump(us, open(sys.argv[1], "w"), indent=1)
for k, v in list(us.items())[:15]: print(" ", k, "|", v)
