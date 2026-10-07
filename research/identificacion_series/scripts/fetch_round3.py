"""Ronda 3: ETFs sectoriales solo ajustados por split; valoracion SPX/SPY con mas campos; UX3/UX4."""
import pickle, sys
import blpapi
import pandas as pd
import bbg

S, E = "20231201", "20251212"
res = {}

def bdh_split_only(tickers):
    """Ajuste solo por split (sin dividendos)."""
    opts = blpapi.SessionOptions(); opts.setServerHost("localhost"); opts.setServerPort(8194)
    s = blpapi.Session(opts); s.start(); s.openService("//blp/refdata")
    req = s.getService("//blp/refdata").createRequest("HistoricalDataRequest")
    for t in tickers: req.getElement("securities").appendValue(t)
    req.getElement("fields").appendValue("PX_LAST")
    req.set("startDate", S); req.set("endDate", E)
    req.set("adjustmentNormal", False); req.set("adjustmentAbnormal", False)
    req.set("adjustmentSplit", True); req.set("adjustmentFollowDPDF", False)
    s.sendRequest(req); out = {}
    while True:
        ev = s.nextEvent(60000)
        for msg in ev:
            if not msg.hasElement("securityData"): continue
            sd = msg.getElement("securityData"); tk = sd.getElementAsString("security")
            fd = sd.getElement("fieldData"); d = {}
            for i in range(fd.numValues()):
                row = fd.getValueAsElement(i)
                if row.hasElement("PX_LAST"):
                    d[pd.Timestamp(row.getElementAsDatetime("date"))] = row.getElementAsFloat("PX_LAST")
            out[(tk, "PX_LAST")] = pd.Series(d, dtype=float)
        if ev.eventType() == blpapi.Event.RESPONSE: break
    s.stop(); return out

SECT = [f"{t} US Equity" for t in "XLK XLE XLU XLB XLY XLP XLV XLI XLF XLC XLRE".split()]
res["split"] = bdh_split_only(SECT)
print("split-only:", len(res["split"]))

FIELDS = ["PE_RATIO", "BEST_PE_RATIO", "PX_TO_BOOK_RATIO", "EARN_YLD", "BEST_PX_BPS_RATIO",
          "T12M_DIL_PE_CONT_OPS", "BEST_PE_NXT_YR", "PX_TO_SALES_RATIO", "DVD_PAYOUT_RATIO",
          "BEST_EARN_YLD", "INDX_ADJ_PE", "INDX_ADJ_PB", "INDX_GENERAL_EARN", "PX_TO_FREE_CASH_FLOW"]
for tk in ["SPX Index", "SPY US Equity", "INDU Index", "NDX Index"]:
    d, err = bbg.bdh([tk], FIELDS, S, E, adjust=None)
    res.setdefault("None", {}).update(d)
    print(tk, "errores:", err)
d, err = bbg.bdh(["UX3 Index", "UX4 Index", "UX5 Index", "VXN Index", "RVX Index", "VXD Index",
                  "VIX9D Index", "VXST Index", "VIX3M Index"], ["PX_LAST", "PX_OPEN", "PX_HIGH"], S, E, adjust=None)
res["None"].update(d); print("vol errores:", err)
d, err = bbg.bdh(["AAIIBULL Index", "AAIIBEAR Index", "AAIINEUT Index", "AAIIBULLBEAR Index",
                  "PCUSEQTR Index", "PCRTEQTY Index", "PCRTINDX Index", "PCRTTOTL Index",
                  "NYA Index", "NYMOAD Index", "NYADV Index", "NYDEC Index", "NYHGH Index", "NYLOW Index"],
                 ["PX_LAST"], S, E, adjust=None)
res["None"].update(d); print("sentimiento errores:", err)
pickle.dump(res, open(sys.argv[1], "wb"))
