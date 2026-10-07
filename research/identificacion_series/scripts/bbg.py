"""Helper minimo para HistoricalDataRequest via blpapi (Terminal local)."""
import blpapi
import pandas as pd


def bdh(tickers, fields, start, end, adjust=None, periodicity="DAILY", timeout_ms=60000):
    """Devuelve dict {(ticker, field): pd.Series}. adjust: None=default terminal,
    False=sin ajustes, True=todos los ajustes."""
    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    s = blpapi.Session(opts)
    if not s.start() or not s.openService("//blp/refdata"):
        raise RuntimeError("No se pudo conectar a Bloomberg")
    svc = s.getService("//blp/refdata")
    req = svc.createRequest("HistoricalDataRequest")
    for t in tickers:
        req.getElement("securities").appendValue(t)
    for f in fields:
        req.getElement("fields").appendValue(f)
    req.set("startDate", start)
    req.set("endDate", end)
    req.set("periodicitySelection", periodicity)
    if adjust == "split":
        req.set("adjustmentNormal", False); req.set("adjustmentAbnormal", False)
        req.set("adjustmentSplit", True); req.set("adjustmentFollowDPDF", False)
    elif adjust is not None:
        for k in ("adjustmentNormal", "adjustmentAbnormal", "adjustmentSplit"):
            req.set(k, bool(adjust))
        req.set("adjustmentFollowDPDF", False)
    s.sendRequest(req)
    out, errors = {}, {}
    while True:
        ev = s.nextEvent(timeout_ms)
        for msg in ev:
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            tk = sd.getElementAsString("security")
            if sd.hasElement("securityError"):
                errors[tk] = sd.getElement("securityError").getElementAsString("message")
                continue
            if sd.hasElement("fieldExceptions") and sd.getElement("fieldExceptions").numValues():
                fe = sd.getElement("fieldExceptions")
                errors[tk] = "; ".join(fe.getValueAsElement(i).getElement("errorInfo")
                                       .getElementAsString("message") for i in range(fe.numValues()))
            fd = sd.getElement("fieldData")
            rows = {f: {} for f in fields}
            for i in range(fd.numValues()):
                row = fd.getValueAsElement(i)
                d = pd.Timestamp(row.getElementAsDatetime("date"))
                for f in fields:
                    if row.hasElement(f):
                        rows[f][d] = row.getElementAsFloat(f)
            for f in fields:
                prev = out.get((tk, f))
                ser = pd.Series(rows[f], dtype=float)
                out[(tk, f)] = ser if prev is None else pd.concat([prev, ser])
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
        if ev.eventType() == blpapi.Event.TIMEOUT:
            errors["__timeout__"] = "timeout"
            break
    s.stop()
    return out, errors
