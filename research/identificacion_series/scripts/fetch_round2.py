"""Ronda 2 de candidatos."""
import pickle, sys
from bbg import bdh

S, E = "20231201", "20251212"
res = {}

def add(tag, tickers, fields, adjust):
    data, err = bdh(tickers, fields, S, E, adjust=adjust)
    res.setdefault(tag, {}).update(data)
    bad = {k: v for k, v in err.items()}
    print(tag, len(tickers), "tickers x", len(fields), "campos | errores:", bad)

# SPY OHLCV
add("False", ["SPY US Equity"], ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"], False)
add("None", ["SPY US Equity"], ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"], None)

# Valoracion
add("None", ["SPX Index", "SPY US Equity"],
    ["PE_RATIO", "BEST_PE_RATIO", "PX_TO_BOOK_RATIO", "EARN_YLD", "PX_TO_SALES_RATIO",
     "EQY_DVD_YLD_12M", "BEST_EPS", "TRAIL_12M_EPS", "PX_TO_CASH_FLOW", "CURRENT_EV_TO_T12M_EBITDA"], None)

ETFS = """DIA MDY IWF IWD IJR IJH VBR VTV VUG RSP XLC XLY XLP XLB XLRE VNQ IYR TLT IEF SHY HYG JNK AGG BND
TIP EMB SMH SOXX IGV XBI IBB KRE KBE XHB ITB XRT GDX GDXJ UNG BNO UUP FXE FXY EWZ EWG EWU INDA KWEB
MCHI VWO IEFA IEMG ACWI VT VTI VOO IVV SPYG SPYV IWB IWN IWO IVE IVW MTUM QUAL USMV VIG SCHD VXX UVXY
SVXY SH SDS SPXU SQQQ TQQQ UPRO SSO IWV OEF MGK VO VB VEA EWC EWA EWY EWT EWH IAU PPLT PALL CPER""".split()
ETFS = [f"{t} US Equity" for t in ETFS]
add("False", ETFS, ["PX_LAST"], False)
add("True", ETFS + ["XLK US Equity", "XLE US Equity", "XLU US Equity", "XLF US Equity", "XLV US Equity",
                    "XLI US Equity", "EFA US Equity", "EEM US Equity", "VGK US Equity", "EWJ US Equity",
                    "FXI US Equity", "QQQ US Equity", "IWM US Equity", "SPY US Equity", "GLD US Equity",
                    "SLV US Equity", "USO US Equity", "DBC US Equity", "DBA US Equity", "LQD US Equity"],
    ["PX_LAST"], True)

IDX = """VXN RVX VVIX SKEW VIX9D VIX6M VXD VXO VXSLV VXGDX VXEWZ VXAPL VXTLT USGG3M USGG6M USGG12M USGG5YR
USGG30YR USGG1M LUACOAS LUACYW LF98YW CSI BARC SPX NDX RTY INDU CCMP MID SML BCOM SPGSCITR SPGSCI
BCOMCL BCOMGC DXY USCRWTIC VIX1D SPXEW RAY RLG RLV CESIUSD SPUSD BDIY MXWD MXEF MXEA NKY SX5E UKX DAX
HSI SHCOMP""".split()
add("None", [f"{t} Index" for t in IDX], ["PX_LAST"], None)
add("None", ["CL1 Comdty", "CO1 Comdty", "NG1 Comdty", "HG1 Comdty", "GC1 Comdty", "SI1 Comdty",
             "UX1 Index", "UX2 Index", "UX3 Index", "UX4 Index", "TY1 Comdty", "ES1 Index",
             "EURUSD Curncy", "USDJPY Curncy", "GBPUSD Curncy", "USDCNH Curncy", "XAU Curncy", "BTC Curncy"],
    ["PX_LAST"], None)
pickle.dump(res, open(sys.argv[1], "wb"))
