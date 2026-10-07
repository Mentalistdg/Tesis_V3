"""Genera activos/spx/series.csv desde el mapeo verificado (mapping.py) y COLUMN_MAPPING de la tesis."""
import ast, re, sys
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(AQUI))
from mapping import MAPPING  # noqa: E402

src = Path(r"C:\Users\itau_lab\Tesis_V3\scripts\build_dataset.py").read_text(encoding="utf-8")
bloque = re.search(r"COLUMN_MAPPING = (\{.*?\n\})", src, re.S).group(1)
CODIGOS = ast.literal_eval(bloque)

# Mediana observada (ECO_RELEASE_DT - periodo) en dias, medida el 2026-10-07; PIB y LEI con rezago fijo.
REZAGO = {"CONCCONF Index": -3, "CONSSENT Index": -3, "CPI YOY Index": 15, "CPMINDX Index": 0,
          "CPUPXCHG Index": 15, "DGNOXTCH Index": 28, "ETSLTOTL Index": 22, "GDP CQOQ Index": 30,
          "INJCJC Index": 6, "INJCSP Index": 13, "IP CHNG Index": 16, "LEI TOTL Index": 20,
          "NAPMNMI Index": 5, "NAPMPMI Index": 1, "NFP TCH Index": 5, "NHSPATOT Index": 18,
          "NHSPSTOT Index": 18, "PCE CYOY Index": 29, "PPI YOY Index": 15, "RSTAMOM Index": 14,
          "USURTOT Index": 5}
SOLO_REZAGO = {"GDP CQOQ Index", "LEI TOTL Index"}
TARDIAS = {"LF98OAS Index", "LF98TRUU Index", "LUACTRUU Index", "LUACOAS Index", "MOVE Index",
           "CVIX Index", "SKEW Index", "PCUSEQTR Index"}

cols = list(pd.read_csv(RAIZ / "activos/spx/historia_congelada.csv", nrows=0).columns[1:])
filas = []
for c in cols:
    tk, campo, ajuste, kind = MAPPING[c]
    if kind == "fundamental":
        tipo = "fundamental"
    elif kind == "macro":
        tipo = "macro"
    elif c == "SPY US Equity (VOLUME)":
        tipo = "volumen"
    elif tk.endswith(" US Equity"):
        tipo = "precio"
    else:
        tipo = "nivel"
    if tipo == "precio" and c.startswith("SPY US Equity ("):
        ajuste = "none"
    elif tipo == "precio":
        ajuste = "none" if tk == "IWF US Equity" else "split"
    elif ajuste == "all":
        ajuste = "split"
    filas.append(dict(
        columna_cruda=c, codigo=CODIGOS.get(c, c), ticker=tk, campo=campo, ajuste=ajuste, tipo=tipo,
        relleno="ffill" if tipo in ("macro", "fundamental") else "no",
        fecha_pub=("rezago" if tk in SOLO_REZAGO else "eco") if tipo == "macro" else "",
        rezago_pub_dias=REZAGO.get(tk) if tipo == "macro" else None,
        publica_tarde="si" if tk in TARDIAS else "no",
        notas="verificado 2026-10-07 contra historia 2000-2025",
    ))
df = pd.DataFrame(filas)
df["rezago_pub_dias"] = df["rezago_pub_dias"].astype("Int64")
df.to_csv(RAIZ / "activos/spx/series.csv", index=False, encoding="utf-8")
print(df.tipo.value_counts().to_dict(), "| tardias:", (df.publica_tarde == "si").sum())
