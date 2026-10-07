"""Corre produccion_lstm.main() usando otro CSV de entrada (sin modificar el paquete)."""
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("p", r"E:\CRONOS_PRODUCCION\produccion_lstm.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
p.DATA_FILE = sys.argv[1]
print("DATA_FILE =", p.DATA_FILE)
p.main()
