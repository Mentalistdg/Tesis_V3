# -*- coding: utf-8 -*-
"""
Audit trades data for date inconsistencies
"""
import json
from collections import Counter

# Load data
with open('app/backend/data/daily_data.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("AUDITORIA DE TRADES - Fechas en P&L Chart")
print("=" * 60)

all_models = list(data.keys())
print(f"\nModelos disponibles: {len(all_models)}")

for model in all_models:
    trades = data[model].get('trades', [])
    if not trades:
        continue

    print(f"\n{'='*60}")
    print(f"Modelo: {model}")
    print(f"{'='*60}")
    print(f"Total trades: {len(trades)}")

    # Check exit dates
    exit_dates = [t['exit_date'] for t in trades]
    unique_dates = set(exit_dates)
    duplicates_count = len(exit_dates) - len(unique_dates)

    print(f"Exit dates unicos: {len(unique_dates)}")
    print(f"Fechas duplicadas: {duplicates_count}")

    # Find duplicates
    if duplicates_count > 0:
        date_counts = Counter(exit_dates)
        duplicates = {d: c for d, c in date_counts.items() if c > 1}
        print(f"\nFechas con multiples trades (primeras 5):")
        for i, (d, c) in enumerate(sorted(duplicates.items())[:5]):
            print(f"  - {d}: {c} trades en la misma fecha")
            # Show trades on this date
            trades_on_date = [t for t in trades if t['exit_date'] == d]
            for t in trades_on_date:
                print(f"      Entry: {t['entry_date']} -> Exit: {t['exit_date']} | Dur: {t['duration']}d | Ret: {t['total_return']*100:+.2f}%")

    # Check date format
    print(f"\nFormato de fechas (primeros 3):")
    for t in trades[:3]:
        print(f"  entry_date: '{t['entry_date']}' | exit_date: '{t['exit_date']}'")

    # Check if sorted
    is_sorted = all(exit_dates[i] <= exit_dates[i+1] for i in range(len(exit_dates)-1))
    print(f"\nTrades ordenados por exit_date: {'SI' if is_sorted else 'NO'}")

    # Show first 5 and last 5 trades
    print(f"\nPrimeros 5 trades:")
    for i, t in enumerate(trades[:5]):
        print(f"  {i+1}. {t['entry_date']} -> {t['exit_date']} ({t['duration']}d) | Pos: {t['entry_position']:.1f}x | Ret: {t['total_return']*100:+.2f}%")

    print(f"\nUltimos 5 trades:")
    for i, t in enumerate(trades[-5:]):
        idx = len(trades) - 5 + i
        print(f"  {idx+1}. {t['entry_date']} -> {t['exit_date']} ({t['duration']}d) | Pos: {t['entry_position']:.1f}x | Ret: {t['total_return']*100:+.2f}%")

print("\n" + "=" * 60)
print("FIN DE AUDITORIA")
print("=" * 60)
