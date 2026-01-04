# -*- coding: utf-8 -*-
import json
from collections import Counter

with open('app/backend/data/daily_data.json', 'r') as f:
    data = json.load(f)

model = 'LightGBM'
trades = data[model].get('trades', [])

print('='*60)
print('LIGHTGBM - Distribucion de trades por mes')
print('='*60)

# Group by year-month
exit_months = [t['exit_date'][:7] for t in trades]
month_counts = Counter(exit_months)

print(f'\nTotal trades: {len(trades)}')
print(f'\nTrades por mes (2024-2025):')
for month in sorted(month_counts.keys()):
    if month >= '2024':
        print(f'  {month}: {month_counts[month]} trades')

print(f'\nUltimos 30 trades de LightGBM:')
for i, t in enumerate(trades[-30:]):
    idx = len(trades) - 30 + i
    entry = t['entry_date']
    exit_d = t['exit_date']
    dur = t['duration']
    ret = t['total_return'] * 100
    pos = t['entry_position']
    print(f'  {idx+1}. {entry} -> {exit_d} ({dur}d) Pos:{pos:.0f}x Ret:{ret:+.2f}%')

# Check if there are gaps
print('\n' + '='*60)
print('Verificando fechas en el area de Dic 2025:')
print('='*60)
dec_trades = [t for t in trades if t['exit_date'] >= '2025-12']
jan_trades = [t for t in trades if '2025-01' <= t['exit_date'] < '2025-02']

print(f'\nTrades con exit en Dic 2025: {len(dec_trades)}')
for t in dec_trades:
    print(f"  {t['entry_date']} -> {t['exit_date']} ({t['duration']}d)")

print(f'\nTrades con exit en Ene 2025: {len(jan_trades)}')
for t in jan_trades[:10]:
    print(f"  {t['entry_date']} -> {t['exit_date']} ({t['duration']}d)")
