# CLAUDE.md — CRONOS Dashboard App

## Overview

**CRONOS** is a dark-themed dashboard for visualizing the performance of 23 ML/DL models predicting S&P 500 returns using a **LONG-ONLY** strategy. Positions are `{0 (CASH), +1 (SPY), +3 (UPRO)}`. There are no short positions.

## Architecture

```
Browser → Vite dev proxy (port 3000) → /api/* → FastAPI (port 8000) → JSON files
                                     → /*     → React SPA (static assets)
```

Production uses Nginx to serve the static build and proxy `/api/*` to Uvicorn.

## How to Run

```bash
# Backend (from repo root — never cd on Windows bash)
.venv/Scripts/python.exe -m uvicorn app.backend.main:app --reload --port 8000

# Frontend (use --prefix to avoid cd)
npm --prefix app/frontend install
npm --prefix app/frontend run dev       # Dev server on port 3000, proxies /api to :8000
npm --prefix app/frontend run build     # Production build (tsc && vite build)
npm --prefix app/frontend run lint      # ESLint
```

## Data Flow

The app does **not** run models or inference. It serves pre-computed JSON files:

```
ML Pipeline (scripts/)
  → python paper/update_backend_data.py   # Reads pipeline results, writes 4 JSON files
  → app/backend/data/
      ├── daily_data.json       # Per-model daily arrays (dates, positions, returns, equity, trades)
      ├── models_summary.json   # All 23 models with summary metrics + config + benchmark
      ├── market_data.json      # SPY benchmark (dates, returns, equity_curve, drawdown)
      └── regime_data.json      # Market regime classification + per-regime performance
```

To regenerate data after re-running the pipeline: `python paper/update_backend_data.py`

## Pages (6)

| Route | Page | Description |
|-------|------|-------------|
| `/overview` | OverviewPage | Model rankings table, top-5 equity curves chart, KPI cards |
| `/detail/:modelName?` | DetailPage | Single model deep-dive: synced equity/position/drawdown charts, metrics, position distribution, date filtering |
| `/trades/:modelName?` | TradesPage | Trade-by-trade log with compounding capital, cost breakdown per trade |
| `/risk/:modelName?` | RiskPage | Risk analysis: drawdown, VaR, volatility, rolling metrics |
| `/regime` | RegimePage | Performance by market regime (bull/bear/sideways/high_vol) |
| `/costs` | CostsPage | Transaction cost impact: expense ratios, bid-ask, volatility drag |

Default route `/` redirects to `/overview`.

## Backend Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | All 23 models with summary metrics, config, benchmark |
| `GET /api/models/{name}` | Daily arrays for one model (dates, positions, returns, equity, trades) |
| `GET /api/models/{name}/metrics?start_date=&end_date=` | Metrics recalculated for a date range |
| `GET /api/models/{name}/trades` | Trade log for a model |
| `GET /api/models/{name}/daily-log` | Daily decision log (one row per trading day) |
| `GET /api/market` | SPY benchmark data |
| `GET /api/regimes` | Regime analysis data |

## Frontend Structure

```
app/frontend/src/
├── App.tsx                          # Router setup (6 routes)
├── main.tsx                         # Entry point
├── index.css                        # Tailwind + custom styles
├── types/index.ts                   # All TypeScript interfaces
├── services/api.ts                  # Axios API layer with 5-min cache
├── utils/transactionCosts.ts        # Cost calculations (source of truth)
├── pages/
│   ├── OverviewPage.tsx             # Rankings + equity chart
│   ├── DetailPage.tsx               # Model deep-dive
│   ├── TradesPage.tsx               # Trade log
│   ├── RiskPage.tsx                 # Risk analysis
│   ├── RegimePage.tsx               # Regime performance
│   └── CostsPage.tsx                # Cost impact
├── components/
│   ├── Layout.tsx                   # Shell: sidebar nav + content area
│   ├── LoadingScreen.tsx            # Loading spinner
│   ├── SortableHeader.tsx           # Clickable table column headers
│   └── DailyLogFilterBar.tsx        # Filter bar for daily log table
└── hooks/
    ├── useTableSort.ts              # Generic table sorting hook
    └── useTableFilters.ts           # Generic table filtering hook
```

## Cost Model (3 components)

All cost calculations live in `utils/transactionCosts.ts`. The backend provides `trade.total_return` as **NET** (costs already deducted). The frontend never re-applies costs — it uses the net returns directly for compounding.

| Component | What it models |
|-----------|---------------|
| **Expense ratio** | UPRO 0.91%/yr, SPY 0.09%/yr, prorated per trade duration |
| **Bid-ask spread** | UPRO 0.05%, SPY 0.02% one-way (doubled for round-trip) |
| **Volatility drag** | Daily rebalancing cost for 3x leveraged ETFs |

## Theme ("Axe Capital" Dark)

- Brand name: **CRONOS**
- Background: `#000000` (pure black)
- Primary accent: `#c41e3a` (red)
- Positive/green: `#00c853`
- Muted text: `#737373`, `#525252`
- Card background: `#111111`
- Borders: `#222222`
- Configuration in `tailwind.config.js` under `axe-*` color keys
- Charts use Lightweight Charts library (TradingView style)

## Dependencies

- **React 18** + TypeScript + Vite
- **Tailwind CSS** for styling
- **lightweight-charts** (TradingView) for financial charts
- **lucide-react** for icons
- **axios** for API calls
- **clsx** for conditional class names
- **react-router-dom** for routing

## Common Tasks

### Regenerate backend data after pipeline re-run
```bash
python paper/update_backend_data.py
```

### Add a new page
1. Create `src/pages/NewPage.tsx`
2. Add route in `src/App.tsx`
3. Add nav link in `src/components/Layout.tsx`
4. Add tab type to `TabType` in `src/types/index.ts`

### Modify metrics displayed
- Summary metrics: edit `ModelSummary` in `types/index.ts` + update `OverviewPage.tsx` table
- Detail metrics: edit `ModelMetrics` in `types/index.ts` + update `DetailPage.tsx` cards

### Update cost model
- Edit `utils/transactionCosts.ts` — all cost logic is centralized there
- `calculateSummaryFromTrades()` is the source of truth used by all pages
