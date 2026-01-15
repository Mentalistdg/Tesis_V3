# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

S&P 500 return prediction system combining Elder's Triple Screen technical analysis with 92 Bloomberg financial variables transformed into 647 features. Trains and compares 19 ML/DL models with a full-stack visualization application.

**Test Period:** October 2020 - December 2025 (1,302 days)

## Quick Start

```bash
# All commands run from this directory (Tesis_V3/)

# Activate virtual environment (from repository root)
..\.venv\Scripts\activate  # Windows
source ../.venv/bin/activate  # Linux/Mac
```

## Repository Structure

```
Tesis_V3/
├── scripts/           # ML pipeline scripts (9 active)
├── data/              # Input CSVs and processed datasets
├── models/            # Trained model artifacts
├── results/           # Pipeline outputs and metrics
├── app/               # Full-stack web application
│   ├── backend/       # FastAPI server (port 8000)
│   └── frontend/      # React/TypeScript SPA (port 3000)
├── paper/             # LaTeX academic paper
├── Bibliography/      # Reference PDFs
└── _archive/          # Archived/deprecated files
```

## Pipeline Execution Order

The scripts must be run in this specific order:

```
1. build_dataset.py          → bloomberg_triple_screen_core.csv (647 features)
2. train_models.py           → trained_artifacts.pkl (19 models)
3. optimize_model_params.py  → optimal_model_params.json
4. evaluate_strategy.py      → strategy_realistic_evaluation.csv
5. generate_app_data_optimized.py → app/backend/data/*.json
```

### Core Pipeline Commands

```bash
# Step 1: Build dataset from Bloomberg data (creates 647 features)
python scripts/build_dataset.py

# Step 2: Train all 19 models (outputs trained_artifacts.pkl)
python scripts/train_models.py

# Step 3: Optimize quantile thresholds per model
python scripts/optimize_model_params.py

# Step 4: Evaluate with realistic costs and risk management
python scripts/evaluate_strategy.py

# Step 5: Generate JSON data for web app
python scripts/generate_app_data_optimized.py
```

### Analysis Scripts (Optional)

```bash
# Validate no data leakage
python scripts/comprehensive_leakage_test.py

# Compare threshold strategies
python scripts/compare_thresholds.py

# Unified model comparison
python scripts/unified_strategy_comparison.py

# Statistical tests for paper (DM, MCS, Clark-West, SHAP)
python scripts/academic_enhancements.py
```

### Web Application

```bash
# Backend (from app/backend/)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (from app/frontend/)
npm install
npm run dev      # Dev server on port 3000
npm run build    # Production build
npm run lint     # ESLint check
```

### Paper Generation

```bash
cd paper
python generate_paper_figures.py
pdflatex paper_triple_screen_ml.tex
```

## Scripts Reference

| Script | Purpose | Output |
|--------|---------|--------|
| `build_dataset.py` | Feature engineering from 92 Bloomberg vars | `data/bloomberg_triple_screen_core.csv` |
| `train_models.py` | Train 23 models, generate predictions | `models/trained_artifacts.pkl` |
| `optimize_model_params.py` | Grid search for optimal thresholds | `results/optimal_model_params.json` |
| `evaluate_strategy.py` | Realistic evaluation with costs | `results/strategy_*.csv` |
| `generate_app_data_optimized.py` | Pre-compute JSON for web app | `app/backend/data/*.json` |
| `comprehensive_leakage_test.py` | Validate no data leakage | `results/leakage_test_results.json` |
| `compare_thresholds.py` | Compare symmetric vs asymmetric | `results/threshold_comparison.csv` |
| `unified_strategy_comparison.py` | Compare all models unified | `results/unified_strategy_*.csv` |
| `academic_enhancements.py` | Statistical tests, SHAP analysis | `results/academic_*.json` |

## Models (23 total)

- **Classical ML:** Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting
- **Gradient Boosting:** XGBoost, LightGBM, CatBoost
- **Time Series:** AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
- **Specialized:** Prophet, GARCH
- **Deep Learning (Darts):** DLinear, N-BEATS, N-HiTS, TCN, TFT
- **RNN Bidirectional (Darts):** Bi-LSTM, Bi-GRU
- **Hybrid/Attention (PyTorch):** CNN-LSTM, LSTM+Attention (Luong)

## Critical Implementation Details

### Data Leakage Prevention
- All features use only past information (t-k, k>=1)
- Rolling windows look backward only
- Train/test split is temporal (no shuffle)
- Preprocessing fitted only on training data
- TimeSeriesSplit with purging gaps for CV
- 252-day warmup period eliminated

### Strategy Logic (UPRO/SPXU Style)
Predictions converted to 5 discrete positions:
- `+3`: UPRO (3x long) - prediction > q_long_extreme percentile
- `+1`: SPY (1x long)
- `0`: Cash (risk-free)
- `-1`: SH (1x inverse)
- `-3`: SPXU (3x short) - prediction < q_short_extreme percentile

**Return Formula:** `return = rf + position * (market_return - rf)`

**Realistic Costs:**
- Expense ratios: UPRO/SPXU (0.89%), SPY (0.09%), SH (0.89%)
- Transaction costs: 15 bps per position change

## Key Data Files

| File | Description |
|------|-------------|
| `data/bloomberg_triple_screen_core.csv` | Main dataset (647 features) |
| `data/BLOOMBERG_RAW_DATA.csv` | Raw Bloomberg data (92 vars) |
| `models/trained_artifacts.pkl` | Trained models + predictions |
| `results/optimal_model_params.json` | Best thresholds per model |
| `app/backend/data/daily_data.json` | Pre-computed app data |

## API Endpoints (FastAPI)

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | List models with summary metrics |
| `GET /api/models/{name}` | Daily data for a model |
| `GET /api/models/{name}/trades` | Trade log |
| `GET /api/signals` | Current signals + consensus |
| `GET /api/market` | SPY benchmark data |
| `GET /api/regimes` | Market regime data |
| `GET /api/compare?models=A,B,C` | Compare models |

## Tech Stack

**Backend:** Python 3.13, FastAPI, scikit-learn, XGBoost, LightGBM, CatBoost, Darts, Prophet, arch (GARCH)

**Frontend:** React 18, TypeScript 5, Vite 5, Tailwind CSS, Recharts, Lightweight Charts, Zustand

**Analysis:** NumPy, Pandas, SciPy, statsmodels, SHAP

## Dataset Usage Example

```python
import pandas as pd

df = pd.read_csv('data/bloomberg_triple_screen_core.csv', index_col=0, parse_dates=True)

# Exclude non-feature columns
exclude = ['date', 'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols]
y = df['market_forward_excess_returns']  # Already has shift(-1) applied

# Temporal split (NEVER shuffle)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```
