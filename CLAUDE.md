# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

S&P 500 return prediction system combining Elder's Triple Screen technical analysis with 92 Bloomberg financial variables transformed into 647 features. Trains and compares 19 ML/DL models with a full-stack visualization application.

**Test Period:** October 2020 - December 2025 (1,302 days)

## Build and Run Commands

### ML Pipeline (from project root)
```bash
# Main pipeline: Train all 19 models with discrete positions [-3,-1,0,+1,+3]
python scripts/ml_pipeline_darts_extended.py

# Generate pre-computed data for web app
python scripts/generate_app_data.py

# Run statistical tests (DM test, MCS, Clark-West)
python scripts/academic_enhancements.py
```

### Web Application

**Backend (app/backend/):**
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend (app/frontend/):**
```bash
npm install
npm run dev      # Dev server on port 3000 (proxies API to 8000)
npm run build    # Production build
npm run lint     # ESLint TypeScript check
```

### Paper Generation
```bash
cd paper
python generate_paper_figures.py
pdflatex paper_triple_screen_ml.tex
```

## Architecture

### Data Flow
```
Bloomberg Data (92 vars) → Feature Engineering (647 features) → ML Models (19) → Results → App Data JSON → Frontend
```

### Key Directories
- `scripts/` - ML pipeline scripts (do not modify without understanding feature engineering)
- `data/` - Input CSVs and processed datasets
- `models/` - Trained model artifacts (.pkl, .json)
- `results/` - Pipeline outputs and model comparisons
- `app/backend/` - FastAPI server (single main.py handles all endpoints)
- `app/frontend/` - React/TypeScript SPA with 9 dashboard panels
- `paper/` - LaTeX academic paper with figures

### Main Scripts
| Script | Purpose |
|--------|---------|
| `feature_engineering_hedge_fund.py` | Generates 647 features from raw Bloomberg data |
| `ml_pipeline_darts_extended.py` | **Main pipeline** - Trains 19 models with discrete positions {-3,-1,0,+1,+3} |
| `generate_app_data.py` | Creates JSON data files for web app |
| `academic_enhancements.py` | Statistical tests (DM, MCS, Clark-West, Kelly) |

### Models (19 total)
- **Classical:** Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting
- **Gradient Boosting:** XGBoost, LightGBM, CatBoost
- **Time Series:** AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
- **Specialized:** Prophet, GARCH
- **Deep Learning:** DLinear, N-BEATS, N-HiTS, TCN, TFT

## Critical Implementation Details

### Data Leakage Prevention
All scripts enforce strict temporal ordering:
- Train/test split is temporal (no shuffle)
- Preprocessing fitted only on training data
- TimeSeriesSplit for cross-validation with purging gaps
- Forward-only feature calculations (no future data)

### Strategy Logic (UPRO/SPXU Style)
- Predictions converted to 5 discrete positions: **{-3, -1, 0, +1, +3}**
  - `-3`: SPXU (3x short)
  - `-1`: SH (1x short)
  - `0`: Cash (risk-free)
  - `+1`: SPY (1x long)
  - `+3`: UPRO (3x long)
- Formula: `return = rf + position * (market_return - rf)`
- Transaction costs: 15 basis points per position change

### Frontend-Backend Communication
- Frontend runs on port 3000, proxies `/api/*` to backend on port 8000
- Data loaded from pre-computed JSON files in `app/backend/data/`
- API endpoints defined in `app/backend/main.py`

## Key Files

| File | Size | Description |
|------|------|-------------|
| `data/bloomberg_triple_screen_core.csv` | 45MB | Main dataset with 647 features |
| `data/final/bloomberg_features_hf.csv` | 42MB | Engineered features output |
| `app/backend/data/daily_data.json` | 4.3MB | Pre-computed daily data for all models |
| `results/model_comparison_extended.csv` | - | 19-model performance comparison |

## Tech Stack

**Backend:** Python 3.13, FastAPI, scikit-learn, XGBoost, LightGBM, CatBoost, Darts, Prophet, arch (GARCH)

**Frontend:** React 18, TypeScript 5, Vite 5, Tailwind CSS, Recharts, Lightweight Charts, Zustand

**Analysis:** NumPy, Pandas, SciPy, statsmodels, SHAP, TA-Lib
