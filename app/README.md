# Strategy Visualizer App

Aplicacion de visualizacion para estrategias de trading basadas en ML.
Interfaz tipo TradingView para analizar el rendimiento de 19 modelos.

## Estructura

```
app/
├── backend/           # FastAPI backend
│   ├── main.py       # API endpoints
│   ├── data/         # Datos pre-computados (JSON)
│   └── requirements.txt
│
└── frontend/          # React + TypeScript frontend
    ├── src/
    │   ├── pages/    # 9 paneles de la app
    │   ├── components/
    │   └── services/
    └── package.json
```

## Requisitos

- Python 3.9+
- Node.js 18+
- npm o yarn

## Instalacion

### 1. Generar datos (si no existen)

```bash
cd scripts
python generate_app_data.py
```

Esto entrena los 19 modelos y genera los archivos JSON en `app/backend/data/`.

### 2. Backend

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

El backend estara disponible en `http://localhost:8000`.

### 3. Frontend

```bash
cd app/frontend
npm install
npm run dev
```

El frontend estara disponible en `http://localhost:3000`.

## Paneles de la App

| Panel | Descripcion |
|-------|-------------|
| **Signals** | Senales actuales de todos los modelos + consenso |
| **Overview** | Dashboard con ranking de modelos y equity curves |
| **Compare** | Comparar 2-4 modelos lado a lado |
| **Detail** | Detalle de un modelo con filtro de fechas |
| **Trades** | Log de operaciones trade-by-trade |
| **Risk** | Analisis de riesgo (VaR, drawdown, volatilidad) |
| **Regime** | Performance por regimen de mercado |
| **Costs** | Impacto de costos de transaccion |
| **Settings** | Configuracion y exportacion |

## API Endpoints

| Endpoint | Descripcion |
|----------|-------------|
| `GET /api/models` | Lista de modelos con metricas |
| `GET /api/models/{name}` | Datos diarios de un modelo |
| `GET /api/models/{name}/metrics?start_date=&end_date=` | Metricas filtradas por fecha |
| `GET /api/market` | Datos del mercado (SPY) |
| `GET /api/signals` | Senales actuales |
| `GET /api/regimes` | Datos de regimen |
| `GET /api/compare?models=A,B,C` | Comparar modelos |

## Modelos Incluidos

**ML (8):** Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost

**StatsForecast (4):** AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

**Prophet (1):** Prophet

**GARCH (1):** GARCH(1,1)

**Darts Deep Learning (5):** DLinear, N-BEATS, N-HiTS, TCN, TFT

## Configuracion

Parametros de estrategia (en `generate_app_data.py`):

```python
CONFIG = {
    'leverage_max': 3.0,      # Maximo leverage (3x como UPRO)
    'sigmoid_scale': 500,     # Escala del sigmoid
    'test_size': 0.20,        # 80% train, 20% test
    'random_state': 42,
}
```

## Notas

- El pipeline de entrenamiento original esta en `scripts/` y NO es modificado por la app
- Los datos de la app son pre-computados y se cargan como JSON
- Para actualizar los datos, re-ejecutar `generate_app_data.py`
