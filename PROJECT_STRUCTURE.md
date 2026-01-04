# Estructura del Proyecto - Tesis V3

## Directorios Protegidos (NO MODIFICAR)

Estos directorios contienen el pipeline de ML y feature engineering original:

```
scripts/                    # Pipeline de entrenamiento y feature engineering
├── ml_pipeline_darts_extended.py       # Pipeline principal (19 modelos, posiciones discretas)
├── feature_engineering_hedge_fund.py   # Feature engineering principal
├── generate_app_data.py                # Genera datos para app web
├── comprehensive_leakage_test.py       # Tests de data leakage
├── academic_enhancements.py            # Tests estadisticos
└── ...

data/                       # Datos procesados
├── final/                  # Dataset final con features
└── prepared/               # Datos preparados

results/                    # Resultados de modelos
├── unified_strategy_3x_comparison.csv  # Metricas de 19 modelos
├── unified_strategy_3x_summary.json    # Resumen
└── ...

models/                     # Modelos entrenados guardados
```

## Directorio de la App (NUEVO)

```
app/                        # Aplicacion de visualizacion
├── backend/                # FastAPI backend
│   ├── main.py            # Entry point
│   ├── api/               # Endpoints
│   ├── services/          # Logica de negocio
│   └── data/              # Datos pre-computados para la app
│
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/    # Componentes React
│   │   ├── pages/         # Paginas (9 paneles)
│   │   ├── hooks/         # Custom hooks
│   │   ├── services/      # API calls
│   │   └── types/         # TypeScript types
│   ├── package.json
│   └── ...
│
└── README.md              # Documentacion de la app
```

## Scripts de Integracion

```
scripts/
└── generate_app_data.py   # Genera datos pre-computados para la app
                           # Lee de results/ y genera app/backend/data/
```

## Flujo de Datos

```
[Pipeline ML]                    [App Visualizacion]
     │                                   │
     ▼                                   │
scripts/*.py                             │
     │                                   │
     ▼                                   │
results/*.csv  ──────────────────►  generate_app_data.py
     │                                   │
     │                                   ▼
     │                           app/backend/data/
     │                                   │
     │                                   ▼
     │                           FastAPI Backend
     │                                   │
     │                                   ▼
     └───────────────────────►   React Frontend
```
