# Guía de Instalación Completa

## Requisitos del Sistema

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| Python | 3.13.x | Pipeline ML y backend |
| Node.js | 22.x | Frontend React |
| Git | 2.x | Control de versiones |
| TeX Live / MiKTeX | 2025 | Compilar paper LaTeX |

## Paso 1: Clonar el repositorio

```bash
git clone https://github.com/Mentalistdg/Tesis_V3.git
cd Tesis_V3
```

## Paso 2: Python y entorno virtual

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Activar (Linux/Mac)
source .venv/bin/activate
```

## Paso 3: TA-Lib (ANTES de requirements.txt)

TA-Lib requiere una librería C nativa. Debe instalarse antes del resto.

### Windows:
```bash
# Descargar el wheel precompilado para Python 3.13 desde:
# https://github.com/cgohlke/talib-build/releases
# Archivo: TA_Lib-0.6.8-cp313-cp313-win_amd64.whl

pip install TA_Lib-0.6.8-cp313-cp313-win_amd64.whl
```

### Linux (Ubuntu/Debian):
```bash
# Instalar la librería C
sudo apt-get update
sudo apt-get install -y ta-lib libta-lib-dev

# Si no está en los repos, compilar desde fuente:
wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4
./configure --prefix=/usr
make
sudo make install
cd ..

pip install TA-Lib==0.6.8
```

### Mac:
```bash
brew install ta-lib
pip install TA-Lib==0.6.8
```

## Paso 4: Dependencias Python

```bash
# PyTorch CPU (por defecto)
pip install -r requirements.txt

# O PyTorch con CUDA (si tienes GPU NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Paso 5: Frontend (Node.js)

```bash
cd app/frontend
npm install
cd ../..
```

## Paso 6: LaTeX (para compilar el paper)

### Windows (MiKTeX - recomendado):
1. Descargar desde https://miktex.org/download
2. Instalar con "Install missing packages on-the-fly = Yes"
3. Los paquetes LaTeX se instalan automáticamente al compilar

### Windows (TeX Live alternativo):
1. Descargar desde https://tug.org/texlive/windows.html
2. Instalar la distribución completa

### Linux:
```bash
# Instalación completa (recomendado, ~5 GB)
sudo apt install texlive-full

# O instalación mínima necesaria (~1 GB)
sudo apt install texlive-base texlive-latex-extra texlive-lang-spanish texlive-science texlive-bibtex-extra texlive-fonts-extra
```

### Mac:
```bash
brew install --cask mactex
```

### Paquetes LaTeX requeridos:
babel (spanish), inputenc, fontenc, amsmath, amsfonts, amssymb, graphicx, booktabs, float, hyperref, geometry, setspace, natbib, caption, subcaption, multirow, array, xcolor, fancyhdr, longtable, pdflscape, titlesec, enumitem, tcolorbox, pifont, threeparttable.

## Verificación

```bash
# Verificar Python
python --version          # 3.13.x

# Verificar paquetes clave
python -c "import pandas; print(pandas.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import darts; print(darts.__version__)"
python -c "import talib; print(talib.__version__)"
python -c "import fastapi; print(fastapi.__version__)"

# Verificar Node.js
node --version            # v22.x

# Verificar LaTeX
pdflatex --version        # pdfTeX (TeX Live 2025)

# Compilar el paper (desde Tesis_V3/)
cd _archive/paper_active
pdflatex paper_triple_screen_ml.tex
```

## Ejecución

### Pipeline ML:
```bash
python scripts/build_dataset.py
python scripts/train_models.py
python scripts/optimize_model_params.py
```

### Web App:
```bash
# Terminal 1 - Backend
cd app/backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd app/frontend
npm run dev
```

### Paper:
```bash
cd _archive/paper_active
pdflatex paper_triple_screen_ml.tex
```
