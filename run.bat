@echo off
REM ============================================================
REM  CRONOS PRODUCCION - LSTM + Attention (modelo ganador tesis)
REM  Usa el Python instalado en el perfil del usuario (disco C:).
REM  Si no existe, lo instala con tools\uv.exe (requiere internet).
REM ============================================================
cd /d "%~dp0"

set "PYROOT=%USERPROFILE%\cronos-python"
set "PY=%PYROOT%\cpython-3.12-windows-x86_64-none\python.exe"

if not exist "%PY%" (
    echo [1/3] Instalando Python 3.12 en %PYROOT% ...
    set "UV_PYTHON_INSTALL_DIR=%PYROOT%"
    tools\uv.exe python install 3.12 --no-bin
    if errorlevel 1 (
        echo [ERROR] No se pudo instalar Python. Se requiere internet.
        pause
        exit /b 1
    )
)

echo [2/3] Verificando dependencias...
tools\uv.exe pip install --quiet --python "%PY%" --break-system-packages -r requirements.txt blpapi ^
    --index-url https://download.pytorch.org/whl/cpu ^
    --extra-index-url https://pypi.org/simple ^
    --extra-index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ ^
    --index-strategy unsafe-best-match
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias. Se requiere internet la primera vez.
    pause
    exit /b 1
)

echo [3/3] Ejecutando el modelo...
echo.
"%PY%" produccion_lstm.py
pause
