@echo off
REM ============================================================
REM  Deja listo el Python portable de CRONOS en el perfil del usuario (C:)
REM  y define CRONOS_PY. Idempotente: solo instala si falta algo o si
REM  requirements.txt cambio. No requiere administrador.
REM ============================================================
set "CRONOS_RAIZ=%~dp0"
set "PYROOT=%USERPROFILE%\cronos-python"
set "CRONOS_PY=%PYROOT%\cpython-3.12-windows-x86_64-none\python.exe"
set "UV=%CRONOS_RAIZ%tools\uv.exe"
set "UV_CACHE_DIR=%TEMP%\uv-cache"

if not exist "%UV%" (
    echo [entorno] Descargando uv 0.12.23...
    if not exist "%CRONOS_RAIZ%tools" mkdir "%CRONOS_RAIZ%tools"
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-pc-windows-msvc.zip -OutFile $env:TEMP\uv.zip -UseBasicParsing; Expand-Archive $env:TEMP\uv.zip -DestinationPath '%CRONOS_RAIZ%tools' -Force" || (echo [ERROR] No se pudo descargar uv & exit /b 1)
)

if not exist "%CRONOS_PY%" (
    echo [entorno] Instalando Python 3.12 en %PYROOT% ...
    set "UV_PYTHON_INSTALL_DIR=%PYROOT%"
    "%UV%" python install 3.12 --no-bin || (echo [ERROR] No se pudo instalar Python & exit /b 1)
)

for /f %%h in ('powershell -NoProfile -Command "(Get-FileHash '%CRONOS_RAIZ%requirements.txt').Hash"') do set "REQHASH=%%h"
set "MARCA=%PYROOT%\.deps_%REQHASH%"
if not exist "%MARCA%" (
    echo [entorno] Instalando dependencias...
    "%UV%" pip install --quiet --python "%CRONOS_PY%" --break-system-packages -r "%CRONOS_RAIZ%requirements.txt" ^
        --index-url https://download.pytorch.org/whl/cpu ^
        --extra-index-url https://pypi.org/simple ^
        --extra-index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ ^
        --index-strategy unsafe-best-match || (echo [ERROR] Fallo la instalacion de dependencias & exit /b 1)
    type nul > "%MARCA%"
)
exit /b 0
