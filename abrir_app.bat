@echo off
REM Abre la app CRONOS local (senales) en http://localhost:8000
cd /d "%~dp0"
call "%~dp0preparar_entorno.bat" || exit /b 1
start "" http://localhost:8000/senales
"%CRONOS_PY%" -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
