@echo off
REM Actualiza datos desde Bloomberg y emite la senal del dia habil siguiente (CRONOS).
REM Uso: actualizar.bat [--ensayo] [--hasta AAAA-MM-DD]
cd /d "%~dp0"
if not exist logs mkdir logs
call "%~dp0preparar_entorno.bat" >> logs\salida.log 2>&1 || exit /b 1
"%CRONOS_PY%" -W ignore -m pipeline.cronos actualizar %* >> logs\salida.log 2>&1
set "CODIGO=%ERRORLEVEL%"
type logs\salida.log | more +0 > nul
exit /b %CODIGO%
