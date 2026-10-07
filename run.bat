@echo off
REM ============================================================
REM  CRONOS PRODUCCION - verificacion de la tesis (LSTM + Attention)
REM  Prepara el entorno (solo la primera vez) y ejecuta produccion_lstm.py,
REM  que reproduce el backtest publicado (+395.4%%, Sharpe 1.319).
REM ============================================================
cd /d "%~dp0"
call "%~dp0preparar_entorno.bat" || exit /b 1
"%CRONOS_PY%" -W ignore produccion_lstm.py
if "%1"=="" pause
