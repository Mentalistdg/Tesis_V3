@echo off
REM ============================================================
REM  Programa la actualizacion diaria de CRONOS en el Programador de tareas
REM  (usuario actual, sin administrador). Horas de Chile, lunes a viernes:
REM  19:30, 21:30, 23:30 (noche NY) y 09:30, 11:30 (manana NY). Cada corrida
REM  recalcula con datos oficiales y solo registra la senal si cambia.
REM  Uso: programar_tarea.bat crear | eliminar | estado | probar
REM ============================================================
setlocal
set "RAIZ=%~dp0"
set "HORAS=19:30 21:30 23:30 09:30 11:30"
if /i "%~1"=="crear" goto crear
if /i "%~1"=="eliminar" goto eliminar
if /i "%~1"=="estado" goto estado
if /i "%~1"=="probar" goto probar
echo Uso: programar_tarea.bat crear ^| eliminar ^| estado ^| probar
exit /b 1

:crear
for %%h in (%HORAS%) do call :crear_una %%h
goto estado

:crear_una
set "HORA=%~1"
set "NOMBRE=CRONOS_actualizar_%HORA::=%"
schtasks /Create /TN "%NOMBRE%" /TR "wscript.exe //B \"%RAIZ%actualizar_oculto.vbs\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST %HORA% /F
exit /b

:eliminar
for %%h in (%HORAS%) do call :eliminar_una %%h
exit /b 0

:eliminar_una
set "HORA=%~1"
schtasks /Delete /TN "CRONOS_actualizar_%HORA::=%" /F
exit /b

:estado
schtasks /Query /FO TABLE | findstr /i "CRONOS_actualizar"
exit /b 0

:probar
schtasks /Run /TN "CRONOS_actualizar_1930"
exit /b 0
