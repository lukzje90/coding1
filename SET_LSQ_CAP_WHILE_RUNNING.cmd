@echo off
setlocal EnableDelayedExpansion
set "capfile=%~dp0least_squares_evaluation_cap.txt"
if not exist "%capfile%" (
    echo No LSQ cap control file exists yet. Start a fit in this model folder first.
    pause
    exit /b 1
)
echo Current cap:
type "%capfile%"
echo.
set "newcap="
set /p "newcap=Enter new total model-evaluation cap [2-10000]: "
if not defined newcap exit /b 1
echo(!newcap!>"!capfile!"
echo Requested cap !newcap!. The LSQ fit reads it before the next evaluation.
echo Check least_squares_active_evaluation.txt for the applied cap.
pause
