@echo off
REM Launcher for FORMA - FFT-based Orbit Response Matrix Analyzer
REM Uses pushd to handle UNC network paths (\\server\share\...)

pushd "%~dp0"

echo ========================================
echo  FORMA
echo  FFT-based Orbit Response Matrix Analyzer
echo ========================================
echo.
echo Starting application...
echo.

python forma.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Application exited with error code %ERRORLEVEL%
    echo.
    pause
)

popd
