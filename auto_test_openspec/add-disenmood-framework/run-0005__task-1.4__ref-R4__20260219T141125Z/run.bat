@echo off
setlocal enabledelayedexpansion

set "BUNDLE_DIR=%~dp0"
set "OMP_NUM_THREADS=1"
set "KMP_USE_SHM=0"
set "KMP_DISABLE_SHM=1"

python "%BUNDLE_DIR%runner.py"
if errorlevel 1 exit /b 1
exit /b 0
