@echo off
setlocal enabledelayedexpansion

rem Resolve bundle dir (directory of this .bat)
set "BUNDLE_DIR=%~dp0"
if "%BUNDLE_DIR:~-1%"=="\" set "BUNDLE_DIR=%BUNDLE_DIR:~0,-1%"
set "BUNDLE_DIR=%BUNDLE_DIR%"

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set KMP_USE_SHM=0

if not exist "%BUNDLE_DIR%\inputs" mkdir "%BUNDLE_DIR%\inputs"
if not exist "%BUNDLE_DIR%\outputs" mkdir "%BUNDLE_DIR%\outputs"
if not exist "%BUNDLE_DIR%\logs" mkdir "%BUNDLE_DIR%\logs"

(
  echo {
  echo   "z_shared_dim": 12,
  echo   "z_pi_dim": 7,
  echo   "property_names": ["qed", "sa", "logp", "affinity"],
  echo   "num_steps": 12,
  echo   "batch_size": 4
  echo }
) > "%BUNDLE_DIR%\inputs\diffusion_config.json"

python "%BUNDLE_DIR%\runner.py"
if errorlevel 1 exit /b 1
exit /b 0
