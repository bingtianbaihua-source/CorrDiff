@echo off
setlocal enabledelayedexpansion

set BUNDLE_DIR=%~dp0
for %%I in ("%BUNDLE_DIR%..\..\..") do set REPO_ROOT=%%~fI

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set KMP_USE_SHM=FALSE
set KMP_SHM_DISABLE=1
set KMP_SHM_LOCK_PATH=%TEMP%
set PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%

cd /d "%BUNDLE_DIR%"
if not exist outputs mkdir outputs

python "%REPO_ROOT%\scripts\train_diffusion_joint.py" ^
  --config "%BUNDLE_DIR%\inputs\train_config_disenmood.yml" ^
  --device cpu ^
  --logdir "%BUNDLE_DIR%\outputs\logs"

python -c "import json,os,sys; p=os.path.join('outputs','train_step_metrics.json'); obj=json.load(open(p)); assert 'latent_diffusion_loss' in obj; print('OK: metrics JSON contains latent_diffusion_loss')"
if errorlevel 1 exit /b 1
exit /b 0
