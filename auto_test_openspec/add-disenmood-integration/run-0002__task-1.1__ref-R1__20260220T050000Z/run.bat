@echo off
setlocal enabledelayedexpansion

set "BUNDLE_DIR=%~dp0"
for %%I in ("%BUNDLE_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"
set "KMP_USE_SHM=0"

set "CONFIG_PATH=%BUNDLE_DIR%inputs\vae_toy_config.yml"
set "OUTPUT_PATH=%BUNDLE_DIR%outputs\vae_checkpoint.pt"

pushd "%REPO_ROOT%" || exit /b 1
python scripts\train_vae.py --config "%CONFIG_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 (popd & exit /b 1)

python -c "import os, sys, torch; p=sys.argv[1]; d=torch.load(p, map_location='cpu'); assert isinstance(d, dict); assert 'encoder_state_dict' in d and 'decoder_state_dict' in d; assert d['encoder_state_dict'] and d['decoder_state_dict']; print('OK: checkpoint contains encoder_state_dict + decoder_state_dict')" "%OUTPUT_PATH%"
if errorlevel 1 (popd & exit /b 1)

popd
exit /b 0
