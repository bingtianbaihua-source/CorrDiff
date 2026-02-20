@echo off
setlocal enabledelayedexpansion

set "BUNDLE_DIR=%~dp0"
for %%I in ("%BUNDLE_DIR%..\\..\\..") do set "REPO_ROOT=%%~fI"

set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"
set "KMP_USE_SHM=0"

set "CONFIG_PATH=%BUNDLE_DIR%inputs\\latent_mode_config.yml"
set "OUTPUT_PATH=%BUNDLE_DIR%outputs\\latent_batch.npz"

if not exist "%BUNDLE_DIR%logs" mkdir "%BUNDLE_DIR%logs"
if not exist "%BUNDLE_DIR%outputs" mkdir "%BUNDLE_DIR%outputs"

pushd "%REPO_ROOT%"
python scripts\\encode_latents.py --config "%CONFIG_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

python -c "import os, sys, numpy as np, yaml; cfg=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8')) or {}; bs=int((cfg.get('encode') or {}).get('batch_size',2)); zsd=int((cfg.get('model') or {}).get('z_shared_dim')); zpd=int((cfg.get('model') or {}).get('z_pi_dim')); obj=np.load(sys.argv[2], allow_pickle=True); assert 'z_shared' in obj; zpi=[k for k in obj.files if k.startswith('z_pi_')]; assert zpi; z_shared=obj['z_shared']; z_pi0=obj[sorted(zpi, key=lambda s: int(s.split('_')[-1]))[0]]; assert tuple(z_shared.shape)==(bs,zsd), (z_shared.shape,(bs,zsd)); assert tuple(z_pi0.shape)==(bs,zpd), (z_pi0.shape,(bs,zpd)); print('OK: latent_batch.npz contains z_shared and z_pi_* with expected shapes')" "%CONFIG_PATH%" "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

popd
exit /b 0
