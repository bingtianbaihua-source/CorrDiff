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

python "%REPO_ROOT%\scripts\sample_disenmood.py" ^
  --config "%BUNDLE_DIR%\inputs\sample_config.yml" ^
  --device cpu ^
  --out-smi "%BUNDLE_DIR%\outputs\generated_molecules.smi" ^
  --out-3d-npz "%BUNDLE_DIR%\outputs\generated_molecules_3d.npz"
if errorlevel 1 exit /b 1

python -c "import numpy as np, os; smi=os.path.join('outputs','generated_molecules.smi'); lines=[ln.strip() for ln in open(smi, encoding='utf-8') if ln.strip()]; assert len(lines)>=1; obj=np.load(os.path.join('outputs','generated_molecules_3d.npz')); xyz=obj['xyz']; an=obj['atomic_nums']; assert xyz.ndim==3 and xyz.shape[-1]==3; assert an.ndim==2 and an.shape[0]==xyz.shape[0] and an.shape[1]==xyz.shape[1]; assert xyz.shape[0]==len(lines)"
if errorlevel 1 exit /b 1
exit /b 0

