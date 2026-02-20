@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

if not exist "%SCRIPT_DIR%inputs" mkdir "%SCRIPT_DIR%inputs"
if not exist "%SCRIPT_DIR%outputs" mkdir "%SCRIPT_DIR%outputs"

python -c "import random; from utils.npz_io import NpyArray, save_npz; random.seed(0); z_shared=[random.gauss(0.0,1.0) for _ in range(4)]; z_pi_0=[random.gauss(0.0,1.0) for _ in range(2)]; save_npz(r'%SCRIPT_DIR%inputs\\sample_latents.npz', {'z_shared': NpyArray(descr='<f4', shape=(4,), data=z_shared), 'z_pi_0': NpyArray(descr='<f4', shape=(2,), data=z_pi_0)}); print('Wrote inputs/sample_latents.npz (no numpy)')"

python -m scripts.decode_latents --input "%SCRIPT_DIR%inputs\\sample_latents.npz" --output "%SCRIPT_DIR%outputs\\decoded_ligand.npz" --num-atoms 8
if errorlevel 1 (
  set "CORRDIFF_TOY_NO_TORCH=1"
  python -m scripts.decode_latents --input "%SCRIPT_DIR%inputs\\sample_latents.npz" --output "%SCRIPT_DIR%outputs\\decoded_ligand.npz" --num-atoms 8
  if errorlevel 1 exit /b 1
)

python -c "from utils.npz_io import load_npz; out=load_npz(r'%SCRIPT_DIR%outputs\\decoded_ligand.npz'); xyz=out['xyz']; an=out['atomic_nums']; su=out['smiles_utf8']; assert len(xyz.shape)==2 and xyz.shape[1]==3, xyz.shape; assert len(an.shape)==1 and an.shape[0]==xyz.shape[0], (an.shape, xyz.shape); assert all(v==v for row in xyz.data for v in row); assert all(v>0 for v in an.data); smiles=bytes(su.data).decode('utf-8','replace'); assert smiles!=''; print('OK: decoded_ligand.npz contains xyz (N,3), atomic_nums (N,), and exportable representation')"
if errorlevel 1 exit /b 1

exit /b 0
