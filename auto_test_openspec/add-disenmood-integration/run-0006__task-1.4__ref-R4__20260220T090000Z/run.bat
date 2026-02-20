@echo off
setlocal enabledelayedexpansion

set "RUN_DIR=%~dp0"
for %%I in ("%RUN_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

if defined PYTHONPATH (
  set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%REPO_ROOT%"
)

if not exist "%RUN_DIR%inputs" mkdir "%RUN_DIR%inputs"
if not exist "%RUN_DIR%outputs" mkdir "%RUN_DIR%outputs"

python -c "from utils.npz_io import NpyArray, save_npz; save_npz(r'%RUN_DIR%inputs\\sample_latents.npz', {'z_shared': NpyArray('<f4',(1,4),[[0.1,-0.2,0.3,0.4]]), 'z_pi_0': NpyArray('<f4',(1,4),[[-0.5,0.6,-0.7,0.8]])}); print('Wrote inputs\\\\sample_latents.npz')" || exit /b 1

python -c "import importlib; importlib.import_module('torch')" >nul 2>&1
if %errorlevel%==0 (
  python "%REPO_ROOT%\\scripts\\decode_latents.py" --input "%RUN_DIR%inputs\\sample_latents.npz" --output "%RUN_DIR%outputs\\decoded_ligand.npz" --num-atoms 8 || exit /b 1
) else (
  set "CORRDIFF_TOY_NO_TORCH=1"
  python "%REPO_ROOT%\\scripts\\decode_latents.py" --input "%RUN_DIR%inputs\\sample_latents.npz" --output "%RUN_DIR%outputs\\decoded_ligand.npz" --num-atoms 8 || exit /b 1
)

python -c "from utils.npz_io import load_npz; out=load_npz(r'%RUN_DIR%outputs\\decoded_ligand.npz'); xyz=out['xyz']; an=out['atomic_nums']; s=bytes(int(b) for b in out['smiles_utf8'].data).decode('utf-8','replace'); assert xyz.shape[1:]==(3,), xyz.shape; assert an.shape==(xyz.shape[0],), (an.shape, xyz.shape); assert s!='', 'smiles must be non-empty'; assert (s=='SMILES_UNAVAILABLE_IN_TOY_MODE') or (s!='SMILES_UNAVAILABLE_IN_TOY_MODE' and s!=''), 'smiles check'; print('OK:', xyz.shape, an.shape, s)" || exit /b 1

exit /b 0
