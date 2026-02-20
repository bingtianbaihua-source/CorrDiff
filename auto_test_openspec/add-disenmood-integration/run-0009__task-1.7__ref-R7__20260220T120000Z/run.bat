@echo off
setlocal enabledelayedexpansion

REM Resolve REPO_ROOT as three levels above this run folder:
set "RUN_DIR=%~dp0"
for %%I in ("%RUN_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

if not exist "%RUN_DIR%inputs" mkdir "%RUN_DIR%inputs"
if not exist "%RUN_DIR%outputs" mkdir "%RUN_DIR%outputs"

> "%RUN_DIR%inputs\\decode_contract.json" (
  echo {
  echo   "required": ["xyz", "atomic_nums"],
  echo   "optional": ["aromatic", "atom_affinity"],
  echo   "reconstruction_fn": "utils.reconstruct.reconstruct_from_generated"
  echo }
)

python "%RUN_DIR%check_decode_contract.py"
if errorlevel 1 exit /b 1

findstr /c:"REQUIRED:" "%RUN_DIR%outputs\\decode_contract_check.txt" >nul || exit /b 1
findstr /c:"RECONSTRUCTION_FN:" "%RUN_DIR%outputs\\decode_contract_check.txt" >nul || exit /b 1
findstr /c:"COMPATIBILITY: PASS" "%RUN_DIR%outputs\\decode_contract_check.txt" >nul || exit /b 1

echo Wrote: %RUN_DIR%outputs\\decode_contract_check.txt
exit /b 0

