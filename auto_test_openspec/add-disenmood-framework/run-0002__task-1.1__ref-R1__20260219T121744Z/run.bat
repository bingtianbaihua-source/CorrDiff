@echo off
setlocal enabledelayedexpansion

set HERE=%~dp0
set HERE=%HERE:~0,-1%
for %%I in ("%HERE%\..\..\..") do set ROOT_DIR=%%~fI

if not exist "%HERE%\logs" mkdir "%HERE%\logs"
if not exist "%HERE%\outputs" mkdir "%HERE%\outputs"

(
  echo date_utc=%DATE% %TIME%
  echo cd=%HERE%
  echo python=^
  where python
  python -V
  echo uv=^
  where uv
) > "%HERE%\logs\env.txt" 2>&1

where uv >nul 2>&1
if %ERRORLEVEL%==0 (
  if not exist "%HERE%\.venv" (
    uv venv "%HERE%\.venv" >> "%HERE%\logs\env.txt" 2>&1
  )
  call "%HERE%\.venv\Scripts\activate.bat"
)

set PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%

python "%ROOT_DIR%\scripts\augment_admet_labels.py" --input "%HERE%\inputs\smiles_list.txt" --output "%HERE%\outputs\augmented_labels.json" --cache "%HERE%\outputs\admet_cache.json"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -c "import json,sys; from pathlib import Path; root=Path(r'%HERE%'); out=root/'outputs'/'augmented_labels.json'; inp=root/'inputs'/'smiles_list.txt'; req=['PAMPA_NCATS','BBB_Martins','logP','Clearance_Microsome_AZ','hERG','affinity','QED','SA','AMES','lipinski']; smiles=[l.strip() for l in inp.read_text(encoding='utf-8').splitlines() if l.strip()]; rows=json.loads(out.read_text(encoding='utf-8')); assert isinstance(rows,list); assert len(rows)==len(smiles); [(__import__('builtins').all([(k in r) for k in req]) or (_ for _ in ()).throw(AssertionError('missing key'))) for r in rows];"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo OK: outputs\augmented_labels.json generated
exit /b 0

