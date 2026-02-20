@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%\..\..\..") do set ROOT_DIR=%%~fI

set CONFIG_PATH=%SCRIPT_DIR%\inputs\eval_config.json
set OUTPUT_PATH=%SCRIPT_DIR%\outputs\eval_report.json

cd /d "%ROOT_DIR%"
set PYTHONPATH=%ROOT_DIR%

python scripts\eval_disenmood.py --config "%CONFIG_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

if not exist "%OUTPUT_PATH%" (
  echo Missing output: %OUTPUT_PATH%
  exit /b 1
)

python - "%OUTPUT_PATH%" ^
  "import json,sys; p=sys.argv[1]; d=json.load(open(p,'r',encoding='utf-8')); req=['HV','Sparsity','correlation_error','intervention_result','docking_scores']; miss=[k for k in req if k not in d]; assert not miss, f'Missing keys: {miss}'; print('OK: required keys present')"
if errorlevel 1 exit /b 1

exit /b 0

