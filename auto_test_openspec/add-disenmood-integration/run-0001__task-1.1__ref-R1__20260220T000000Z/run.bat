@echo off
setlocal enabledelayedexpansion

set "BUNDLE_DIR=%~dp0"
for %%I in ("%BUNDLE_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

set "CONFIG_PATH=%BUNDLE_DIR%inputs\\vae_toy_config.yml"
set "OUTPUT_PATH=%BUNDLE_DIR%outputs\\vae_checkpoint.pt"

if not exist "%BUNDLE_DIR%outputs" mkdir "%BUNDLE_DIR%outputs"

cd /d "%REPO_ROOT%"

python scripts\\train_vae.py --config "%CONFIG_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

python -c "p=r'%OUTPUT_PATH%'; import torch; ckpt=torch.load(p,map_location='cpu'); enc=ckpt.get('encoder_state_dict'); dec=ckpt.get('decoder_state_dict'); assert isinstance(enc,dict) and len(enc)>0, 'encoder_state_dict missing/empty'; assert isinstance(dec,dict) and len(dec)>0, 'decoder_state_dict missing/empty'; print(f'OK: wrote {p} (encoder_keys={len(enc)}, decoder_keys={len(dec)})')"
if errorlevel 1 exit /b 1

exit /b 0
