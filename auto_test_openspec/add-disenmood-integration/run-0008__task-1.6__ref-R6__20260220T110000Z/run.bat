@echo off
setlocal enabledelayedexpansion

set BUNDLE_DIR=%~dp0
for %%I in ("%BUNDLE_DIR%..\..\..") do set REPO_ROOT=%%~fI

set PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%

cd /d "%BUNDLE_DIR%"
if not exist inputs mkdir inputs
if not exist outputs mkdir outputs
if not exist logs mkdir logs

> inputs\workflow_checklist.txt (
  echo stage_1_vae_pretraining
  echo stage_2_latent_diffusion
  echo disenmood_mode_switch
  echo legacy_3d_path_disabled
)

python verify_workflow_docs.py
if errorlevel 1 exit /b 1

python -c "p=r'outputs\\workflow_summary.txt'; s=open(p,encoding='utf-8').read(); req=['stage_1_vae_pretraining','stage_2_latent_diffusion','disenmood_mode_switch','legacy_3d_path_disabled','Stage 1: VAE pretraining','Stage 2: BranchDiffusion latent diffusion training','disenmood_mode: true','DISABLED (no fallback route)']; assert all(r in s for r in req)"
if errorlevel 1 exit /b 1

exit /b 0

