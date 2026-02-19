@echo off
setlocal

set BUNDLE_DIR=%~dp0
set OMP_NUM_THREADS=1
set KMP_USE_SHM=0
python "%BUNDLE_DIR%runner.py"

