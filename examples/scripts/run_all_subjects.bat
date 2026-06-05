@echo off
REM Train CHB-MIT subjects across multiple models using the new seizure_pred CLI.
REM Prerequisite (repo root):
REM   pip install -e ".[cli]"
REM Optional extras depending on models/features:
REM   pip install -e ".[gnn]"    (graph models)
REM   pip install -e ".[signal]" (scipy-based transforms/models)
REM   pip install -e ".[eeg]"    (mne-connectivity)

setlocal enabledelayedexpansion

set DATASET_DIR=data\BIDS_CHB-MIT
set EPOCHS=20
set BATCH_SIZE=32
set LR=1e-3
set SUFFIX=fd_5s_szx5_prex5

REM CHB-MIT subjects (common subset; edit as needed)
set SUBJECTS=01 02 03 04 05 06 07 08 09 10 13 14 15 16 17 18 19 20 22 23 24

REM Model registry names (see `python -m seizure_pred.cli.main list --models`)
set MODELS=eegwavenet ce_stsenet mb_dmgc_cwtffnet

set ROOT_DIR=%~dp0\..\..
pushd %ROOT_DIR%

set CONFIG_DIR=examples\scripts\_configs
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

set BASE_CFG=%CONFIG_DIR%\base_chbmit.yaml
python examples\scripts\config_template.py --out "%BASE_CFG%" --task prediction --dataset-dir "%DATASET_DIR%" --subject 01 --model ce_stsenet >nul
if errorlevel 1 (
  echo Failed to generate base config via examples\scripts\config_template.py
  popd
  exit /b 1
)

for %%M in (%MODELS%) do (
  for %%S in (%SUBJECTS%) do (
    echo ================================================================
    echo Training model=%%M subject=%%S
    echo ================================================================

    set OVERRIDE=%CONFIG_DIR%\override_%%M_%%S.yaml
    > "!OVERRIDE!" (
      echo epochs: %EPOCHS%
      echo data:
      echo   dataset_dir: "%DATASET_DIR%"
      echo   subject_id: "%%S"
      echo   suffix: "%SUFFIX%"
      echo   batch_size: %BATCH_SIZE%
      echo model:
      echo   name: %%M
      echo optim:
      echo   lr: %LR%
      echo run_name: "%%M_sub%%S_%SUFFIX%"
    )

    python -m seizure_pred.cli.main train --config "%BASE_CFG%" --override "!OVERRIDE!" --split-index 0 --n-folds 5
  )
)

echo.
echo ================================================================
echo Running analysis for all runs under runs/
echo ================================================================
for /d %%D in (runs\*) do (
  if exist "%%D\split_0" (
    echo Analyzing %%D\split_0
    python examples\scripts\analyze_results.py --run-dir "%%D\split_0"
  )
)

popd
echo Done.
pause
