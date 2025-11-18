@echo off
if "%1"=="" (
    echo Usage: train_phase_transfer.bat [phase_number]
    echo Example: train_phase_transfer.bat 1
    pause
    exit /b 1
)

set PHASE=%1
set /a PREV_PHASE=%PHASE%-1
set SOURCE_DIR=checkpoints\ct_phase%PREV_PHASE%_generator_5090
set TARGET_DIR=checkpoints\ct_phase%PHASE%_generator_5090

echo ========================================
echo CT Phase %PHASE% - Transfer Learning
echo ========================================
echo.
echo Loading from Phase %PREV_PHASE% model...
echo.

REM Check if previous phase model exists
if not exist "%SOURCE_DIR%\latest_net_G.pth" (
    echo ERROR: Phase %PREV_PHASE% model not found at %SOURCE_DIR%\latest_net_G.pth
    echo Please complete Phase %PREV_PHASE% training first!
    pause
    exit /b 1
)

echo ✓ Found Phase %PREV_PHASE% model: %SOURCE_DIR%\latest_net_G.pth
echo Copying Phase %PREV_PHASE% weights for initialization...

REM Create target directory
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
    echo ✓ Created directory: %TARGET_DIR%
)

REM Copy generator weights (required)
copy /Y "%SOURCE_DIR%\latest_net_G.pth" "%TARGET_DIR%\latest_net_G.pth" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Copied Generator weights
) else (
    echo ✗ Failed to copy Generator weights
    pause
    exit /b 1
)

REM Copy discriminator weights (required)
copy /Y "%SOURCE_DIR%\latest_net_D.pth" "%TARGET_DIR%\latest_net_D.pth" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Copied Discriminator weights
) else (
    echo ✗ Failed to copy Discriminator weights
    pause
    exit /b 1
)

REM Create epoch marker
echo 1 > "%TARGET_DIR%\latest_net_G.pth.epoch"
echo ✓ Set epoch counter to 1
if %PHASE% EQU 1 (
    set EPOCHS=25
    set DECAY_EPOCHS=25
    echo Target: 50 epochs (25 + 25 decay) - First transfer
) else (
    set EPOCHS=20
    set DECAY_EPOCHS=20
    echo Target: 40 epochs (20 + 20 decay) - Sequential transfer
)
echo.

python train.py --dataroot D:\ct_phases_datasets\ct_phase%PHASE%_dataset --name ct_phase%PHASE%_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 16 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 16 --continue_train --epoch_count 1 --n_epochs %EPOCHS% --n_epochs_decay %DECAY_EPOCHS%

echo.
echo Phase %PHASE% transfer learning completed!
pause