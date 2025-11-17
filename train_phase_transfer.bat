@echo off
if "%1"=="" (
    echo Usage: train_phase_transfer.bat [phase_number]
    echo Example: train_phase_transfer.bat 1
    pause
    exit /b 1
)

set PHASE=%1
echo ========================================
echo CT Phase %PHASE% - Transfer Learning
echo ========================================
echo.
echo Using Phase 0 weights as initialization...
echo Target: 50 epochs (vs 100 from scratch)
echo.

python train.py --dataroot D:\ct_phases_datasets\ct_phase%PHASE%_dataset --name ct_phase%PHASE%_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 32 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 32 --continue_train --load_pretrain ct_phase0_generator_5090 --epoch_count 1 --n_epochs 50 --n_epochs_decay 0

echo.
echo Phase %PHASE% transfer learning completed!
pause