@echo off
echo ========================================
echo CT Phase 0 Generator - RTX 5090 Optimized
echo ========================================
echo.
echo Hardware: Ryzen 9950X + RTX 5090 24GB
echo Dataset: Phase 0 (80 patients, 8971 slices)
echo.

python train.py --dataroot D:\ct_phases_datasets\ct_phase0_dataset --name ct_phase0_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 32 --lambda_L1 50 --lr 0.0002 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 32 --n_epochs 100 --n_epochs_decay 0

echo.
echo Training completed!
pause