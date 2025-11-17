@echo off
echo ========================================
echo CT Phase 0 - Continue Training 100-200
echo ========================================
echo.
echo Continuing from epoch 100 to 200...
echo.

python train.py --dataroot D:\ct_phases_datasets\ct_phase0_dataset --name ct_phase0_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 32 --lambda_L1 50 --lr 0.0002 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 32 --continue_train --epoch_count 101 --n_epochs 150 --n_epochs_decay 50

echo.
echo Extended training completed!
pause