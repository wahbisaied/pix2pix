@echo off
echo ========================================
echo CT Phase 0 - Continue from Latest Model
echo ========================================
echo.
echo Using latest saved model (safer option)
echo Reduced settings to prevent freeze
echo.

python train.py --dataroot D:\ct_phases_datasets\ct_phase0_dataset --name ct_phase0_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 16 --lambda_L1 50 --lr 0.0002 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 16 --continue_train --n_epochs 150 --n_epochs_decay 50

echo.
echo Training completed successfully!
pause