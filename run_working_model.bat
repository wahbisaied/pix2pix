@echo off
echo ========================================
echo CT Phase Generator - Working Model
echo ========================================
echo.
echo This will start training with the exact configuration that worked successfully.
echo Training details:
echo - 80 volume pairs, 9,062 valid slices
echo - Stable convergence achieved
echo - ~36 minutes per epoch on GPU
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause

echo.
echo Starting training...
echo.

python train.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator_optimized --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 4 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 100 --save_epoch_freq 5 --n_epochs 100 --n_epochs_decay 100

echo.
echo Training completed!
echo Check the logs folder for training details.
echo Model saved in: checkpoints/ct_phase0_generator_optimized/
pause