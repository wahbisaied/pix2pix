@echo off
echo ========================================
echo CT Phase 0 - Continue from Epoch 150 (DECAY PHASE)
echo ========================================
echo.
echo Continuing decay phase:
echo - From epoch 150 to 200 (50 epochs remaining)
echo - LR decay: 0.0002 → 0.0000
echo - batch_size: 16, num_threads: 16
echo.

cd "C:\Users\Wahbi Saied\Documents\GitHub\pix2pix"
conda activate torch_env

python train.py --dataroot D:\ct_phases_datasets\ct_phase0_dataset --name ct_phase0_generator_5090 --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 16 --lambda_L1 50 --lr 0.0002 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 16 --continue_train --epoch_count 151 --n_epochs 150 --n_epochs_decay 50

echo.
echo Phase 0 training completed! Ready for transfer learning.
pause