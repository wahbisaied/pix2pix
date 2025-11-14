# Quick Conda Fix

## Problem
`CondaError: Run 'conda init' before 'conda activate'`

## Solution

### Step 1: Initialize Conda
```bash
conda init cmd.exe
```

### Step 2: Restart Command Prompt
Close and reopen your command prompt

### Step 3: Activate Environment
```bash
conda activate pytorch-img2img
```

## Alternative: Direct Python Path
If conda still doesn't work, use direct path:
```bash
# Find your conda environment path
C:\Users\[YourUsername]\miniconda3\envs\pytorch-img2img\python.exe train.py [parameters...]
```

## Quick Training Command (Updated Path)
```bash
python train.py --dataroot D:\ct_phases_dataset --name ct_phase0_generator_op --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 4 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 100 --save_epoch_freq 5
```

## Verify Setup
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```