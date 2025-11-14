# CT Phase Generator - Quick Reference

## Setup
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
conda env create -f environment.yml
conda activate pytorch-img2img
```

## Dataset Structure
```
datasets/ct_phases_dataset/
├── trainA/          # Input CT (average phase)
└── trainB/          # Target CT (phase 0)
```

## Training Command
```bash
python train.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator_optimized --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 4 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 100 --save_epoch_freq 5
```

## Testing
```bash
python test.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator_optimized --model pix2pix --dataset_mode robust_nifti --preprocess none --input_nc 1 --output_nc 1 --axial_slice --resize_to 512 --epoch latest
```

## Single Volume Inference
```bash
python use_model.py --input_nifti "./path/to/input.nii.gz" --model_name ct_phase0_generator_optimized --epoch latest --output_dir ./results
```

## Required Custom Files
- `data/robust_nifti_dataset.py` - NIfTI dataset loader
- `use_model.py` - Volume inference script

## Key Results
- 80 volume pairs → 9,062 valid slices
- 14+ epochs, ~36 min/epoch
- G_L1 loss: 29 → 3-7 range