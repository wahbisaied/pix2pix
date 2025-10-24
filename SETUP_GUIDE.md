# Complete Setup Guide: PyTorch CycleGAN and pix2pix on New Desktop

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Testing & Inference](#testing--inference)
6. [Model Management](#model-management)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3060+ with 12GB+ VRAM)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for datasets and models
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+)

### Software
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8-3.11
- **CUDA**: 11.3+ (if using GPU)
- **Git**: For cloning repository

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### Step 2: Environment Setup

#### Option A: Conda (Recommended)
```bash
# Create environment from provided file
conda env create -f environment.yml

# Activate environment
conda activate pytorch-img2img
```

#### Option B: Manual Installation
```bash
# Create new environment
conda create -n pytorch-img2img python=3.10
conda activate pytorch-img2img

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install visdom dominate
pip install nibabel  # For NIfTI medical images
pip install pillow numpy matplotlib
```

### Step 3: Verify Installation
```python
# Test CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Data Preparation

### For Medical Images (CT/MRI - NIfTI format)

#### 1. Required Files
The project uses these custom files for medical image processing:
- `data/robust_nifti_dataset.py` - Robust NIfTI dataset loader with error handling
- `train.py` - Modified training script with logging
- `use_model.py` - Custom inference script for medical images

#### 2. Data Structure
Create your dataset folder:
```
datasets/
└── ct_phases_dataset/           # Your dataset name
    ├── trainA/                  # Input images (e.g., average CT)
    │   ├── patient_1.nii.gz
    │   ├── patient_2.nii.gz
    │   ├── patient_3.nii.gz
    │   └── ...
    ├── trainB/                  # Target images (e.g., phase 0 CT)
    │   ├── patient_1.nii.gz     # MUST match trainA names
    │   ├── patient_2.nii.gz
    │   ├── patient_3.nii.gz
    │   └── ...
    ├── testA/                   # Test input images (optional)
    │   └── ...
    └── testB/                   # Test target images (optional)
        └── ...
```

#### 3. Data Preprocessing (Automatic)
The `robust_nifti_dataset.py` handles:
- **File validation**: Checks for corrupted files
- **Slice extraction**: Extracts 2D axial slices from 3D volumes
- **Intensity normalization**: Normalizes to [0, 255] range
- **Error handling**: Skips corrupted slices automatically
- **Resizing**: Resizes to specified dimensions (512x512)

#### 4. Data Requirements
- **File format**: NIfTI (.nii.gz or .nii)
- **Naming**: Corresponding files MUST have identical names
- **Dimensions**: 3D volumes (any size, will be processed to 2D slices)
- **Intensity**: Any range (automatically normalized)
- **Minimum**: At least 10-20 patient pairs for meaningful training

### For Standard Images (PNG/JPG)

#### 1. Download Existing Datasets
```bash
# CycleGAN datasets
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
bash ./datasets/download_cyclegan_dataset.sh maps

# pix2pix datasets  
bash ./datasets/download_pix2pix_dataset.sh facades
bash ./datasets/download_pix2pix_dataset.sh edges2shoes
```

#### 2. Custom Image Datasets
For pix2pix (paired data):
```bash
# Combine A and B images into pairs
python datasets/combine_A_and_B.py --fold_A /path/to/A --fold_B /path/to/B --fold_AB /path/to/combined
```

---

## Training

### Basic Training Commands

#### pix2pix (Paired Translation)
```bash
# Medical images (CT phases) - WORKING CONFIGURATION
python train.py \
    --dataroot ./datasets/ct_phases_dataset \
    --name ct_phase0_generator_optimized \
    --model pix2pix \
    --dataset_mode robust_nifti \
    --skip_corrupted \
    --preprocess none \
    --input_nc 1 \
    --output_nc 1 \
    --axial_slice \
    --norm instance \
    --netG unet_512 \
    --batch_size 4 \
    --lambda_L1 50 \
    --lr 0.0001 \
    --resize_to 512 \
    --display_freq 100 \
    --save_epoch_freq 5 \
    --n_epochs 100 \
    --n_epochs_decay 100

# Standard images (facades)
python train.py \
    --dataroot ./datasets/facades \
    --name facades_pix2pix \
    --model pix2pix \
    --direction BtoA \
    --batch_size 4 \
    --n_epochs 200
```

#### CycleGAN (Unpaired Translation)
```bash
# Horse to Zebra
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_cyclegan \
    --model cycle_gan \
    --batch_size 1 \
    --n_epochs 100 \
    --n_epochs_decay 100
```

### Training Parameters Explained

| Parameter | Description | Medical Images | Standard Images |
|-----------|-------------|----------------|----------------|
| `--dataroot` | Path to dataset | `./datasets/ct_phases_dataset` | `./datasets/facades` |
| `--dataset_mode` | Dataset loader | `robust_nifti` | `aligned` |
| `--preprocess` | Preprocessing | `none` | `resize_and_crop` |
| `--resize_to` | Fixed resize | `512` | Not used |
| `--netG` | Generator | `unet_512` | `unet_256` |
| `--batch_size` | Batch size | `4` (512px) | `1-8` |
| `--norm` | Normalization | `instance` | `batch` |
| `--axial_slice` | Slice axis | Required | Not used |
| `--skip_corrupted` | Error handling | `True` | Not used |
| `--lambda_L1` | L1 loss weight | `50` | `100` |

### Multi-GPU Training
```bash
# Use multiple GPUs
torchrun --nproc_per_node=2 train.py \
    --dataroot ./datasets/your_data \
    --name multi_gpu_experiment \
    --model pix2pix \
    --norm sync_instance \
    --batch_size 16
```

### Resume Training
```bash
python train.py \
    --continue_train \
    --epoch_count 50 \
    --name existing_experiment \
    [other parameters...]
```

---

## Testing & Inference

### Test Trained Model
```bash
# Test pix2pix model
python test.py \
    --dataroot ./datasets/facades \
    --name facades_pix2pix \
    --model pix2pix \
    --direction BtoA \
    --epoch latest

# Test CycleGAN model
python test.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_cyclegan \
    --model cycle_gan \
    --epoch 200
```

### Single Image Inference
```bash
# Use custom inference script (for medical images)
python use_model.py \
    --input_nifti "./datasets/ct_phases_dataset/trainA/patient_1.nii.gz" \
    --model_name ct_phase0_generator_optimized \
    --epoch latest \
    --output_dir ./results

# The script will:
# 1. Load the trained model
# 2. Process each 2D slice from the 3D volume
# 3. Generate corresponding output slices
# 4. Save individual PNG images
# 5. Reconstruct and save output NIfTI volume
```

### Pre-trained Models
```bash
# Download pre-trained CycleGAN models
bash ./scripts/download_cyclegan_model.sh horse2zebra
bash ./scripts/download_cyclegan_model.sh style_monet

# Download pre-trained pix2pix models  
bash ./scripts/download_pix2pix_model.sh facades_label2photo
bash ./scripts/download_pix2pix_model.sh edges2shoes
```

---

## Model Management

### File Structure
```
checkpoints/
└── experiment_name/
    ├── latest_net_G.pth      # Latest generator
    ├── latest_net_D.pth      # Latest discriminator
    ├── 100_net_G.pth        # Epoch 100 generator
    ├── train_opt.txt         # Training options
    └── web/                  # Training visualizations
        └── index.html

results/
└── experiment_name/
    └── test_latest/
        ├── images/           # Generated images
        └── index.html        # Results webpage

logs/
└── train_log_YYYYMMDD_HHMMSS.txt  # Training logs
```

### Save/Load Models
```python
# Load model in your code
import torch
from models.networks import define_G

# Load generator
model_path = "checkpoints/experiment_name/latest_net_G.pth"
generator = define_G(input_nc=1, output_nc=1, ngf=64, 
                    netG="unet_256", norm="instance", 
                    use_dropout=False, init_gain=0.02, gpu_ids=[0])

state_dict = torch.load(model_path, map_location='cuda:0')
generator.load_state_dict(state_dict)
generator.eval()
```

### Monitor Training
```bash
# Start Visdom server for visualization
python -m visdom.server

# View training progress
# Open browser: http://localhost:8097

# Or use Weights & Biases
python train.py --use_wandb [other parameters...]
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Solution**: Reduce batch size or image size
```bash
--batch_size 1 --crop_size 256
```

#### 2. Model Loading Errors
**Solution**: Match training and test parameters
```bash
# Ensure same architecture
--netG unet_256 --norm instance --no_dropout
```

#### 3. Dataset Loading Issues
**Solution**: Check file paths and formats
```bash
# For medical images, ensure NIfTI format
# For standard images, ensure PNG/JPG format
```

#### 4. Slow Training
**Solutions**:
- Use smaller images: `--crop_size 256`
- Reduce dataset size: `--max_dataset_size 1000`
- Use multiple threads: `--num_threads 8`

#### 5. Poor Results
**Solutions**:
- Train longer: `--n_epochs 200`
- Adjust loss weights: `--lambda_L1 100`
- Try different architectures: `--netG resnet_9blocks`

### Performance Optimization

#### For Medical Images
```bash
# Optimized for CT/MRI
--preprocess resize_and_crop \
--load_size 286 \
--crop_size 256 \
--netG unet_256 \
--norm instance \
--batch_size 8 \
--lambda_L1 50
```

#### For High-Resolution Images
```bash
# Memory-efficient high-res training
--preprocess scale_width_and_crop \
--load_size 1024 \
--crop_size 512 \
--batch_size 1 \
--netG unet_512
```

### Hardware Recommendations

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB | `batch_size=4, crop_size=256` |
| 12GB | `batch_size=8, crop_size=256` or `batch_size=4, crop_size=512` |
| 16GB+ | `batch_size=16, crop_size=256` or `batch_size=8, crop_size=512` |

---

## Quick Start Checklist

### For Medical Images (CT/MRI)
- [ ] Install Python 3.8-3.11 + PyTorch + CUDA
- [ ] Clone repository
- [ ] Install nibabel: `pip install nibabel`
- [ ] Create dataset folder: `datasets/ct_phases_dataset/`
- [ ] Add NIfTI files to `trainA/` and `trainB/` with matching names
- [ ] Verify files: Check that `robust_nifti_dataset.py` exists
- [ ] Test run: `--n_epochs 5` first
- [ ] Full training: Use the working command above
- [ ] Monitor: Check `logs/train_log_*.txt`
- [ ] Inference: Use `use_model.py` for testing

### Key Files for Medical Images
- `data/robust_nifti_dataset.py` - Dataset loader
- `train.py` - Training script (with logging)
- `use_model.py` - Inference script
- `logs/train_log_*.txt` - Training logs

## Support Resources

- **Documentation**: [docs/](docs/)
- **FAQ**: [docs/qa.md](docs/qa.md)
- **Training Tips**: [docs/tips.md](docs/tips.md)
- **GitHub Issues**: [pytorch-CycleGAN-and-pix2pix/issues](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues)

---

*This guide covers the complete setup process for running PyTorch CycleGAN and pix2pix on a new desktop. For specific use cases or advanced configurations, refer to the detailed documentation in the `docs/` folder.*