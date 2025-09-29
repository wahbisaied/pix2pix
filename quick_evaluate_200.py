import os
import torch
import nibabel as nib
import numpy as np
from PIL import Image
import subprocess
import sys

# Add project root to path
sys.path.append('.')

from options.test_options import TestOptions
from models import create_model

def nifti_to_png(nifti_path, output_path):
    """Convert single NIfTI slice to PNG"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Take middle slice
    middle_slice = data[:, :, data.shape[2] // 2]
    
    # Normalize to 0-255
    normalized = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255
    normalized = normalized.astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(normalized, mode='L').save(output_path)

def tensor_to_png(tensor, output_path):
    """Convert tensor to PNG"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.size(0) == 3:  # RGB to grayscale
        tensor = tensor.mean(dim=0)
    elif tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(np_img, mode='L').save(output_path)

# Configuration
MODEL_PATH = "./checkpoints/ct_phase0_generator_optimized/200_net_G.pth"
DATASET_A = "./datasets/ct_phases_dataset/trainA"
DATASET_B = "./datasets/ct_phases_dataset/trainB"
OUTPUT_DIR = "./model_200_quick_eval"
REAL_DIR = os.path.join(OUTPUT_DIR, "real")
FAKE_DIR = os.path.join(OUTPUT_DIR, "fake")

# Create directories
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

print("=== Quick Model 200 Evaluation ===")

# Step 1: Convert reference images (trainB)
print("Converting reference images...")
nifti_files = [f for f in os.listdir(DATASET_B) if f.endswith('.nii.gz')][:20]

for i, nifti_file in enumerate(nifti_files):
    nifti_path = os.path.join(DATASET_B, nifti_file)
    png_path = os.path.join(REAL_DIR, f"real_{i:04d}.png")
    nifti_to_png(nifti_path, png_path)

print(f"Converted {len(nifti_files)} reference images")

# Step 2: Load model and generate fake images
print("Loading model and generating fake images...")

# Setup test options
sys.argv = ['quick_evaluate_200.py', '--dataroot', './datasets/ct_phases_dataset', '--load_size', '512', '--crop_size', '512', '--preprocess', 'none', '--epoch', '200']
opt = TestOptions().parse()
opt.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1
opt.name = "ct_phase0_generator_optimized"
opt.model = "cycle_gan"

# Create model
model = create_model(opt)
model.setup(opt)

# Model loaded automatically with epoch 200
model.eval()

# Generate fake images from trainA using dataset
from data import create_dataset
opt.dataroot = "./datasets/ct_phases_dataset"
opt.phase = "train"
dataset = create_dataset(opt)

with torch.no_grad():
    for i, data in enumerate(dataset):
        if i >= 20:  # Limit to 20 samples
            break
            
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        
        # Save fake_B image
        if 'fake_B' in visuals:
            fake_B = visuals['fake_B']
            fake_path = os.path.join(FAKE_DIR, f"fake_{i:04d}.png")
            tensor_to_png(fake_B, fake_path)

print(f"Generated 20 fake images")

# Step 3: Calculate FID
print("Calculating FID score...")
try:
    result = subprocess.run([
        "python", "-m", "pytorch_fid", 
        REAL_DIR, FAKE_DIR
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        fid_score = result.stdout.strip().split()[-1]
        print(f"\n=== RESULTS ===")
        print(f"Model 200 FID Score: {fid_score}")
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
            f.write(f"Model: 200\n")
            f.write(f"FID Score: {fid_score}\n")
            f.write(f"Real images: {len(nifti_files)}\n")
            f.write(f"Fake images: 20\n")
    else:
        print(f"FID calculation failed: {result.stderr}")
        
except Exception as e:
    print(f"Error calculating FID: {e}")

print(f"\nEvaluation complete! Check {OUTPUT_DIR} for results")