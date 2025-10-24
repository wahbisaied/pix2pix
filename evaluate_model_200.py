import os
import torch
import nibabel as nib
import numpy as np
from PIL import Image
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import subprocess

# Configuration
MODEL_EPOCH = "200"
CHECKPOINT_DIR = "./checkpoints/ct_phase0_generator_optimized"
DATASET_DIR = "./datasets/ct_phases_dataset"
OUTPUT_DIR = "./model_200_evaluation"
REAL_PNG_DIR = os.path.join(OUTPUT_DIR, "real_images")
FAKE_PNG_DIR = os.path.join(OUTPUT_DIR, "fake_images")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REAL_PNG_DIR, exist_ok=True)
os.makedirs(FAKE_PNG_DIR, exist_ok=True)

def nifti_to_png_slices(nifti_path, output_dir, prefix, max_slices=10):
    """Convert NIfTI to PNG slices"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Normalize to 0-255
    data = (data - data.min()) / (data.max() - data.min()) * 255
    data = data.astype(np.uint8)
    
    # Take middle slices
    z_dim = data.shape[2]
    start_slice = max(0, z_dim // 2 - max_slices // 2)
    end_slice = min(z_dim, start_slice + max_slices)
    
    png_files = []
    for i in range(start_slice, end_slice):
        slice_data = data[:, :, i]
        img_pil = Image.fromarray(slice_data)
        png_path = os.path.join(output_dir, f"{prefix}_slice_{i:03d}.png")
        img_pil.save(png_path)
        png_files.append(png_path)
    
    return png_files

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    np_img = tensor.cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)

print("=== Model 200 Evaluation Pipeline ===")

# Step 1: Create test directory structure
print("Step 1: Setting up test directory...")
test_dir = os.path.join(DATASET_DIR, "test")
os.makedirs(test_dir, exist_ok=True)

# Copy some trainA files to test directory for evaluation
train_a_dir = os.path.join(DATASET_DIR, "trainA")
if os.path.exists(train_a_dir):
    train_files = [f for f in os.listdir(train_a_dir) if f.endswith('.nii.gz')][:10]
    for file in train_files:
        src = os.path.join(train_a_dir, file)
        dst = os.path.join(test_dir, file)
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
    print(f"Copied {len(train_files)} files to test directory")

# Step 2: Convert test data (use subset of trainB as reference)
print("Step 2: Converting reference images...")
trainB_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "trainB")) if f.endswith('.nii.gz')][:5]

for nifti_file in trainB_files:
    patient_id = nifti_file.replace('.nii.gz', '')
    nifti_path = os.path.join(DATASET_DIR, "trainB", nifti_file)
    nifti_to_png_slices(nifti_path, REAL_PNG_DIR, patient_id)

print(f"Converted {len(trainB_files)} reference files to PNG")

# Step 3: Setup test options
print("Step 3: Setting up model...")
# Create minimal args list to avoid command line parsing
import sys
original_argv = sys.argv
sys.argv = [
    'evaluate_model_200.py', 
    '--dataroot', DATASET_DIR,
    '--name', 'ct_phase0_generator_optimized',
    '--model', 'test',
    '--dataset_mode', 'single',
    '--no_flip',
    '--preprocess', 'none',
    '--load_size', '512',
    '--crop_size', '512',
    '--epoch', MODEL_EPOCH
]

opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1
opt.name = "ct_phase0_generator_optimized"
opt.model = "test"
opt.dataset_mode = "single"
opt.dataroot = DATASET_DIR
opt.phase = "test"
opt.preprocess = "none"  # No preprocessing
opt.load_size = 512
opt.crop_size = 512
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restore original argv
sys.argv = original_argv

# Create dataset and model
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)

# Load specific checkpoint
model.load_networks(MODEL_EPOCH)
model.eval()

print(f"Loaded model from epoch {MODEL_EPOCH}")

# Step 4: Generate fake images
print("Step 4: Generating fake images...")
fake_count = 0

with torch.no_grad():
    for i, data in enumerate(dataset):
        if i >= 50:  # Limit to 50 samples
            break
            
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        
        # Save fake_B image
        if 'fake_B' in visuals:
            fake_B = visuals['fake_B']
            fake_pil = tensor_to_pil(fake_B)
            fake_path = os.path.join(FAKE_PNG_DIR, f"fake_{i:04d}.png")
            fake_pil.save(fake_path)
            fake_count += 1

print(f"Generated {fake_count} fake images")

# Step 5: Calculate FID
print("Step 5: Calculating FID score...")
try:
    result = subprocess.run([
        "python", "-m", "pytorch_fid", 
        REAL_PNG_DIR, FAKE_PNG_DIR
    ], capture_output=True, text=True, cwd=".")
    
    if result.returncode == 0:
        fid_score = result.stdout.strip().split()[-1]
        print(f"FID Score for Model {MODEL_EPOCH}: {fid_score}")
        
        # Save result
        with open(os.path.join(OUTPUT_DIR, "fid_result.txt"), "w") as f:
            f.write(f"Model: {MODEL_EPOCH}\n")
            f.write(f"FID Score: {fid_score}\n")
            f.write(f"Real images: {len(os.listdir(REAL_PNG_DIR))}\n")
            f.write(f"Fake images: {fake_count}\n")
    else:
        print(f"FID calculation failed: {result.stderr}")
        
except Exception as e:
    print(f"Error calculating FID: {e}")

print(f"\nEvaluation complete! Results saved in: {OUTPUT_DIR}")
print(f"Real images: {REAL_PNG_DIR}")
print(f"Fake images: {FAKE_PNG_DIR}")