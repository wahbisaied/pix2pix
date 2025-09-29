import os
import torch
import nibabel as nib
import numpy as np
from PIL import Image
import subprocess
from networks import define_G

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
    Image.fromarray(normalized).save(output_path)

def tensor_to_png(tensor, output_path):
    """Convert tensor to PNG"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(np_img).save(output_path)

# Configuration
MODEL_PATH = "./checkpoints/ct_phase0_generator_optimized/200_net_G.pth"
DATASET_A = "./datasets/ct_phases_dataset/trainA"
DATASET_B = "./datasets/ct_phases_dataset/trainB"
OUTPUT_DIR = "./simple_eval_200"
REAL_DIR = os.path.join(OUTPUT_DIR, "real")
FAKE_DIR = os.path.join(OUTPUT_DIR, "fake")

# Create directories
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

print("=== Simple Model 200 Evaluation ===")

# Step 1: Convert reference images (trainB)
print("Converting reference images...")
nifti_files = [f for f in os.listdir(DATASET_B) if f.endswith('.nii.gz')][:20]

for i, nifti_file in enumerate(nifti_files):
    nifti_path = os.path.join(DATASET_B, nifti_file)
    png_path = os.path.join(REAL_DIR, f"real_{i:04d}.png")
    nifti_to_png(nifti_path, png_path)

print(f"Converted {len(nifti_files)} reference images")

# Step 2: Load generator model
print("Loading generator model...")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Create generator (adjust parameters to match your training)
netG = define_G(
    input_nc=1,
    output_nc=1, 
    ngf=64,
    netG='resnet_9blocks',
    norm='instance',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=[0] if torch.cuda.is_available() else []
)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
netG.load_state_dict(checkpoint)
netG.eval()

print("Generator loaded successfully")

# Step 3: Generate fake images from trainA
print("Generating fake images...")
trainA_files = [f for f in os.listdir(DATASET_A) if f.endswith('.nii.gz')][:20]

with torch.no_grad():
    for i, nifti_file in enumerate(trainA_files):
        # Load and preprocess input
        nifti_path = os.path.join(DATASET_A, nifti_file)
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Take middle slice and normalize to [-1, 1]
        middle_slice = data[:, :, data.shape[2] // 2]
        
        # Resize to 512x512 if needed
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(middle_slice.astype(np.float32))
        pil_img = pil_img.resize((512, 512), PILImage.LANCZOS)
        resized = np.array(pil_img)
        
        # Normalize to [-1, 1]
        normalized = (resized - resized.min()) / (resized.max() - resized.min())
        normalized = (normalized * 2.0) - 1.0
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Generate fake image
        fake_B = netG(input_tensor)
        
        # Save fake image
        fake_path = os.path.join(FAKE_DIR, f"fake_{i:04d}.png")
        tensor_to_png(fake_B, fake_path)

print(f"Generated {len(trainA_files)} fake images")

# Step 4: Calculate FID
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
            f.write(f"Fake images: {len(trainA_files)}\n")
    else:
        print(f"FID calculation failed: {result.stderr}")
        
except Exception as e:
    print(f"Error calculating FID: {e}")

print(f"\nEvaluation complete! Check {OUTPUT_DIR} for results")