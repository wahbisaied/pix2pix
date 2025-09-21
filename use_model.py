#!/usr/bin/env python3
"""
Simple script to use the trained pix2pix model for CT phase generation.

Usage:
    python use_model.py --input_nifti path/to/average_ct.nii.gz --output_dir ./results/

This script:
1. Loads a trained model (.pth file)
2. Takes a NIfTI file as input (average CT)
3. Generates phase 0 CT slices
4. Saves results as individual PNG images and reconstructed NIfTI
"""

import os
import argparse
import torch
import nibabel as nib
import numpy as np
from PIL import Image
from models import create_model
from options.test_options import TestOptions
from data.base_dataset import get_transform

def load_model(model_name, epoch='latest'):
    """Load the trained pix2pix model"""
    import sys
    
    # Temporarily modify sys.argv to pass arguments to TestOptions
    original_argv = sys.argv.copy()
    sys.argv = ['use_model.py', '--dataroot', './dummy', '--name', model_name, 
                '--model', 'pix2pix', '--dataset_mode', 'robust_nifti',
                '--preprocess', 'resize_and_crop', '--input_nc', '1', '--output_nc', '1',
                '--netG', 'unet_256', '--crop_size', '256', '--load_size', '286',
                '--norm', 'instance', '--epoch', epoch]
    
    try:
        # Set up options (mimicking training setup)
        opt = TestOptions().parse()
        
        # Add missing attributes that BaseModel expects
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create and load model
        model = create_model(opt)
        try:
            model.setup(opt)
        except (AttributeError, KeyError) as e:
            print(f"Warning: Error during model setup: {e}")
            print("Trying alternative loading method...")
            # Try to load manually without the problematic patching
            for name in model.model_names:
                if isinstance(name, str):
                    net = getattr(model, 'net' + name)
                    load_filename = f"{epoch}_net_{name}.pth"
                    load_path = model.save_dir / load_filename
                    if load_path.exists():
                        print(f"Loading {load_path}")
                        state_dict = torch.load(load_path, map_location=str(opt.device), weights_only=True)
                        net.load_state_dict(state_dict, strict=False)
                        net.to(opt.device)
                    else:
                        print(f"Warning: Model file not found: {load_path}")
                        return None, None
        model.eval()
        
        return model, opt
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def preprocess_slice(slice_2d, opt):
    """Preprocess a 2D slice for model input"""
    # Normalize to [0, 255]
    denom = np.max(slice_2d) - np.min(slice_2d)
    if denom == 0:
        slice_2d = np.zeros_like(slice_2d)
    else:
        slice_2d = (slice_2d - np.min(slice_2d)) / denom * 255.0
    
    # Convert to PIL Image
    img = Image.fromarray(np.uint8(slice_2d)).convert('L')
    
    # Resize to model's expected input size (256x256 based on crop_size)
    target_size = opt.crop_size if hasattr(opt, 'crop_size') else 256
    img = img.resize((target_size, target_size), Image.LANCZOS)
    
    # Apply transforms
    transform = get_transform(opt, grayscale=True)
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return tensor

def generate_phase_ct(input_nifti_path, model_name='ct_phase0_generator_optimized', epoch='latest', output_dir='./results'):
    """
    Generate phase 0 CT from average CT
    
    Args:
        input_nifti_path: Path to input NIfTI file (average CT)
        model_name: Name of trained model
        epoch: Which epoch to use ('latest', '200', etc.)
        output_dir: Where to save results
    """
    
    print(f"Loading model: {model_name}, epoch: {epoch}")
    model, opt = load_model(model_name, epoch)
    
    print(f"Loading input NIfTI: {input_nifti_path}")
    # Normalize path for cross-platform compatibility
    input_nifti_path = os.path.normpath(input_nifti_path)
    # Load input NIfTI
    nifti_img = nib.load(input_nifti_path)
    volume = nifti_img.get_fdata()
    
    print(f"Input volume shape: {volume.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each axial slice
    num_slices = volume.shape[2]
    generated_volume = np.zeros_like(volume)
    
    print(f"Processing {num_slices} slices...")
    
    for slice_idx in range(num_slices):
        if slice_idx % 10 == 0:
            print(f"Processing slice {slice_idx}/{num_slices}")
            
        # Extract slice
        input_slice = volume[:, :, slice_idx]
        
        # Preprocess
        input_tensor = preprocess_slice(input_slice, opt)
        
        # Generate
        with torch.no_grad():
            model.set_input({'A': input_tensor, 'B': input_tensor, 'A_paths': [''], 'B_paths': ['']})
            model.test()
            visuals = model.get_current_visuals()
            
            # Get generated image
            fake_B = visuals['fake_B']
            
            # Convert back to numpy
            generated_slice = fake_B.cpu().numpy().squeeze()
            
            # Debug: print range of generated values
            if slice_idx == 0:
                print(f"Generated slice range: [{generated_slice.min():.3f}, {generated_slice.max():.3f}]")
            
            # Normalize to [0, 255] - handle different output ranges
            if generated_slice.min() >= -1 and generated_slice.max() <= 1:
                # Model outputs [-1, 1]
                generated_slice = (generated_slice + 1) / 2 * 255.0
            elif generated_slice.min() >= 0 and generated_slice.max() <= 1:
                # Model outputs [0, 1]
                generated_slice = generated_slice * 255.0
            else:
                # Unknown range, normalize to [0, 255]
                generated_slice = (generated_slice - generated_slice.min()) / (generated_slice.max() - generated_slice.min()) * 255.0
            
            generated_slice = np.clip(generated_slice, 0, 255)
            
            # Resize back to original slice dimensions
            original_shape = input_slice.shape
            generated_img = Image.fromarray(np.uint8(generated_slice))
            generated_img = generated_img.resize(original_shape[::-1], Image.LANCZOS)  # PIL uses (width, height)
            generated_slice_resized = np.array(generated_img)
            
            # Store in volume
            generated_volume[:, :, slice_idx] = generated_slice_resized
            
            # Save individual slice as PNG
            slice_img = Image.fromarray(np.uint8(generated_slice))
            slice_img.save(os.path.join(output_dir, f'generated_slice_{slice_idx:03d}.png'))
    
    # Save as NIfTI
    output_nifti = nib.Nifti1Image(generated_volume, nifti_img.affine, nifti_img.header)
    output_path = os.path.join(output_dir, 'generated_phase0_ct.nii.gz')
    nib.save(output_nifti, output_path)
    
    print(f"Results saved to: {output_dir}")
    print(f"Generated NIfTI: {output_path}")
    print(f"Individual slices: {output_dir}/generated_slice_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate phase 0 CT from average CT')
    parser.add_argument('--input_nifti', required=True, help='Path to input NIfTI file (average CT)')
    parser.add_argument('--model_name', default='ct_phase0_generator_optimized', help='Name of trained model')
    parser.add_argument('--epoch', default='latest', help='Which epoch to use (latest, 200, etc.)')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    generate_phase_ct(args.input_nifti, args.model_name, args.epoch, args.output_dir)