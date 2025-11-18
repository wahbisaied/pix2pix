#!/usr/bin/env python3
"""
Convert NIfTI files to PNG slices for HTML viewer
"""

import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image

def convert_nifti_to_png_slices(nifti_path, output_dir, filename_prefix):
    """Convert a NIfTI file to individual PNG slices"""
    
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()
    
    print(f"Processing {nifti_path}")
    print(f"Volume shape: {volume.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each axial slice
    num_slices = volume.shape[2]
    
    for slice_idx in range(num_slices):
        # Extract slice
        slice_2d = volume[:, :, slice_idx]
        
        # Normalize to [0, 255]
        if slice_2d.max() > slice_2d.min():
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255.0
        else:
            slice_2d = np.zeros_like(slice_2d)
        
        # Convert to PIL Image
        img = Image.fromarray(np.uint8(slice_2d)).convert('L')
        
        # Save as PNG
        slice_filename = f"{filename_prefix}_slice_{slice_idx:03d}.png"
        output_path = os.path.join(output_dir, slice_filename)
        img.save(output_path)
    
    print(f"Saved {num_slices} slices to {output_dir}")

def main():
    # Paths
    testA_dir = r"D:\ct_phases_datasets\ct_phase0_dataset\testA"
    testB_dir = r"D:\ct_phases_datasets\ct_phase0_dataset\testB"
    
    # Output directories
    testA_png_dir = r"D:\viewer\testA_png"
    testB_png_dir = r"D:\viewer\testB_png"
    
    # Patient files from your results
    patient_files = [
        'patient_12', 'patient_13', 'patient_2', 'patient_21', 'patient_22', 
        'patient_25', 'patient_35', 'patient_38', 'patient_41', 'patient_59', 
        'patient_72', 'patient_75'
    ]
    
    print("Converting testA files...")
    for patient in patient_files:
        # Try different extensions
        nifti_patterns = [f"{patient}.nii.gz", f"{patient}.nii"]
        
        for pattern in nifti_patterns:
            nifti_path = os.path.join(testA_dir, pattern)
            if os.path.exists(nifti_path):
                convert_nifti_to_png_slices(nifti_path, testA_png_dir, patient)
                break
        else:
            print(f"Warning: {patient} not found in testA")
    
    print("\nConverting testB files...")
    for patient in patient_files:
        # Try different extensions
        nifti_patterns = [f"{patient}.nii.gz", f"{patient}.nii"]
        
        for pattern in nifti_patterns:
            nifti_path = os.path.join(testB_dir, pattern)
            if os.path.exists(nifti_path):
                convert_nifti_to_png_slices(nifti_path, testB_png_dir, patient)
                break
        else:
            print(f"Warning: {patient} not found in testB")
    
    print(f"\nConversion complete!")
    print(f"testA PNG files: {testA_png_dir}")
    print(f"testB PNG files: {testB_png_dir}")

if __name__ == "__main__":
    main()