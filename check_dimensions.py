#!/usr/bin/env python3
import nibabel as nib
import os

# Check dimensions of a few NIfTI files
dataset_dir = "./datasets/ct_phases_dataset"
trainA_dir = os.path.join(dataset_dir, "trainA")
trainB_dir = os.path.join(dataset_dir, "trainB")

print("Checking NIfTI file dimensions...")
print("=" * 50)

# Check first few files
for i in range(1, 4):
    file_a = os.path.join(trainA_dir, f"patient_{i}.nii.gz")
    file_b = os.path.join(trainB_dir, f"patient_{i}.nii.gz")
    
    if os.path.exists(file_a) and os.path.exists(file_b):
        img_a = nib.load(file_a)
        img_b = nib.load(file_b)
        
        print(f"Patient {i}:")
        print(f"  TrainA shape: {img_a.shape}")
        print(f"  TrainB shape: {img_b.shape}")
        print(f"  Voxel size A: {img_a.header.get_zooms()}")
        print(f"  Voxel size B: {img_b.header.get_zooms()}")
        print()