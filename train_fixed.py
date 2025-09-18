#!/usr/bin/env python3
"""
Fixed training script for CT phase generation with proper normalization settings.
This script addresses the batch normalization issue when using batch_size=1.
"""

import subprocess
import sys

def main():
    # Fixed training command with instance normalization instead of batch normalization
    cmd = [
        sys.executable, "train.py",
        "--dataroot", "./datasets/ct_phases_dataset",
        "--name", "ct_phase0_generator",
        "--model", "pix2pix",
        "--dataset_mode", "nifti_aligned",
        "--preprocess", "none",
        "--input_nc", "1",
        "--output_nc", "1",
        "--axial_slice",
        "--norm", "instance",  # Changed from batch to instance normalization
        "--batch_size", "1"    # Explicitly set batch size to 1
    ]
    
    print("Starting training with fixed normalization settings...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return 1
    
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())