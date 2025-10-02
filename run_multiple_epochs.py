#!/usr/bin/env python3
"""
Run batch processing for multiple epochs (50, 100, 150) for patients 1-5
"""

import subprocess
import sys

def run_epoch(epoch):
    """Run batch processing for a specific epoch"""
    cmd = [
        sys.executable, "batch_process_patients.py",
        "--data_dir", r"C:\Users\wahbi\OneDrive\Desktop\pytorch-CycleGAN-and-pix2pix\datasets\ct_phases_dataset\trainA",
        "--model_name", "ct_phase0_generator_optimized",
        "--epoch", str(epoch),
        "--output_dir", f"./results_epoch_{epoch}",
        "--checkpoints_dir", r"C:\Users\wahbi\Downloads\seg\DG-20250924T214348Z-1-001\DG",
        "--patient_range", "1", "6"
    ]
    
    print(f"\n{'='*50}")
    print(f"Running epoch {epoch}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    epochs = [50, 100, 150]
    
    for epoch in epochs:
        success = run_epoch(epoch)
        if not success:
            print(f"Failed to process epoch {epoch}")
        else:
            print(f"Successfully completed epoch {epoch}")

if __name__ == "__main__":
    main()