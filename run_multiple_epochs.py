#!/usr/bin/env python3
"""
Run use_model.py for multiple epochs on your CT dataset
"""

import subprocess
import sys
import os
import glob

def run_epoch(epoch, input_files):
    """Run use_model.py for a specific epoch on all input files"""
    success_count = 0
    
    for input_file in input_files:
        filename = os.path.basename(input_file).replace('.nii.gz', '').replace('.nii', '')
        output_dir = f"./results_epoch_{epoch}/{filename}"
        
        cmd = [
            sys.executable, "use_model.py",
            "--input_nifti", input_file,
            "--model_name", "ct_phase0_generator_5090",
            "--epoch", str(epoch),
            "--output_dir", output_dir,
            "--checkpoints_dir", r"C:\Users\Wahbi Saied\Documents\GitHub\pix2pix\checkpoints"
        ]
        
        print(f"\nProcessing: {filename} with epoch {epoch}")
        print(f"Output: {output_dir}")
        
        result = subprocess.run(cmd)
        if result.returncode == 0:
            success_count += 1
            print(f"✓ Successfully processed {filename}")
        else:
            print(f"✗ Failed to process {filename}")
    
    return success_count

def main():
    # Your dataset path
    data_dir = r"D:\ct_phases_datasets\ct_phase0_dataset\testA"
    
    # Find all NIfTI files in the dataset
    nifti_patterns = ["*.nii.gz", "*.nii"]
    input_files = []
    
    for pattern in nifti_patterns:
        input_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    if not input_files:
        print(f"No NIfTI files found in {data_dir}")
        print("Looking for files with extensions: .nii.gz, .nii")
        return
    
    print(f"Found {len(input_files)} NIfTI files:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    
    epochs = [200]
    
    for epoch in epochs:
        print(f"\n{'='*60}")
        print(f"Running epoch {epoch} on {len(input_files)} files")
        print(f"{'='*60}")
        
        success_count = run_epoch(epoch, input_files)
        print(f"\nEpoch {epoch} completed: {success_count}/{len(input_files)} files processed successfully")

if __name__ == "__main__":
    main()