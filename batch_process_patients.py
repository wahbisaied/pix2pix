#!/usr/bin/env python3
r"""
Batch processing script to generate fake_B for multiple patients using the trained model.

Usage:
    python batch_process_patients.py --data_dir C:\Users\wahbi\Downloads\seg\data --model_name ct_phase0_generator_optimized --epoch 50 --output_dir .\best_results --checkpoints_dir C:\Users\wahbi\Downloads\seg\DG-20250924T214348Z-1-001\DG

This script processes patients 1-5 by calling use_model.py for each patient file.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def find_patient_files(data_dir, patient_range=(1, 6)):
    """Find patient files in the data directory"""
    patient_files = []
    data_path = Path(data_dir)
    
    for patient_num in range(patient_range[0], patient_range[1]):
        # Look for patient_X.nii.gz files
        pattern = f"patient_{patient_num}.nii.gz"
        patient_file = data_path / pattern
        
        if patient_file.exists():
            patient_files.append(str(patient_file))
            print(f"Found: {patient_file}")
        else:
            print(f"Warning: Patient file not found: {patient_file}")
    
    return patient_files

def process_patient(input_file, model_name, epoch, output_dir, checkpoints_dir):
    """Process a single patient file using use_model.py"""
    patient_name = Path(input_file).stem
    patient_output_dir = os.path.join(output_dir, patient_name)
    
    # Prepare command
    cmd = [
        sys.executable, "use_model.py",
        "--input_nifti", input_file,
        "--model_name", model_name,
        "--epoch", str(epoch),
        "--output_dir", patient_output_dir
    ]
    
    print(f"\nProcessing {patient_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Set environment variable for checkpoints directory
        env = os.environ.copy()
        if checkpoints_dir:
            # Modify the model loading to use custom checkpoints directory
            # This requires updating the use_model.py or passing it as an environment variable
            env['CHECKPOINTS_DIR'] = checkpoints_dir
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"✓ Successfully processed {patient_name}")
            print(f"  Output saved to: {patient_output_dir}")
        else:
            print(f"✗ Error processing {patient_name}")
            print(f"  Error: {result.stderr}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Exception processing {patient_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch process multiple patients to generate fake_B')
    parser.add_argument('--data_dir', required=True, help='Directory containing patient_X.nii.gz files')
    parser.add_argument('--model_name', default='ct_phase0_generator_optimized', help='Name of trained model')
    parser.add_argument('--epoch', default='50', help='Which epoch to use')
    parser.add_argument('--output_dir', default='./best_results', help='Base output directory')
    parser.add_argument('--checkpoints_dir', help='Custom checkpoints directory')
    parser.add_argument('--patient_range', nargs=2, type=int, default=[1, 6], 
                       help='Patient range to process (start, end+1). Default: 1 6 (processes 1-5)')
    
    args = parser.parse_args()
    
    print("=== Batch Patient Processing ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model_name}")
    print(f"Epoch: {args.epoch}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Patient range: {args.patient_range[0]}-{args.patient_range[1]-1}")
    
    # Find patient files
    patient_files = find_patient_files(args.data_dir, tuple(args.patient_range))
    
    if not patient_files:
        print("No patient files found!")
        return
    
    print(f"\nFound {len(patient_files)} patient files to process")
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each patient
    successful = 0
    failed = 0
    
    for patient_file in patient_files:
        success = process_patient(
            patient_file, 
            args.model_name, 
            args.epoch, 
            args.output_dir,
            args.checkpoints_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(patient_files)}")
    
    if successful > 0:
        print(f"\nResults saved in: {args.output_dir}")
        print("Each patient has its own subdirectory with:")
        print("  - generated_phase0_ct.nii.gz (fake_B volume)")
        print("  - generated_slice_*.png (individual slices)")

if __name__ == "__main__":
    main()