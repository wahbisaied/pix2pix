#!/usr/bin/env python3
"""
Quick Setup Script for Working CT Phase Generator Model
This script helps set up the environment and verify requirements for the working model.
"""

import os
import sys
import subprocess
import torch
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version should be 3.8-3.11")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available - training will be slow on CPU")
            return False
    except Exception as e:
        print(f"✗ PyTorch not properly installed: {e}")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pillow', 'matplotlib', 'nibabel'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def check_directory_structure():
    """Check if required directories and files exist."""
    required_dirs = [
        'data', 'models', 'options', 'util', 'datasets'
    ]
    
    required_files = [
        'train.py', 'test.py', 'data/robust_nifti_dataset.py'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' missing")
            missing_items.append(dir_name)
    
    # Check files
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File '{file_name}' exists")
        else:
            print(f"✗ File '{file_name}' missing")
            missing_items.append(file_name)
    
    return missing_items

def check_dataset():
    """Check if dataset is properly structured."""
    dataset_path = "datasets/ct_phases_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset directory '{dataset_path}' not found")
        return False
    
    trainA_path = os.path.join(dataset_path, "trainA")
    trainB_path = os.path.join(dataset_path, "trainB")
    
    if not os.path.exists(trainA_path):
        print(f"✗ trainA directory missing: {trainA_path}")
        return False
    
    if not os.path.exists(trainB_path):
        print(f"✗ trainB directory missing: {trainB_path}")
        return False
    
    # Count files
    trainA_files = [f for f in os.listdir(trainA_path) if f.endswith(('.nii', '.nii.gz'))]
    trainB_files = [f for f in os.listdir(trainB_path) if f.endswith(('.nii', '.nii.gz'))]
    
    print(f"✓ Dataset found: {len(trainA_files)} files in trainA, {len(trainB_files)} files in trainB")
    
    if len(trainA_files) != len(trainB_files):
        print("⚠ Warning: trainA and trainB have different number of files")
    
    return len(trainA_files) > 0 and len(trainB_files) > 0

def generate_training_command():
    """Generate the exact training command."""
    command = """
# Exact working training command:
python train.py \\
    --dataroot ./datasets/ct_phases_dataset \\
    --name ct_phase0_generator_optimized \\
    --model pix2pix \\
    --dataset_mode robust_nifti \\
    --skip_corrupted \\
    --preprocess none \\
    --input_nc 1 \\
    --output_nc 1 \\
    --axial_slice \\
    --norm instance \\
    --netG unet_512 \\
    --batch_size 4 \\
    --lambda_L1 50 \\
    --lr 0.0001 \\
    --resize_to 512 \\
    --display_freq 100 \\
    --save_epoch_freq 5 \\
    --n_epochs 100 \\
    --n_epochs_decay 100
"""
    return command

def main():
    """Main setup verification function."""
    print("=" * 60)
    print("CT Phase Generator Model - Setup Verification")
    print("=" * 60)
    
    print("\n1. Checking Python version...")
    python_ok = check_python_version()
    
    print("\n2. Checking PyTorch and CUDA...")
    pytorch_ok = check_pytorch()
    
    print("\n3. Checking required packages...")
    missing_packages = check_required_packages()
    
    print("\n4. Checking directory structure...")
    missing_items = check_directory_structure()
    
    print("\n5. Checking dataset...")
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if python_ok and pytorch_ok and not missing_packages and not missing_items and dataset_ok:
        print("✓ All requirements met! Ready to train.")
        print("\nTo start training, run:")
        print(generate_training_command())
    else:
        print("✗ Some requirements are missing:")
        
        if missing_packages:
            print(f"\nInstall missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
        
        if missing_items:
            print(f"\nMissing files/directories: {', '.join(missing_items)}")
        
        if not dataset_ok:
            print("\nDataset not properly configured. Check dataset structure.")
    
    print("\n" + "=" * 60)
    print("System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("Recommended batch_size: 4 (for 8GB+ GPU)")
    else:
        print("No GPU detected - consider using Google Colab or cloud GPU")

if __name__ == "__main__":
    main()