#!/usr/bin/env python3
"""
Optimized training script for CT phase generation with improved settings
"""
import subprocess
import sys
import torch

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected. Training will be slow on CPU.")
        return False, 0
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU detected: {gpu_name}")
    print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    return True, gpu_memory

def get_optimal_settings(gpu_memory_gb):
    """Get optimal training settings based on GPU memory"""
    if gpu_memory_gb >= 8:
        return {
            'batch_size': 4,
            'image_size': 512,
            'netG': 'unet_512',
            'lambda_L1': 50
        }
    elif gpu_memory_gb >= 4:
        return {
            'batch_size': 2,
            'image_size': 256,
            'netG': 'unet_256',
            'lambda_L1': 50
        }
    else:
        return {
            'batch_size': 1,
            'image_size': 128,
            'netG': 'unet_128',
            'lambda_L1': 100
        }

def main():
    print("🚀 Starting optimized CT phase generation training...")
    
    # Check GPU
    has_gpu, gpu_memory = check_gpu()
    
    # Get optimal settings
    settings = get_optimal_settings(gpu_memory)
    print(f"\n📊 Optimal settings for your hardware:")
    print(f"   Image size: {settings['image_size']}x{settings['image_size']}")
    print(f"   Batch size: {settings['batch_size']}")
    print(f"   Generator: {settings['netG']}")
    print(f"   L1 weight: {settings['lambda_L1']}")
    
    # Build training command
    cmd = [
        'python', 'train.py',
        '--dataroot', './datasets/ct_phases_dataset',
        '--name', 'ct_phase0_generator_optimized',
        '--model', 'pix2pix',
        '--dataset_mode', 'nifti_aligned',
        '--preprocess', 'none',
        '--input_nc', '1',
        '--output_nc', '1',
        '--axial_slice',
        '--norm', 'instance',
        '--netG', settings['netG'],
        '--batch_size', str(settings['batch_size']),
        '--lambda_L1', str(settings['lambda_L1']),
        '--lr', '0.0001',  # Lower learning rate for stability
        '--resize_to', str(settings['image_size']),
        '--display_freq', '100',
        '--save_epoch_freq', '5'
    ]
    
    print(f"\n🔧 Training command:")
    print(' '.join(cmd))
    print(f"\n⏳ Starting training...")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())