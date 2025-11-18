#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_phase_transfer.py [phase_number]")
        print("Example: python train_phase_transfer.py 1")
        return 1
    
    phase = sys.argv[1]
    source_dir = "checkpoints/ct_phase0_generator_5090"
    target_dir = f"checkpoints/ct_phase{phase}_generator_5090"
    
    # Check if Phase 0 model exists
    source_g_path = f"{source_dir}/latest_net_G.pth"
    if not os.path.exists(source_g_path):
        print(f"ERROR: Phase 0 model not found at {source_g_path}")
        print("Make sure Phase 0 training is completed first!")
        return 1
    
    print("=" * 40)
    print(f"CT Phase {phase} - Transfer Learning")
    print("=" * 40)
    print(f"✓ Found Phase 0 model: {source_g_path}")
    
    # Copy Phase 0 weights to Phase N directory
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(f"{source_dir}/latest_net_G.pth", f"{target_dir}/latest_net_G.pth")
    
    # Copy discriminator if exists
    source_d_path = f"{source_dir}/latest_net_D.pth"
    if os.path.exists(source_d_path):
        shutil.copy2(source_d_path, f"{target_dir}/latest_net_D.pth")
    
    # Create epoch marker
    with open(f"{target_dir}/latest_net_G.pth.epoch", "w") as f:
        f.write("1")
    
    print("✓ Copied Phase 0 weights for initialization")
    print("Target: 50 epochs (25 + 25 decay)")
    print()
    
    # Training command
    cmd = [
        "python", "train.py",
        "--dataroot", f"D:\\ct_phases_datasets\\ct_phase{phase}_dataset",
        "--name", f"ct_phase{phase}_generator_5090",
        "--model", "pix2pix",
        "--dataset_mode", "robust_nifti",
        "--skip_corrupted",
        "--preprocess", "none",
        "--input_nc", "1",
        "--output_nc", "1",
        "--axial_slice",
        "--norm", "instance",
        "--netG", "unet_512",
        "--batch_size", "16",
        "--lambda_L1", "50",
        "--lr", "0.0001",
        "--resize_to", "512",
        "--display_freq", "50",
        "--save_epoch_freq", "5",
        "--num_threads", "16",
        "--continue_train",
        "--epoch_count", "1",
        "--n_epochs", "25",
        "--n_epochs_decay", "25"
    ]
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print(f"✓ Phase {phase} transfer learning completed!")
    else:
        print(f"✗ Phase {phase} training failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    sys.exit(main())