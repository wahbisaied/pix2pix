import os
import shutil
import glob
import random

def split_dataset(source_dir, train_ratio=0.8):
    """Split dataset into train/val folders"""
    
    trainA_dir = os.path.join(source_dir, 'trainA')
    trainB_dir = os.path.join(source_dir, 'trainB')
    
    # Create val directories
    valA_dir = os.path.join(source_dir, 'valA')
    valB_dir = os.path.join(source_dir, 'valB')
    os.makedirs(valA_dir, exist_ok=True)
    os.makedirs(valB_dir, exist_ok=True)
    
    # Get all patient files
    patient_files = glob.glob(os.path.join(trainA_dir, 'patient_*.nii*'))
    patient_files.sort()
    
    print(f"Found {len(patient_files)} patient files")
    
    # Shuffle and split
    random.seed(42)  # For reproducible splits
    random.shuffle(patient_files)
    
    split_idx = int(len(patient_files) * train_ratio)
    train_files = patient_files[:split_idx]
    val_files = patient_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Move validation files
    for file_path in val_files:
        filename = os.path.basename(file_path)
        
        # Move A file
        shutil.move(file_path, os.path.join(valA_dir, filename))
        
        # Move corresponding B file
        b_file = os.path.join(trainB_dir, filename)
        if os.path.exists(b_file):
            shutil.move(b_file, os.path.join(valB_dir, filename))
    
    print("Dataset split complete!")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

if __name__ == "__main__":
    split_dataset("D:\\ct_phases_dataset")