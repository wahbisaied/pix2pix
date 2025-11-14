import os
import shutil
import glob
import random

def create_train_test_split(source_dir, train_ratio=0.8):
    """Split dataset into train/test folders only"""
    
    trainA_dir = os.path.join(source_dir, 'trainA')
    trainB_dir = os.path.join(source_dir, 'trainB')
    
    # Create test directories
    testA_dir = os.path.join(source_dir, 'testA')
    testB_dir = os.path.join(source_dir, 'testB')
    os.makedirs(testA_dir, exist_ok=True)
    os.makedirs(testB_dir, exist_ok=True)
    
    # Get all patient files
    patient_files = glob.glob(os.path.join(trainA_dir, 'patient_*.nii*'))
    patient_files.sort()
    
    print(f"Total patients: {len(patient_files)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(patient_files)
    
    split_idx = int(len(patient_files) * train_ratio)
    train_files = patient_files[:split_idx]
    test_files = patient_files[split_idx:]
    
    print(f"Train: {len(train_files)} patients")
    print(f"Test: {len(test_files)} patients")
    
    # Move test files
    for file_path in test_files:
        filename = os.path.basename(file_path)
        
        # Move A file
        shutil.move(file_path, os.path.join(testA_dir, filename))
        
        # Move corresponding B file
        b_file = os.path.join(trainB_dir, filename)
        if os.path.exists(b_file):
            shutil.move(b_file, os.path.join(testB_dir, filename))
    
    print("\nSplit terminé !")
    print(f"Entraînement: {len(train_files)} patients")
    print(f"Test: {len(test_files)} patients")

if __name__ == "__main__":
    create_train_test_split("D:\\ct_phases_dataset", train_ratio=0.8)