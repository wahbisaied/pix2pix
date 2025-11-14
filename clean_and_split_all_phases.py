import os
import shutil
import glob
import random
import nibabel as nib

def validate_nifti_file(file_path):
    """Check if NIfTI file is valid"""
    try:
        nib.load(file_path)
        return True
    except Exception as e:
        print(f"❌ Fichier corrompu: {file_path} - {e}")
        return False

def get_valid_patients(trainA_dir, trainB_dir):
    """Get list of valid patient pairs"""
    patient_files_A = glob.glob(os.path.join(trainA_dir, 'patient_*.nii*'))
    valid_patients = []
    
    for file_A in patient_files_A:
        filename = os.path.basename(file_A)
        file_B = os.path.join(trainB_dir, filename)
        
        if os.path.exists(file_B):
            if validate_nifti_file(file_A) and validate_nifti_file(file_B):
                patient_num = filename.split('patient_')[1].split('.')[0]
                valid_patients.append(int(patient_num))
            else:
                # Delete corrupted files
                if os.path.exists(file_A):
                    os.remove(file_A)
                if os.path.exists(file_B):
                    os.remove(file_B)
    
    return sorted(valid_patients)

def create_phase_datasets():
    """Create datasets for all 10 phases with same train/test split"""
    
    base_dir = "D:\\ct_phases_datasets"  # Note: plural
    os.makedirs(base_dir, exist_ok=True)
    
    # Get valid patients from phase 0 dataset
    phase0_dir = "D:\\ct_phases_dataset"
    trainA_dir = os.path.join(phase0_dir, 'trainA')
    trainB_dir = os.path.join(phase0_dir, 'trainB')
    
    print("🔍 Validation des fichiers Phase 0...")
    valid_patients = get_valid_patients(trainA_dir, trainB_dir)
    print(f"✅ {len(valid_patients)} patients valides trouvés")
    
    # Create consistent train/test split
    random.seed(42)  # Same seed for reproducible split
    random.shuffle(valid_patients)
    
    split_idx = int(len(valid_patients) * 0.8)
    train_patients = valid_patients[:split_idx]
    test_patients = valid_patients[split_idx:]
    
    print(f"📊 Split: {len(train_patients)} train, {len(test_patients)} test")
    print(f"Train patients: {train_patients}")
    print(f"Test patients: {test_patients}")
    
    # Create phase directories
    for phase in range(10):
        phase_dir = os.path.join(base_dir, f"phase_{phase}")
        
        # Create train/test folders
        for split in ['train', 'test']:
            for domain in ['A', 'B']:
                folder = os.path.join(phase_dir, f"{split}{domain}")
                os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Créé: {phase_dir}")
    
    # Save split info
    split_file = os.path.join(base_dir, "train_test_split.txt")
    with open(split_file, 'w') as f:
        f.write(f"Train patients ({len(train_patients)}): {train_patients}\n")
        f.write(f"Test patients ({len(test_patients)}): {test_patients}\n")
    
    print(f"💾 Split sauvé dans: {split_file}")
    print(f"🎯 Structure créée pour 10 phases dans: {base_dir}")
    
    return train_patients, test_patients

if __name__ == "__main__":
    create_phase_datasets()