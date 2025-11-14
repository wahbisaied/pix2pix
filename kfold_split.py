import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import KFold

def create_kfold_splits(source_dir, k=5):
    """Create k-fold cross-validation splits"""
    
    trainA_dir = os.path.join(source_dir, 'trainA')
    trainB_dir = os.path.join(source_dir, 'trainB')
    
    # Get all patient files
    patient_files = glob.glob(os.path.join(trainA_dir, 'patient_*.nii*'))
    patient_files.sort()
    patient_nums = [int(f.split('patient_')[1].split('.')[0]) for f in patient_files]
    
    print(f"Found {len(patient_files)} patients")
    
    # Create k-fold splits
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_nums)):
        print(f"\nFold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Create fold directories
        fold_dir = os.path.join(source_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create train/val splits for this fold
        for split, indices in [('train', train_idx), ('val', val_idx)]:
            for domain in ['A', 'B']:
                split_dir = os.path.join(fold_dir, f'{split}{domain}')
                os.makedirs(split_dir, exist_ok=True)
                
                for idx in indices:
                    patient_num = patient_nums[idx]
                    filename = f'patient_{patient_num}.nii.gz'
                    
                    src_file = os.path.join(source_dir, f'train{domain}', filename)
                    dst_file = os.path.join(split_dir, filename)
                    
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    create_kfold_splits("D:\\ct_phases_dataset", k=5)