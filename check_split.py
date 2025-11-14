import os
import glob

def check_dataset_split(dataset_dir):
    """Check current dataset split"""
    
    folders = ['trainA', 'trainB', 'testA', 'testB']
    
    print("=== Dataset Split Status ===")
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, 'patient_*.nii*'))
            print(f"{folder}: {len(files)} patients")
        else:
            print(f"{folder}: Folder doesn't exist")
    
    # Check if split is needed
    trainA_path = os.path.join(dataset_dir, 'trainA')
    testA_path = os.path.join(dataset_dir, 'testA')
    
    if os.path.exists(trainA_path) and not os.path.exists(testA_path):
        train_files = glob.glob(os.path.join(trainA_path, 'patient_*.nii*'))
        print(f"\n⚠️  Pas de split - tous les {len(train_files)} patients sont dans train/")
        print("Exécuter: python simple_train_test_split.py")
    elif os.path.exists(testA_path):
        print("\n✅ Split déjà fait")

if __name__ == "__main__":
    check_dataset_split("D:\\ct_phases_dataset")