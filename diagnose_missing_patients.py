import os
import glob
import nibabel as nib

def diagnose_dataset():
    """Diagnostiquer les patients manquants"""
    
    trainA_dir = "D:\\ct_phases_dataset\\trainA"
    trainB_dir = "D:\\ct_phases_dataset\\trainB"
    
    print("🔍 Diagnostic des patients manquants...")
    
    # Check all files in trainA
    all_files_A = []
    for root, dirs, files in os.walk(trainA_dir):
        for file in files:
            if file.startswith('patient_') and file.endswith(('.nii', '.nii.gz')):
                all_files_A.append(os.path.join(root, file))
    
    print(f"📁 Total fichiers trouvés dans trainA: {len(all_files_A)}")
    
    # Check corresponding B files
    valid_pairs = []
    corrupted_files = []
    missing_B_files = []
    
    for file_A in all_files_A:
        filename = os.path.basename(file_A)
        
        # Find corresponding B file
        file_B = None
        for root, dirs, files in os.walk(trainB_dir):
            if filename in files:
                file_B = os.path.join(root, filename)
                break
        
        if not file_B:
            missing_B_files.append(filename)
            continue
        
        # Validate both files
        try:
            nib.load(file_A)
            nib.load(file_B)
            patient_num = filename.split('patient_')[1].split('.')[0]
            valid_pairs.append(int(patient_num))
            print(f"✅ Patient {patient_num}: OK")
        except Exception as e:
            corrupted_files.append((filename, str(e)))
            print(f"❌ Patient {filename}: {e}")
    
    print(f"\n📊 Résumé:")
    print(f"Total fichiers A: {len(all_files_A)}")
    print(f"Fichiers B manquants: {len(missing_B_files)}")
    print(f"Fichiers corrompus: {len(corrupted_files)}")
    print(f"Paires valides: {len(valid_pairs)}")
    
    if missing_B_files:
        print(f"\n❌ Fichiers B manquants:")
        for f in missing_B_files:
            print(f"  - {f}")
    
    if corrupted_files:
        print(f"\n❌ Fichiers corrompus:")
        for f, error in corrupted_files:
            print(f"  - {f}: {error}")
    
    # Check for files in subdirectories
    print(f"\n🔍 Vérification des sous-dossiers...")
    
    for domain, dir_path in [('A', trainA_dir), ('B', trainB_dir)]:
        subdirs = []
        for root, dirs, files in os.walk(dir_path):
            if root != dir_path:  # It's a subdirectory
                patient_files = [f for f in files if f.startswith('patient_')]
                if patient_files:
                    subdirs.append((root, patient_files))
        
        if subdirs:
            print(f"📁 Sous-dossiers trouvés dans train{domain}:")
            for subdir, files in subdirs:
                print(f"  - {subdir}: {len(files)} fichiers")
                for f in files[:3]:  # Show first 3
                    print(f"    * {f}")
                if len(files) > 3:
                    print(f"    * ... et {len(files)-3} autres")
    
    return valid_pairs

if __name__ == "__main__":
    diagnose_dataset()