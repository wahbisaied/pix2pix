import os
import shutil
import glob

def prepare_phase_dataset(phase_num, train_patients, test_patients):
    """Prepare dataset for specific phase using predefined train/test split"""
    
    # Source directories
    avg_src_dir = r"C:\Users\wahbi\OneDrive\avg"
    phases_src_dir = r"C:\Users\wahbi\OneDrive\final_data"
    
    # Destination directory
    base_dir = "D:\\ct_phases_datasets"
    phase_dir = os.path.join(base_dir, f"phase_{phase_num}")
    
    # Create directories
    trainA_dest = os.path.join(phase_dir, "trainA")
    trainB_dest = os.path.join(phase_dir, "trainB")
    testA_dest = os.path.join(phase_dir, "testA")
    testB_dest = os.path.join(phase_dir, "testB")
    
    for dir_path in [trainA_dest, trainB_dest, testA_dest, testB_dest]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"🚀 Préparation Phase {phase_num}...")
    
    # Get scan folders
    scan_folders = os.listdir(avg_src_dir)
    processed = 0
    
    for scan_id in scan_folders:
        avg_scan_folder = os.path.join(avg_src_dir, scan_id)
        phases_scan_folder = os.path.join(phases_src_dir, scan_id)
        
        if not os.path.isdir(avg_scan_folder) or not os.path.isdir(phases_scan_folder):
            continue
        
        # Find AVG file
        avg_files = glob.glob(os.path.join(avg_scan_folder, '*.nii*'))
        if not avg_files:
            continue
        avg_src_path = avg_files[0]
        
        # Find Phase file (adjust pattern for each phase)
        phase_pattern = os.path.join(phases_scan_folder, f'*Gated {phase_num/10:.1f}*.nii*')
        phase_files = glob.glob(phase_pattern)
        if not phase_files:
            continue
        phase_src_path = phase_files[0]
        
        # Determine patient number
        processed += 1
        patient_num = processed
        
        # Determine if train or test
        if patient_num in train_patients:
            dest_A = os.path.join(trainA_dest, f"patient_{patient_num}.nii.gz")
            dest_B = os.path.join(trainB_dest, f"patient_{patient_num}.nii.gz")
        elif patient_num in test_patients:
            dest_A = os.path.join(testA_dest, f"patient_{patient_num}.nii.gz")
            dest_B = os.path.join(testB_dest, f"patient_{patient_num}.nii.gz")
        else:
            continue  # Skip if not in our valid patients list
        
        # Copy files
        try:
            shutil.copy2(avg_src_path, dest_A)
            shutil.copy2(phase_src_path, dest_B)
            print(f"✅ Patient {patient_num}: {scan_id}")
        except Exception as e:
            print(f"❌ Erreur patient {patient_num}: {e}")
    
    print(f"🎉 Phase {phase_num} terminée!")

def prepare_all_phases():
    """Prepare datasets for all phases"""
    
    # Read train/test split
    split_file = "D:\\ct_phases_datasets\\train_test_split.txt"
    if not os.path.exists(split_file):
        print("❌ Exécuter d'abord: python clean_and_split_all_phases.py")
        return
    
    with open(split_file, 'r') as f:
        lines = f.readlines()
        train_patients = eval(lines[0].split(': ')[1])
        test_patients = eval(lines[1].split(': ')[1])
    
    print(f"📊 Train: {len(train_patients)}, Test: {len(test_patients)}")
    
    # Prepare each phase
    for phase in range(10):
        prepare_phase_dataset(phase, train_patients, test_patients)

if __name__ == "__main__":
    prepare_all_phases()