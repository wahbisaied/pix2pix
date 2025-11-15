import os
import shutil
import glob

def prepare_phase_dataset_robust(phase_num, base_output_dir):
    """Prepare dataset for specific phase with robust handling"""
    
    # Source directories
    avg_src_dir = r"C:\Users\Wahbi Saied\OneDrive\avg"
    phases_src_dir = r"C:\Users\Wahbi Saied\OneDrive\final_data"
    
    # Excluded patients (problematic ones)
    excluded_patients = {
        "08-10-1998-NA-NA-85576",
        "06-09-1998-NA-NA-78411", 
        "12-10-1999-NA-p4-47741"
    }
    
    # Output directories
    phase_dir = os.path.join(base_output_dir, f"ct_phase{phase_num}_dataset")
    trainA_dest = os.path.join(phase_dir, "trainA")
    trainB_dest = os.path.join(phase_dir, "trainB")
    
    os.makedirs(trainA_dest, exist_ok=True)
    os.makedirs(trainB_dest, exist_ok=True)
    
    print(f"🚀 Préparation Phase {phase_num}...")
    
    if not os.path.exists(avg_src_dir) or not os.path.exists(phases_src_dir):
        print(f"❌ Source directories not found")
        return 0
    
    scan_folders = os.listdir(avg_src_dir)
    file_counter = 1
    
    for scan_id in scan_folders:
        # Skip excluded patients
        if scan_id in excluded_patients:
            print(f"  ⚠️  Skipping excluded patient: {scan_id}")
            continue
            
        avg_scan_folder = os.path.join(avg_src_dir, scan_id)
        phases_scan_folder = os.path.join(phases_src_dir, scan_id)
        
        if not os.path.isdir(avg_scan_folder) or not os.path.isdir(phases_scan_folder):
            continue
        
        # Find AVG file
        avg_files = glob.glob(os.path.join(avg_scan_folder, '*.nii*'))
        if not avg_files:
            continue
        avg_src_path = avg_files[0]
        
        # Find Phase file with multiple patterns
        phase_files = []
        
        # Try different patterns for phase files
        patterns = []
        if phase_num == 0:
            patterns = [
                f'*Gated 0.0A*.nii*',
                f'*Gated 0.0B*.nii*',
                f'*Gated 0.0*.nii*'  # Without A/B
            ]
        else:
            patterns = [
                f'*Gated {phase_num}0.0A*.nii*',
                f'*Gated {phase_num}0.0B*.nii*',
                f'*Gated {phase_num}0.0*.nii*'  # Without A/B
            ]
        
        for pattern in patterns:
            phase_pattern = os.path.join(phases_scan_folder, pattern)
            found_files = glob.glob(phase_pattern)
            if found_files:
                phase_files.extend(found_files)
                break
        
        if not phase_files:
            print(f"  ❌ No phase {phase_num} file found for {scan_id}")
            continue
            
        phase_src_path = phase_files[0]
        
        # Copy files
        extension = '.nii.gz' if phase_src_path.endswith('.nii.gz') else '.nii'
        new_filename = f"patient_{file_counter}{extension}"
        
        dest_A_path = os.path.join(trainA_dest, new_filename)
        dest_B_path = os.path.join(trainB_dest, new_filename)
        
        try:
            shutil.copy(avg_src_path, dest_A_path)
            shutil.copy(phase_src_path, dest_B_path)
            print(f"  ✅ Patient {file_counter}: {scan_id}")
            file_counter += 1
        except Exception as e:
            print(f"  ❌ Erreur patient {file_counter}: {e}")
    
    patients_created = file_counter - 1
    print(f"🎉 Phase {phase_num}: {patients_created} patients créés")
    return patients_created

def create_all_phase_datasets_robust():
    """Create datasets for all 10 phases with robust handling"""
    
    base_output_dir = "D:\\ct_phases_datasets"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("🚀 Création robuste de tous les datasets de phases...")
    print("⚠️  Exclusion de 3 patients problématiques")
    print("🎯 Objectif: 80 patients par phase")
    
    total_patients = {}
    
    # Create datasets for phases 0-9
    for phase in range(10):
        patients_count = prepare_phase_dataset_robust(phase, base_output_dir)
        total_patients[phase] = patients_count
        print()
    
    print("📊 Résumé final:")
    for phase, count in total_patients.items():
        status = "✅" if count == 80 else "⚠️"
        print(f"Phase {phase}: {count} patients {status}")
    
    # Check if we got 80 patients for all phases
    if all(count == 80 for count in total_patients.values()):
        print("\n🎉 Succès ! 80 patients pour toutes les phases")
    else:
        print("\n⚠️  Certaines phases n'ont pas 80 patients")
    
    print(f"\n📁 Tous les datasets créés dans: {base_output_dir}")

if __name__ == "__main__":
    create_all_phase_datasets_robust()