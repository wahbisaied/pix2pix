import os
import shutil
import glob

def prepare_phase_dataset(phase_num, base_output_dir):
    """Prepare dataset for specific phase"""
    
    # Source directories (update these paths for your laptop)
    avg_src_dir = r"C:\Users\Wahbi Saied\OneDrive\avg"
    phases_src_dir = r"C:\Users\Wahbi Saied\OneDrive\final_data"
    
    # Output directories
    phase_dir = os.path.join(base_output_dir, f"ct_phase{phase_num}_dataset")
    trainA_dest = os.path.join(phase_dir, "trainA")
    trainB_dest = os.path.join(phase_dir, "trainB")
    
    os.makedirs(trainA_dest, exist_ok=True)
    os.makedirs(trainB_dest, exist_ok=True)
    
    print(f"🚀 Préparation Phase {phase_num}...")
    
    # Get scan folders
    if not os.path.exists(avg_src_dir):
        print(f"❌ AVG directory not found: {avg_src_dir}")
        return 0
    
    if not os.path.exists(phases_src_dir):
        print(f"❌ Phases directory not found: {phases_src_dir}")
        return 0
    
    scan_folders = os.listdir(avg_src_dir)
    file_counter = 1
    
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
        
        # Find Phase file - adjust pattern for each phase
        phase_pattern = os.path.join(phases_scan_folder, f'*Gated {phase_num/10:.1f}*.nii*')
        phase_files = glob.glob(phase_pattern)
        if not phase_files:
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

def create_all_phase_datasets():
    """Create datasets for all 10 phases"""
    
    base_output_dir = "D:\\ct_phases_datasets"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("🚀 Création de tous les datasets de phases...")
    
    total_patients = {}
    
    # Create datasets for phases 0-9
    for phase in range(10):
        patients_count = prepare_phase_dataset(phase, base_output_dir)
        total_patients[phase] = patients_count
        print()
    
    print("📊 Résumé:")
    for phase, count in total_patients.items():
        print(f"Phase {phase}: {count} patients")
    
    print(f"\n📁 Tous les datasets créés dans: {base_output_dir}")
    print("Structure:")
    print("├── ct_phase0_dataset/")
    print("├── ct_phase1_dataset/")
    print("├── ...")
    print("└── ct_phase9_dataset/")

if __name__ == "__main__":
    create_all_phase_datasets()