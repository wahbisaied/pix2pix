import os
import shutil
import glob

def replace_patient_in_phase(phase_num, patient_to_replace, replacement_patient):
    """Replace a specific patient in a phase dataset"""
    
    # Source directories
    avg_src_dir = r"C:\Users\Wahbi Saied\OneDrive\avg"
    phases_src_dir = r"C:\Users\Wahbi Saied\OneDrive\final_data"
    
    # Dataset directory
    phase_dir = f"D:\\ct_phases_datasets\\ct_phase{phase_num}_dataset"
    trainA_dest = os.path.join(phase_dir, "trainA")
    trainB_dest = os.path.join(phase_dir, "trainB")
    
    # Find which file number corresponds to the patient to replace
    # This assumes you know the patient number or can identify it
    patient_file = f"patient_{patient_to_replace}.nii.gz"  # or .nii
    
    # Get replacement patient files
    avg_scan_folder = os.path.join(avg_src_dir, replacement_patient)
    phases_scan_folder = os.path.join(phases_src_dir, replacement_patient)
    
    # Find AVG file
    avg_files = glob.glob(os.path.join(avg_scan_folder, '*.nii*'))
    if not avg_files:
        print(f"❌ No AVG file found for {replacement_patient}")
        return False
    
    # Find Phase file
    patterns = []
    if phase_num == 0:
        patterns = [f'*Gated 0.0A*.nii*', f'*Gated 0.0B*.nii*', f'*Gated 0.0*.nii*']
    else:
        patterns = [f'*Gated {phase_num}0.0A*.nii*', f'*Gated {phase_num}0.0B*.nii*', f'*Gated {phase_num}0.0*.nii*']
    
    phase_files = []
    for pattern in patterns:
        found_files = glob.glob(os.path.join(phases_scan_folder, pattern))
        if found_files:
            phase_files = found_files
            break
    
    if not phase_files:
        print(f"❌ No phase {phase_num} file found for {replacement_patient}")
        return False
    
    # Replace files
    dest_A_path = os.path.join(trainA_dest, patient_file)
    dest_B_path = os.path.join(trainB_dest, patient_file)
    
    try:
        shutil.copy(avg_files[0], dest_A_path)
        shutil.copy(phase_files[0], dest_B_path)
        print(f"✅ Replaced patient_{patient_to_replace} with {replacement_patient} in phase {phase_num}")
        return True
    except Exception as e:
        print(f"❌ Error replacing patient: {e}")
        return False

# Example usage:
# replace_patient_in_phase(0, 42, "01-01-2000-NA-NA-12345")  # Replace patient_42 with new patient