import os
import shutil
import glob

# --- Configuration ---
avg_src_dir = r"C:\Users\wahbi\OneDrive\avg"
phases_src_dir = r"C:\Users\wahbi\OneDrive\final_data"
train_A_dest = r"c:\Users\wahbi\OneDrive\Desktop\pytorch-CycleGAN-and-pix2pix\datasets\ct_phases_dataset\trainA"
train_B_dest = r"c:\Users\wahbi\OneDrive\Desktop\pytorch-CycleGAN-and-pix2pix\datasets\ct_phases_dataset\trainB"

print("Starting dataset preparation with updated logic...")

os.makedirs(train_A_dest, exist_ok=True)
os.makedirs(train_B_dest, exist_ok=True)

try:
    # NEW LOGIC: We assume avg_src_dir contains subdirectories, one for each scan.
    scan_folders = os.listdir(avg_src_dir)
except FileNotFoundError:
    print(f"--> ERROR: Source directory not found: {avg_src_dir}")
    exit()

file_counter = 1
print(f"Found {len(scan_folders)} potential scan folders in the source directory.")
print("Pairing assumption: A folder 'res_avg/SCAN_01/' is paired with 'res_phases/SCAN_01/'.")

# Iterate through each of the scan folders (e.g., '01-02-2001-NA-p4-43956')
for scan_id in scan_folders:
    avg_scan_folder_path = os.path.join(avg_src_dir, scan_id)
    phases_scan_folder_path = os.path.join(phases_src_dir, scan_id)

    # Check if it's actually a directory
    if not os.path.isdir(avg_scan_folder_path):
        print(f"--> INFO: Skipping item that is not a directory: '{scan_id}'")
        continue

    # Check for a matching directory in the phases folder
    if not os.path.isdir(phases_scan_folder_path):
        print(f"--> WARNING: No matching phase folder found for '{scan_id}'. Looked for '{phases_scan_folder_path}'. Skipping.")
        continue

    # --- Find the AVG scan file ---
    # Find any .nii or .nii.gz file in the avg scan folder
    avg_files = glob.glob(os.path.join(avg_scan_folder_path, '*.nii*'))
    if not avg_files:
        print(f"--> WARNING: No NIfTI file found in '{avg_scan_folder_path}'. Skipping.")
        continue
    avg_src_path = avg_files[0] # Use the first NIfTI file found

    # --- Find the PHASE 0 scan file ---
    search_pattern = os.path.join(phases_scan_folder_path, '*Gated 0.0*.nii*')
    phase_0_files = glob.glob(search_pattern)
    if not phase_0_files:
        print(f"--> WARNING: No 'Gated 0.0A' file found in '{phases_scan_folder_path}'. Skipping.")
        continue
    phase_0_src_path = phase_0_files[0]

    # --- Copy and Rename ---
    extension = '.nii.gz' if phase_0_src_path.endswith('.nii.gz') else '.nii'
    new_filename = f"patient_{file_counter}{extension}"
    
    dest_A_path = os.path.join(train_A_dest, new_filename)
    dest_B_path = os.path.join(train_B_dest, new_filename)

    print(f"  - Pairing and copying for patient {file_counter} (ID: {scan_id}):")
    print(f"    - AVG: '{os.path.basename(avg_src_path)}' -> '{new_filename}'")
    shutil.copy(avg_src_path, dest_A_path)

    print(f"    - PHS: '{os.path.basename(phase_0_src_path)}' -> '{new_filename}'")
    shutil.copy(phase_0_src_path, dest_B_path)

    file_counter += 1

print(f"\nPreparation complete. {file_counter - 1} pairs of files were copied and renamed.")