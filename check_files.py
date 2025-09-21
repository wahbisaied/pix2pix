# Reminder of the script (check_files.py)
import os
import nibabel as nib
import gzip

dataroot = './datasets/ct_phases_dataset' 

print("--- Starting scan for corrupted NIfTI files... ---")
corrupted_count = 0
for root, _, files in os.walk(dataroot):
    for filename in files:
        if filename.endswith(('.nii', '.nii.gz')):
            file_path = os.path.join(root, filename)
            try:
                img = nib.load(file_path)
                img.get_fdata() # This forces the file to be read
            except (gzip.BadGzipFile, EOFError) as e:
                print(f"❌ CORRUPTED: {file_path}")
                corrupted_count += 1

print(f"\nScan complete. Found {corrupted_count} corrupted file(s).")