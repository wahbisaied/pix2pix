import os
import nibabel as nib
import gzip
from pathlib import Path

def check_nifti_file(file_path):
    """Check if a NIfTI file is corrupted"""
    try:
        # Try to load the file
        img = nib.load(file_path)
        data = img.get_fdata()
        return True, f"OK - Shape: {data.shape}"
    except Exception as e:
        return False, str(e)

def check_gzip_file(file_path):
    """Check if a gzipped file is corrupted"""
    try:
        with gzip.open(file_path, 'rb') as f:
            f.read(1024)  # Try to read first 1KB
        return True, "Gzip OK"
    except Exception as e:
        return False, f"Gzip error: {str(e)}"

def scan_dataset(dataset_path):
    """Scan all NIfTI files in dataset for corruption"""
    print("🔍 Scanning for corrupted NIfTI files...")
    print("=" * 60)
    
    corrupted_files = []
    total_files = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(root, file)
                total_files += 1
                
                print(f"\nChecking: {file_path}")
                
                # Check gzip integrity first if .gz file
                if file.endswith('.gz'):
                    gzip_ok, gzip_msg = check_gzip_file(file_path)
                    print(f"  Gzip: {gzip_msg}")
                    if not gzip_ok:
                        corrupted_files.append((file_path, gzip_msg))
                        continue
                
                # Check NIfTI integrity
                nifti_ok, nifti_msg = check_nifti_file(file_path)
                print(f"  NIfTI: {nifti_msg}")
                
                if not nifti_ok:
                    corrupted_files.append((file_path, nifti_msg))
    
    print("\n" + "=" * 60)
    print(f"📊 SCAN RESULTS:")
    print(f"Total files scanned: {total_files}")
    print(f"Corrupted files found: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n❌ CORRUPTED FILES:")
        for file_path, error in corrupted_files:
            print(f"  {file_path}")
            print(f"    Error: {error}")
        
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print(f"1. Delete corrupted files:")
        for file_path, _ in corrupted_files:
            print(f'   del "{file_path}"')
        
        print(f"\n2. Re-download or regenerate these files")
        print(f"3. Check available disk space")
        
    else:
        print("✅ All files are intact!")

if __name__ == "__main__":
    dataset_path = "./datasets/ct_phases_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable in this script")
    else:
        scan_dataset(dataset_path)