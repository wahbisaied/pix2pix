import os
import glob

def check_phase_file_patterns():
    """Check what phase files actually exist"""
    
    phases_src_dir = r"C:\Users\Wahbi Saied\OneDrive\final_data"
    
    if not os.path.exists(phases_src_dir):
        print(f"❌ Directory not found: {phases_src_dir}")
        return
    
    # Take first few scan folders to check patterns
    scan_folders = os.listdir(phases_src_dir)[:3]
    
    print("🔍 Vérification des patterns de fichiers de phases...")
    
    for scan_id in scan_folders:
        scan_folder = os.path.join(phases_src_dir, scan_id)
        
        if not os.path.isdir(scan_folder):
            continue
            
        print(f"\n📁 {scan_id}:")
        
        # Get all .nii files in this scan
        nii_files = glob.glob(os.path.join(scan_folder, '*.nii*'))
        
        for file_path in nii_files:
            filename = os.path.basename(file_path)
            print(f"  - {filename}")
            
            # Check if it matches any phase pattern
            if 'gated' in filename.lower() or 'phase' in filename.lower():
                print(f"    🎯 Possible phase file!")

if __name__ == "__main__":
    check_phase_file_patterns()