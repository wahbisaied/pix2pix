import os

def find_data_directories():
    """Find where the data is located"""
    
    base_paths = [
        r"C:\Users\Wahbi Saied\OneDrive",
        r"C:\Users\Wahbi Saied\Documents",
        r"C:\Users\Wahbi Saied\Desktop",
        r"D:",
        r"E:"
    ]
    
    print("🔍 Recherche des données...")
    
    for base_path in base_paths:
        if os.path.exists(base_path):
            print(f"\n📁 Vérification: {base_path}")
            
            # Look for directories containing "avg", "final_data", "phase", etc.
            for root, dirs, files in os.walk(base_path):
                # Limit depth to avoid too much searching
                depth = root.replace(base_path, '').count(os.sep)
                if depth > 3:
                    continue
                
                for dir_name in dirs:
                    if any(keyword in dir_name.lower() for keyword in ['avg', 'final_data', 'phase', 'ct']):
                        full_path = os.path.join(root, dir_name)
                        # Check if it contains .nii files
                        nii_files = []
                        try:
                            for f in os.listdir(full_path)[:5]:  # Check first 5 files
                                if f.endswith(('.nii', '.nii.gz')):
                                    nii_files.append(f)
                        except:
                            continue
                        
                        if nii_files:
                            print(f"  ✅ {full_path} ({len(nii_files)} fichiers .nii)")
                        else:
                            # Check subdirectories
                            subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
                            if subdirs:
                                print(f"  📂 {full_path} ({len(subdirs)} sous-dossiers)")

if __name__ == "__main__":
    find_data_directories()