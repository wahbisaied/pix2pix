import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Dossier contenant tes fichiers .nii
nii_folder = r"C:\Users\wahbi\OneDrive\final_data"

print(f"Checking folder: {nii_folder}")
print(f"Folder exists: {os.path.exists(nii_folder)}")

if os.path.exists(nii_folder):
    files = os.listdir(nii_folder)
    print(f"Total files in folder: {len(files)}")
    
    # Check file extensions
    extensions = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    print(f"File extensions found: {extensions}")
    print(f"First 10 files: {files[:10]}")
    
    nii_files = [f for f in files if f.endswith(".nii") or f.endswith(".nii.gz")]
    print(f"NIfTI files found: {len(nii_files)}")
    print(f"NIfTI files: {nii_files[:5]}")  # Show first 5
    
    # Check if these are directories
    print("\nChecking if items are directories:")
    for item in files[:5]:  # Check first 5
        item_path = os.path.join(nii_folder, item)
        is_dir = os.path.isdir(item_path)
        print(f"{item}: {'Directory' if is_dir else 'File'}")
        if is_dir:
            sub_files = os.listdir(item_path)
            nii_in_sub = [f for f in sub_files if f.endswith('.nii') or f.endswith('.nii.gz')]
            print(f"  Contains {len(nii_in_sub)} NIfTI files: {nii_in_sub[:3]}")
    
    # Try to find NIfTI files recursively
    print("\nSearching for NIfTI files recursively...")
    nii_files_found = []
    for root, dirs, files_in_dir in os.walk(nii_folder):
        for file in files_in_dir:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files_found.append(os.path.join(root, file))
    
    print(f"Total NIfTI files found recursively: {len(nii_files_found)}")
    
    # Process first few NIfTI files found
    for nii_path in nii_files_found[:3]:
        print(f"\nProcessing: {nii_path}")
        try:
            img = sitk.ReadImage(nii_path)
            arr = sitk.GetArrayFromImage(img)
            print(f"Shape: {arr.shape}")
            slice_idx = arr.shape[0] // 2
            slice_img = arr[slice_idx]
            plt.imshow(slice_img, cmap="gray")
            plt.title(f"{os.path.basename(nii_path)} - Shape: {arr.shape}")
            plt.show()
        except Exception as e:
            print(f"Error processing {nii_path}: {e}")
else:
    print("Folder does not exist!")
