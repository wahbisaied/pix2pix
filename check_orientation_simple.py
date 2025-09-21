import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

nii_folder = r"C:\Users\wahbi\OneDrive\final_data"

# Find first few NIfTI files
nii_files = []
for root, dirs, files in os.walk(nii_folder):
    for file in files:
        if file.endswith('.nii.gz'):
            nii_files.append(os.path.join(root, file))
            if len(nii_files) >= 3:  # Just check first 3 files
                break
    if len(nii_files) >= 3:
        break

print(f"Found {len(nii_files)} files to check")

for i, nii_path in enumerate(nii_files):
    print(f"\nProcessing: {os.path.basename(nii_path)}")
    
    img = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(img)
    print(f"Shape: {arr.shape}")
    
    # Middle slice
    slice_idx = arr.shape[0] // 2
    slice_img = arr[slice_idx]
    
    # Save image instead of showing
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_img, cmap='gray')
    plt.title(f"{os.path.basename(nii_path)}\nShape: {arr.shape}")
    plt.axis('off')
    plt.savefig(f'orientation_check_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: orientation_check_{i+1}.png")

print("\nDone! Check the saved PNG files to verify orientation.")