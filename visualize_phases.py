import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def visualize_phases(folder_path):
    """Visualize middle slices from all cardiac phases"""
    
    # Get all phase files
    phase_files = glob.glob(os.path.join(folder_path, "*Gated*.nii.gz"))
    phase_files.sort()
    
    if not phase_files:
        print("No phase files found!")
        return
    
    # Create subplot grid
    n_phases = len(phase_files)
    cols = 5
    rows = (n_phases + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, phase_file in enumerate(phase_files):
        row = i // cols
        col = i % cols
        
        # Extract phase number from filename
        phase_name = os.path.basename(phase_file)
        phase_num = phase_name.split("Gated ")[1].split("A")[0]
        
        # Load NIfTI file
        img = nib.load(phase_file)
        data = img.get_fdata()
        
        # Get middle slice
        middle_slice = data.shape[2] // 2
        slice_data = data[:, :, middle_slice]
        
        # Display slice
        axes[row, col].imshow(slice_data.T, cmap='gray', origin='lower')
        axes[row, col].set_title(f'Phase {phase_num}%')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_phases, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Cardiac Phases - {os.path.basename(folder_path)}', y=1.02)
    plt.show()

if __name__ == "__main__":
    folder_path = r"C:\Users\Wahbi Saied\OneDrive\final_data\12-10-1999-NA-p4-47741"
    visualize_phases(folder_path)