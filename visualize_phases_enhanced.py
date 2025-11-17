import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def visualize_all_phases(folder_path):
    """Visualize middle slices from all cardiac phases"""
    
    phase_files = glob.glob(os.path.join(folder_path, "*Gated*.nii.gz"))
    phase_files.sort()
    
    if not phase_files:
        print("No phase files found!")
        return
    
    print(f"Found {len(phase_files)} phase files")
    
    # Create subplot grid (2 rows x 5 cols for 10 phases)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, phase_file in enumerate(phase_files):
        row = i // 5
        col = i % 5
        
        # Extract phase number
        phase_name = os.path.basename(phase_file)
        phase_num = phase_name.split("Gated ")[1].split("A")[0]
        
        print(f"Loading phase {phase_num}%...")
        
        # Load and display
        img = nib.load(phase_file)
        data = img.get_fdata()
        
        # Get middle slice
        middle_slice = data.shape[2] // 2
        slice_data = data[:, :, middle_slice]
        
        # Normalize for better visualization
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        axes[row, col].imshow(slice_data.T, cmap='gray', origin='lower')
        axes[row, col].set_title(f'Phase {phase_num}%', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Cardiac Phases - Patient {os.path.basename(folder_path)}', fontsize=16, y=0.98)
    plt.savefig('cardiac_phases_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_single_phase_slices(folder_path, phase_percent=0):
    """Visualize multiple slices from a single phase"""
    
    pattern = f"*Gated {phase_percent}.0A*.nii.gz"
    phase_files = glob.glob(os.path.join(folder_path, pattern))
    
    if not phase_files:
        print(f"No phase {phase_percent}% file found!")
        return
    
    phase_file = phase_files[0]
    print(f"Loading {os.path.basename(phase_file)}...")
    
    img = nib.load(phase_file)
    data = img.get_fdata()
    
    # Show multiple slices
    n_slices = 9
    slice_indices = np.linspace(10, data.shape[2]-10, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, slice_idx in enumerate(slice_indices):
        row = i // 3
        col = i % 3
        
        slice_data = data[:, :, slice_idx]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        axes[row, col].imshow(slice_data.T, cmap='gray', origin='lower')
        axes[row, col].set_title(f'Slice {slice_idx}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Phase {phase_percent}% - Multiple Slices', fontsize=16)
    plt.savefig(f'phase_{phase_percent}_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    folder_path = r"C:\Users\Wahbi Saied\OneDrive\final_data\12-10-1999-NA-p4-47741"
    
    print("=== Visualizing All Cardiac Phases ===")
    visualize_all_phases(folder_path)
    
    print("\n=== Visualizing Phase 0% Multiple Slices ===")
    visualize_single_phase_slices(folder_path, 0)
    
    print("\n=== Visualizing Phase 50% Multiple Slices ===")
    visualize_single_phase_slices(folder_path, 50)