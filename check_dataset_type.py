"""
Quick script to check if your dataset contains CBCT or CT data
"""

import os
import nibabel as nib
import numpy as np

def analyze_dataset(dataset_path):
    """Analyze dataset to identify CBCT vs CT characteristics"""
    
    train_a_path = os.path.join(dataset_path, 'trainA')
    train_b_path = os.path.join(dataset_path, 'trainB')
    
    if not os.path.exists(train_a_path):
        print(f"Error: {train_a_path} not found!")
        return
    
    patient_files = [f for f in os.listdir(train_a_path) if f.endswith('.nii.gz')]
    print(f"Found {len(patient_files)} patient files")
    print(f"\nFirst 10 patient files:")
    
    for i, filename in enumerate(patient_files[:10]):
        print(f"  {i+1}. {filename}")
        
        # Check filename for CBCT indicators
        if any(keyword in filename.lower() for keyword in ['cbct', 'cone', 'beam']):
            print(f"     ⚠️  CBCT indicator in filename!")
    
    print(f"\nAnalyzing first 3 files for imaging characteristics...")
    
    for i, filename in enumerate(patient_files[:3]):
        try:
            print(f"\n--- {filename} ---")
            
            # Load TrainA file
            file_a = os.path.join(train_a_path, filename)
            nifti_a = nib.load(file_a)
            data_a = nifti_a.get_fdata()
            
            # Load corresponding TrainB file
            file_b = os.path.join(train_b_path, filename)
            if os.path.exists(file_b):
                nifti_b = nib.load(file_b)
                data_b = nifti_b.get_fdata()
            else:
                print(f"  Warning: {filename} not found in trainB")
                continue
            
            # Analyze characteristics
            print(f"  Shape: {data_a.shape}")
            print(f"  Voxel size: {nifti_a.header.get_zooms()}")
            print(f"  Data range A: {data_a.min():.1f} to {data_a.max():.1f}")
            print(f"  Data range B: {data_b.min():.1f} to {data_b.max():.1f}")
            
            # Check for CBCT characteristics
            voxel_size = nifti_a.header.get_zooms()
            
            # CBCT typically has:
            # - Isotropic voxels (same size in all dimensions)
            # - Smaller field of view
            # - Different intensity ranges
            
            is_isotropic = abs(voxel_size[0] - voxel_size[1]) < 0.1 and abs(voxel_size[1] - voxel_size[2]) < 0.1
            
            if is_isotropic:
                print(f"  📋 Isotropic voxels detected (possible CBCT characteristic)")
            else:
                print(f"  📋 Anisotropic voxels (typical CT characteristic)")
            
            # Check intensity characteristics
            if data_a.max() > 3000 or data_a.min() < -1000:
                print(f"  📋 Standard CT Hounsfield units detected")
            else:
                print(f"  📋 Non-standard intensity range (possible CBCT)")
                
        except Exception as e:
            print(f"  Error analyzing {filename}: {e}")

def check_specific_patient(dataset_path, patient_name):
    """Check a specific patient file"""
    
    file_path = os.path.join(dataset_path, 'trainA', patient_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Analyzing: {patient_name}")
    
    try:
        nifti = nib.load(file_path)
        data = nifti.get_fdata()
        
        print(f"Shape: {data.shape}")
        print(f"Voxel size: {nifti.header.get_zooms()}")
        print(f"Data range: {data.min():.1f} to {data.max():.1f}")
        print(f"Data type: {data.dtype}")
        
        # Sample a middle slice
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]
        
        print(f"Middle slice ({mid_slice}) stats:")
        print(f"  Mean: {slice_data.mean():.1f}")
        print(f"  Std: {slice_data.std():.1f}")
        print(f"  Non-zero pixels: {np.count_nonzero(slice_data)}/{slice_data.size}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dataset_path = "D:\\ct_phases_dataset"
    
    print("=== DATASET TYPE ANALYSIS ===")
    analyze_dataset(dataset_path)
    
    print(f"\n" + "="*50)
    print("SUMMARY:")
    print("- Check filenames for CBCT indicators")
    print("- Check voxel sizes (isotropic = possible CBCT)")
    print("- Check intensity ranges (HU units = CT)")
    print("- Visual inspection of epoch images recommended")