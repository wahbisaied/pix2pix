"""
Script to identify which patient/slice corresponds to training visualization images
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from data.robust_nifti_dataset import RobustNiftiDataset
from options.train_options import TrainOptions

def load_png_as_array(png_path):
    """Load PNG and convert to numpy array"""
    img = Image.open(png_path).convert('L')  # Convert to grayscale
    return np.array(img)

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr

def find_matching_slice(target_image, dataset_path):
    """Find which patient/slice matches the target image"""
    
    # Get all patient files
    train_a_path = os.path.join(dataset_path, 'trainA')
    patient_files = [f for f in os.listdir(train_a_path) if f.endswith('.nii.gz')]
    
    target_normalized = normalize_array(target_image)
    best_match = None
    best_similarity = -1
    
    print(f"Searching through {len(patient_files)} patient files...")
    
    for i, patient_file in enumerate(patient_files):
        if i % 10 == 0:
            print(f"Checking patient {i+1}/{len(patient_files)}: {patient_file}")
            
        try:
            # Load NIfTI file
            nifti_path = os.path.join(train_a_path, patient_file)
            nifti_img = nib.load(nifti_path)
            volume_data = nifti_img.get_fdata()
            
            # Check each axial slice
            for slice_idx in range(volume_data.shape[2]):
                slice_data = volume_data[:, :, slice_idx]
                
                # Resize to 512x512 (same as training)
                from PIL import Image
                slice_pil = Image.fromarray(slice_data.astype(np.float32))
                slice_resized = slice_pil.resize((512, 512), Image.LANCZOS)
                slice_array = np.array(slice_resized)
                
                # Normalize
                slice_normalized = normalize_array(slice_array)
                
                # Calculate similarity (correlation coefficient)
                if slice_normalized.std() > 0 and target_normalized.std() > 0:
                    correlation = np.corrcoef(slice_normalized.flatten(), target_normalized.flatten())[0, 1]
                    
                    if correlation > best_similarity:
                        best_similarity = correlation
                        best_match = {
                            'patient': patient_file,
                            'slice': slice_idx,
                            'correlation': correlation
                        }
                        
        except Exception as e:
            print(f"Error processing {patient_file}: {e}")
            continue
    
    return best_match

def identify_epoch_images(epoch_num, dataset_path, images_path):
    """Identify which patients correspond to epoch images"""
    
    print(f"=== IDENTIFYING EPOCH {epoch_num} IMAGES ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Images path: {images_path}")
    
    # Load the epoch images
    real_a_path = os.path.join(images_path, f'epoch{epoch_num:03d}_real_A.png')
    real_b_path = os.path.join(images_path, f'epoch{epoch_num:03d}_real_B.png')
    fake_b_path = os.path.join(images_path, f'epoch{epoch_num:03d}_fake_B.png')
    
    if not os.path.exists(real_a_path):
        print(f"Error: {real_a_path} not found!")
        return
    
    print(f"\nLoading epoch {epoch_num} images...")
    real_a_img = load_png_as_array(real_a_path)
    
    print(f"Real A image shape: {real_a_img.shape}")
    print(f"Real A image range: {real_a_img.min()} - {real_a_img.max()}")
    
    # Find matching slice
    print(f"\nSearching for matching patient/slice...")
    match = find_matching_slice(real_a_img, dataset_path)
    
    if match:
        print(f"\n=== MATCH FOUND ===")
        print(f"Patient: {match['patient']}")
        print(f"Slice: {match['slice']}")
        print(f"Correlation: {match['correlation']:.4f}")
        
        # Check if it looks like CBCT
        if 'cbct' in match['patient'].lower() or match['correlation'] < 0.7:
            print(f"\n⚠️  WARNING: This might be CBCT data!")
            print(f"   - Patient filename: {match['patient']}")
            print(f"   - Low correlation: {match['correlation']:.4f}")
        else:
            print(f"\n✓ This appears to be regular CT data")
            
    else:
        print(f"\n❌ No matching slice found!")
        print(f"This could indicate:")
        print(f"  - Data preprocessing differences")
        print(f"  - CBCT vs CT mismatch")
        print(f"  - Corrupted data")

if __name__ == "__main__":
    # Paths
    dataset_path = "D:\\ct_phases_dataset"  # Your dataset path
    images_path = "C:\\Users\\Wahbi Saied\\Documents\\GitHub\\pix2pix\\checkpoints\\ct_phase0_generator_5090\\web\\images"
    
    # Check epoch 184
    identify_epoch_images(184, dataset_path, images_path)