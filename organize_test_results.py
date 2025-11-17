import os
import shutil
import glob
from PIL import Image
import re

def organize_test_results(results_dir):
    """Organize test results by patient"""
    
    images_dir = os.path.join(results_dir, "images")
    if not os.path.exists(images_dir):
        print("❌ No images directory found!")
        return
    
    # Get all result images
    fake_images = glob.glob(os.path.join(images_dir, "*_fake_B.png"))
    real_images = glob.glob(os.path.join(images_dir, "*_real_A.png"))
    
    print(f"📊 Found {len(fake_images)} generated images")
    print(f"📊 Found {len(real_images)} input images")
    
    # Create organized directory
    organized_dir = os.path.join(results_dir, "organized_by_patient")
    os.makedirs(organized_dir, exist_ok=True)
    
    # Group by patient
    patients = {}
    for img_path in fake_images:
        filename = os.path.basename(img_path)
        # Extract patient info from filename
        parts = filename.split('_')
        if len(parts) >= 2:
            patient_id = parts[0]  # Assuming format like "000001_fake_B.png"
            if patient_id not in patients:
                patients[patient_id] = []
            patients[patient_id].append(img_path)
    
    print(f"📋 Found {len(patients)} unique patients")
    
    # Create patient folders and copy images
    for patient_id, image_paths in patients.items():
        patient_dir = os.path.join(organized_dir, f"patient_{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)
        
        print(f"📁 Organizing patient {patient_id}: {len(image_paths)} images")
        
        # Copy images for this patient
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            new_path = os.path.join(patient_dir, filename)
            shutil.copy2(img_path, new_path)
            
            # Also copy corresponding real_A image
            real_filename = filename.replace('_fake_B.png', '_real_A.png')
            real_path = os.path.join(images_dir, real_filename)
            if os.path.exists(real_path):
                real_new_path = os.path.join(patient_dir, real_filename)
                shutil.copy2(real_path, real_new_path)
    
    print(f"✅ Results organized in: {organized_dir}")
    return organized_dir

def create_patient_summary(organized_dir):
    """Create summary images for each patient"""
    
    patient_dirs = [d for d in os.listdir(organized_dir) 
                   if os.path.isdir(os.path.join(organized_dir, d))]
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(organized_dir, patient_dir)
        
        # Get all fake images for this patient
        fake_images = glob.glob(os.path.join(patient_path, "*_fake_B.png"))
        fake_images.sort()
        
        if len(fake_images) > 0:
            print(f"📊 Patient {patient_dir}: {len(fake_images)} slices")
            
            # Show a few representative slices
            if len(fake_images) >= 5:
                representative_slices = [
                    fake_images[0],  # First slice
                    fake_images[len(fake_images)//4],  # 25%
                    fake_images[len(fake_images)//2],  # Middle
                    fake_images[3*len(fake_images)//4],  # 75%
                    fake_images[-1]  # Last slice
                ]
                print(f"  Representative slices: {len(representative_slices)}")

def main():
    results_dir = r"results\ct_phase0_generator_5090\test_latest"
    
    print("=" * 60)
    print("🗂️  ORGANIZING TEST RESULTS")
    print("=" * 60)
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Run the test command first!")
        return
    
    # Organize results
    organized_dir = organize_test_results(results_dir)
    
    # Create patient summaries
    if organized_dir:
        create_patient_summary(organized_dir)
    
    print("\n" + "=" * 60)
    print("✅ ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Check organized results in: {organized_dir}")
    print("💡 Each patient now has their own folder with all slices")

if __name__ == "__main__":
    main()