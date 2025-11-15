import os
import shutil
import glob
import random

def split_phase_dataset(phase_num, train_ratio=0.85):
    """Split one phase dataset into train/test"""
    
    base_dir = "D:\\ct_phases_datasets"
    phase_dir = os.path.join(base_dir, f"ct_phase{phase_num}_dataset")
    
    trainA_dir = os.path.join(phase_dir, "trainA")
    trainB_dir = os.path.join(phase_dir, "trainB")
    
    # Create test directories
    testA_dir = os.path.join(phase_dir, "testA")
    testB_dir = os.path.join(phase_dir, "testB")
    os.makedirs(testA_dir, exist_ok=True)
    os.makedirs(testB_dir, exist_ok=True)
    
    # Get all patient files
    patient_files = glob.glob(os.path.join(trainA_dir, 'patient_*.nii*'))
    patient_files.sort()
    
    if not patient_files:
        print(f"❌ Phase {phase_num}: No files found")
        return 0, 0
    
    # Use same seed for consistent splits across phases
    random.seed(42)
    random.shuffle(patient_files)
    
    split_idx = int(len(patient_files) * train_ratio)
    train_files = patient_files[:split_idx]
    test_files = patient_files[split_idx:]
    
    # Move test files
    moved_count = 0
    for file_path in test_files:
        filename = os.path.basename(file_path)
        
        # Move A file
        dest_A = os.path.join(testA_dir, filename)
        shutil.move(file_path, dest_A)
        
        # Move corresponding B file
        b_file = os.path.join(trainB_dir, filename)
        if os.path.exists(b_file):
            dest_B = os.path.join(testB_dir, filename)
            shutil.move(b_file, dest_B)
            moved_count += 1
    
    train_count = len(train_files)
    test_count = moved_count
    
    print(f"✅ Phase {phase_num}: {train_count} train, {test_count} test")
    return train_count, test_count

def split_all_phases():
    """Split all phase datasets into train/test"""
    
    print("🚀 Split train/test pour toutes les phases...")
    print("📊 Ratio: 85% train, 15% test")
    print("🎯 Même split pour toutes les phases (seed=42)")
    print()
    
    total_stats = {}
    
    for phase in range(10):
        train_count, test_count = split_phase_dataset(phase)
        total_stats[phase] = (train_count, test_count)
    
    print("\n📊 Résumé final:")
    print("Phase | Train | Test")
    print("------|-------|-----")
    
    for phase, (train, test) in total_stats.items():
        print(f"  {phase}   |  {train:2d}   | {test:2d}")
    
    # Check consistency
    train_counts = [stats[0] for stats in total_stats.values()]
    test_counts = [stats[1] for stats in total_stats.values()]
    
    if len(set(train_counts)) == 1:
        print(f"\n✅ Split cohérent pour l'entraînement!")
        print(f"📈 Training: {train_counts[0]} patients par phase")
        print(f"🧪 Testing: {min(test_counts)}-{max(test_counts)} patients par phase")
    else:
        print(f"\n⚠️  Split incohérent entre les phases")

if __name__ == "__main__":
    split_all_phases()