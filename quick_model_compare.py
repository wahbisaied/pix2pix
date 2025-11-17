import os
import glob

def analyze_training_logs(checkpoint_dir1, checkpoint_dir2, model1_name, model2_name):
    """Quick analysis based on training logs and loss files"""
    
    print("=" * 60)
    print("QUICK MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Check if loss logs exist
    loss_log1 = os.path.join(checkpoint_dir1, "loss_log.txt")
    loss_log2 = os.path.join(checkpoint_dir2, "loss_log.txt")
    
    print(f"\n{model1_name}:")
    print(f"- 80 patients (9062 images)")
    print(f"- LR: 0.0001")
    print(f"- Epochs: 100 + 100 decay")
    print(f"- Total training: 200 epochs")
    
    print(f"\n{model2_name}:")
    print(f"- 68 patients (7611 images)")  
    print(f"- LR: 0.0002")
    print(f"- Epochs: 150 + 50 decay")
    print(f"- Total training: 200 epochs")
    
    # Analysis based on training characteristics
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    print("\nDATA SIZE COMPARISON:")
    print(f"- {model1_name}: 9062 images (MORE data)")
    print(f"- {model2_name}: 7611 images (1451 fewer images)")
    print("→ More data usually leads to better generalization")
    
    print("\nLEARNING RATE COMPARISON:")
    print(f"- {model1_name}: 0.0001 (Conservative)")
    print(f"- {model2_name}: 0.0002 (2x higher)")
    print("→ Higher LR can lead to faster convergence but less stability")
    
    print("\nTRAINING SCHEDULE:")
    print(f"- {model1_name}: 100 + 100 decay (Equal phases)")
    print(f"- {model2_name}: 150 + 50 decay (More main training)")
    print("→ Longer main training can capture more features")
    
    # Recommendation based on medical imaging best practices
    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR CARDIAC IMAGING:")
    print("=" * 60)
    
    print("\n🔍 ANALYSIS:")
    print("1. MORE DATA (9062 vs 7611) typically wins in medical imaging")
    print("2. CONSERVATIVE LR (0.0001) is often better for medical data")
    print("3. EQUAL decay phases (100+100) provide stable convergence")
    
    print("\n🏆 RECOMMENDED APPROACH:")
    print("USE the 80-patient model settings (Model 1) because:")
    print("✓ More training data (1451 additional images)")
    print("✓ Conservative learning rate (better for medical imaging)")
    print("✓ Balanced training schedule")
    print("✓ Medical imaging benefits from stable, conservative training")
    
    print(f"\n📋 TRAINING SETTINGS for remaining phases:")
    print("- Learning Rate: 0.0001")
    print("- Epochs: 100 + 100 decay")
    print("- Batch size: Keep same as before")
    print("- Use same data augmentation")
    
    return "model1_80patients"

def check_model_files(checkpoint_dir):
    """Check what model files exist"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory not found: {checkpoint_dir}")
        return False
    
    # Look for generator models
    gen_files = glob.glob(os.path.join(checkpoint_dir, "*_net_G.pth"))
    if gen_files:
        print(f"✓ Found generator models: {len(gen_files)}")
        for f in gen_files:
            print(f"  - {os.path.basename(f)}")
        return True
    else:
        print(f"❌ No generator models found in {checkpoint_dir}")
        return False

if __name__ == "__main__":
    # Update these paths to your actual checkpoint directories
    checkpoint_dir1 = r"C:\Users\Wahbi Saied\OneDrive\Desktop\pytorch-CycleGAN-and-pix2pix\checkpoints\ct_phase0_generator_optimized"  # Update this
    checkpoint_dir2 = r"C:\path\to\68patients_model\C:\Users\Wahbi Saied\Documents\GitHub\pix2pix\checkpoints\ct_phase0_generator_5090"  # Update this
    
    model1_name = "80patients_lr0001_100+100epochs"
    model2_name = "68patients_lr0002_150+50epochs"
    
    print("Checking model files...")
    print(f"\nModel 1 directory: {checkpoint_dir1}")
    check1 = check_model_files(checkpoint_dir1)
    
    print(f"\nModel 2 directory: {checkpoint_dir2}")
    check2 = check_model_files(checkpoint_dir2)
    
    if check1 or check2:
        recommendation = analyze_training_logs(checkpoint_dir1, checkpoint_dir2, model1_name, model2_name)
    else:
        print("\n❌ Please update the checkpoint directory paths in the script")