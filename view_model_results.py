import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from torchvision import transforms

def load_and_display_results(checkpoint_dir, test_data_dir=None, num_samples=6):
    """Load and display model results from web folder or test on new data"""
    
    print("=" * 60)
    print("PIX2PIX MODEL RESULTS VIEWER")
    print("=" * 60)
    
    # Check web folder for training results
    web_images_dir = os.path.join(checkpoint_dir, "web", "images")
    
    if os.path.exists(web_images_dir):
        print(f"📁 Found web images in: {web_images_dir}")
        display_web_results(web_images_dir, num_samples)
    else:
        print("❌ No web images found")
    
    # If test data is provided, run inference
    if test_data_dir and os.path.exists(test_data_dir):
        print(f"\n📁 Test data directory: {test_data_dir}")
        # This would require loading the model and running inference
        print("ℹ️  For inference on new data, use the test.py script")

def display_web_results(web_images_dir, num_samples=6):
    """Display results from the web images folder"""
    
    # Get all epoch images
    fake_images = glob.glob(os.path.join(web_images_dir, "*_fake_B.png"))
    real_a_images = glob.glob(os.path.join(web_images_dir, "*_real_A.png"))
    real_b_images = glob.glob(os.path.join(web_images_dir, "*_real_B.png"))
    
    if not fake_images:
        print("❌ No training result images found")
        return
    
    # Sort by epoch number
    fake_images.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('epoch', '')))
    real_a_images.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('epoch', '')))
    real_b_images.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('epoch', '')))
    
    print(f"📊 Found {len(fake_images)} training result images")
    
    # Show progression through training
    epochs_to_show = np.linspace(0, len(fake_images)-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    
    for i, epoch_idx in enumerate(epochs_to_show):
        epoch_num = int(os.path.basename(fake_images[epoch_idx]).split('_')[0].replace('epoch', ''))
        
        # Load images
        real_a = Image.open(real_a_images[epoch_idx])
        fake_b = Image.open(fake_images[epoch_idx])
        real_b = Image.open(real_b_images[epoch_idx])
        
        # Display
        axes[0, i].imshow(real_a)
        axes[0, i].set_title(f'Input A\nEpoch {epoch_num}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(fake_b)
        axes[1, i].set_title(f'Generated B\nEpoch {epoch_num}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(real_b)
        axes[2, i].set_title(f'Real B\nEpoch {epoch_num}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Training Progress: Input → Generated → Target', y=1.02, fontsize=16)
    plt.show()
    
    # Show latest results
    show_latest_results(web_images_dir)

def show_latest_results(web_images_dir):
    """Show the latest training results"""
    
    # Get the latest epoch images
    fake_images = glob.glob(os.path.join(web_images_dir, "*_fake_B.png"))
    if not fake_images:
        return
    
    # Get the latest epoch
    latest_fake = max(fake_images, key=lambda x: int(os.path.basename(x).split('_')[0].replace('epoch', '')))
    epoch_num = int(os.path.basename(latest_fake).split('_')[0].replace('epoch', ''))
    
    latest_real_a = os.path.join(web_images_dir, f"epoch{epoch_num:03d}_real_A.png")
    latest_real_b = os.path.join(web_images_dir, f"epoch{epoch_num:03d}_real_B.png")
    
    if all(os.path.exists(f) for f in [latest_fake, latest_real_a, latest_real_b]):
        print(f"\n🎯 LATEST RESULTS (Epoch {epoch_num}):")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load and display
        real_a = Image.open(latest_real_a)
        fake_b = Image.open(latest_fake)
        real_b = Image.open(latest_real_b)
        
        axes[0].imshow(real_a)
        axes[0].set_title('Input (Real A)', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(fake_b)
        axes[1].set_title('Generated (Fake B)', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(real_b)
        axes[2].set_title('Target (Real B)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Latest Results - Epoch {epoch_num}', fontsize=16)
        plt.show()

def analyze_model_files(checkpoint_dir):
    """Analyze available model files"""
    
    print("\n" + "=" * 60)
    print("📁 AVAILABLE MODEL FILES:")
    print("=" * 60)
    
    # Find all generator models
    gen_models = glob.glob(os.path.join(checkpoint_dir, "*_net_G.pth"))
    
    # Filter out non-numeric epochs and sort
    numeric_models = []
    for model in gen_models:
        epoch_str = os.path.basename(model).split('_')[0]
        if epoch_str.isdigit():
            numeric_models.append(model)
    
    numeric_models.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    gen_models = numeric_models
    
    if gen_models:
        print(f"Found {len(gen_models)} generator models:")
        
        # Show first few, middle few, and last few
        if len(gen_models) <= 10:
            for model in gen_models:
                epoch = os.path.basename(model).split('_')[0]
                size = os.path.getsize(model) / (1024*1024)  # MB
                print(f"  ✅ {epoch}_net_G.pth ({size:.1f} MB)")
        else:
            # Show first 3
            for model in gen_models[:3]:
                epoch = os.path.basename(model).split('_')[0]
                size = os.path.getsize(model) / (1024*1024)
                print(f"  ✅ {epoch}_net_G.pth ({size:.1f} MB)")
            
            print("  ...")
            
            # Show last 3
            for model in gen_models[-3:]:
                epoch = os.path.basename(model).split('_')[0]
                size = os.path.getsize(model) / (1024*1024)
                print(f"  ✅ {epoch}_net_G.pth ({size:.1f} MB)")
        
        # Show latest model info
        if gen_models:
            latest_model = gen_models[-1]
            latest_epoch = os.path.basename(latest_model).split('_')[0]
            print(f"\n🎯 LATEST MODEL: {latest_epoch}_net_G.pth")
            print(f"📊 TOTAL EPOCHS TRAINED: {latest_epoch}")
            
            # Check for latest.pth
            latest_pth = os.path.join(checkpoint_dir, "latest_net_G.pth")
            if os.path.exists(latest_pth):
                print(f"✅ Also found: latest_net_G.pth (symlink to latest epoch)")
        
    else:
        print("❌ No generator models found!")

def main():
    checkpoint_dir = r"C:\Users\Wahbi Saied\Documents\GitHub\pix2pix\checkpoints\ct_phase0_generator_5090"
    
    print("🔍 Analyzing checkpoint directory:")
    print(f"📁 {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print("❌ Checkpoint directory not found!")
        return
    
    # Analyze model files
    analyze_model_files(checkpoint_dir)
    
    # Display training results
    load_and_display_results(checkpoint_dir)
    
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS:")
    print("=" * 60)
    print("1. Run: python analyze_loss.py (to find the best epoch)")
    print("2. Use test.py to evaluate on test data:")
    print("   python test.py --dataroot ./datasets/your_test_data \\")
    print("                  --name ct_phase0_generator_5090 \\")
    print("                  --model pix2pix \\")
    print("                  --epoch [best_epoch_number]")
    print("3. Check results in ./results/ folder")
    print("\n🎯 QUICK TEST: Use 'latest' or '200' as epoch number")

if __name__ == "__main__":
    main()