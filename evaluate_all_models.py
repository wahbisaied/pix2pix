import os
import glob
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# Configuration
CHECKPOINTS_DIR = "./checkpoints/ct_phase0_generator"  # Your model directory
REAL_IMAGES_DIR = "./datasets/ct_phases_dataset/test/testB_png"  # Convert testB to PNG first
RESULTS_BASE_DIR = "./fid_evaluation_results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Get all model files (every 5 epochs)
model_files = glob.glob(f"{CHECKPOINTS_DIR}/*_net_G.pth")
model_files.sort(key=lambda x: int(x.split('_')[-3]))  # Sort by epoch number

print(f"Found {len(model_files)} models to evaluate")

# Setup test options (modify as needed)
opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1

dataset = create_dataset(opt)

# Open log file
with open("fid_scores.txt", "w") as log_file:
    log_file.write("Model,Epoch,FID_Score\n")
    
    for i, model_file in enumerate(model_files):
        # Extract epoch number from filename
        epoch = model_file.split('_')[-3]
        model_name = f"epoch_{epoch}"
        
        print(f"\n--- Evaluating {i+1}/{len(model_files)}: {model_name} ---")
        
        # Create output directory
        output_dir = os.path.join(RESULTS_BASE_DIR, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model = create_model(opt)
        model.setup(opt)
        
        # Load specific checkpoint
        model.load_networks(epoch)
        model.eval()
        
        # Generate images
        print(f"Generating images...")
        for j, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            
            # Save only fake_B images as PNG
            fake_B = visuals['fake_B']
            save_path = os.path.join(output_dir, f"fake_{j:04d}.png")
            # Convert tensor to PIL and save
            # You'll need to implement tensor_to_pil conversion
            
        # Calculate FID
        print("Calculating FID...")
        fid_command = f"python -m pytorch_fid {REAL_IMAGES_DIR} {output_dir}"
        fid_output = os.popen(fid_command).read()
        
        try:
            fid_score = float(fid_output.strip().split()[-1])
            print(f"FID Score: {fid_score}")
        except:
            fid_score = "Error"
            print("Error calculating FID")
        
        # Log result
        log_file.write(f"{model_name},{epoch},{fid_score}\n")
        log_file.flush()
        
        # Optional: Clean up generated images to save space
        # shutil.rmtree(output_dir)

print("\nEvaluation complete! Check fid_scores.txt for results")