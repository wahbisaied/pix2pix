import torch
import numpy as np
from PIL import Image
import os
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import matplotlib.pyplot as plt
from models import create_model
from options.test_options import TestOptions
from data import create_dataset
import argparse

def load_model(checkpoint_path, model_name):
    """Load a trained pix2pix model"""
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.name = model_name
    opt.checkpoints_dir = os.path.dirname(checkpoint_path)
    opt.model = 'pix2pix'
    opt.netG = 'unet_256'
    opt.direction = 'AtoB'
    opt.dataset_mode = 'aligned'
    opt.norm = 'batch'
    
    model = create_model(opt)
    model.setup(opt)
    return model, opt

def calculate_metrics(real_img, fake_img):
    """Calculate SSIM, PSNR, and MAE between real and generated images"""
    # Convert to numpy arrays
    if isinstance(real_img, torch.Tensor):
        real_img = real_img.cpu().numpy()
    if isinstance(fake_img, torch.Tensor):
        fake_img = fake_img.cpu().numpy()
    
    # Normalize to [0, 1]
    real_img = (real_img + 1) / 2.0
    fake_img = (fake_img + 1) / 2.0
    
    # Calculate metrics
    ssim_score = ssim(real_img, fake_img, data_range=1.0)
    psnr_score = psnr(real_img, fake_img, data_range=1.0)
    mae_score = np.mean(np.abs(real_img - fake_img))
    
    return ssim_score, psnr_score, mae_score

def evaluate_model(model_path, test_data_path, model_name):
    """Evaluate a single model on test data"""
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    model, opt = load_model(model_path, model_name)
    
    # Set test data path
    opt.dataroot = test_data_path
    dataset = create_dataset(opt)
    
    ssim_scores = []
    psnr_scores = []
    mae_scores = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= 50:  # Limit to 50 test images for speed
                break
                
            model.set_input(data)
            model.forward()
            
            # Get images
            real_A = model.real_A[0].cpu().numpy()
            real_B = model.real_B[0].cpu().numpy()
            fake_B = model.fake_B[0].cpu().numpy()
            
            # Calculate metrics for each channel
            for c in range(real_B.shape[0]):
                ssim_score, psnr_score, mae_score = calculate_metrics(
                    real_B[c], fake_B[c]
                )
                ssim_scores.append(ssim_score)
                psnr_scores.append(psnr_score)
                mae_scores.append(mae_score)
    
    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'num_samples': len(ssim_scores)
    }

def compare_models(model1_path, model2_path, test_data_path, model1_name, model2_name):
    """Compare two models and recommend which one to use"""
    
    print("=" * 60)
    print("MODEL COMPARISON FOR PIX2PIX CARDIAC PHASE TRANSLATION")
    print("=" * 60)
    
    # Evaluate both models
    results1 = evaluate_model(model1_path, test_data_path, model1_name)
    results2 = evaluate_model(model2_path, test_data_path, model2_name)
    
    # Print results
    print(f"\n{model1_name.upper()} RESULTS:")
    print(f"SSIM: {results1['ssim_mean']:.4f} ± {results1['ssim_std']:.4f}")
    print(f"PSNR: {results1['psnr_mean']:.2f} ± {results1['psnr_std']:.2f} dB")
    print(f"MAE:  {results1['mae_mean']:.4f} ± {results1['mae_std']:.4f}")
    
    print(f"\n{model2_name.upper()} RESULTS:")
    print(f"SSIM: {results2['ssim_mean']:.4f} ± {results2['ssim_std']:.4f}")
    print(f"PSNR: {results2['psnr_mean']:.2f} ± {results2['psnr_std']:.2f} dB")
    print(f"MAE:  {results2['mae_mean']:.4f} ± {results2['mae_std']:.4f}")
    
    # Calculate improvements
    ssim_diff = results2['ssim_mean'] - results1['ssim_mean']
    psnr_diff = results2['psnr_mean'] - results1['psnr_mean']
    mae_diff = results1['mae_mean'] - results2['mae_mean']  # Lower MAE is better
    
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS:")
    print("=" * 60)
    
    # Score each metric
    scores = {'model1': 0, 'model2': 0}
    
    if ssim_diff > 0.01:  # Significant SSIM improvement
        print(f"✓ {model2_name} has BETTER SSIM (+{ssim_diff:.4f})")
        scores['model2'] += 2
    elif ssim_diff < -0.01:
        print(f"✓ {model1_name} has BETTER SSIM (+{-ssim_diff:.4f})")
        scores['model1'] += 2
    else:
        print(f"≈ SSIM is similar (diff: {ssim_diff:.4f})")
        scores['model1'] += 1
        scores['model2'] += 1
    
    if psnr_diff > 1.0:  # Significant PSNR improvement
        print(f"✓ {model2_name} has BETTER PSNR (+{psnr_diff:.2f} dB)")
        scores['model2'] += 2
    elif psnr_diff < -1.0:
        print(f"✓ {model1_name} has BETTER PSNR (+{-psnr_diff:.2f} dB)")
        scores['model1'] += 2
    else:
        print(f"≈ PSNR is similar (diff: {psnr_diff:.2f} dB)")
        scores['model1'] += 1
        scores['model2'] += 1
    
    if mae_diff > 0.01:  # Significant MAE improvement (lower is better)
        print(f"✓ {model2_name} has BETTER MAE (-{mae_diff:.4f})")
        scores['model2'] += 2
    elif mae_diff < -0.01:
        print(f"✓ {model1_name} has BETTER MAE (-{-mae_diff:.4f})")
        scores['model1'] += 2
    else:
        print(f"≈ MAE is similar (diff: {mae_diff:.4f})")
        scores['model1'] += 1
        scores['model2'] += 1
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION:")
    print("=" * 60)
    
    if scores['model2'] > scores['model1']:
        winner = model2_name
        winner_path = model2_path
        print(f"🏆 USE {model2_name.upper()} for training other phases!")
        print(f"   Score: {scores['model2']}/6 vs {scores['model1']}/6")
    elif scores['model1'] > scores['model2']:
        winner = model1_name
        winner_path = model1_path
        print(f"🏆 USE {model1_name.upper()} for training other phases!")
        print(f"   Score: {scores['model1']}/6 vs {scores['model2']}/6")
    else:
        print(f"🤝 BOTH MODELS perform similarly!")
        print(f"   Recommend using {model2_name} (more recent training)")
        winner = model2_name
        winner_path = model2_path
    
    # Training recommendations
    print(f"\nRECOMMENDED TRAINING SETTINGS for remaining phases:")
    if winner == model1_name:
        print("- Learning Rate: 0.0001")
        print("- Epochs: 100 + 100 decay")
        print("- Reason: Conservative approach worked better")
    else:
        print("- Learning Rate: 0.0002") 
        print("- Epochs: 150 + 50 decay")
        print("- Reason: Higher LR with more epochs showed better results")
    
    return winner, winner_path, results1, results2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_path', required=True, help='Path to first model checkpoint')
    parser.add_argument('--model2_path', required=True, help='Path to second model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test dataset')
    parser.add_argument('--model1_name', default='Model_80patients_lr0001', help='Name for first model')
    parser.add_argument('--model2_name', default='Model_68patients_lr0002', help='Name for second model')
    
    args = parser.parse_args()
    
    winner, winner_path, results1, results2 = compare_models(
        args.model1_path, 
        args.model2_path, 
        args.test_data,
        args.model1_name,
        args.model2_name
    )