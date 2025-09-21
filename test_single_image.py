#!/usr/bin/env python3
"""
Test script for single PNG/JPG images
Usage: python test_single_image.py --input_image path/to/image.png --model_name ct_phase0_generator_optimized
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from models import create_model
from options.test_options import TestOptions
from data.base_dataset import get_transform
import sys

def load_model(model_name, epoch='latest'):
    """Load the trained model"""
    original_argv = sys.argv.copy()
    sys.argv = ['test_single_image.py', '--dataroot', './dummy', '--name', model_name, 
                '--model', 'pix2pix', '--dataset_mode', 'single',
                '--preprocess', 'resize_and_crop', '--input_nc', '1', '--output_nc', '1',
                '--load_size', '286', '--crop_size', '256', '--epoch', epoch]
    
    try:
        opt = TestOptions().parse()
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        
        return model, opt
    finally:
        sys.argv = original_argv

def test_single_image(input_path, model_name='ct_phase0_generator_optimized', epoch='latest', output_dir='./single_test_results'):
    """Test on a single PNG/JPG image"""
    
    print(f"Loading model: {model_name}")
    model, opt = load_model(model_name, epoch)
    
    print(f"Loading image: {input_path}")
    # Load and preprocess image
    img = Image.open(input_path).convert('L')  # Convert to grayscale
    
    # Apply same transforms as training
    transform = get_transform(opt, grayscale=True)
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate
    with torch.no_grad():
        model.set_input({'A': input_tensor, 'B': input_tensor, 'A_paths': [''], 'B_paths': ['']})
        model.test()
        visuals = model.get_current_visuals()
        
        # Get results
        real_A = visuals['real_A']
        fake_B = visuals['fake_B']
        
        # Convert to PIL images
        def tensor_to_pil(tensor):
            # Convert from [-1, 1] to [0, 255]
            array = (tensor.cpu().numpy().squeeze() + 1) / 2 * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array)
        
        input_img = tensor_to_pil(real_A)
        output_img = tensor_to_pil(fake_B)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        input_img.save(os.path.join(output_dir, f'{base_name}_input.png'))
        output_img.save(os.path.join(output_dir, f'{base_name}_generated.png'))
        
        # Create side-by-side comparison
        comparison = Image.new('L', (input_img.width * 2, input_img.height))
        comparison.paste(input_img, (0, 0))
        comparison.paste(output_img, (input_img.width, 0))
        comparison.save(os.path.join(output_dir, f'{base_name}_comparison.png'))
        
        print(f"Results saved to: {output_dir}")
        print(f"Input: {base_name}_input.png")
        print(f"Generated: {base_name}_generated.png")
        print(f"Comparison: {base_name}_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test single image')
    parser.add_argument('--input_image', required=True, help='Path to input image (PNG/JPG)')
    parser.add_argument('--model_name', default='ct_phase0_generator_optimized', help='Model name')
    parser.add_argument('--epoch', default='latest', help='Epoch to use')
    parser.add_argument('--output_dir', default='./single_test_results', help='Output directory')
    
    args = parser.parse_args()
    test_single_image(args.input_image, args.model_name, args.epoch, args.output_dir)