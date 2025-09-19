#!/usr/bin/env python3
"""
Enhanced model saving with CT-specific metadata
"""
import torch
import json
from pathlib import Path

def save_model_with_metadata(model, opt, epoch, save_dir):
    """Save model with CT-specific metadata for better reproducibility"""
    
    # Create metadata dictionary
    metadata = {
        'model_type': 'pix2pix_ct_phase_generator',
        'dataset_info': {
            'input_type': 'average_ct_slices',
            'output_type': 'phase0_ct_slices', 
            'image_size': getattr(opt, 'resize_to', opt.crop_size),
            'input_channels': opt.input_nc,
            'output_channels': opt.output_nc,
            'voxel_spacing': '0.9766x0.9766x3.0mm',  # From your data
            'slice_thickness': '3.0mm'
        },
        'training_info': {
            'epoch': epoch,
            'batch_size': opt.batch_size,
            'learning_rate': opt.lr,
            'lambda_L1': getattr(opt, 'lambda_L1', 100.0),
            'generator_arch': opt.netG,
            'normalization': opt.norm
        },
        'preprocessing': {
            'preprocess_mode': opt.preprocess,
            'axial_slice': getattr(opt, 'axial_slice', False),
            'normalization_range': '[-1, 1]'  # Standard for pix2pix
        }
    }
    
    # Save metadata
    metadata_path = save_dir / f"{epoch}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved CT metadata to: {metadata_path}")
    return metadata

def load_model_with_metadata(model_path):
    """Load model and return associated metadata"""
    model_dir = Path(model_path).parent
    epoch = Path(model_path).stem.split('_')[0]
    
    metadata_path = model_dir / f"{epoch}_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded CT metadata from: {metadata_path}")
        return metadata
    else:
        print(f"⚠️ No metadata found for {model_path}")
        return None

# Example usage in your training script:
if __name__ == "__main__":
    # This would be integrated into the training loop
    pass