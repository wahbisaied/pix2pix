import os
import sys
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data.base_dataset import get_transform
from options.train_options import TrainOptions

# Load a sample NIfTI file
nii_path = r"C:\Users\wahbi\OneDrive\Desktop\pytorch-CycleGAN-and-pix2pix\datasets\ct_phases_dataset\trainA\patient_1.nii.gz"
vol = nib.load(nii_path).get_fdata()
slice_img = vol[:, :, vol.shape[2] // 2]

# Normalize like in the dataset
slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255.0
pil_img = Image.fromarray(np.uint8(slice_img)).convert('L')

# Create options for transforms
sys.argv = ['test_augmentation.py', '--dataroot', './datasets/ct_phases_dataset', '--name', 'test']
opt = TrainOptions().parse()

# Get transform with flipping enabled (default)
transform_with_flip = get_transform(opt, grayscale=True)

# Get transform with flipping disabled
opt.no_flip = True
transform_no_flip = get_transform(opt, grayscale=True)

# Apply transforms multiple times to see the difference
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original
axes[0, 0].imshow(pil_img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# With flipping (may randomly flip)
for i in range(2):
    transformed = transform_with_flip(pil_img)
    axes[0, i+1].imshow(transformed.squeeze(), cmap='gray')
    axes[0, i+1].set_title(f'With Flip #{i+1}')
    axes[0, i+1].axis('off')

# Without flipping (consistent)
for i in range(3):
    transformed = transform_no_flip(pil_img)
    axes[1, i].imshow(transformed.squeeze(), cmap='gray')
    axes[1, i].set_title(f'No Flip #{i+1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('augmentation_test.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved augmentation_test.png - check if flipping occurs randomly in top row")