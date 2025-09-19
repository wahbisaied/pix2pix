# Project Summary: CT Scan Phase Generation

This document summarizes the process of adapting the `pytorch-CycleGAN-and-pix2pix` repository for a medical imaging task: generating phase 0 CT scans from an average CT scan.

### 1. Project Goal

The primary objective is to train a `pix2pix` model to perform an image-to-image translation from a 2D slice of an average CT scan (input) to a corresponding 2D slice of a phase 0 CT scan (output).

### 2. Core Implementation Steps

To achieve this, we performed the following key actions:

#### a. Custom NIfTI Dataset Loader

A new dataset loader was created to handle the specific requirements of NIfTI medical imaging files.

-   **File Created:** `data/nifti_aligned_dataset.py`
-   **Functionality:**
    -   Loads paired 3D NIfTI volumes from `trainA` (average) and `trainB` (phase) directories.
    -   Treats each 2D axial slice from the volumes as an individual training sample.
    -   Converts the slices to tensors suitable for the `pix2pix` model.
    -   Includes a `--preprocess none` option to disable resizing and preserve the original image dimensions.

#### b. Data Preparation Automation

The source data was located in `C:\Users\wahbi\OneDrive\res_avg` and `C:\Users\wahbi\OneDrive\res_phases` with a complex, unpaired structure. A script was created to automate the entire preparation process.

-   **File Created:** `prepare_dataset.py`
-   **Functionality:**
    -   Identifies pairs of scans by matching subdirectory names between the `res_avg` and `res_phases` folders.
    -   Specifically searches for the phase 0 file within each phase subfolder using the `*Gated 0.0A*.nii*` pattern.
    -   Copies the paired average and phase 0 scans into the project's `datasets/ct_phases_dataset/trainA` and `datasets/ct_phases_dataset/trainB` folders.
    -   Renames the files to a consistent, matched naming scheme (e.g., `patient_1.nii.gz`).
-   **Debugging:** The script was refined to handle `PermissionError` issues related to OneDrive's Files On-Demand feature and corrected to handle the directory-based structure of the source data.

### 3. Final Status & Usage

-   **Dataset:** The `prepare_dataset.py` script successfully processed the source data, creating a training set of **75 paired scans**. A list of 8 scans that could not be paired was generated for user review.
-   **Training Command:** The model can be trained using the following command:
    ```bash
    python train.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator --model pix2pix --dataset_mode nifti_aligned --preprocess none --input_nc 1 --output_nc 1 --axial_slice
    ```
-   **Testing Command:** Once trained, the model can be tested on new data using:
    ```bash

    ** to run the train file 
    python test.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator --model pix2pix --dataset_mode nifti_aligned --preprocess none --input_nc 1 --output_nc 1 --axial_slice
    ```

### 4. Troubleshooting

During the setup, we resolved the following issues:

-   **`ModuleNotFoundError: No module named 'wandb'`**: This was fixed by installing the `wandb` dependency, which is required by the project's visualizer utility.

-   **`ValueError: num_samples should be a positive integer value, but got num_samples=0`**: This error occurred because the framework did not recognize NIfTI files. It was resolved by modifying `data/image_folder.py` to include `.nii` and `.nii.gz` in the list of valid image extensions.

-   **`ValueError: Expected more than 1 spatial element when training, got input size torch.Size([1, 512, 1, 1])`**: This instance normalization error occurred because the default U-Net architectures (unet_128, unet_256) were designed for larger input sizes, but our CT slices are 64x64 pixels. The solution was to:
    1. Add a new `unet_64` architecture option in `models/networks.py` with 6 downsampling layers instead of 7 or 8
    2. Use `--netG unet_64` in the training command

### 5. Resolution Analysis

**IMPORTANT FINDING**: The model is actually training on **512x512 pixel** images, not 64x64 as initially suspected.

**Evidence from training configuration:**
- `--resize_to 512`: Images are resized to 512x512 pixels in the dataset loader
- `--netG unet_512`: Using U-Net architecture designed for 512x512 images (9 downsampling layers)
- `--batch_size 4`: Larger batch size possible due to higher resolution training

**Original CT scan dimensions**: The original NIfTI files contain CT slices that are likely around 64x64 pixels natively, but the dataset loader (`nifti_aligned_dataset.py`) automatically resizes them to 512x512 using PIL's LANCZOS interpolation before feeding them to the model.

**Why images might appear low resolution:**
1. The original CT data has inherently low spatial resolution (64x64 native)
2. Upsampling from 64x64 to 512x512 doesn't add real detail, just interpolated pixels
3. The model learns on the upsampled 512x512 images but the underlying information content is still limited by the original 64x64 resolution

### 6. Training Commands

**Original working command:**
```bash
python train.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator --model pix2pix --dataset_mode nifti_aligned --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --batch_size 1 --netG unet_64
```

**Optimized command (currently running):**
```bash
python train.py --dataroot ./datasets/ct_phases_dataset --name ct_phase0_generator_optimized --model pix2pix --dataset_mode nifti_aligned --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 4 --lambda_L1 50 --lr 0.0001 --resize_to 512 --display_freq 100 --save_epoch_freq 5
```

### 7. CT-Specific Metadata Saving

To improve model reproducibility and tracking for medical imaging applications, we implemented enhanced model saving with CT-specific metadata.

#### a. Enhanced Model Saving Script

-   **File Created:** `save_ct_metadata.py`
-   **Functionality:**
    -   `save_model_with_metadata()`: Saves comprehensive metadata alongside model checkpoints including:
        - Model type and architecture details
        - Dataset information (input/output types, image dimensions, voxel spacing)
        - Training parameters (epoch, batch size, learning rate, lambda values)
        - Preprocessing settings and normalization ranges
        - CT-specific parameters (slice thickness: 3.0mm, voxel spacing: 0.9766x0.9766x3.0mm)
    -   `load_model_with_metadata()`: Loads and returns metadata associated with saved models
    -   Metadata is saved as JSON files with naming pattern: `{epoch}_metadata.json`

#### b. Integration Benefits

The metadata saving system provides:
-   **Reproducibility**: Complete record of training configuration and data specifications
-   **Medical Compliance**: Tracks critical CT scan parameters (voxel spacing, slice thickness)
-   **Model Versioning**: Easy comparison between different training runs and epochs
-   **Debugging Support**: Detailed information for troubleshooting model performance issues

#### c. Usage Example

```python
# In training loop
from save_ct_metadata import save_model_with_metadata

# Save model with metadata
metadata = save_model_with_metadata(model, opt, epoch, save_dir)

# Later, load model with metadata
from save_ct_metadata import load_model_with_metadata
metadata = load_model_with_metadata('path/to/model.pth')
```

This enhancement ensures that all CT-specific training parameters and medical imaging specifications are preserved with each model checkpoint, making the training process more transparent and reproducible for medical applications.