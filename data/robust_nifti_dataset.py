import os
import nibabel as nib
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import logging

class RobustNiftiDataset(BaseDataset):
    """
    A robust version of NiftiAlignedDataset that handles corrupted files gracefully.
    
    Features:
    - Pre-validates all files during initialization
    - Skips corrupted files/slices automatically
    - Provides detailed logging of issues
    - Maintains training stability even with some corrupted data
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options"""
        parser.add_argument('--axial_slice', action='store_true', help='if specified, load slices along the axial dimension')
        parser.add_argument('--resize_to', type=int, default=None, help='resize images to this size (e.g., 256, 512)')
        parser.add_argument('--skip_corrupted', action='store_true', default=True, help='skip corrupted files instead of crashing')
        parser.add_argument('--min_slice_variance', type=float, default=1e-6, help='minimum variance for valid slices')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class with robust error handling."""
        BaseDataset.__init__(self, opt)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        assert(len(self.A_paths) == len(self.B_paths)), "The number of images in trainA and trainB must be the same."

        self.slice_maps = []
        self.total_slices = 0
        self.corrupted_files = []
        self.corrupted_slices = []

        # Pre-validate all files and build slice maps
        self._validate_and_build_slice_maps(opt)
        
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))
        
        # Report statistics
        self.logger.info(f"Dataset initialized successfully:")
        self.logger.info(f"  - Total volume pairs: {len(self.A_paths)}")
        self.logger.info(f"  - Total valid slices: {self.total_slices}")
        self.logger.info(f"  - Corrupted files: {len(self.corrupted_files)}")
        self.logger.info(f"  - Corrupted slices: {len(self.corrupted_slices)}")

    def _validate_and_build_slice_maps(self, opt):
        """Validate all files and build slice maps, skipping corrupted data."""
        
        for volume_idx in range(len(self.A_paths)):
            A_path = self.A_paths[volume_idx]
            B_path = self.B_paths[volume_idx]
            
            try:
                # Load volumes
                img_A = nib.load(A_path)
                img_B = nib.load(B_path)
                
                data_A = img_A.get_fdata()
                data_B = img_B.get_fdata()
                
                # Check shapes match
                if data_A.shape != data_B.shape:
                    self.logger.warning(f"Shape mismatch in pair {volume_idx}: A={data_A.shape}, B={data_B.shape}")
                    self.corrupted_files.append((A_path, B_path, "shape_mismatch"))
                    continue
                
                # Check for NaN/Inf values
                if np.any(np.isnan(data_A)) or np.any(np.isinf(data_A)) or \
                   np.any(np.isnan(data_B)) or np.any(np.isinf(data_B)):
                    self.logger.warning(f"NaN/Inf values in pair {volume_idx}")
                    self.corrupted_files.append((A_path, B_path, "nan_inf_values"))
                    continue
                
                # Determine number of slices
                if opt.axial_slice:
                    num_slices = data_A.shape[2]
                else:
                    raise ValueError("Please specify an axis to slice along, e.g., --axial_slice")
                
                # Validate each slice
                valid_slices = 0
                for slice_idx in range(num_slices):
                    if self._is_valid_slice(data_A, data_B, slice_idx, opt):
                        self.slice_maps.append({
                            'volume_idx': volume_idx, 
                            'slice_idx': slice_idx,
                            'A_path': A_path,
                            'B_path': B_path
                        })
                        valid_slices += 1
                    else:
                        self.corrupted_slices.append((A_path, B_path, slice_idx))
                
                self.total_slices += valid_slices
                self.logger.info(f"Volume {volume_idx}: {valid_slices}/{num_slices} valid slices")
                
            except Exception as e:
                self.logger.error(f"Error processing volume pair {volume_idx}: {str(e)}")
                self.corrupted_files.append((A_path, B_path, str(e)))
                continue

    def _is_valid_slice(self, data_A, data_B, slice_idx, opt):
        """Check if a slice is valid for training."""
        try:
            if opt.axial_slice:
                slice_A = data_A[:, :, slice_idx]
                slice_B = data_B[:, :, slice_idx]
            else:
                return False
            
            # Check for empty or constant slices
            if np.all(slice_A == 0) or np.all(slice_B == 0):
                return False
            
            # Check for sufficient variance
            min_variance = getattr(opt, 'min_slice_variance', 1e-6)
            if np.var(slice_A) < min_variance or np.var(slice_B) < min_variance:
                return False
            
            # Check for reasonable intensity ranges
            if np.max(slice_A) == np.min(slice_A) or np.max(slice_B) == np.min(slice_B):
                return False
            
            return True
            
        except Exception:
            return False

    def __getitem__(self, index):
        """Return a data point with robust error handling."""
        max_retries = 5
        
        for retry in range(max_retries):
            try:
                # Get slice info
                slice_info = self.slice_maps[index]
                volume_idx = slice_info['volume_idx']
                slice_idx = slice_info['slice_idx']
                A_path = slice_info['A_path']
                B_path = slice_info['B_path']

                # Load volumes (cached loading could be added here)
                vol_A = nib.load(A_path).get_fdata()
                vol_B = nib.load(B_path).get_fdata()

                # Extract slices
                if self.opt.axial_slice:
                    slice_A = vol_A[:, :, slice_idx]
                    slice_B = vol_B[:, :, slice_idx]
                else:
                    raise ValueError("Please specify an axis to slice along, e.g., --axial_slice")

                # Normalize intensity values
                slice_A = self._normalize_slice(slice_A)
                slice_B = self._normalize_slice(slice_B)
                
                # Convert to PIL Images
                A = Image.fromarray(np.uint8(slice_A)).convert('L')
                B = Image.fromarray(np.uint8(slice_B)).convert('L')
                
                # Resize if specified
                if self.opt.resize_to is not None:
                    A = A.resize((self.opt.resize_to, self.opt.resize_to), Image.LANCZOS)
                    B = B.resize((self.opt.resize_to, self.opt.resize_to), Image.LANCZOS)

                # Apply transformations
                A = self.transform(A)
                B = self.transform(B)

                # Create unique identifiers for each slice
                patient_name = os.path.splitext(os.path.basename(A_path))[0]
                slice_identifier = f"{patient_name}_slice{slice_idx:03d}"
                
                return {
                    'A': A, 
                    'B': B, 
                    'A_paths': slice_identifier, 
                    'B_paths': slice_identifier
                }
                
            except Exception as e:
                self.logger.warning(f"Error loading sample {index} (retry {retry+1}): {str(e)}")
                
                # Try a different random sample
                if retry < max_retries - 1:
                    index = np.random.randint(0, len(self.slice_maps))
                else:
                    # Last resort: return a dummy sample
                    self.logger.error(f"Failed to load any sample after {max_retries} retries")
                    return self._get_dummy_sample()
    
    def _normalize_slice(self, slice_data):
        """Normalize slice intensity values to [0, 255] range."""
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        
        if slice_max == slice_min:
            return np.zeros_like(slice_data)
        
        return (slice_data - slice_min) / (slice_max - slice_min) * 255.0
    
    def _get_dummy_sample(self):
        """Return a dummy sample in case of complete failure."""
        dummy_size = self.opt.resize_to if self.opt.resize_to else 64
        dummy_image = torch.zeros(1, dummy_size, dummy_size)
        
        return {
            'A': dummy_image, 
            'B': dummy_image, 
            'A_paths': 'dummy', 
            'B_paths': 'dummy'
        }

    def __len__(self):
        """Return the total number of valid 2D slices in the dataset."""
        return self.total_slices
    
    def get_corruption_report(self):
        """Return a detailed report of corrupted files and slices."""
        report = {
            'corrupted_files': self.corrupted_files,
            'corrupted_slices': self.corrupted_slices,
            'total_corrupted_files': len(self.corrupted_files),
            'total_corrupted_slices': len(self.corrupted_slices),
            'valid_slices': self.total_slices
        }
        return report