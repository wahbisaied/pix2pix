import os
import nibabel as nib
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class NiftiAlignedDataset(BaseDataset):
    """
    A dataset class for paired NIfTI files (e.g., average CT and phase CT).
    It assumes that the file names for paired data are identical and stored in separate directories.

    The dataset is structured to treat each axial slice as an individual 2D image.

    Expected directory structure:
    /path/to/data/
    ├── trainA/  (e.g., contains avg_scan1.nii.gz, avg_scan2.nii.gz)
    └── trainB/  (e.g., contains phase0_scan1.nii.gz, phase0_scan2.nii.gz)

    (And similarly for valA, valB, etc.)
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--axial_slice', action='store_true', help='if specified, load slices along the axial dimension')
        # You can add other axes (sagittal, coronal) here if needed
        # parser.add_argument('--sagittal_slice', action='store_true', help='load slices along the sagittal dimension')
        # parser.add_argument('--coronal_slice', action='store_true', help='load slices along the coronal dimension')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # get the images directory for domain A
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # get the images directory for domain B

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        assert(len(self.A_paths) == len(self.B_paths)), "The number of images in trainA and trainB must be the same."

        self.slice_maps = []
        self.total_slices = 0

        # Pre-calculate the number of slices in each volume
        for index in range(len(self.A_paths)):
            A_path = self.A_paths[index]
            img_A = nib.load(A_path)
            
            # Determine which axis to slice along
            if opt.axial_slice:
                num_slices = img_A.shape[2] # Axial slices
            # elif opt.sagittal_slice:
            #     num_slices = img_A.shape[0] # Sagittal slices
            # elif opt.coronal_slice:
            #     num_slices = img_A.shape[1] # Coronal slices
            else:
                raise ValueError("Please specify an axis to slice along, e.g., --axial_slice")

            for slice_idx in range(num_slices):
                self.slice_maps.append({'volume_idx': index, 'slice_idx': slice_idx})
            
            self.total_slices += num_slices

        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point (a pair of 2D slices).
        Parameters:
            index (int) -- a random integer for data indexing
        
        Returns a dictionary that contains A, B, A_paths, and B_paths
            A (tensor) -- an image in the input domain
            B (tensor) -- its corresponding image in the target domain
            A_paths (str) -- image path
            B_paths (str) -- image path
        """
        # Determine which volume and slice this index corresponds to
        slice_info = self.slice_maps[index]
        volume_idx = slice_info['volume_idx']
        slice_idx = slice_info['slice_idx']

        A_path = self.A_paths[volume_idx]
        B_path = self.B_paths[volume_idx]

        # Load the 3D NIfTI volumes
        vol_A = nib.load(A_path).get_fdata()
        vol_B = nib.load(B_path).get_fdata()

        # Extract the 2D slice
        if self.opt.axial_slice:
            slice_A = vol_A[:, :, slice_idx]
            slice_B = vol_B[:, :, slice_idx]
        # Add other axes if needed
        # elif self.opt.sagittal_slice:
        #     slice_A = vol_A[slice_idx, :, :]
        #     slice_B = vol_B[slice_idx, :, :]
        # elif self.opt.coronal_slice:
        #     slice_A = vol_A[:, slice_idx, :]
        #     slice_B = vol_B[:, slice_idx, :]
        else:
            raise ValueError("Please specify an axis to slice along, e.g., --axial_slice")

        # --- Data Preprocessing ---
        # 1. Normalize the intensity values (e.g., to [0, 255] or [-1, 1])
        # This is a basic example; you might need more sophisticated windowing/leveling for CT
        slice_A = (slice_A - np.min(slice_A)) / (np.max(slice_A) - np.min(slice_A)) * 255.0
        slice_B = (slice_B - np.min(slice_B)) / (np.max(slice_B) - np.min(slice_B)) * 255.0
        
        # Convert to PIL Image to use existing transforms
        # Convert to PIL Image to use existing transforms. 
        # 'L' mode is for single-channel grayscale.
        A = Image.fromarray(np.uint8(slice_A)).convert('L')
        B = Image.fromarray(np.uint8(slice_B)).convert('L')

        # Apply the transformations (e.g., normalization to tensor)
        A = self.transform(A)
        B = self.transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of 2D slices in the dataset."""
        return self.total_slices
