# 

# pred_DS.py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import rasterio as rio
import torch

class PredTwoMonths(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 16 bands for the input image.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 16 bands as the input image
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan=0.0001)
            
            # Get geolocation information
            geotransform = src.transform
            crs = src.crs.to_dict()  # Convert CRS to dictionary
            
        # Create a sample dictionary
        sample = {'image': image, 'geotransform': geotransform, 'crs': crs, 'path': str(image_path)}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
