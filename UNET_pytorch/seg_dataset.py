from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import rasterio as rio
import torch
from torchvision.transforms import functional as F

class SegmentationDataset(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 8 bands for the input image and the last band as the mask.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform
        self.weights = self.calculate_weights()
        self.class_weights = self.compute_class_weights_for_crossentropy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 8 bands as the input image
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan =0.000000)
            # Bands in rasterio are indexed starting from 1, so the 9th band is the mask
            mask = src.read(9)
            
            # Move the channel axis to the last dimension for the image
            # image = np.moveaxis(image, 0, -1)
            # Squeeze the mask to remove the channel axis, if it's single-channel
            # mask = np.squeeze(mask)

        # Create a sample dictionary
        sample = {'image': image, 'mask': mask}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# return the weights for each image
    def calculate_weights(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask, return_counts=True)
                for label, count in zip(labels, counts):
                    if label in class_counts:
                        class_counts[label] += count
                    else:
                        class_counts[label] = count

        total_counts = sum(class_counts.values())
        class_weights = {label: total_counts / count for label, count in class_counts.items()}
        image_weights = []

        # Calculate weights for each image based on its mask
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask, return_counts=True)
                image_weight = sum(class_weights[label] for label in labels)
                image_weights.append(image_weight)

        return image_weights
    
    def compute_class_weights_for_crossentropy(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask.astype(int), return_counts=True)  # Ensure labels are integers
                for label, count in zip(labels, counts):
                    class_counts[label] = class_counts.get(label, 0) + count

        total_counts = sum(class_counts.values())
        num_classes = int(max(class_counts.keys()) + 1)  # Cast max label to int before adding 1
        weights = [total_counts / class_counts.get(i, 1) for i in range(num_classes)]  # default to 1 if no samples for a class
        min_weight = min(weights)
        normalized_weights = [w / min_weight for w in weights]

        return torch.tensor(normalized_weights, dtype=torch.float32)
    


    

class SegmentationDatasetTwoMonths(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 8 bands for the input image and the last band as the mask.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform
        # self.weights = self.calculate_weights()
        self.class_weights = self.compute_class_weights_for_crossentropy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 8 bands as the input image
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan =0.0001)
            # Bands in rasterio are indexed starting from 1, so the 9th band is the mask
            mask = src.read(17)
            
            # Move the channel axis to the last dimension for the image
            # image = np.moveaxis(image, 0, -1)
            # Squeeze the mask to remove the channel axis, if it's single-channel
            # mask = np.squeeze(mask)

        # Create a sample dictionary
        sample = {'image': image, 'mask': mask}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# return the weights for each image
    def calculate_weights(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(17)
                labels, counts = np.unique(mask, return_counts=True)
                for label, count in zip(labels, counts):
                    if label in class_counts:
                        class_counts[label] += count
                    else:
                        class_counts[label] = count

        total_counts = sum(class_counts.values())
        class_weights = {label: total_counts / count for label, count in class_counts.items()}
        image_weights = []

        # Calculate weights for each image based on its mask
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(17)
                labels, counts = np.unique(mask, return_counts=True)
                image_weight = sum(class_weights[label] for label in labels)
                image_weights.append(image_weight)

        return image_weights
    
    def compute_class_weights_for_crossentropy(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(17)
                labels, counts = np.unique(mask.astype(int), return_counts=True)  # Ensure labels are integers
                for label, count in zip(labels, counts):
                    class_counts[label] = class_counts.get(label, 0) + count

        total_counts = sum(class_counts.values())
        num_classes = int(max(class_counts.keys()) + 1)  # Cast max label to int before adding 1
        weights = [total_counts / class_counts.get(i, 1) for i in range(num_classes)]  # default to 1 if no samples for a class
        min_weight = min(weights)
        normalized_weights = [w / min_weight for w in weights]

        return torch.tensor(normalized_weights, dtype=torch.float32)
    
    
class SegmentationDatasetSeasonal(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 8 bands for the input image and the last band as the mask.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform
        # self.weights = self.calculate_weights()
        self.class_weights = self.compute_class_weights_for_crossentropy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 8 bands as the input image
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,
                              17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan =0.0001)
            # Bands in rasterio are indexed starting from 1, so the 9th band is the mask
            mask = src.read(33)
            
            # Move the channel axis to the last dimension for the image
            # image = np.moveaxis(image, 0, -1)
            # Squeeze the mask to remove the channel axis, if it's single-channel
            # mask = np.squeeze(mask)

        # Create a sample dictionary
        sample = {'image': image, 'mask': mask}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# return the weights for each image
    def calculate_weights(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(33)
                labels, counts = np.unique(mask, return_counts=True)
                for label, count in zip(labels, counts):
                    if label in class_counts:
                        class_counts[label] += count
                    else:
                        class_counts[label] = count

        total_counts = sum(class_counts.values())
        class_weights = {label: total_counts / count for label, count in class_counts.items()}
        image_weights = []

        # Calculate weights for each image based on its mask
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(33)
                labels, counts = np.unique(mask, return_counts=True)
                image_weight = sum(class_weights[label] for label in labels)
                image_weights.append(image_weight)

        return image_weights
    
    def compute_class_weights_for_crossentropy(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(33)
                labels, counts = np.unique(mask.astype(int), return_counts=True)  # Ensure labels are integers
                for label, count in zip(labels, counts):
                    class_counts[label] = class_counts.get(label, 0) + count

        total_counts = sum(class_counts.values())
        num_classes = int(max(class_counts.keys()) + 1)  # Cast max label to int before adding 1
        weights = [total_counts / class_counts.get(i, 1) for i in range(num_classes)]  # default to 1 if no samples for a class
        min_weight = min(weights)
        normalized_weights = [w / min_weight for w in weights]

        return torch.tensor(normalized_weights, dtype=torch.float32)
    
    
    
    

    

class SegmentationDatasetRGB(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 8 bands for the input image and the last band as the mask.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform
        self.weights = self.calculate_weights()
        self.class_weights = self.compute_class_weights_for_crossentropy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 8 bands as the input image
            image = src.read([1, 2, 3])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan =0.000000)
            # Bands in rasterio are indexed starting from 1, so the 9th band is the mask
            mask = src.read(9)
            
            # Move the channel axis to the last dimension for the image
            # image = np.moveaxis(image, 0, -1)
            # Squeeze the mask to remove the channel axis, if it's single-channel
            # mask = np.squeeze(mask)

        # Create a sample dictionary
        sample = {'image': image, 'mask': mask}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# return the weights for each image
    def calculate_weights(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask, return_counts=True)
                for label, count in zip(labels, counts):
                    if label in class_counts:
                        class_counts[label] += count
                    else:
                        class_counts[label] = count

        total_counts = sum(class_counts.values())
        class_weights = {label: total_counts / count for label, count in class_counts.items()}
        image_weights = []

        # Calculate weights for each image based on its mask
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask, return_counts=True)
                image_weight = sum(class_weights[label] for label in labels)
                image_weights.append(image_weight)

        return image_weights
    
    def compute_class_weights_for_crossentropy(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(9)
                labels, counts = np.unique(mask.astype(int), return_counts=True)  # Ensure labels are integers
                for label, count in zip(labels, counts):
                    class_counts[label] = class_counts.get(label, 0) + count

        total_counts = sum(class_counts.values())
        num_classes = int(max(class_counts.keys()) + 1)  # Cast max label to int before adding 1
        weights = [total_counts / class_counts.get(i, 1) for i in range(num_classes)]  # default to 1 if no samples for a class
        min_weight = min(weights)
        normalized_weights = [w / min_weight for w in weights]

        return torch.tensor(normalized_weights, dtype=torch.float32)
    


class SegmentationDatasetFusion(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images with multiple bands, apply augmentations,
    and process transformations. Use the first 8 bands for the input image and the last band as the mask.

    Args:
        data_path (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob('**/*.tif'))
        self.transform = transform
        # self.weights = self.calculate_weights()
        self.class_weights = self.compute_class_weights_for_crossentropy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the rasterio dataset
        with rio.open(image_path) as src:
            # Read the first 8 bands as the input image
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,
                              17,18,19,20,21])  # Adjust the band numbers if needed
            image = np.nan_to_num(image, nan =0.0001)
            # Bands in rasterio are indexed starting from 1, so the 9th band is the mask
            mask = src.read(22)
            
            # Move the channel axis to the last dimension for the image
            # image = np.moveaxis(image, 0, -1)
            # Squeeze the mask to remove the channel axis, if it's single-channel
            # mask = np.squeeze(mask)

        # Create a sample dictionary
        # sample = {'image': image, 'mask': mask}
        sample = {'image': torch.tensor(image, dtype=torch.float32),
                  'mask': torch.tensor(mask, dtype=torch.long)}

        # Apply transformations, if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# return the weights for each image
    def calculate_weights(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(22)
                labels, counts = np.unique(mask, return_counts=True)
                for label, count in zip(labels, counts):
                    if label in class_counts:
                        class_counts[label] += count
                    else:
                        class_counts[label] = count

        total_counts = sum(class_counts.values())
        class_weights = {label: total_counts / count for label, count in class_counts.items()}
        image_weights = []

        # Calculate weights for each image based on its mask
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(22)
                labels, counts = np.unique(mask, return_counts=True)
                image_weight = sum(class_weights[label] for label in labels)
                image_weights.append(image_weight)

        return image_weights
    
    def compute_class_weights_for_crossentropy(self):
        class_counts = {}
        for image_path in self.image_paths:
            with rio.open(image_path) as src:
                mask = src.read(22)
                labels, counts = np.unique(mask.astype(int), return_counts=True)  # Ensure labels are integers
                for label, count in zip(labels, counts):
                    class_counts[label] = class_counts.get(label, 0) + count

        total_counts = sum(class_counts.values())
        num_classes = int(max(class_counts.keys()) + 1)  # Cast max label to int before adding 1
        weights = [total_counts / class_counts.get(i, 1) for i in range(num_classes)]  # default to 1 if no samples for a class
        min_weight = min(weights)
        normalized_weights = [w / min_weight for w in weights]

        return torch.tensor(normalized_weights, dtype=torch.float32)
    
    
    

class RandomFlipRotate:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        if torch.rand(1) > 0.5:
            image = torch.flip(image, [2])  # Flip horizontally
            mask = torch.flip(mask, [1])

        if torch.rand(1) > 0.5:
            image = torch.flip(image, [1])  # Flip vertically
            mask = torch.flip(mask, [0])

        if torch.rand(1) > 0.5:
            image = torch.rot90(image, 1, [1, 2])  # Rotate 90 degrees
            mask = torch.rot90(mask, 1, [0, 1])

        return {'image': image, 'mask': mask}