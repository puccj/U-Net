import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
    """Dataset for segmentation tasks. 
    The images and masks should be placed in two different directories and have the same file names.
    
    Attributes
    ----------
    image_dir : str
        Path to the directory containing the images
    mask_dir : str
        Path to the directory containing the masks
    transform : Callable
        Tranformation function to be applied to the images and masks in order to perform data augmentation.
    images : List[str]
        List of image file names

    Methods
    -------
    __len__(self)
        Return the length of the dataset
    __getitem__(self, idx)
        Return the image and mask at the given index
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        """Initialize the dataset

        Parameters
        ----------
        images_dir : str
            Path to the directory containing the images
        masks_dir : str
            Path to the directory containing the masks
        transform : Callable
            Tranformation function to be applied to the images and masks in order to perform data augmentation.
        """
        
        self.image_dir = images_dir
        self.mask_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return the image and mask at the given index

        Parameters
        ----------
        idx : int
            Index of the image and mask to return

        Returns
        -------
        image : np.ndarray
            Image as a numpy array
        mask : np.ndarray
            Mask as a numpy array
        """

        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)   # so [0-255]
        mask = (mask > 128).astype(np.float32)  # threshold the mask to get binary values (0 or 1)
        

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask