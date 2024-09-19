import numpy as np
from dataset import SegmentationDataset


# Test the SegmentationDataset class

def test_images_names_loaded():
    """Check if images name are loaded correctly

    GIVEN: a test directory containing two images
    WHEN: the SegmentationDataset class is initialized
    THEN: the images attribute should contain the two images names
    """

    dataset = SegmentationDataset('testing/test_data/test_red_dot/test_images', 'testing/test_data/test_red_dot/test_masks')
    expected = ['3x3.png', '5x5.png']

    assert sorted(dataset.images) == sorted(expected)

def test_empty_images():
    """Check if the dataset is empty when loading images from an empty directory
    
    GIVEN: a directory containing no images
    WHEN: the SegmentationDataset class is initialized
    THEN: the images attribute should be an empty list
    """

    dataset = SegmentationDataset('testing/test_data/empty_dir', 'testing/test_data/empty_dir', transform=None)
    assert dataset.images == []

def test_images_data_loaded():
    """Check if images data are loaded correctly
    
    GIVEN: a test directory containing two images with a red dot in the center
    WHEN: the SegmentationDataset class is initialized and __getitem__() is called
    THEN: the image data should be as expected
    """

    dataset = SegmentationDataset('testing/test_data/test_red_dot/test_images', 'testing/test_data/test_red_dot/test_masks')
    image = dataset.__getitem__(0)[0]
    mask = dataset.__getitem__(0)[1]

    expected_image = np.zeros((5, 5, 3))
    expected_image[2, 2, 0] = 255
    expected_mask = np.zeros((5, 5))
    expected_mask[2, 2] = 1

    assert np.array_equal(image, expected_image)
    assert np.array_equal(mask, expected_mask)
