from dataset import SegmentationDataset
from unittest.mock import patch

# Test the SegmentationDataset class constructor (__init__ method)

# Test Initialization with Valid Directories
@patch('os.listdir')
def test_images_loaded(mock_listdir):
    # Check if images are loaded correctly

    mock_listdir.side_effect = [    # Simulate files in the directory
        ['image1.png', 'image2.png'],
        ['mask1.png', 'mask2.png']]
    dataset = SegmentationDataset('path/to/images', 'path/to/masks')
    assert dataset.images == ['image1.png', 'image2.png']

@patch('os.listdir')
def test_masks_loaded(mock_listdir):
    # Check if masks paths are set correctly

    mock_listdir.side_effect = [
        ['image1.png', 'image2.png'],
        ['mask1.png', 'mask2.png']]
    dataset = SegmentationDataset('path/to/images', 'path/to/masks')
    assert dataset.mask_dir == 'path/to/masks'

@patch('os.listdir')
def test_empty_images(mock_listdir):
    # Check if images are empty when the directory is empty

    mock_listdir.side_effect = [
        [],
        []]
    dataset = SegmentationDataset('path/to/images', 'path/to/masks', transform=None)
    assert dataset.images == []

@patch('os.listdir')
def test_empty_masks(mock_listdir):
    # Check if masks are empty when the directory is empty

    mock_listdir.side_effect = [
        [],
        []]
    dataset = SegmentationDataset('path/to/images', 'path/to/masks', transform=None)
    assert dataset.masks == []