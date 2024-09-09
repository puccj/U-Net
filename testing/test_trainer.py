import pytest
from unittest.mock import MagicMock, patch, call
from hypothesis import given, example
import hypothesis.strategies as st

from trainer import ensure_directory_exists, Trainer

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_test_deterministic(seed=42):
    "Set the random seed and ensure all tests are deterministic"
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Test ensure_directiory_exists function

@patch("os.makedirs")
@patch("os.path.exists", return_value=False)
def test_ensure_directory_exists_creates_directory(mock_exists, mock_makedirs):
    # Check that os.makedirs is called (with the correct path) when the directory does not exist
    file_path = "test_dir/file.txt"
    dir_path = os.path.dirname(file_path)
    ensure_directory_exists(file_path)

    mock_exists.assert_called_once_with(dir_path)
    mock_makedirs.assert_called_once_with(dir_path)

@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
def test_ensure_directory_exists_directory_already_exists(mock_exists, mock_makedirs):
    # Check that os.makedirs is NOT called when the directory already exists
    file_path = "test_dir/file.txt"
    dir_path = os.path.dirname(file_path)
    ensure_directory_exists(file_path)

    mock_exists.assert_called_once_with(dir_path)
    mock_makedirs.assert_not_called()

@patch("os.makedirs")
@patch("os.path.exists")
def test_ensure_directory_exists_empty_path(mock_exists, mock_makedirs):
    # Check that neither os.path.exists nor os.makedirs is called when the path is empty
    file_path = ""
    ensure_directory_exists(file_path)

    mock_exists.assert_not_called()
    mock_makedirs.assert_not_called()


@patch("os.makedirs")
@patch("os.path.exists", return_value=False)
def test_ensure_directory_exists_with_file_not_specified(mock_exists, mock_makedirs):
    # Check that os.makedirs is called with the correct path when the file is not specified
    file_path = "test_dir/"     # Note the trailing slash
    dir_path = os.path.dirname(file_path)
    ensure_directory_exists(file_path)

    mock_exists.assert_called_once_with(dir_path)
    mock_makedirs.assert_called_once_with(dir_path)



# Test the Trainer class constructor (__init__ method)

def test_trainer_valid_loss_binary():
    # Check that default parameter for loss leads to correct loss function for binary segmentation
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.is_binary = True
    mock_model.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2, 2))])  # Mock model parameters
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, loss_fn='default')

    assert isinstance(trainer.loss_fn, nn.BCEWithLogitsLoss)

def test_trainer_valid_loss_multi_class():
    # Check that default parameter for loss leads to correct loss function for multi-class segmentation
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.is_binary = False
    mock_model.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2, 2))])  # Mock model parameters
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, loss_fn='default')

    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)

def test_trainer_valid_device():
    # Check that the default device is either 'cuda' or 'cpu' depending on availability
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.is_binary = True
    mock_model.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2, 2))])  # Mock model parameters
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, device='default')
    assert trainer.device == 'cuda' if torch.cuda.is_available() else 'cpu'


# Test Trainer train_step method

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)  # Mock tqdm to disable progress bar during testing
def test_train_step_train_mode(mock_tqdm):
    # Check that the train_step method set the model to training mode
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))  # Example batch for binary segmentation
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))  # Return a dummy loss value
    mock_optimizer = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = MagicMock()
    
    trainer.train_step()

    mock_model.train.assert_called_once()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_train_step_loss_called(mock_tqdm):
    # Check that the loss function is called during training
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))  # Example batch for binary segmentation
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))  # Return a dummy loss value
    mock_optimizer = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = MagicMock()
    
    trainer.train_step()

    mock_loss_fn.assert_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_train_step_optimizer_zero_grad(mock_tqdm):
    # Check that the optimizer.zero_grad method is called during training
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = MagicMock()
    
    trainer.train_step()

    mock_optimizer.zero_grad.assert_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_train_step_scaler_called(mock_tqdm):
    # Check that the scaler methods (scale, step, update) are called during training
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()
    
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.train_step()

    mock_scaler.scale.assert_called()
    mock_scaler.step.assert_called()
    mock_scaler.update.assert_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
@patch("torchvision.utils.save_image")  # Mock save_image to avoid actual file saving
def test_train_step_save_image_not_called(mock_tqdm, mock_save_image):
    # Check that save_image is not called when save_img_dir is not provided
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()
    
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.train_step()

    mock_save_image.assert_not_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
@patch("torchvision.utils.save_image")  # Mock save_image to avoid actual file saving
@patch("trainer.ensure_directory_exists")       # Mock ensure_directory_exists to avoid creating directories
def test_train_step_save_image_called(mock_tqdm, mock_save_image, mock_ensure_directory_exists):
    # Check that save_image is called when save_img_dir is provided
    make_test_deterministic()
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_train_loader.__len__.return_value = 1
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()
    
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.train_step('save_img_dir')

    mock_save_image.assert_called()


# Test Trainer val_step method

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_val_step_eval_mode(mock_tqdm):
    # Check that the val_step method set the model to eval mode
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))  # Return a dummy loss value
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step()

    mock_model.eval.assert_called_once()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_val_step_loss_called(mock_tqdm):
    # Check that the loss function is called during validation
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step()

    mock_loss_fn.assert_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_val_step_optimizer_zero_grad(mock_tqdm):
    # Check that the optimizer.zero_grad method is NOT called during validation
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step()

    mock_optimizer.zero_grad.assert_not_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
def test_val_step_scaler_called(mock_tqdm):
    # Check that the scaler methods (scale, step, update) are NOT called during validation
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step()

    mock_scaler.scale.assert_not_called()
    mock_scaler.step.assert_not_called()
    mock_scaler.update.assert_not_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
@patch("torchvision.utils.save_image")  # Mock save_image to avoid actual file saving
def test_val_step_save_image_not_called(mock_tqdm, mock_save_image):
    # Check that save_image is not called when save_img_dir is not provided
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step()

    mock_save_image.assert_not_called()

@patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)
@patch("torchvision.utils.save_image")  # Mock save_image to avoid actual file saving
@patch("trainer.ensure_directory_exists")       # Mock ensure_directory_exists to avoid creating directories
def test_val_step_save_image_called(mock_tqdm, mock_save_image, mock_ensure_directory_exists):
    # Check that save_image is called when save_img_dir is provided
    make_test_deterministic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.return_value = torch.randn(4, 1, 32, 32).to(device)

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4, 32, 32)))
    ])
    mock_val_loader.__len__.return_value = 1

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()
    mock_scaler = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)
    trainer.scaler = mock_scaler

    trainer.val_step('save_img_dir')

    mock_save_image.assert_called()


# Test Trainer save_checkpoint method

@patch("torch.save")
@patch("trainer.ensure_directory_exists")
def test_save_checkpoint_called(mock_save, mock_ensure_directory_exists):
    # Check that torch.save is called when save_model_dir is provided
    mock_model = MagicMock()

    mock_train_loader = MagicMock(spec=DataLoader)
    mock_val_loader = MagicMock(spec=DataLoader)

    mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()

    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, mock_loss_fn, mock_optimizer)

    trainer.save_checkpoint('save_model_dir')

    mock_save.assert_called()


# Test Trainer load_checkpoint method

@patch("os.path.isfile", return_value=True)
@patch("os.path.exists", return_value=True)
@patch("torch.load")
def test_load_checkpoint_from_file(mock_torch_load, mock_exists, mock_isfile):
    # Test that torch.load is called when load_checkpoint is called with a file path
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_torch_load.return_value = {'model': MagicMock(), 'optimizer': MagicMock()}  # Simulate loaded checkpoint data
    trainer = Trainer(mock_model, MagicMock(), MagicMock(), optimizer=mock_optimizer)

    trainer.load_checkpoint('some_path/checkpoint.pth')

    mock_torch_load.assert_called_once_with('some_path/checkpoint.pth', weights_only=True)
    mock_model.load_state_dict.assert_called_once()
    mock_optimizer.load_state_dict.assert_called_once()

@patch("os.path.isfile", return_value=False)
@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["checkpoint1.pth", "checkpoint2.pth"])
@patch("os.path.getmtime", side_effect=[1, 2])  # Mock timestamps to simulate file modification times
@patch("torch.load")
def test_load_checkpoint_from_directory(mock_torch_load, mock_getmtime, mock_listdir, mock_isdir, mock_isfile):
    # Test that torch.load is called with the path to the most recent checkpoint 
    # file when load_checkpoint is called with a directory path
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_torch_load.return_value = {'model': MagicMock(), 'optimizer': MagicMock()}
    trainer = Trainer(mock_model, MagicMock(), MagicMock(), optimizer=mock_optimizer)

    trainer.load_checkpoint('some_path/')

    mock_torch_load.assert_called_once_with('some_path/checkpoint2.pth', weights_only=True)
    mock_model.load_state_dict.assert_called_once()
    mock_optimizer.load_state_dict.assert_called_once()

@patch("os.path.isfile", return_value=True)
@patch("os.path.exists", return_value=False)
def test_load_checkpoint_file_not_exists(mock_exists, mock_isfile):
    # Check that a FileNotFoundError is raised when the file does not exist
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    trainer = Trainer(mock_model, MagicMock(), MagicMock(), optimizer=mock_optimizer)

    with pytest.raises(FileNotFoundError, match="File some_path/checkpoint.pth does not exist"):
        trainer.load_checkpoint('some_path/checkpoint.pth')

@patch("os.path.isfile", return_value=False)
@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=[])
def test_load_checkpoint_empty_directory(mock_listdir, mock_isdir, mock_isfile):
    # Check that a FileNotFoundError is raised when the directory is empty
    mock_model = MagicMock()
    mock_optimizer = MagicMock()

    trainer = Trainer(mock_model, MagicMock(), MagicMock(), optimizer=mock_optimizer)

    with pytest.raises(FileNotFoundError, match="Directory some_path/ is empty"):
        trainer.load_checkpoint('some_path/')


# Test Trainer train method

@patch.object(Trainer, 'train_step', return_value=0.5)              # Mock train_step to return a fixed loss value
@patch.object(Trainer, 'val_step', return_value=(0.4, 0.85, 0.75))  # Mock val_step to return fixed values
@patch.object(Trainer, 'save_checkpoint')                           # Mock save_checkpoint to avoid actual file operations
def test_train_calling_n_times(mock_save_checkpoint, mock_val_step, mock_train_step):
    # Check train_step and val_step were called the right number of times
    mock_model = MagicMock()
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_optimizer = MagicMock()
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, optimizer=mock_optimizer)
    
    trainer.train(num_epochs=11, save_interval=5, early_stop_patience=None, save_val_img=False, save_train_img=False)

    assert mock_train_step.call_count == 11
    assert mock_val_step.call_count == 11

@patch.object(Trainer, 'train_step', return_value=0.5)
@patch.object(Trainer, 'val_step', return_value=(0.4, 0.85, 0.75))
@patch.object(Trainer, 'save_checkpoint')
def test_train_checkpoints_saved(mock_save_checkpoint, mock_val_step, mock_train_step):
    # Check that checkpoints are saved correctly at intervals and for best model
    mock_model = MagicMock()
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_optimizer = MagicMock()
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, optimizer=mock_optimizer)

    trainer.train(num_epochs=11, save_interval=5, early_stop_patience=None, save_val_img=True, save_train_img=True)

    mock_save_checkpoint.assert_has_calls([
        call('checkpoints/0.pth'),
        call('checkpoints/5.pth'),
        call('checkpoints/10.pth'),
        call('checkpoints/best.pth')
    ], any_order=True)

@patch.object(Trainer, 'train_step', return_value=0.5)
@patch.object(Trainer, 'val_step', return_value=(0.4, 0.85, 0.75))  # Simulate no improvements
@patch.object(Trainer, 'save_checkpoint')
def test_train_early_stopping(mock_save_checkpoint, mock_val_step, mock_train_step):
    # Check that early stopping is triggered after 'patience' epochs of no improvement
    mock_model = MagicMock()
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_optimizer = MagicMock()
    trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, optimizer=mock_optimizer)
    
    trainer.train(num_epochs=10, save_interval=5, early_stop_patience=3, save_val_img=False, save_train_img=False)

    assert mock_train_step.call_count == 3
    assert mock_val_step.call_count == 3