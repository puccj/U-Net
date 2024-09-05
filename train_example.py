import torch
import torchvision
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from unet import UNet
from dataset import SegmentationDataset

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 4
NUM_CLASSES = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not directory:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_binary_accuracy(loader, model, device='cuda'):
    """Check the accuracy of the model with binary segmentation task using Dice score and accuracy metrics

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        Data loader to iterate over the dataset
    model: torch.nn.Module
        Model to evaluate
    device: str
        Device to run the model on (default is 'cuda')

    Returns
    -------
    float
        Accuracy of the model on the dataset (number of correctly classified pixels divided by the total number of pixels)
    float
        Dice score of the model on the dataset
    """
    num_correct = 0 
    num_pixels = 0
    dice_score = 0
    model.eval()    # set the model to evaluation mode
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x = x.float()
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()   # binary thresholding (change for multi-class segmentation)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)  # add epsilon to avoid division by zero
                                                                                # change for multi-class segmentation
    
    model.train()   # re-set the model back to training mode

    dice_score /= len(loader)
    accuracy = num_correct/num_pixels

    return accuracy, dice_score


def save_predictions_as_imgs(loader, model, dir="saved_images/", device="cuda"):
    """Save the model's predictions as images along with the ground truth masks

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        Data loader to iterate over the dataset
    model: torch.nn.Module
        Model to evaluate
    dir: str
        Directory path where to save the images
    device: str
        Device to run the model on (default is 'cuda')

    Returns
    -------
    None
    """
    
    ensure_directory_exists(dir)

    model.eval()    # set the model to evaluation mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()   # binary thresholding (change for multi-class segmentation)
        
        x = x.float()
        x = (x - x.min()) / (x.max() - x.min())
        torchvision.utils.save_image(x, f"{dir}/img_{idx}.png")
        torchvision.utils.save_image(preds, f"{dir}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{dir}/mask_{idx}.png")
    
    model.train()   # re-set the model back to training mode


def train(loader, model, optimizer, loss_fn, scaler, device="default"):
    """Train the model

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        Data loader for the training dataset, containing images and masks
    model: torch.nn.Module
        Model to be trained on the dataset
    optimizer: torch.optim.Optimizer
        Optimizer to update the model's weights based on the loss
    loss_fn: torch.nn.Module
        Loss function to compute the difference between the model's predictions and targets
    scaler: torch.cuda.amp.GradScaler
        Gradient scaler to prevent underflow and overflow during training
    device: str
        Device to run the model on ('cuda' or 'cpu'). "default" will use 'cuda' if available, otherwise 'cpu'
    """

    if device == "default":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.amp.autocast(device):
            data = data.float()
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # Initialize the model, loss function, optimizer, and data loaders
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    if NUM_CLASSES == 1:
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = nn.MSELoss()
    elif NUM_CLASSES > 1:
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),    # 50% chance of flipping the image
        A.Rotate(5),                # rotate +/- 5 degrees
        A.RandomResizedCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, scale=(0.1, 1.0)),  # crop the image
        # A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH), # crop the image
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        # A.CenterCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.RandomResizedCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, scale=(0.1, 1.0)),  # crop the image
        # A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    
    scaler = torch.amp.GradScaler(DEVICE)

    starting_epoch = 0
    # Load model from checkpoint
    if LOAD_MODEL:
        checkpoint_dir = "checkpoints/"
        checkpoint_files = os.listdir(checkpoint_dir)
        last_epoch = max([int(file.split("epoch")[1].split(".")[0]) for file in checkpoint_files])
        starting_epoch = last_epoch + 1
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch{last_epoch}.pth")

        #TODO: Replace with get_last_checkpoint

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

        # Save model
        ensure_directory_exists("checkpoints/")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, f"checkpoints/unet_epoch{starting_epoch + epoch}.pth")

        # Check accuracy
        check_binary_accuracy(val_loader, model, device=DEVICE)

        # Print some examples to a folder
        save_predictions_as_imgs(val_loader, model, dir="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()