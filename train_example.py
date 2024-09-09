from trainer import Trainer
from unet import UNet
from dataset import SegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 100

def main():
    model = UNet(in_channels=3, out_channels=1)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),    # 50% chance of flipping the image
        A.Rotate(5),                # rotate +/- 5 degrees
        A.RandomResizedCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, scale=(0.1, 1.0)),  # crop the image
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.CenterCrop(IMAGE_HEIGHT*8, IMAGE_WIDTH*8),
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        # A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(num_epochs=NUM_EPOCHS, save_interval=1, save_val_img=True, save_train_img=True)


if __name__ == "__main__":
    main()