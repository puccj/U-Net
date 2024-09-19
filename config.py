# Directories

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


# Training parameters

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 100
SAVE_INTERVAL = 5
SAVE_VAL_IMG = True
SAVE_TRAIN_IMG = True

# Model parameters

IN_CHANNELS = 3
OUT_CHANNELS = 1
FEATURES = [64, 128, 256, 512]

# Data augmentation

CROP_MIN_AREA = 0.1
CROP_MAX_AREA = 1.0

# Save data

TRAIN_LOSS_PATH = "train_loss.npy"
VAL_LOSS_PATH = "val_loss.npy"
VAL_ACC_PATH = "val_accuracy.npy"
VAL_DICE_PATH = "val_dice.npy"

# Save plots

LOSS_PLOT_PATH = "loss_plot.png"
ACC_PLOT_PATH = "acc_plot.png"
DICE_PLOT_PATH = "dice_plot.png"