import os
import torch
import torchvision
from tqdm import tqdm


def ensure_directory_exists(file_path):
    """Ensures that the directory of the file path exists, and creates it if it doesn't.
    
    Parameters
    ----------
    file_path : str
        Path to the file whose directory needs to be created if it doesn't exist.
    """

    directory = os.path.dirname(file_path)
    if not directory:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)

class Trainer:
    """Trainer class for training and validating a PyTorch model on a dataset.

    This class provides methods to train the model on a training dataset, validate it on a 
    validation dataset, and save/load model checkpoints for resuming training or inference. 
    During training, the model is saved at regular intervals and the 
    best model based on the validation loss is saved separately. 
    During validation, the model's predictions can be saved as images for visualization.

    Attributes
    ----------
    model : torch.nn.Module
        Model to be trained and validated.
    optimizer : torch.optim.Optimizer
        Optimizer to be used for training.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    model_is_binary : bool
        True if the model is a binary segmentation model, False if it is a multi-class segmentation model.
    train_loss : list
        List of training losses for each epoch.
    val_loss : list
        List of validation losses for each epoch.
    val_accuracy : list
        List of accuracies on the validation set for each epoch.
    val_dice : list
        List of Dice scores on the validation set for each epoch.
    loss_fn : torch.nn.Module
        Loss function to be used for training.
    device : str
        Device to run the model on ('cuda' or 'cpu').
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler to prevent underflow and overflow during training.

    Methods
    -------
    train_step()
        Performs a single training step (forward and backward pass for one epoch) on the training dataset.
    val_step(save_img_dir=None)
        Performs validation on the validation dataset and optionally saves images of predictions.
    save_checkpoint(file_path='checkpoints/last.pth')
        Save the model and optimizer state to a checkpoint file.
    load_checkpoint(path)
        Load the model and optimizer state from a checkpoint file or directory.
    train(num_epochs, save_interval=5, early_stop_patience=None, save_img=True)
        Train the model for a specified number of epochs, saving the best model based on the validation loss.
    """

    def __init__(self,
                 model, 
                 train_loader, 
                 val_loader, 
                 loss_fn='default', 
                 optimizer='default', 
                 learning_rate=1e-4,
                 device='default'
                ):
        """Initialize the Trainer class with the model, data loaders, loss function, optimizer, and device.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be trained and validated.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        loss_fn : torch.nn.Module or str, optional
            Loss function to be used for training. Default is 'default', which uses BCEWithLogitsLoss for binary segmentation and CrossEntropyLoss for multi-class segmentation.
            Note that if it cannot be determined whether the model is binary or multi-class, it is assumed to be multi-class.
        optimizer : torch.optim.Optimizer or str, optional
            Optimizer to be used for training. Default is 'default', which uses Adam with a learning rate of 1e-4.
        learning_rate : float, optional
            Learning rate to be used by the optimizer. Default is 1e-4.
        device : str, optional
            Device to run the model on ('cuda' or 'cpu'). Default is 'default', which uses 'cuda' if available, otherwise 'cpu'.
        """

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_is_binary = (   (hasattr(self.model, 'is_binary') and self.model.is_binary) 
                                or (hasattr(self.model, 'final_conv') and self.model.final_conv.out_channels == 1))
        self.train_loss = []
        self.val_loss = []
        self.val_accuracy = []
        self.val_dice = []
        
        self.loss_fn = loss_fn
        if loss_fn == 'default':
            if self.model_is_binary:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.optimizer = optimizer
        if optimizer == 'default':
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        self.device = device
        if device == 'default':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scaler = torch.amp.GradScaler(self.device)
        self.model.to(self.device)

    def train_step(self, save_img_dir = None):
        """Performs a single training step (forward and backward pass for one epoch) on the training dataset.
        Optionally saves images of the input, predictions, and masks during training.

        Parameters
        ----------
        save_img_dir : str, optional
            If provided, directory where the input images, predictions, and masks will be saved.
        Returns
        -------
        float
            Average loss over the training set.
        """

        self.model.train()
        loop = tqdm(self.train_loader, desc='Training')
        total_loss = 0
        for index, (x, y) in enumerate(loop):
            x = x.to(self.device).float()
            y = y.to(self.device).float()

            if self.model_is_binary:
                y = y.unsqueeze(1)  # Add channel dimension for binary segmentation
            
            # Forward pass
            with torch.amp.autocast(self.device):
                prediction = self.model(x)
                loss = self.loss_fn(prediction, y)
                total_loss += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update tqdm loop
            loop.set_postfix(loss=loss.item())

            # Save images
            if save_img_dir:
                path_with_slash = os.path.join(save_img_dir, '')   # Ensure the path ends with a slash
                ensure_directory_exists(path_with_slash)
                # Normalize the input image for visualization
                x = (x - x.min()) / (x.max() - x.min())
                torchvision.utils.save_image(x, f"{path_with_slash}img_{index}.png")
                torchvision.utils.save_image(prediction, f"{path_with_slash}pred_{index}.png")
                torchvision.utils.save_image(y, f"{path_with_slash}mask_{index}.png")

        return total_loss / len(self.train_loader)

    def val_step(self, save_img_dir = None):
        """Performs validation on the validation dataset.
        Optionally saves images of the input, predictions, and masks during validation.

        Parameters
        ----------
        save_img_dir : str, optional
            If provided, directory where the input images, predictions, and masks will be saved.

        Returns
        -------
        float
            Average loss over the validation set.
        float
            Accuracy of the model on the validation set (ratio of correct predictions to total pixels).
        float
            Average Dice score of the model on the validation set.
        """

        self.model.eval()
        total_loss = 0
        num_correct = 0
        num_pixels = 0
        total_dice = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc='Validation')
            for index, (x, y) in enumerate(loop):
                x = x.to(self.device).float()
                y = y.to(self.device)

                if self.model_is_binary:
                    y = y.float().unsqueeze(1)  # Add channel dimension for binary segmentation
                else:   # If the model is multi-class, 
                        # remove the channel dimension from the target tensor if it exists
                    if y.dim() == 4 and y.size(1) == 1:
                        y = y.squeeze(1)

                prediction = self.model(x)      # prediction = torch.sigmoid(self.model(x))
                loss = self.loss_fn(prediction, y)
                total_loss += loss.item()

                if self.model_is_binary:
                    prediction = torch.sigmoid(prediction)
                    prediction = (prediction > 0.5).float()
                else:
                    prediction = torch.softmax(prediction, dim=1)
                    prediction = torch.argmax(prediction, dim=1)

                num_correct += (prediction == y).sum().item()
                num_pixels += y.numel()
                
                total_dice += ((2 * (prediction*y).sum()) / ((prediction + y).sum() + 1e-8)).item() # add epsilon to avoid division by zero

                # Update tqdm loop
                loop.set_postfix(loss=loss.item())

                # Save images
                if save_img_dir:
                    path_with_slash = os.path.join(save_img_dir, '')   # Ensure the path ends with a slash
                    ensure_directory_exists(path_with_slash)
                    # Normalize the input image for visualization
                    x = (x - x.min()) / (x.max() - x.min())
                    torchvision.utils.save_image(x, f"{path_with_slash}img_{index}.png")
                    torchvision.utils.save_image(prediction, f"{path_with_slash}pred_{index}.png")
                    torchvision.utils.save_image(y, f"{path_with_slash}mask_{index}.png")

        # Compute metrics
        loss = total_loss / len(self.val_loader)
        accuracy = num_correct / num_pixels
        dice_score = total_dice / len(self.val_loader)

        return loss, accuracy, dice_score

    def save_checkpoint(self, file_path = 'checkpoints/last.pth'):
        """Save the model and optimizer state to a checkpoint file.

        Parameters
        ----------
        file_path : str, optional
            Path to the checkpoint file. Default is 'checkpoints/last.pth'.
        """

        ensure_directory_exists(file_path)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, path):
        """Load the model and optimizer state from a checkpoint file or directory.
        If a directory is provided, the most recent checkpoint file in the directory is loaded.

        Parameters
        ----------
        path : str
            Path to the checkpoint file or directory containing the checkpoint files.

        Raises
        ------
        FileNotFoundError
            If the file does not exist or the directory is empty.
        """
        
        if os.path.isfile(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist")
            checkpoint_path = path
        elif os.path.isdir(path):
            files = os.listdir(path)
            if not files:
                raise FileNotFoundError(f"Directory {path} is empty")
            files.sort(key=os.path.getmtime)
            checkpoint_path = os.path.join(path, files[-1]) # Most recent file
        
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, num_epochs, save_interval=5, early_stop_patience=None, save_val_img=True, save_train_img=False):
        """Train the model for a specified number of epochs, saving the last model and the best model based on the validation loss.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model for.
        save_interval : int, optional
            Interval at which to save the model checkpoint. Default is 5. 
            If set to 0, only the best and last models are saved. If set to 1, the model is saved after every epoch.
        early_stop_patience : int, optional
            If provided, training will stop if the validation loss does not improve after this number of epochs.
        save__val_img : bool, optional
            If True, save the input images, predictions, and masks during validation. Default is True.
        save_train_img : bool, optional
            If True, save the input images, predictions, and masks during training. Default is False.

        Returns
        -------
        list
            List of training losses for each epoch.
        list
            List of validation losses for each epoch.
        list
            List of accuracies on the validation set for each epoch.
        list
            List of Dice scores on the validation set for each epoch.
        """

        best_val_loss = 0
        patience_counter = 0
        save_img_val_dir = None
        save_img_train_dir = None

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch}/{num_epochs-1}]")

            if save_train_img and epoch % save_interval == 0:
                save_img_train_dir = f"saved_images/epoch_{epoch}/train"
            else:
                save_img_train_dir = None

            if save_val_img and epoch % save_interval == 0:
                save_img_val_dir = f"saved_images/epoch_{epoch}/val"
            else:
                save_img_val_dir = None
            
            # Perform a training step
            train_loss = self.train_step(save_img_train_dir)
            print(f"Training Loss: {train_loss:.4f}")
            self.train_loss.append(train_loss)

            # Perform a validation step
            val_loss, val_accuracy, val_dice = self.val_step(save_img_val_dir)
            print(f"Validation Loss: {val_loss:.4f}  -  Accuracy: {val_accuracy:.4f}  -  Dice Score: {val_dice:.4f}")
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_accuracy)
            self.val_dice.append(val_dice)

            # Save the model checkpoint    
            if save_interval > 0 and epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoints/{epoch}.pth')
            
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('checkpoints/best.pth')

            # Early Stopping check
            if early_stop_patience:
                if val_loss < best_val_loss:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
        # Save the last model
        self.save_checkpoint('checkpoints/last.pth')

        return self.train_loss, self.val_loss, self.val_accuracy, self.val_dice