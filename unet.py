import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConvolution(nn.Module):
    """Applies double convolution to the input
    
    Double convolution is a sequence of two convolutional layers with batch normalization and ReLU activation function.
    """
    def __init__(self, in_channels, out_channels):
        """Initializes the DoubleConvolution module
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(DoubleConvolution, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),   #original code has no batch normalization (since it came from a paper in 2016)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Perform the forward propagation

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to be processed by the double convolution

        Returns
        -------
        torch.Tensor
            Output tensor after applying the double convolution
        """
        return self.double_conv(x)
    
class UNet(nn.Module):
    """UNet model

    UNet is a convolutional neural network used for image segmentation. It consists of an encoder and a decoder. The encoder
    downsamples the input image and the decoder upsamples the encoder's output to the original size. The encoder and decoder
    are connected by so-called skip connections, which concatenate the output of the encoder to the input of the decoder at the
    same resolution. This helps the decoder to recover the spatial information lost during downsampling. The skip connections
    are concatenated channel-wise, which means that the number of channels is doubled after each concatenation. The UNet model
    has a contracting path (encoder) and an expansive path (decoder): the contracting path follows the typical architecture of
    a convolutional neural network, with a series of convolutional layers followed by a max-pooling layer. The expansive path
    consists of a series of up-convolutions, which increase the spatial resolution of the input, followed by a series of
    convolutional layers. The final layer of the UNet model is a 1x1 convolutional layer that maps each pixel to the desired
    number of classes.
    """
    def __init__(self, in_channel=3, out_channels=1, features=[64, 128, 256, 512]):
        """Initializes the UNet model

        Parameters
        ----------
        in_channel: int
            Number of input channels
        out_channels: int
            Number of output channels
        features: list
            List of features in the encoder. The length of the list determines the iteration in the encoder and decoder
        """

        if len(features) < 2:
            raise ValueError("The number of features should be at least 2")
        if any(f < 0 for f in features):
            raise ValueError("The number of features should be positive")

        super(UNet, self).__init__()
        self.num_features = len(features)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConvolution(in_channel, feature))
            in_channel = feature
        
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConvolution(feature*2, feature))
        
        self.bottleneck = DoubleConvolution(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Perform the forward propagation

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to be processed by the UNet model

        Returns
        -------
        torch.Tensor
            Output tensor after applying the UNet model
        """

        if x.shape[-1] < 2**(self.num_features+1) or x.shape[-2] < 2**(self.num_features+1):
            raise ValueError(f"The input tensor sizes must be at least 2^(len(features)+1) = {2**(self.num_features+1)}")

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # In order to make the implementation more general, if the input size is not divisible by 16, we need to resize the skip connection
            # For instance, if the input size is 161x161, the output size will be 160x160, so we need to resize the skip connection to 160x160
            if x.shape != skip_connection.shape:    
                # we can cut one or add padding to the image, but we will resize it for simplicity 
                # since it's just one pixel difference, it won't affect the performance
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)