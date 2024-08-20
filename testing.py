import pytest
from unittest.mock import patch

import torch
import torch.nn as nn

from unet import DoubleConvolution, UNet


def make_test_deterministic(seed=42):
    """Set the random seed and ensure all tests are deterministic"""
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


############## Test DoubleConvolution class ##############
# Test DoubleConvolution constructor (init method)

def test_double_convolution_in_channels():
    # Check if the in_channels parameter is correctly set
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    assert double_conv.double_conv[0].in_channels == 3

def test_double_convolution_out_channels():
    # Check if the out_channels parameter is correctly set
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    assert double_conv.double_conv[0].out_channels == 64

def test_double_convolution_length():
    # Check if the length of the DoubleConvolution layer is 6
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    assert len(double_conv.double_conv) == 6

def test_double_convolution_negative_in_channels():
    # Check if a RuntimeError is raised when in_channels is negative
    with pytest.raises(RuntimeError):
        _ = DoubleConvolution(in_channels=-3, out_channels=64)

def test_double_convolution_negative_out_channels():
    # Check if a RuntimeError is raised when out_channels is negative
    with pytest.raises(RuntimeError):
        _ = DoubleConvolution(in_channels=3, out_channels=-64)


# Test DoubleConvolution forward method

def test_double_convolution_forward_called():
    # Check that the forward method was called once
    # forward method should be triggered by the __call__ method of nn.Module
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    with patch.object(DoubleConvolution, 'forward', wraps=double_conv.forward) as mock_forward:
        x = torch.randn(1, 3, 128, 128)
        double_conv(x)
        mock_forward.assert_called_once()

def test_double_convolution_forward_called_with_correct_input():
    # Check that the forward method was called once with the correct input
    # forward method should be triggered by the __call__ method of nn.Module
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    with patch.object(DoubleConvolution, 'forward', wraps=double_conv.forward) as mock_forward:
        x = torch.randn(1, 3, 128, 128)
        double_conv(x)
        mock_forward.assert_called_once_with(x)

def test_double_convolution_same_channel():
    # Check if the output has the same number of channels as the out_channels parameter
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 128, 128)
    output = double_conv(x)
    assert output.shape[1] == 64

def test_double_convolution_same_size():
    # Check if the output size matches the input size
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 128, 128)
    output = double_conv(x)
    assert output.shape[2:] == x.shape[2:]

def test_double_convolution_correct_shape():
    # Check if the output shape is as expected
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 128, 128)
    output = double_conv(x)
    assert output.shape == (1, 64, 128, 128)

def test_double_convolution_correct_shape_for_odd_dimension():
    # Check if the output shape is as expected for odd dimension input tensor (161x161)
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 161, 161)  # odd dimension input tensor
    output = double_conv(x)
    assert output.shape == (1, 64, 161, 161)

def test_double_convolution_large_input():
    # Check if the forward method can handle very big input sizes (1024x1024)
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 1024, 1024)
    output = double_conv(x)
    assert output.shape == (1, 64, 1024, 1024)

def test_double_convolution_zero_sizes():
    # Check if a RuntimeError is raised when the input tensor size is zero
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 0, 0)
    with pytest.raises(RuntimeError):
        double_conv(x)

def test_double_convolution_non_tensor_input():
    # Check if a TypeError is raised when the input is not a tensor
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = 1
    with pytest.raises(TypeError):
        double_conv(x)

def test_double_convolution_non_4d_input():
    # Check if a RuntimeError is raised when the input is not a 4D-tensor
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 128)
    with pytest.raises(RuntimeError):
        double_conv(x)

def test_double_convolution_wrong_channel_input():
    # Check if a RuntimeError is raised when the number of channel of input (1) is different from the model (3)
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 1, 128, 128)
    with pytest.raises(RuntimeError):
        double_conv(x)

def test_double_convolution_wrong_channel_input2():
    # Check if a RuntimeError is raised when the number of channel of input (3) is different from the model (1)
    double_conv = DoubleConvolution(in_channels=1, out_channels=64)
    x = torch.randn(1, 3, 128, 128)
    with pytest.raises(RuntimeError):
        double_conv(x)

def test_double_convolution_small_input():
    # Check if a ValueError is raised when input is tensor sizes is 1
    double_conv = DoubleConvolution(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 1, 1)
    with pytest.raises(ValueError):
        double_conv(x)



############## Test UNet class ##############
# Test UNet constructor (init method)

def test_downs_length():
    # Check if the length of self.downs matches the length of the features list
    model = UNet(in_channel=3, out_channels=1, features=[64, 128, 256, 512])
    assert len(model.downs) == 4

def test_ups_length():
    # Check if the length of self.ups is twice the length of the features list
    # because each up layer has a ConvTranspose2d followed by a DoubleConvolution
    model = UNet(in_channel=3, out_channels=1, features=[64, 128, 256, 512])
    assert len(model.ups) == 8

def test_pooling_layer():
    # Check if pooling layer is correctly initialized
    model = UNet(in_channel=3, out_channels=1, features=[64, 128, 256, 512])
    assert isinstance(model.pool, nn.MaxPool2d)
    assert model.pool.kernel_size == 2
    assert model.pool.stride == 2

def test_bottleneck_layer():
    # Check if the bottleneck layer is correctly initialized
    model = UNet(in_channel=3, out_channels=1, features=[64, 128, 256, 512])
    assert isinstance(model.bottleneck, DoubleConvolution)
    assert model.bottleneck.double_conv[0].in_channels == 512
    assert model.bottleneck.double_conv[0].out_channels == 1024

def test_final_conv_layer():
    # Check if the final convolution layer is correctly initialized
    model = UNet(in_channel=3, out_channels=1, features=[64, 128, 256, 512])
    assert isinstance(model.final_conv, nn.Conv2d)
    assert model.final_conv.in_channels == 64
    assert model.final_conv.out_channels == 1

def test_negative_in_channels():
    # Check if a RuntimeError is raised when the input channels are negative
    with pytest.raises(RuntimeError):
        _ = UNet(in_channel=-3, out_channels=1, features=[64, 128, 256, 512])

def test_negative_out_channels():
    # Check if a RuntimeError is raised when the output channels are negative
    with pytest.raises(RuntimeError):
        _ = UNet(in_channel=3, out_channels=-1, features=[64, 128, 256, 512])

def test_min_features():
    # Check if a ValueError is raised when the number of features is less than 2
    with pytest.raises(ValueError):
        _ = UNet(in_channel=3, out_channels=1, features=[64])

def test_negative_features():
    # Check if a ValueError is raised when the features are negative
    with pytest.raises(ValueError):
        _ = UNet(in_channel=3, out_channels=1, features=[-64, 128, 256, 512])


# Test UNet forward method

def test_forward_called():
    # Check that the forward method was called once
    # forward method should be triggered by the __call__ method of nn.Module
    make_test_deterministic()
    model = UNet()
    with patch.object(UNet, 'forward', wraps=model.forward) as mock_forward:
        x = torch.randn(1, 3, 128, 128)  # dummy input tensor
        model(x)
        mock_forward.assert_called_once()

def test_forward_called_with_correct_input():
    # Check that the forward method was called once with the correct input
    # forward method should be triggered by the __call__ method of nn.Module
    make_test_deterministic()
    model = UNet()
    with patch.object(UNet, 'forward', wraps=model.forward) as mock_forward:
        x = torch.randn(1, 3, 128, 128)
        model(x)
        mock_forward.assert_called_once_with(x)

def test_forward_same_channel():
    # Check if the output has the correct number of channels
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    assert output.shape[1] == 1

def test_forward_same_size():
    # Check if the output size matches the input size
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    assert output.shape[2:] == x.shape[2:]

def test_forward_correct_shape():
    # Check if the output shape is as expected
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    assert output.shape == (1, 1, 128, 128)

def test_forward_correct_shape_for_odd_dimension():
    # Check if the output shape is as expected for odd dimension input tensor (161x161)
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 161, 161) # odd dimension input tensor
    output = model(x)
    assert output.shape == (1, 1, 161, 161)

def test_forward_large_input():
    # Check if the forward method can handle very big input sizes (1024x1024)
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 1024, 1024)
    output = model(x)
    assert output.shape == (1, 1, 1024, 1024)


def test_forward_zero_sizes():
    # Check if a RuntimeError is raised when the input tensor size is zero
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 0, 0)
    with pytest.raises(RuntimeError):
        model(x)

def test_forward_non_tensor_input():
    # Check if a TypeError is raised when the input is not a tensor
    make_test_deterministic()
    model = UNet()
    x = 1
    with pytest.raises(TypeError):
        model(x)

def test_forward_non_4d_input():
    # Check if a RuntimeError is raised when the input is not a 4D-tensor
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 128)
    with pytest.raises(RuntimeError):
        model(x)

def test_forward_wrong_channel_input():
    # Check if a RuntimeError is raised when the number of channel of input (1) is different from the model (3)
    model = UNet()  # default input channel is 3
    x = torch.randn(1, 1, 128, 128)
    with pytest.raises(RuntimeError):
        model(x)

def test_forward_wrong_channel_input2():
    # Check if a RuntimeError is raised when the number of channel of input (3) is different from the model (1)
    model = UNet(in_channel=1, out_channels=1, features=[64, 128, 256, 512])
    x = torch.randn(1, 3, 128, 128)
    with pytest.raises(RuntimeError):
        model(x)

def test_forward_small_input():
    # Check if a ValueError is raised when input is tensor sizes is 1
    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 1, 1)
    with pytest.raises(ValueError):
        model(x)


if __name__ == '__main__':
    pass