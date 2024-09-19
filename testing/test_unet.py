import pytest
from hypothesis import given, example, settings
import hypothesis.strategies as st

from unet import DoubleConvolution, UNet

import torch
import torch.nn as nn

def make_test_deterministic(seed=42):
    "Set the random seed and ensure all tests are deterministic"
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Identity(nn.Module):
    """Identity module that returns the input tensor, used for testing purposes"""        
    def forward(self, x):
        return x

############## Test DoubleConvolution class ##############
# Test DoubleConvolution constructor (init method)

@given(input = st.integers(min_value=1, max_value=512),
       output = st.integers(min_value=1, max_value=512))
@example(input=3, output=64)
def test_double_convolution_in_channels(input, output):
    """Check if the in_channels parameter is correctly set when the DoubleConvolution class is initialized

    GIVEN: input and output are integers
    WHEN: the DoubleConvolution class is initialized
    THEN: the in_channels parameter is correctly set
    """

    double_conv = DoubleConvolution(in_channels=input, out_channels=output)
    assert double_conv.double_conv[0].in_channels == input

@given(input = st.integers(min_value=1, max_value=512),
       output = st.integers(min_value=1, max_value=512))
@example(input=3, output=64)
def test_double_convolution_out_channels(input, output):
    """Check if the out_channels parameter is correctly set when the DoubleConvolution class is initialized

    GIVEN: input and output are integers
    WHEN: the DoubleConvolution class is initialized
    THEN: the out_channels parameter is correctly set
    """

    double_conv = DoubleConvolution(in_channels=input, out_channels=output)
    assert double_conv.double_conv[0].out_channels == output

def test_double_convolution_negative_in_channels():
    """Check if a ValueError is raised when DoubleConvolution is initialized with negative in_channels

    GIVEN: in_channels is negative
    WHEN: the DoubleConvolution class is initialized with the negative in_channels
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = DoubleConvolution(in_channels=-3, out_channels=64)

def test_double_convolution_negative_out_channels():
    """Check if a ValueError is raised when DoubleConvolution is initialized with negative out_channels

    GIVEN: out_channels is negative
    WHEN: the DoubleConvolution class is initialized with the negative out_channels
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = DoubleConvolution(in_channels=3, out_channels=-64)

def test_double_convolution_zero_in_channels():
    """Check if a RuntimeError is raised when in_channels is zero

    GIVEN: in_channels is 0
    WHEN: the DoubleConvolution class is initialized with in_channels=0
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = DoubleConvolution(in_channels=0, out_channels=64)

def test_double_convolution_zero_out_channels():
    """Check if a RuntimeError is raised when out_channels is zero

    GIVEN: out_channels is 0
    WHEN: the DoubleConvolution class is initialized with out_channels=0
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = DoubleConvolution(in_channels=3, out_channels=0)


# Test DoubleConvolution forward method

@given(inp = st.integers(min_value=1, max_value=512),
       out = st.integers(min_value=1, max_value=512),
       batch = st.integers(min_value=1, max_value=16),
       width = st.integers(min_value=5, max_value=128),
       height = st.integers(min_value=5, max_value=128))
@settings(deadline=None)
@example(inp=3, out=64, batch = 1, width = 128, height = 128)
def test_double_convolution_same_channel(inp, out, batch, width, height):
    """Check if the output of the DoubleConvolution block has the same number of channels as the out_channels parameter
    
    GIVEN: inp, out, batch, width, height are integers
    WHEN: the DoubleConvolution block is applied to a tensor
    THEN: the output has the same number of channels as the out_channels parameter
    """

    make_test_deterministic()
    double_conv = DoubleConvolution(in_channels=inp, out_channels=out)
    x = torch.randn(batch, inp, width, height)
    output = double_conv(x)
    assert output.shape[1] == out

@given(batch = st.integers(min_value=1, max_value=16),
       channels = st.integers(min_value=1, max_value=5),
       width = st.integers(min_value=5, max_value=128),
       height = st.integers(min_value=5, max_value=128))
@settings(deadline=None)
@example(batch = 1, channels = 3, width = 128, height = 128)
@example(batch = 1, channels = 3, width = 161, height = 161)    # odd dimension input tensor
def test_double_convolution_same_size(batch, channels, width, height):
    """Check if the output of the DoubleConvolution block has the same size as the input tensor
    
    GIVEN: batch, channels, width, height are integers
    WHEN: the DoubleConvolution block is applied to a tensor
    THEN: the output size (width, height) matches the input size
    """

    make_test_deterministic()
    double_conv = DoubleConvolution(in_channels=channels, out_channels=64)
    x = torch.randn(batch, channels, width, height)
    output = double_conv(x)
    assert output.shape[2:] == x.shape[2:]

def test_double_convolution_3x3dot_zeros_weights():
    """Check the result of a DoubleConvolution block with weights=0 applied to a 3x3 image
    
    GIVEN: a 3x3 image tensor with a single dot in the middle and zero elsewhere.
           A DoubleConvolution block with weights initialized to zero
    WHEN: the DoubleConvolution block is applied to the 3x3 image tensor
    THEN: the output is a 3x3 tensor with zeros
    """

    double_conv = DoubleConvolution(in_channels=1, out_channels=1)
    double_conv.double_conv[0].weight.data.fill_(0)
    double_conv.double_conv[3].weight.data.fill_(0)

    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 1, 1] = 255
    output = double_conv(x)
    expected = torch.zeros(1, 1, 3, 3)
    
    assert torch.equal(output, expected)

def test_double_convolution_3x3dot_ones_weights():
    """Check the result of a DoubleConvolution block with weights=1 applied to a 3x3 image
    For testing purposes, the BatchNorm2d layers are removed (replaced by Identity layers)
    
    GIVEN: a 3x3 image tensor with a single dot in the middle and zero elsewhere.
           A DoubleConvolution block with weights initialized to ones and without batch normalization
    WHEN: the DoubleConvolution block is applied to the 3x3 image tensor
    THEN: the output is a 3x3 tensor with the same values as the input
    """

    double_conv = DoubleConvolution(in_channels=1, out_channels=1)
    double_conv.double_conv[0].weight.data.fill_(1)
    double_conv.double_conv[3].weight.data.fill_(1)
    double_conv.double_conv[1] = Identity()
    double_conv.double_conv[4] = Identity()

    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 1, 1] = 255
    output = double_conv(x)
    expected = torch.tensor([[[[1020, 1530, 1020],
                               [1530, 2295, 1530],
                               [1020, 1530, 1020]]]])
    
    assert torch.equal(output, expected)

def test_double_convolution_5x5dot_zeros_weights():
    """Check the result of a DoubleConvolution block with weights=0 applied to a 5x5 image
    
    GIVEN: a 5x5 image tensor with a single dot (255) in the middle and zero elsewhere.
           A DoubleConvolution block with weights initialized to zero
    WHEN: the DoubleConvolution block is applied to the 5x5 image tensor
    THEN: the output is a 5x5 tensor with zeros
    """

    double_conv = DoubleConvolution(in_channels=1, out_channels=1)
    double_conv.double_conv[0].weight.data.fill_(0)
    double_conv.double_conv[3].weight.data.fill_(0)

    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 255
    output = double_conv(x)
    expected = torch.zeros(1, 1, 5, 5)
    
    assert torch.equal(output, expected)

def test_double_convolution_5x5dot_ones_weights():
    """Check the result of a DoubleConvolution block with weights=1 applied to a 5x5 image
    For testing purposes, the BatchNorm2d layers are removed (replaced by Identity layers)
    
    GIVEN: a 5x5 image tensor with a single dot in the middle and zero elsewhere.
           A DoubleConvolution block with weights initialized to ones and without batch normalization
    WHEN: the DoubleConvolution block is applied to the 5x5 image tensor
    THEN: the output is a 5x5 tensor with the same values as the input
    """

    double_conv = DoubleConvolution(in_channels=1, out_channels=1)
    double_conv.double_conv[0].weight.data.fill_(1)
    double_conv.double_conv[3].weight.data.fill_(1)
    double_conv.double_conv[1] = Identity()
    double_conv.double_conv[4] = Identity()

    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 255
    output = double_conv(x)
    expected = torch.tensor([[[[255,  510,  765,  510, 255],
                               [510, 1020, 1530, 1020, 510],
                               [765, 1530, 2295, 1530, 765],
                               [510, 1020, 1530, 1020, 510],
                               [255,  510,  765,  510, 255]]]])
    
    assert torch.equal(output, expected)



############## Test UNet class ##############
# Test UNet constructor (init method)

def test_negative_in_channels():
    """Check if a ValueError is raised when the input channels are negative
    
    GIVEN: in_channels is negative
    WHEN: the UNet class is initialized with the negative in_channels
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = UNet(in_channels=-3, out_channels=1, features=[64, 128, 256, 512])

def test_negative_out_channels():
    """Check if a ValueError is raised when the output channels are negative

    GIVEN: out_channels is negative
    WHEN: the UNet class is initialized with the negative out_channels
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = UNet(in_channels=3, out_channels=-1, features=[64, 128, 256, 512])

def test_min_features():
    """Check if a ValueError is raised when the features list length is less than 2

    GIVEN: the features list has only one element
    WHEN: the UNet class is initialized with the features list
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = UNet(in_channels=3, out_channels=1, features=[64])

def test_negative_features():
    """Check if a ValueError is raised when the features list contains negative values

    GIVEN: the features list contains negative values
    WHEN: the UNet class is initialized with the features list
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    with pytest.raises(ValueError):
        _ = UNet(in_channels=3, out_channels=1, features=[-64, 128, 256, 512])

# Test UNet forward method


@given(batch = st.integers(min_value=1, max_value=16),
       channels = st.integers(min_value=1, max_value=5),
       width = st.integers(min_value=32, max_value=128),
       height = st.integers(min_value=32, max_value=128))
@settings(deadline=None)
@example(batch = 1, channels = 3, width = 128, height = 128)
def test_forward_correct_channel(batch, channels, width, height):
    """Check if the output has the correct number of channels
    
    GIVEN: batch, channels, width, height are integers
    WHEN: the UNet model is applied to a tensor
    THEN: the output has the correct number of channels
    """

    make_test_deterministic()
    model = UNet(in_channels=channels)
    x = torch.randn(batch, channels, width, height)
    output = model(x)
    assert output.shape[1] == 1

@given(batch = st.integers(min_value=1, max_value=16),
       channels = st.integers(min_value=1, max_value=5),
       width = st.integers(min_value=32, max_value=128),
       height = st.integers(min_value=32, max_value=128))
@settings(deadline=None)
@example(batch = 1, channels = 3, width = 128, height = 128)
@example(batch = 1, channels = 3, width = 161, height = 161)    # odd dimension input tensor
def test_forward_same_size(batch, channels, width, height):
    """Check if the output has the same size as the input tensor

    GIVEN: batch, channels, width, height are integers
    WHEN: the UNet model is applied to a tensor
    THEN: the output size (width, height) matches the input size
    """

    make_test_deterministic()
    model = UNet(in_channels=channels)
    x = torch.randn(batch, channels, width, height)
    output = model(x)
    assert output.shape[2:] == x.shape[2:]

def test_forward_small_input():
    """Check if a ValueError is raised when input tensor sizes is less then 2^(len(features)+1)
    This is because the tensor sizes are halved in each downsampling step.

    GIVEN: the input tensor size is 1x1
    WHEN: the UNet model is applied to the input tensor
    THEN: a ValueError is raised, as PyTorch does not raise an error
    """

    make_test_deterministic()
    model = UNet()
    x = torch.randn(1, 3, 1, 1)
    with pytest.raises(ValueError):
        model(x)


if __name__ == '__main__':
    pass