from dataclasses import dataclass
from typing import Tuple, Literal, Optional

from ...activations import Activation, ReLU


@dataclass
class Conv2DConfig:
    """
    Configuration for a Conv2D layer.
    
    Attributes:
    - num_filters (int): Number of filters in the convolutional layer.
    - kernel_size (Tuple[int, int]): Size of the convolutional kernel.
    - stride (Tuple[int, int]): Stride of the convolution.
    - padding (Literal["same", "valid"]): Padding type, either 'same' or 'valid'.
    - activation (Activation): Activation function to use (default: ReLU).
    """
    
    num_filters: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Literal["same", "valid"] = "same"
    activation: Optional[Activation] = ReLU()


@dataclass
class ConvTranspose2DConfig:
    """
    Configuration for a ConvTranspose2D layer.
    
    Attributes:
    - num_filters (int): Number of filters in the transposed convolutional layer.
    - kernel_size (Tuple[int, int]): Size of the convolutional kernel.
    - stride (Tuple[int, int]): Stride of the convolution.
    - padding (Tuple[int, int]): Padding applied to reduce output size.
    - output_padding (Tuple[int, int]): Additional size added to output.
    - activation (Activation): Activation function to use (default: ReLU).
    """

    num_filters: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    output_padding: tuple[int, int] = (0, 0)
    activation: Optional[Activation] = ReLU()


@dataclass
class DenseConfig:
    """
    Configuration for a Dense layer.
    
    Attributes:
    - num_units (int): Number of units in the dense layer.
    - activation (Activation): Activation function to use (default: ReLU).
    """
    
    num_units: int
    activation: Optional[Activation] = ReLU()


@dataclass
class VAEEncoderConfig:
    """
    Configuration for the VAE Encoder architecture.
    
    Attributes:
    - conv_1 (Conv2DConfig): Configuration for the first convolutional layer.
    - conv_2 (Conv2DConfig): Configuration for the second convolutional layer.
    - fc (DenseConfig): Configuration for the fully connected layer.
    """

    conv_1: Conv2DConfig
    conv_2: Conv2DConfig
    fc: DenseConfig


@dataclass
class VAEDecoderConfig:
    """
    Configuration for the VAE Decoder architecture.
    
    Attributes:
    - fc (DenseConfig): Configuration for the fully connected layer (activation only, units computed automatically).
    - deconv_1 (ConvTranspose2DConfig): Configuration for the first transposed convolutional layer.
    - deconv_2 (ConvTranspose2DConfig): Configuration for the second transposed convolutional layer.
    - deconv_3 (ConvTranspose2DConfig): Configuration for the third transposed convolutional layer.
    """
    
    fc: DenseConfig
    deconv_1: ConvTranspose2DConfig
    deconv_2: ConvTranspose2DConfig
    deconv_3: ConvTranspose2DConfig


@dataclass
class VAEConfig:
    """
    Configuration for the Variational Autoencoder (VAE) architecture.
    
    Attributes:
    - encoder (VAEEncoderConfig): Configuration for the encoder part of the VAE.
    - latent_dim (int): Dimensionality of the latent space.
    - decoder (VAEDecoderConfig): Configuration for the decoder part of the VAE
    """
    
    encoder: VAEEncoderConfig
    latent_dim: int
    decoder: VAEDecoderConfig