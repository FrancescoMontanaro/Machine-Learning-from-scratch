from dataclasses import dataclass, field
from typing import Tuple, List, Literal, Optional

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
    - conv (List[Conv2DConfig]): Configuration for additional convolutional layers.
    - fc (List[DenseConfig]): Configuration for the fully connected layers.
    """

    conv: List[Conv2DConfig] = field(default_factory=list)
    fc: List[DenseConfig] = field(default_factory=list)


@dataclass
class VAEDecoderConfig:
    """
    Configuration for the VAE Decoder architecture.
    
    Attributes:
    - fc (List[DenseConfig]): Configuration for the fully connected layers.
    - deconv (List[ConvTranspose2DConfig]): Configuration for additional transposed convolutional layers.
    """
    
    fc: List[DenseConfig] = field(default_factory=list)
    deconv: List[ConvTranspose2DConfig] = field(default_factory=list)


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