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
class LocalResponseNormConfig:
    """
    Configuration for Local Response Normalization (LRN) layer.
    
    Attributes:
    - size (int): Size of the local neighborhood (number of adjacent channels).
    - alpha (float): Scaling parameter.
    - beta (float): Exponent.
    - k (float): Bias parameter.
    """
    
    size: int = 5
    alpha: float = 1e-4
    beta: float = 0.75
    k: float = 1.0
    

@dataclass
class MaxPoolConfig:
    """
    Configuration for max pooling layers.
    
    Attributes:
    - size (Tuple[int, int]): Size of the pooling window.
    - stride (Tuple[int, int]): Stride of the pooling operation.
    - padding (Literal["same", "valid"]): Padding type, either 'same' or 'valid'.
    """
    
    size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (2, 2)
    padding: Literal["same", "valid"] = "valid"


@dataclass
class AlexNetConfig:
    """
    Configuration for the AlexNet architecture.
    
    Attributes:
    - num_classes (int): Number of output classes (default: 1000).
    - dropout_rate (float): Dropout rate for the dropout layer (default: 0.1).
    - conv_layers (List[Conv2DConfig]): List of configurations for convolutional layers.
    - dense_layers (List[DenseConfig]): List of configurations for dense layers.
    - local_response_norm (Optional[LocalResponseNormConfig]): Configuration for local response normalization layer.
    - max_pool (PoolConfig): Configuration for pooling layers.
    """
    
    # General parameters
    num_classes: int = 1000
    dropout_rate: float = 0.1
    
    # Layers configurations
    conv_layers: List[Conv2DConfig] = field(default_factory=list)
    dense_layers: List[DenseConfig] = field(default_factory=list)
    local_response_norm: Optional[LocalResponseNormConfig] = None
    max_pool: Optional[MaxPoolConfig] = None
    
    
    # Post-initialization checks and default values
    def __post_init__(self) -> None:
        """
        Post-initialization checks to ensure the configuration is valid and set default values.
        
        Raises:
        - ValueError: If the number of convolutional layers is not exactly 5.
        - ValueError: If the number of dense layers is not at least 2.
        """
        
        # Set default convolutional layers if empty
        if not self.conv_layers:
            self.conv_layers = [
                Conv2DConfig(num_filters=96, kernel_size=(11, 11), stride=(4, 4), padding="valid"),
                Conv2DConfig(num_filters=256, kernel_size=(5, 5), stride=(1, 1)),
                Conv2DConfig(num_filters=384, kernel_size=(3, 3), stride=(1, 1)),
                Conv2DConfig(num_filters=384, kernel_size=(3, 3), stride=(1, 1)),
                Conv2DConfig(num_filters=256, kernel_size=(3, 3), stride=(1, 1))
            ]
            
        # Set default dense layers if empty
        if not self.dense_layers:
            self.dense_layers = [
                DenseConfig(num_units=4096),
                DenseConfig(num_units=4096)
            ]
            
        # Set default local response normalization if not provided
        if self.local_response_norm is None:
            self.local_response_norm = LocalResponseNormConfig()
            
        # Set default pool config if not provided
        if self.max_pool is None:
            self.max_pool = MaxPoolConfig()
        
        # Check if the number of convolutional layers is at exactly 5
        if len(self.conv_layers) != 5:
            raise ValueError("AlexNet requires exactly 5 convolutional layers.")
        
        # Check if the number of dense layers is at least 2
        if len(self.dense_layers) != 2:
            raise ValueError("AlexNet requires at least 2 dense layers.")