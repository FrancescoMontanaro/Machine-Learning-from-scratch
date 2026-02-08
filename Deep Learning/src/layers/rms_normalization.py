import numpy as np

from ..core import Tensor, Module
from ..core.utils.constants import EPSILON


class RMSNorm(Module):
    
    ### Magic methods ###

    def __init__(self, epsilon: float = EPSILON, *args, **kwargs) -> None:
        """
        Initialize the RMS normalization layer.
        
        Parameters:
        - epsilon (float): A small value to avoid division by zero. Default is EPSILON.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        
        # Initialize the scale parameter
        self.gamma: Tensor
     
     
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the RMS normalization layer.
        
        Parameters:
        - x (Tensor): The input tensor. Shape: (Batch size, ..., Features)
        
        Returns:
        - Tensor: The normalized tensor.
        """
        
        # Compute the RMS value
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.epsilon) ** 0.5

        # Scale and return the normalized tensor
        return self.gamma * (x / rms)

    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, ..., Features)
        """
        
        # Extract the shape of the parameters: all except the batch dimension
        feature_shape = (x.shape[-1],)
        
        # Initialize the scale parameter
        self.gamma = Tensor(
            data = np.ones(feature_shape),
            requires_grad = True,
            is_parameter = True
        )