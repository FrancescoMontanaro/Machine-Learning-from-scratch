import numpy as np

from ..core import Tensor, Module


class LayerNormalization(Module):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, *args, **kwargs) -> None:
        """
        Initialize the layer normalization layer.
        
        Parameters:
        - epsilon (float): The epsilon parameter for numerical stability.
        - momentum (float): The momentum parameter for the moving average.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize the scale and shift parameters
        self.gamma: Tensor
        self.beta: Tensor
     
     
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the layer normalization layer.
        
        Parameters:
        - x (Tensor): The input tensor. Shape: (Batch size, ..., Features)
        
        Returns:
        - Tensor: The normalized tensor.
        """
        
        # Compute mean and variance along the feature dimension for each sample
        layer_mean = x.mean(axis=-1, keepdims=True)
        layer_var = x.var(axis=-1, keepdims=True, ddof=0)
        
        # Scale and shift
        return self.gamma * ((x - layer_mean) * (1 / (layer_var + self.epsilon).sqrt())) + self.beta

    
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
        
        # Initialize the shift parameter
        self.beta = Tensor(
            data = np.zeros(feature_shape),
            requires_grad = True,
            is_parameter = True
        )