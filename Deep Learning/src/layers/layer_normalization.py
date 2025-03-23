import numpy as np
from typing import Optional

from ..core import Tensor, Module


class LayerNormalization(Module):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the layer normalization layer.
        
        Parameters:
        - epsilon (float): The epsilon parameter for numerical stability.
        - momentum (float): The momentum parameter for the moving average.
        - name (str): The name of the layer.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize the scale and shift parameters
        self.gamma: Tensor
        self.beta: Tensor
     
     
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer normalization layer.
        
        Parameters:
        - x (Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        Returns:
        - Tensor: The normalized tensor.
        """
        
        # Store the input dimension
        self.input_shape = x.shape()
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params()
        
        # Extract the axis along which to compute the mean and variance: all except the batch dimension
        axes = tuple(range(1, len(x.shape())))
        
        # Compute mean and variance along the feature dimension for each sample
        layer_mean = x.mean(axis=axes, keepdims=True)
        layer_var = x.var(axis=axes, keepdims=True, ddof=0)
        
        # Scale and shift
        return self.gamma * ((x - layer_mean) * (1 / (layer_var + self.epsilon).sqrt())) + self.beta
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # The output shape is the same as the input shape
        return self.input_shape

    
    def init_params(self) -> None:
        """
        Method to initialize the parameters of the layer
        
        Raises:
        - AssertionError: If the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with input data."
        
        # Extract the shape of the parameters: all except the batch dimension
        feature_shape = self.input_shape[1:]
        
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
        
        # Call the parent class method to set the layer as initialized
        super().init_params()