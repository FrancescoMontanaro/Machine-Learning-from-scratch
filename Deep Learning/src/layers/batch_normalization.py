import numpy as np
from typing import Optional

from ..core import Tensor, Module


class BatchNormalization(Module):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the batch normalization layer.
        
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
        
        # Initialize the mean and variance parameters
        self.running_mean: Tensor
        self.running_var: Tensor

     
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the batch normalization layer.
        
        Parameters:
        - x (Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        Returns:
        - Tensor: The normalized tensor.
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized or the running mean and variance are not set.
        """
        
        # Store the input dimension
        self.input_shape = x.shape()
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params()
        
        # Assert that the gamma and beta parameters are initialized and the running mean and variance are set
        assert isinstance(self.gamma, Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.running_mean, Tensor), "Running mean is not initialized. Please call the layer with input data."
        assert isinstance(self.running_var, Tensor), "Running variance is not initialized. Please call the layer with input data."
        
        # Determine axes to compute mean and variance: all except the last dimension (features)
        axes = tuple(range(len(x.shape()) - 1))
        
        # The layer is in training phase
        if self.training:
            # Calculate batch mean and variance
            batch_mean = x.mean(axis=axes, keepdims=True)
            batch_var = x.var(axis=axes, keepdims=True)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
           
        # The layer is in inference phase 
        else:
            # Use running statistics for normalization
            mean = self.running_mean
            var = self.running_var
            
        # Normalize the input
        self.X_centered = x - mean
        self.stddev_inv = 1 / (var + self.epsilon).sqrt()
        self.X_norm = self.X_centered * self.stddev_inv
            
        # Scale and shift the normalized input
        return self.gamma * self.X_norm + self.beta
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Return the input shape
        return self.input_shape
    
    
    def init_params(self) -> None:
        """
        Method to initialize the parameters of the layer
        
        Raises:
        - AssertionError: If the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with input data."
        
        # Extract the shape of the parameters
        num_features = self.input_shape[-1]
        
        # Initialize the scale parameter
        self.gamma = Tensor(
            data = np.ones(num_features),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the shift parameter
        self.beta = Tensor(
            data = np.zeros(num_features),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the running mean and variance
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(num_features), requires_grad=False, is_parameter=False)
        
        # Call the parent class method to set the layer as initialized
        super().init_params()