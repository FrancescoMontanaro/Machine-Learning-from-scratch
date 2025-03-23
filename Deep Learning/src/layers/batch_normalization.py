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
        
        # Initialize the running mean and variance
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
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True, ddof=0)
            
            # Update running mean and variance
            self.running_mean.data = self.running_mean.data * (1 - self.momentum) + mean.data * self.momentum
            self.running_var.data = self.running_var.data * (1 - self.momentum) + var.data * self.momentum
            
            # Normalize and return the output
            return (x - mean) / (var + self.epsilon).sqrt() * self.gamma + self.beta
           
        # The layer is in inference phase 
        else:
            # Use running statistics for normalization
            return self.gamma * ((x - self.running_mean) * (1 / (self.running_var + self.epsilon).sqrt())) + self.beta
    
    
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
        running_mean = Tensor(np.zeros(num_features), requires_grad=False, is_parameter=False)
        running_var = Tensor(np.ones(num_features), requires_grad=False, is_parameter=False)
        
        # Store the running mean and variance as buffers
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)
        
        # Call the parent class method to set the layer as initialized
        super().init_params()